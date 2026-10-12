#!/usr/bin/env python3
"""C5-B two-step Code-only OPD Actor update on frozen fresh C5-B Agent data.

Both rollout groups were sampled before this process. Step 1 has LR=0, so
the Step 2 group is still on-policy at its update: Student parameters have
not changed between either group's sampling and the Step 2 update.
"""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, reduce_metrics
from experiments.ExecOPD.agent.c2c_tensor import tensorize, batch_pad
from experiments.ExecOPD.agent.c5b_worker import C5BWorker
from experiments.ExecOPD.agent.run_c5a_credit_diagnostic import assign_credit

REPO = REPO_ROOT
BASE = REPO / 'experiments/ExecOPD'
REPORTS = BASE / 'reports'
TMP = BASE / 'agent/tmp/c5b'
MODEL = STUDENT_MODEL
DATA = TRAIN_PARQUET
PYTHON = PYTHON_EXE
K = 16


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def gate(condition, status, detail):
    if not condition:
        raise RuntimeError(status + ': ' + detail)


def rank_zero(records):
    gate(isinstance(records, list) and len(records) == 2,
         'C5_B_RUNTIME_BLOCKED', 'two distributed rank outputs required')
    chosen = [x for x in records if x['rank'] == 0]
    gate(len(chosen) == 1, 'C5_B_RUNTIME_BLOCKED', 'rank-zero output missing')
    return chosen[0]


def config(slice_path):
    cfg = OmegaConf.load(REPO / 'verl/trainer/config/ppo_trainer.yaml')
    values = {
        'data.train_files': str(slice_path),
        'data.val_files': str(BASE / 'c1/data/c1b_val.parquet'),
        'data.train_batch_size': 1,
        'data.max_prompt_length': 512,
        'data.max_response_length': 512,
        'data.shuffle': False,
        'actor_rollout_ref.model.path': str(MODEL),
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 1,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.ppo_epochs': 1,
        'actor_rollout_ref.actor.optim.lr': 5e-7,
        'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.entropy_coeff': 0.001,
        'actor_rollout_ref.actor.loss_agg_mode': 'token-mean',
        'actor_rollout_ref.actor.fsdp_config.param_offload': True,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.rollout.n': 4,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 2,
        'actor_rollout_ref.rollout.gpu_memory_utilization': 0.15,
        'actor_rollout_ref.rollout.max_model_len': 2048,
        'actor_rollout_ref.rollout.max_num_batched_tokens': 2048,
        'actor_rollout_ref.rollout.temperature': 1.0,
        'actor_rollout_ref.rollout.top_p': 1.0,
        'actor_rollout_ref.rollout.top_k': -1,
        'algorithm.adv_estimator': 'token_reward_direct',
        'algorithm.use_kl_in_reward': False,
        'reward_model.enable': False,
        'trainer.total_epochs': 1,
        'trainer.total_training_steps': 2,
        'trainer.n_gpus_per_node': 2,
        'trainer.nnodes': 1,
        'trainer.logger': ['console'],
        'trainer.save_freq': -1,
        'trainer.test_freq': -1,
        'trainer.resume_mode': 'disable',
        'trainer.project_name': 'ExecOPD',
        'trainer.experiment_name': 'c5b_credit_aware_opd_two_step',
    }
    for key, value in values.items():
        OmegaConf.update(cfg, key, value)
    OmegaConf.update(cfg, 'trainer.val_before_train', False, force_add=True)
    gate(cfg.algorithm.adv_estimator == 'token_reward_direct' and
         not cfg.actor_rollout_ref.actor.use_kl_loss and
         cfg.actor_rollout_ref.rollout.n == 4 and
         cfg.trainer.total_training_steps == 2,
         'C5_B_RUNTIME_BLOCKED', 'Vanilla OPD configuration changed')
    write(REPORTS / 'c5b_effective_config.json', OmegaConf.to_container(cfg, resolve=True))
    return cfg


def load_evidence(tokenizer):
    trajectory_path = REPORTS / 'c5b_trajectories.jsonl'
    student = json.loads((REPORTS / 'c5b_student_topk.json').read_text())
    teacher = json.loads((REPORTS / 'c5b_teacher_scores.json').read_text())
    credit_file = json.loads((REPORTS / 'c5b_turn_credit.json').read_text())
    gate(credit_file['status'] == 'PASS' and credit_file['rule_source'] == 'C5-A unchanged',
         'C5_B_RUNTIME_BLOCKED', 'turn credit capture incomplete')
    gate(student['k'] == teacher['k'] == K and student['trajectory_count'] == 8,
         'C5_B_RUNTIME_BLOCKED', 'Top-K cache wrong shape')
    gate(hashlib.sha256(trajectory_path.read_bytes()).hexdigest() ==
         student['trajectories_sha256'] == teacher['trajectories_sha256'],
         'C5_B_RUNTIME_BLOCKED', 'fresh trajectory hashes changed')
    gate(student['trajectories_sha256'] != hashlib.sha256(
         (REPORTS / 'c3b_trajectories.jsonl').read_bytes()).hexdigest(),
         'C5_B_RUNTIME_BLOCKED', 'C3-B frozen trajectories cannot be reused')
    trajectories = [json.loads(x) for x in trajectory_path.read_text().splitlines()]
    gate(len(trajectories) == len(student['records']) == len(teacher['records']) == 8,
         'C5_B_RUNTIME_BLOCKED', 'expected exactly eight fresh Agent trajectories')
    gate(student['trajectories_sha256'] != hashlib.sha256(
         (REPORTS / 'c4b4_trajectories.jsonl').read_bytes()).hexdigest(),
         'C5_B_RUNTIME_BLOCKED', 'C4-B4 frozen trajectories cannot be reused')
    recorded_credit = {(r['global_step'], r['trajectory_index'], r['turn_id']): r
                       for r in credit_file['turns']}
    gate(len(recorded_credit) == credit_file['turn_count'] ==
         sum(len(t['turns']) for t in trajectories),
         'C5_B_RUNTIME_BLOCKED', 'saved turn credit count mismatch')
    for i, (t, s, r) in enumerate(zip(trajectories, student['records'], teacher['records'])):
        gate(t['global_step'] == s['global_step'] == r['global_step'] == i // 4 + 1 and
             t['trajectory_index'] == s['trajectory_index'] == r['trajectory_index'] == i % 4,
             'C5_B_RUNTIME_BLOCKED', 'trajectory order differs')
        item, detail = tensorize(tokenizer, t)
        for turn in t['turns']:
            j = turn['turn_id']
            computed, reason = assign_credit(
                t['turns'], j, t['state']['terminal_reason'] == 'PASS')
            saved = recorded_credit[(t['global_step'], t['trajectory_index'], j)]
            gate(saved['credit'] == computed and saved['credit_reason'] == reason,
                 'C5_B_MASK_BLOCKED', 'saved turn credit differs from C5-A rule')
        gate(s['input_ids_sha256'] == r['input_ids_sha256'] ==
             hashlib.sha256(item['input_ids'].numpy().tobytes()).hexdigest() and
             detail['generated_count'] == r['teacher_scored_count'] == len(s['positions']) == len(r['positions']) and
             detail['code_count'] == s['code_count'] == r['code_count'] ==
             r['teacher_scored_code_count'] and r['noncode_target_count'] == 0 and
             detail['code_count'] > 0,
             'C5_B_MASK_BLOCKED', 'same-state or target count mismatch')
        for a, b in zip(s['positions'], r['positions']):
            gate(a['position'] == b['position'] and a['turn_id'] == b['turn_id'] and
                 a['student_topk_ids'] == b['gather_ids'] and
                 int(item['generated_token_mask'][a['position']]) == 1 and
                 int(item['execution_observation_mask'][a['position']]) == 0 and
                 b['code_target'] == bool(item['code_token_mask'][a['position']]) and
                 (any(v != 0 for v in b['opd_signal']) if b['code_target'] else
                  all(v == 0 for v in b['opd_signal'])),
                 'C5_B_MASK_BLOCKED', 'teacher/Student Top-K state-ID alignment failure')
    return trajectories, student['records'], teacher['records']


class Trainer(RayPPOTrainer):
    def __init__(self, config, *args, **kwargs):
        # Code-R1's older RayPPOTrainer constructor only recognizes GAE/GRPO
        # roles. Use GRPO solely for its no-Critic worker setup; fit() below
        # computes the exact Rethinking-OPD token_reward_direct advantage.
        gate(config.algorithm.adv_estimator == 'token_reward_direct',
             'C5_B_RUNTIME_BLOCKED', 'direct OPD estimator not selected')
        config.algorithm.adv_estimator = 'grpo'
        try:
            super().__init__(config, *args, **kwargs)
        finally:
            config.algorithm.adv_estimator = 'token_reward_direct'

    def fit(self):
        gate(self.total_training_steps == 2 and not self.use_critic and not self.use_rm and
             not self.use_reference_policy and self.val_reward_fn is None,
             'C5_B_RUNTIME_BLOCKED', 'unexpected trainer role enabled')
        tokenizer = self.tokenizer
        trajectories, student, teacher = load_evidence(tokenizer)
        snapshots, loss_records, audits, advantage_records, steps = {}, [], [], [], []

        def snapshot(phase):
            rec = rank_zero(self.actor_rollout_wg.c5b_parameters(phase))
            snapshots[phase] = rec
            write(REPORTS / 'c5b_parameter_snapshots.json', snapshots)
            return rec

        def lr_audit():
            records = self.actor_rollout_wg.c5b_lr_audit()
            gate(len(records) == 2 and records[0]['lrs'] == records[1]['lrs'] and
                 records[0]['scheduler_last_epoch'] == records[1]['scheduler_last_epoch'],
                 'C5_B_RUNTIME_BLOCKED', 'rank LR/scheduler disagreement')
            return records[0]

        snapshot('before_step1')
        gate(lr_audit()['lrs'] == [0.0], 'C5_B_RUNTIME_BLOCKED', 'Step 1 LR must be 0')
        for step in (1, 2):
            subset_t = trajectories[(step - 1) * 4:step * 4]
            subset_s = student[(step - 1) * 4:step * 4]
            subset_r = teacher[(step - 1) * 4:step * 4]
            items, alignments = [], []
            for trajectory in subset_t:
                item, detail = tensorize(tokenizer, trajectory)
                items.append(item)
                alignments.append(detail)
            tensors, prompt_width, response_len = batch_pad(tokenizer, items, alignments)
            generated_mask = tensors['generated_token_mask'][:, -response_len:]
            response_mask = tensors['code_token_mask'][:, -response_len:]
            tool_mask = tensors['execution_observation_mask'][:, -response_len:]
            noncode_mask = generated_mask.bool() & ~response_mask.bool()
            credit_full = torch.zeros(tensors['input_ids'].shape, dtype=torch.float32)
            turn_credits = []
            for row, (trajectory, item, detail) in enumerate(zip(subset_t, items, alignments)):
                left = prompt_width - detail['initial_prompt_length']
                by_turn = {}
                for turn in trajectory['turns']:
                    j = turn['turn_id']
                    value, reason = assign_credit(
                        trajectory['turns'], j,
                        trajectory['state']['terminal_reason'] == 'PASS')
                    gate(value in (0.0, 0.25, 0.5, 1.0),
                         'C5_B_MASK_BLOCKED', 'undefined turn credit')
                    by_turn[j] = value
                    turn_credits.append({'trajectory_index': trajectory['trajectory_index'],
                                         'turn_id': j, 'credit': value,
                                         'reason': reason,
                                         'execution_status': turn['execution_observation']['status']})
                for position in torch.nonzero(item['code_token_mask'], as_tuple=False).flatten().tolist():
                    j = int(item['turn_ids'][position])
                    gate(j in by_turn, 'C5_B_MASK_BLOCKED', 'code token without turn credit')
                    credit_full[row, left + position] = by_turn[j]
            credit_response = credit_full[:, -response_len:]
            gate(bool(torch.all(credit_full[~tensors['code_token_mask'].bool()] == 0)) and
                 not bool(torch.any(credit_full * tensors['execution_observation_mask'])) and
                 not bool(torch.any(credit_full * (tensors['generated_token_mask'] -
                                                  tensors['code_token_mask']))),
                 'C5_B_MASK_BLOCKED', 'credit leaked outside code mask')
            ids = torch.zeros((4, response_len, K), dtype=torch.long)
            student_logp = torch.zeros((4, response_len, K), dtype=torch.float32)
            signal = torch.zeros((4, response_len, K), dtype=torch.float32)
            for row, (item, detail, s, r) in enumerate(zip(items, alignments, subset_s, subset_r)):
                left = prompt_width - detail['initial_prompt_length']
                for src, tgt in zip(s['positions'], r['positions']):
                    padded = left + src['position']
                    j = padded - prompt_width
                    gate(0 <= j < response_len and generated_mask[row, j] == 1 and
                         tool_mask[row, j] == 0 and
                         tgt['code_target'] == bool(response_mask[row, j]),
                         'C5_B_MASK_BLOCKED', 'OPD signal mapped into tool/padding')
                    ids[row, j] = torch.tensor(src['student_topk_ids'], dtype=torch.long)
                    student_logp[row, j] = torch.tensor(src['student_logprobs'], dtype=torch.float32)
                    signal[row, j] = torch.tensor(tgt['opd_signal'], dtype=torch.float32)
            gate(signal.shape == (4, response_len, K) and bool(torch.isfinite(signal).all()) and
                 bool(torch.any(signal[response_mask.bool()] != 0)) and
                 bool(torch.all(response_mask <= generated_mask)) and
                 bool(torch.all(torch.any(signal[response_mask.bool()] != 0, dim=-1))) and
                 not bool(torch.any(signal * (1 - response_mask.unsqueeze(-1)))),
                 'C5_B_MASK_BLOCKED', 'OPD signal nonfinite, zero, or outside Student mask')
            # Original C4-B4 OPD signal stays unchanged. Apply C5-A turn credit
            # once, to the direct token advantage only. The Actor PPO objective
            # and its entropy mask remain those of C4-B4.
            a_opd = signal * response_mask.unsqueeze(-1)
            advantage = a_opd * credit_response.unsqueeze(-1)
            returns = advantage.clone()
            gate(advantage.shape == signal.shape and bool(torch.isfinite(a_opd).all()) and
                 bool(torch.isfinite(advantage).all()) and bool(torch.any(advantage != 0)) and
                 not bool(torch.any(advantage * tool_mask.unsqueeze(-1))) and
                 not bool(torch.any(advantage * noncode_mask.unsqueeze(-1))) and
                 not bool(torch.any(advantage[response_mask.bool() &
                                                 (credit_response == 0)] != 0)),
                 'C5_B_NUMERICAL_BLOCKED', 'credit-adjusted advantage invalid')
            audit = {'step': step, 'trajectory_count': 4,
                     'assistant_turn_count': sum(len(t['turns']) for t in subset_t),
                     'terminal_status_counts': dict(Counter(
                         t['state']['terminal_reason'] for t in subset_t)),
                     'turn_credit_counts': {str(tier): sum(x['credit'] == tier
                         for x in turn_credits) for tier in (0.0, 0.25, 0.5, 1.0)},
                     'generated_tokens': int(generated_mask.sum()),
                     'code_tokens': int(response_mask.sum()),
                     'credit_active_code_tokens': int(torch.count_nonzero(credit_response)),
                     'zero_credit_code_tokens': int((response_mask.bool() &
                         (credit_response == 0)).sum()),
                     'code_to_generated_ratio': float(response_mask.sum() / generated_mask.sum()),
                     'tool_tokens': int(tensors['execution_observation_mask'].sum()),
                     'teacher_scored_tokens': sum(r['teacher_scored_count'] for r in subset_r),
                     'teacher_scored_code_tokens': sum(r['teacher_scored_code_count'] for r in subset_r),
                     'generated_tool_overlap': int((generated_mask & tool_mask).sum()),
                     'code_tool_overlap': int((response_mask & tool_mask).sum()),
                     'tool_target_count': sum(r['tool_target_count'] for r in subset_r),
                     'noncode_target_count': sum(r['noncode_target_count'] for r in subset_r),
                     'tool_opd_signal_count': int(torch.count_nonzero(signal * tool_mask.unsqueeze(-1))),
                     'tool_advantage_count': int(torch.count_nonzero(advantage * tool_mask.unsqueeze(-1))),
                     'noncode_opd_signal_count': int(torch.count_nonzero(signal * noncode_mask.unsqueeze(-1))),
                     'noncode_advantage_count': int(torch.count_nonzero(advantage * noncode_mask.unsqueeze(-1))),
                     'credit_tool_overlap_count': int(torch.count_nonzero(
                         credit_response * tool_mask)),
                     'credit_noncode_overlap_count': int(torch.count_nonzero(
                         credit_response * noncode_mask)),
                     'zero_credit_code_advantage_nonzero_count': int(torch.count_nonzero(
                         advantage[response_mask.bool() & (credit_response == 0)])),
                     'credit_weight_per_token': credit_full.tolist(),
                     'turn_credits': turn_credits,
                     'assistant_turns': [len(t['turns']) for t in subset_t],
                     'boundary_crossing_token_count': sum(
                         int(span.get('boundary_crossing', False))
                         for detail in alignments for span in detail['code_spans']),
                     'per_trajectory': [{'trajectory_index': t['trajectory_index'],
                                         'generated_tokens': x['generated_count'],
                                         'code_tokens': x['code_count'],
                                         'code_to_generated_ratio': x['code_count'] / x['generated_count'],
                                         'tool_context_tokens': x['tool_count'],
                                         'code_boundary_spans': x['code_spans']}
                                        for t, x in zip(subset_t, alignments)],
                     'mask_checks': [x['checks'] for x in alignments]}
            gate(audit['generated_tokens'] == audit['teacher_scored_tokens'] and
                 audit['code_tokens'] == audit['teacher_scored_code_tokens'] and
                 audit['generated_tool_overlap'] == audit['tool_target_count'] ==
                 audit['code_tool_overlap'] == audit['noncode_target_count'] ==
                 audit['tool_opd_signal_count'] == audit['tool_advantage_count'] ==
                 audit['noncode_opd_signal_count'] == audit['noncode_advantage_count'] == 0,
                 'C5_B_MASK_BLOCKED', 'tool target or Student/Teacher count mismatch')
            gate(audit['credit_tool_overlap_count'] ==
                 audit['credit_noncode_overlap_count'] ==
                 audit['zero_credit_code_advantage_nonzero_count'] == 0,
                 'C5_B_MASK_BLOCKED', 'credit or advantage leaked outside active code')
            audits.append(audit)
            write(REPORTS / 'c5b_credit_mask_audit.json', audits)
            code_values = a_opd[response_mask.bool()].reshape(-1).tolist()
            credit_values = advantage[response_mask.bool()].reshape(-1).tolist()
            advantage_records.append({
                'step': step,
                'application': 'A_credit = A_opd * credit_weight_per_token, once',
                'original_a_opd': {'mean': statistics.mean(code_values),
                                   'std_population': statistics.pstdev(code_values),
                                   'nonzero': sum(v != 0 for v in code_values)},
                'credit_a': {'mean': statistics.mean(credit_values),
                             'std_population': statistics.pstdev(credit_values),
                             'nonzero': sum(v != 0 for v in credit_values)},
                'zero_credit_code_advantage_nonzero_count':
                    audit['zero_credit_code_advantage_nonzero_count'],
                'tool_advantage_nonzero_count': audit['tool_advantage_count'],
                'noncode_advantage_nonzero_count': audit['noncode_advantage_count']})
            write(REPORTS / 'c5b_advantage.json', advantage_records)
            tensors.update({'prompts': tensors['input_ids'][:, :prompt_width],
                            'responses': tensors['input_ids'][:, prompt_width:],
                            'response_mask': response_mask,
                            'student_top_k_ids': ids,
                            'student_top_k_log_probs': student_logp,
                            'old_log_probs': student_logp,
                            'token_level_scores': signal,
                            'token_level_rewards': signal,
                            'credit_weight_per_token': credit_full,
                            'advantages': advantage,
                            'returns': returns})
            batch = DataProto(batch=TensorDict(tensors, batch_size=[4]),
                              non_tensor_batch={'uid': np.array([subset_t[0]['task_id']] * 4,
                                                                 dtype=object)},
                              meta_info={'global_token_num': tensors['attention_mask'].sum(-1).tolist(),
                                         'c5b_step': step,
                                         'temperature': 1.0,
                                         'top_k_strategy': 'only_stu',
                                         'reward_weight_mode': 'student_p',
                                         'log_prob_top_k': K})
            print(f'C5B_PHASE=step{step}_actor_update_start', flush=True)
            actor_output = self.actor_rollout_wg.update_actor(batch)
            metrics = reduce_metrics(actor_output.meta_info['metrics'])
            gate(all(np.isfinite(v) for v in metrics.values()) and
                 metrics.get('c5b/tool_loss_overlap') == 0 and
                 metrics.get('c5b/noncode_assistant_loss_overlap') == 0 and
                 metrics.get('c5b/zero_credit_code_advantage_nonzero') == 0 and
                 metrics.get('c5b/entropy_mask_unweighted_code_only') == 1 and
                 metrics.get('c5b/any_nonzero_gradient') > 0 and
                 metrics.get('actor/grad_norm') > 0,
                 'C5_B_NUMERICAL_BLOCKED', 'actor loss/gradient nonfinite or tool loss')
            expected = 0.0 if step == 1 else 5e-7
            gate(metrics['c5b/actual_optimizer_lr'] == expected and
                 metrics['c5b/optimizer_step_count'] == step,
                 'C5_B_RUNTIME_BLOCKED', 'actual LR/optimizer step mismatch')
            after = snapshot('after_step' + str(step))
            changed = any(p['changed'] for p in after['parameters'].values())
            gate(changed == (step == 2), 'C5_B_PARAMETER_BLOCKED',
                 'Step 1 must be unchanged and Step 2 must update Student')
            next_lr = lr_audit()
            gate(next_lr['optimizer_step_count'] == step and next_lr['lrs'] == [5e-7],
                 'C5_B_RUNTIME_BLOCKED', 'scheduler progression mismatch')
            values = signal[response_mask.bool()].reshape(-1).tolist()
            record = {'step': step, 'generated_tokens': audit['generated_tokens'],
                      'code_tokens': audit['code_tokens'],
                      'credit_active_code_tokens': audit['credit_active_code_tokens'],
                      'assistant_turn_count': audit['assistant_turn_count'],
                      'terminal_status_counts': audit['terminal_status_counts'],
                      'turn_credit_counts': audit['turn_credit_counts'],
                      'code_to_generated_ratio': audit['code_to_generated_ratio'],
                      'tool_tokens': audit['tool_tokens'],
                      'teacher_scored_tokens': audit['teacher_scored_tokens'],
                      'teacher_scored_code_tokens': audit['teacher_scored_code_tokens'],
                      'opd_signal': {'mean': statistics.mean(values),
                                     'std_population': statistics.pstdev(values),
                                     'abs_mean': statistics.mean(abs(x) for x in values)},
                      'advantage': {'nonzero': int(torch.count_nonzero(advantage)),
                                    'mean': float(advantage[response_mask.bool()].mean()),
                                    'std': float(advantage[response_mask.bool()].std(unbiased=False))},
                      'a_opd': advantage_records[-1]['original_a_opd'],
                      'a_credit': advantage_records[-1]['credit_a'],
                      'entropy': {'mask': 'code_token_mask',
                                  'credit_weight_applied': False,
                                  'entropy_loss_metric': metrics['actor/entropy_loss'],
                                  'zero_credit_code_tokens_still_in_entropy_mask':
                                      audit['zero_credit_code_tokens']},
                      'actor_metrics': metrics, 'lr_audit': next_lr,
                      'parameter_changed': changed,
                      'tool_loss_count': int(metrics['c5b/tool_loss_overlap']),
                      'noncode_assistant_loss_count': int(
                          metrics['c5b/noncode_assistant_loss_overlap'])}
            loss_records.append(record)
            write(REPORTS / 'c5b_loss.json', loss_records)
            steps.append({'step': step, 'fresh_trajectories': 4,
                          'assistant_turn_count': audit['assistant_turn_count'],
                          'generated_tokens': audit['generated_tokens'],
                          'code_tokens': audit['code_tokens'],
                          'credit_active_code_tokens': audit['credit_active_code_tokens'],
                          'teacher_scored_tokens': audit['teacher_scored_tokens'],
                          'teacher_scored_code_tokens': audit['teacher_scored_code_tokens'],
                          'actual_lr': expected, 'parameter_changed': changed})
            print('C5B_STEP_COMPLETE=' + json.dumps(steps[-1]), flush=True)
        return {'status': 'C5_B_PASS', 'global_steps': 2, 'trajectories': 8,
                'student_parameter_updated_step2': True,
                'teacher_parameter_unchanged': True,
                'teacher_generation_count': 0,
                'tool_target_count': 0,
                'tool_loss_count': 0,
                'noncode_assistant_loss_count': 0,
                'backward_steps': 2,
                'steps': steps}


def main():
    gate(sys.executable == PYTHON and os.environ.get('PYTHONPATH') == str(REPO),
         'C5_B_RUNTIME_BLOCKED', 'wrong isolated Python/PYTHONPATH')
    gate(os.environ.get('CUDA_VISIBLE_DEVICES') == '2,3',
         'C5_B_GPU_BLOCKED', 'only preflighted physical GPUs 2,3 permitted')
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), use_fast=True, local_files_only=True)
    gate(tokenizer.is_fast, 'C5_B_RUNTIME_BLOCKED', 'Student tokenizer not fast')
    fixed = json.loads((REPORTS / 'c2a_fixed_tasks.json').read_text())
    gate(hashlib.sha256(DATA.read_bytes()).hexdigest() == fixed['source_file_sha256'],
         'C5_B_RUNTIME_BLOCKED', 'fixed Code-R1 task source changed')
    slice_path = BASE / 'agent/data/c2c_row2_train.parquet'
    gate(slice_path.exists() and len(pd.read_parquet(slice_path)) == 1,
         'C5_B_RUNTIME_BLOCKED', 'fixed one-row task slice absent')
    cfg = config(slice_path)
    ray.init(num_gpus=2, include_dashboard=False,
             runtime_env={'env_vars': {'PYTHONNOUSERSITE': '1', 'PYTHONPATH': str(REPO),
                                      'TOKENIZERS_PARALLELISM': 'false', 'NCCL_DEBUG': 'WARN'}})
    try:
        from verl.utils import hf_tokenizer, hf_processor
        mapping = {Role.ActorRollout: ray.remote(C5BWorker)}
        pool = ResourcePoolManager({'c5b_pool': [2]}, {Role.ActorRollout: 'c5b_pool'})
        trainer = Trainer(cfg, hf_tokenizer(str(MODEL)), mapping, pool, RayWorkerGroup,
                          processor=hf_processor(str(MODEL), use_fast=True),
                          reward_fn=None, val_reward_fn=None)
        print('C5B_PHASE=init_workers_start', flush=True)
        trainer.init_workers()
        print('C5B_PHASE=init_workers_complete', flush=True)
        return trainer.fit()
    finally:
        ray.shutdown()


if __name__ == '__main__':
    start = time.time()
    try:
        result = main()
    except Exception as exc:
        message = str(exc)
        status = next((x for x in ('C5_B_MASK_BLOCKED', 'C5_B_NUMERICAL_BLOCKED',
                                   'C5_B_PARAMETER_BLOCKED', 'C5_B_GPU_BLOCKED',
                                   'C5_B_MEMORY_BLOCKED', 'C5_B_RUNTIME_BLOCKED')
                       if x in message), None)
        if status is None:
            status = 'C5_B_MEMORY_BLOCKED' if 'out of memory' in message.lower() else 'C5_B_RUNTIME_BLOCKED'
        result = {'status': status, 'error': message, 'traceback': traceback.format_exc()}
    result['duration_seconds'] = round(time.time() - start, 2)
    write(REPORTS / 'c5b_result.json', result)
    print('C5B_FINAL=' + json.dumps(result, ensure_ascii=False), flush=True)
    if result['status'] != 'C5_B_PASS':
        sys.exit(1)
