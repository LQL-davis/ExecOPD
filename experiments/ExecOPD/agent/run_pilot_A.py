#!/usr/bin/env python3
"""Method A 64-step Resume Pilot: Student-only execution-interactive Agent GRPO."""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
import uuid
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
from verl.trainer.ppo import core_algos
from experiments.ExecOPD.agent.c2c_rollout import collect_one
from experiments.ExecOPD.agent.pilot_turn_tensor import tensorize, batch_pad
from experiments.ExecOPD.agent.pilot_a_worker import PilotAWorker
from experiments.ExecOPD.agent.secure_execution_tool import SecureExecutionTool

REPO = REPO_ROOT
BASE = REPO / 'experiments/ExecOPD'
from experiments.ExecOPD.agent.pilot_paths import REPORTS, LOGS, RAY_RUN_ENV
MODEL = STUDENT_MODEL
DATA = TRAIN_PARQUET
PYTHON = PYTHON_EXE


def seed_everything():
    seed = 20261006
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {'python_seed': seed, 'numpy_seed': seed, 'torch_cpu_initial_seed': torch.initial_seed(),
            'torch_cuda_seed': seed, 'ray_worker_seed_env': os.environ.get('D1_SEED')}


def persist(name, value):
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / ('resume_pilot_A_' + name + '.json')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    return str(path)


def gate(condition, status, reason):
    if not condition:
        raise RuntimeError(status + ': ' + reason)


def one_rank(records):
    gate(isinstance(records, list) and len(records) == 2,
         'RESUME_PILOT_A_RUNTIME_BLOCKED', 'two TP rank outputs required')
    found = [x for x in records if x['rank'] == 0]
    gate(len(found) == 1, 'RESUME_PILOT_A_RUNTIME_BLOCKED', 'missing TP rank zero')
    return found[0]


def config(slice_path):
    cfg = OmegaConf.load(REPO / 'verl/trainer/config/ppo_trainer.yaml')
    values = {
        'data.train_files': str(slice_path),
        'data.val_files': str(BASE / 'c1/data/c1b_val.parquet'),
        'data.train_batch_size': 1, 'data.max_prompt_length': 512,
        'data.max_response_length': 512, 'data.shuffle': False,
        'actor_rollout_ref.model.path': str(MODEL),
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 1,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.ppo_epochs': 1,
        'actor_rollout_ref.actor.optim.lr': 5e-7,
        'actor_rollout_ref.actor.use_kl_loss': False,
        'actor_rollout_ref.actor.kl_loss_coef': 0.001,
        'actor_rollout_ref.actor.kl_loss_type': 'low_var_kl',
        'actor_rollout_ref.actor.entropy_coeff': 0.001,
        'actor_rollout_ref.actor.fsdp_config.param_offload': True,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': True,
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,
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
        'actor_rollout_ref.rollout.ignore_eos': False,
        'algorithm.adv_estimator': 'grpo', 'algorithm.use_kl_in_reward': False, 'reward_model.enable': False,
        'trainer.total_epochs': 1, 'trainer.total_training_steps': 64,
        'trainer.n_gpus_per_node': 2, 'trainer.nnodes': 1,
        'trainer.logger': ['console'], 'trainer.save_freq': -1,
        'trainer.test_freq': -1, 'trainer.resume_mode': 'disable',
        'trainer.project_name': 'ExecOPD',
        'trainer.experiment_name': 'resume_pilot_A_checkpoint_compatibility',
    }
    for key, value in values.items():
        OmegaConf.update(cfg, key, value)
    OmegaConf.update(cfg, 'trainer.val_before_train', False, force_add=True)
    gate(cfg.actor_rollout_ref.rollout.n == 4 and cfg.trainer.total_training_steps == 64,
         'RESUME_PILOT_A_RUNTIME_BLOCKED', 'tiny training configuration changed')
    persist('effective_config', OmegaConf.to_container(cfg, resolve=True))
    return cfg


from experiments.ExecOPD.agent.pilot_common import frozen_train_task as frozen_task, ensure_task_slice, split


def reward_of(trajectory):
    status = trajectory['state']['terminal_reason']
    if status == 'SECURE_EXECUTOR_ERROR':
        raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: secure Docker executor error')
    if status == 'PASS':
        return 1.1
    if status == 'MAX_TURNS' and any(x['extraction_status'] == 'SELECTED'
                                     for x in trajectory['state']['code_history']):
        return 0.1
    if status in ('ACTION_FORMAT_ERROR', 'NO_EXECUTABLE_CODE'):
        return -1.1
    raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: unsupported terminal status ' + str(status))


def stats_of(trajectory, reward):
    turns = trajectory['turns']
    history = trajectory['state']['execution_history']
    revisions = [x.get('revision_from_previous_code') for x in turns[1:]]
    return {'global_step': trajectory['global_step'], 'trajectory_index': trajectory['trajectory_index'],
            'execution_turns': len(history), 'terminal_status': trajectory['state']['terminal_reason'],
            'terminal_reward': reward,
            'code_changed_between_turns': any(x and x['code_changed_between_turns'] for x in revisions),
            'failure_to_pass_recovery': any(a['status'] != 'PASS' and b['status'] == 'PASS'
                                            for a, b in zip(history, history[1:])),
            'repeat_code': any(x and x['exactly_identical'] for x in revisions),
            'context_tokens': [x['input_context_tokens'] for x in turns],
            'generated_token_count': sum(len(x['raw_student_token_ids']) for x in turns),
            'tool_token_count': None}


MODE = os.environ.get('PILOT_A_MODE', 'UNSET')
CHECKPOINT = REPORTS / 'resume_pilot_artifacts/A/checkpoint_step64'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def driver_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all()}


def save_a_checkpoint(trainer, results, seeds, step):
    gate(MODE == 'SAVE' and len(results['steps']) == step and step in (8, 16, 32, 48, 64),
         'RESUME_PILOT_A_RUNTIME_BLOCKED', 'Method A checkpoint step differs')
    target = REPORTS / f'resume_pilot_artifacts/A/checkpoint_step{step}'
    gate(not target.exists(), 'RESUME_PILOT_A_RUNTIME_BLOCKED',
         'A checkpoint path already exists')
    started = time.perf_counter()
    target.mkdir(parents=True)
    actor = target / 'actor'
    ranks = trainer.actor_rollout_wg.save_checkpoint(str(actor), step)
    gate(len(ranks) == 2 and all(x['global_step'] == step for x in ranks),
         'RESUME_PILOT_A_RUNTIME_BLOCKED', 'A rank checkpoint incomplete')
    call_count = len(seeds['vllm_requests'])
    state = {'global_step': step, 'next_task_cursor': step+1,
             'generation_call': call_count, 'rng': driver_rng()}
    torch.save(state, target / 'driver_state.pt')
    audit = trainer.actor_rollout_wg.pilot_a_state_audit()
    gate(len(audit) == 2 and all(x['global_step'] == step and
         x['generation_call'] == call_count for x in audit),
         'RESUME_PILOT_A_RUNTIME_BLOCKED', 'A worker state inconsistent')
    (target / 'worker_state.json').write_text(json.dumps(audit, indent=2) + '\n')
    metadata = {'global_step': step, 'next_task_cursor': step+1,
                'generation_call': call_count,
                'split_sha256': sha(BASE / 'reports/d1_data_split.json'),
                'model_config_sha256': sha(MODEL / 'config.json'),
                'tokenizer_sha256': sha(MODEL / 'tokenizer.json'),
                'seed': 20261006, 'world_size': 2,
                'config_sha256': sha(REPORTS / 'resume_pilot_A_effective_config.json'),
                'method': 'A', 'teacher_state_present': False}
    (target / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    shutil_source = REPORTS / 'resume_pilot_A_effective_config.json'
    (target / 'frozen_effective_config.json').write_bytes(shutil_source.read_bytes())
    files = [{'path': str(x.relative_to(target)), 'bytes': x.stat().st_size,
              'sha256': sha(x)} for x in sorted(target.rglob('*')) if x.is_file()]
    for rank in (0, 1):
        for stem in ('model', 'optim', 'extra_state'):
            gate(any(f['path'] == f'actor/{stem}_world_size_2_rank_{rank}.pt'
                     for f in files), 'RESUME_PILOT_A_RUNTIME_BLOCKED',
                 f'Method A missing {stem} rank {rank}')
    manifest = {'files': files, 'total_bytes': sum(x['bytes'] for x in files),
                'write_seconds': time.perf_counter()-started}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    marker = target / 'CHECKPOINT_COMPLETE.tmp'
    marker.write_text(sha(target / 'manifest.json') + '\n')
    os.replace(marker, target / 'CHECKPOINT_COMPLETE')
    return {'path': str(target), 'bytes': manifest['total_bytes'],
            'step': step, 'write_seconds': manifest['write_seconds'], 'worker_state': audit}


def finish_a_checkpoint(trainer, results, seeds):
    check = save_a_checkpoint(trainer, results, seeds, 64)
    export = REPORTS / 'resume_pilot_artifacts/A/export_step64'
    rank_export = trainer.actor_rollout_wg.pilot_a_export_hf(str(export))
    gate(len(rank_export) == 2 and (export / 'model.safetensors').is_file(),
         'RESUME_PILOT_A_RUNTIME_BLOCKED', 'A trained model export incomplete')
    return {'status': 'RESUME_PILOT_A64_PASS', 'global_step': 64,
            'next_task_cursor': 65, 'generation_call': len(seeds['vllm_requests']),
            'teacher_loaded': False, 'checkpoint_bytes': check['bytes'],
            'checkpoint_complete': True, 'worker_state': check['worker_state'],
            'periodic_checkpoints': results.get('periodic_checkpoints', []) + [check],
            'eval_model_path': str(export), 'model_export': rank_export,
            'steps': results['steps'],
            'frozen_train_task_count': 64,
            'fresh_trajectory_count': 256}


class Trainer(RayPPOTrainer):
    def fit(self):
        gate(self.total_training_steps == 64 and not self.use_critic and not self.use_rm and
             not self.use_reference_policy and self.val_reward_fn is None,
             'RESUME_PILOT_A_RUNTIME_BLOCKED', 'unexpected trainer roles or steps')
        tokenizer = self.tokenizer
        gate(MODE == 'SAVE', 'RESUME_PILOT_A_RUNTIME_BLOCKED',
             'only fresh independent A64 run is permitted')
        gate(json.loads((REPORTS / 'd1g3_result.json').read_text())['status'] == 'D1_G3_PASS',
             'RESUME_PILOT_A_RUNTIME_BLOCKED', 'D1-G3 must PASS first')
        task = frozen_task(tokenizer, 1)
        physical = os.environ['CUDA_VISIBLE_DEVICES']
        results = {'physical_gpu_ids': [int(x) for x in physical.split(',')],
                   'steps': [], 'teacher_loaded': False, 'opd_enabled': False,
                   'baseline_artifacts_modified': False}
        seeds = {'base_seed': 20261006, 'driver': seed_everything(),
                 'ray_workers': self.actor_rollout_wg.pilot_a_seed_audit(),
                 'vllm_requests': []}
        persist('seed_audit', seeds)
        trajectory_path = REPORTS / 'resume_pilot_A_trajectories.jsonl'
        gate(not trajectory_path.exists(), 'RESUME_PILOT_A_RUNTIME_BLOCKED',
             'A trajectory output exists; no reroll')
        trajectory_path.touch(exist_ok=False)
        reward_groups, mask_audit, loss_report, snapshots = [], [], [], {}
        tool_calls = []
        tool = SecureExecutionTool(log_event=tool_calls.append)

        def generate(prompt_ids):
            records = self.actor_rollout_wg.pilot_a_generate(list(prompt_ids))
            first = one_rank(records)
            gate(records[0]['sampled_ids'] == records[1]['sampled_ids'],
                 'RESUME_PILOT_A_RUNTIME_BLOCKED', 'TP rank sampled IDs differ')
            gate(records[0]['request_seed'] == records[1]['request_seed'] and
                 records[0]['generation_call'] == records[1]['generation_call'],
                 'RESUME_PILOT_A_RUNTIME_BLOCKED', 'TP rank sampling seeds differ')
            seeds['vllm_requests'].append({'call': first['generation_call'],
                                            'seed': first['request_seed']})
            persist('seed_audit', seeds)
            return first

        def parameter_snapshot(phase):
            rec = one_rank(self.actor_rollout_wg.pilot_a_parameters(phase))
            key = 'actor_' + phase
            snapshots[key] = rec
            persist('parameter_snapshots', snapshots)
            return rec

        def lr_audit():
            records = self.actor_rollout_wg.pilot_a_lr_audit()
            gate(len(records) == 2 and records[0]['lrs'] == records[1]['lrs'] and
                 records[0]['scheduler_last_epoch'] == records[1]['scheduler_last_epoch'],
                 'RESUME_PILOT_A_RUNTIME_BLOCKED', 'rank learning rate disagreement')
            return records[0]

        parameter_snapshot('before_step1')
        gate(lr_audit()['lrs'] == [0.0], 'RESUME_PILOT_A_RUNTIME_BLOCKED', 'initial LR must be zero')
        case_z = False
        case_v = False
        for step in range(1, 65):
            task = frozen_task(tokenizer, step)
            print('D1G1RA_PHASE=step' + str(step) + '_fresh_rollout_start', flush=True)
            trajectories = []
            call_start = len(tool_calls)
            for index in range(4):
                trajectory = collect_one(tokenizer, generate, tool, task,
                                         step=step, sample_index=index)
                trajectory['terminal_reward'] = reward_of(trajectory)
                trajectories.append(trajectory)
                with trajectory_path.open('a') as stream:
                    stream.write(json.dumps(trajectory, ensure_ascii=False) + '\n')
                print('D1G1RA_TRAJECTORY=' + json.dumps({'step': step, 'index': index,
                     'turns': len(trajectory['turns']),
                     'terminal': trajectory['state']['terminal_reason'],
                     'reward': trajectory['terminal_reward']}), flush=True)
            rewards = [float(t['terminal_reward']) for t in trajectories]
            group = {'step': step, 'rewards': rewards, 'mean': statistics.mean(rewards),
                     'population_std': statistics.pstdev(rewards),
                     'sample_std': statistics.stdev(rewards),
                     'zero_variance_group': statistics.pstdev(rewards) == 0,
                     'task_id': task['task_id'],
                     'secure_executor_calls': len(tool_calls) - call_start,
                     'trajectory_count': 4}
            reward_groups.append(group)
            persist('reward_groups', reward_groups)
            items, aligned = [], []
            for trajectory in trajectories:
                item, detail = tensorize(tokenizer, trajectory)
                items.append(item)
                aligned.append(detail)
            tensors, prompt_width, continuation_width = batch_pad(tokenizer, items, aligned)
            for record, detail in zip(trajectories, aligned):
                metrics = stats_of(record, record['terminal_reward'])
                metrics['tool_token_count'] = detail['tool_count']
                detail['diagnostics'] = metrics
            audit = {'step': step, 'trajectories': aligned, 'batch_shape': list(tensors['input_ids'].shape),
                     'prompt_width': prompt_width, 'continuation_width': continuation_width,
                     'generated_tool_overlap': int((tensors['generated_token_mask'] &
                                                    tensors['execution_observation_mask']).sum()),
                     'code_tool_overlap': int((tensors['code_token_mask'] &
                                               tensors['execution_observation_mask']).sum())}
            gate(audit['generated_tool_overlap'] == audit['code_tool_overlap'] == 0,
                 'RESUME_PILOT_A_MASK_BLOCKED', 'tool tokens overlap Student/code mask')
            mask_audit.append(audit)
            persist('mask_audit', mask_audit)
            response_len = continuation_width
            response_mask = tensors['generated_token_mask'][:, -response_len:]
            scores = torch.zeros((4, response_len), dtype=torch.float32)
            for index, value in enumerate(rewards):
                positions = torch.where(response_mask[index] == 1)[0]
                gate(len(positions) > 0, 'RESUME_PILOT_A_MASK_BLOCKED', 'trajectory has no sampled Student token')
                scores[index, positions[-1]] = value
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=scores, eos_mask=response_mask,
                index=np.array([task['task_id']] * 4, dtype=object))
            gate(not bool((advantages * (1 - response_mask)).any()) and
                 bool(torch.isfinite(advantages).all()),
                 'RESUME_PILOT_A_MASK_BLOCKED', 'advantage leaked beyond generated Student mask')
            advantage_nonzero_count = int((advantages[response_mask.bool()] != 0).sum())
            if group['zero_variance_group']:
                gate(advantage_nonzero_count == 0, 'RESUME_PILOT_A_NUMERICAL_BLOCKED',
                     'zero-variance group has nonzero GRPO advantage')
            tensors.update({'prompts': tensors['input_ids'][:, :prompt_width],
                            'responses': tensors['input_ids'][:, prompt_width:],
                            'token_level_scores': scores, 'token_level_rewards': scores,
                            'advantages': advantages, 'returns': returns})
            batch = DataProto(batch=TensorDict(tensors, batch_size=[4]),
                              non_tensor_batch={'uid': np.array([task['task_id']] * 4,
                                                                 dtype=object)},
                              meta_info={'global_token_num': tensors['turn_attention_mask'].sum((-1, -2)).tolist(),
                                         'pilot_a_step': step,
                                         'zero_variance_group': group['zero_variance_group']})
            print('D1G1RA_PHASE=step' + str(step) + '_old_ref_logprob_start', flush=True)
            batch = batch.union(self.actor_rollout_wg.compute_log_prob(batch))
            logprobs = {}
            for key in ('old_log_probs',):
                value = batch.batch[key]
                logprobs[key] = {'shape': list(value.shape),
                                 'nan_count': int(torch.isnan(value).sum()),
                                 'inf_count': int(torch.isinf(value).sum())}
                gate(value.shape == scores.shape and bool(torch.isfinite(value).all()),
                     'RESUME_PILOT_A_NUMERICAL_BLOCKED', key + ' shape/nonfinite failure')
            print('D1G1RA_PHASE=step' + str(step) + '_actor_update_start', flush=True)
            actor_output = self.actor_rollout_wg.update_actor(batch)
            metrics = reduce_metrics(actor_output.meta_info['metrics'])
            gate(all(np.isfinite(v) for v in metrics.values()) and
                 metrics.get('pilot_a/tool_loss_overlap') == 0,
                 'RESUME_PILOT_A_NUMERICAL_BLOCKED', 'actor metric nonfinite or tool-loss overlap')
            expected_lr = 0.0 if step == 1 else 5e-7
            gate(metrics['pilot_a/actual_optimizer_lr'] == expected_lr and
                 metrics['pilot_a/optimizer_step_count'] == step,
                 'RESUME_PILOT_A_RUNTIME_BLOCKED', 'LR/optimizer step audit failed')
            after = parameter_snapshot('after_step' + str(step))
            if step == 1:
                gate(not any(p['changed'] for p in after['parameters'].values()),
                     'RESUME_PILOT_A_PARAMETER_UPDATE_BLOCKED', 'Actor changed at LR zero')
            else:
                gate(metrics['actor/grad_norm'] > 0 and metrics['pilot_a/any_nonzero_gradient'] > 0,
                     'RESUME_PILOT_A_PARAMETER_UPDATE_BLOCKED', 'nonzero-LR step has no gradient')
            next_lr = lr_audit()
            gate(next_lr['optimizer_step_count'] == step and
                 next_lr['lrs'] == [5e-7],
                 'RESUME_PILOT_A_RUNTIME_BLOCKED', 'scheduler advancement mismatch')
            loss_record = {'step': step, 'reward_group': group,
                           'advantage_nonzero_count': advantage_nonzero_count,
                           'zero_variance_group': group['zero_variance_group'],
                           'advantage_generated_count': int(response_mask.sum()),
                           'tool_advantage_nonzero_count': int((advantages *
                               tensors['execution_observation_mask'][:, -response_len:]).count_nonzero()),
                           'logprobs': logprobs, 'actor_metrics': metrics, 'lr_audit': next_lr}
            loss_report.append(loss_record)
            persist('loss', loss_report)
            results['steps'].append({'step': step, 'trajectory_count': 4,
                                     'reward_std': group['population_std'],
                                     'generated_token_count': int(response_mask.sum()),
                                     'tool_token_count': int(tensors['execution_observation_mask'].sum()),
                                     'actual_optimizer_lr': expected_lr,
                                     'parameter_changed': any(p['changed'] for p in after['parameters'].values()),
                                     'grpo_advantage_nonzero_count': advantage_nonzero_count,
                                     'grpo_policy_loss_contribution': metrics['pilot_a/grpo_policy_contribution'],
                                     'entropy_contribution': metrics['pilot_a/entropy_contribution'],
                                     'total_actor_loss': metrics['pilot_a/total_loss'],
                                     'entropy': metrics['actor/entropy_loss'],
                                     'grad_norm': metrics['actor/grad_norm'],
                                     'parameter_delta_l2': {k: v.get('l2_delta') for k, v in after['parameters'].items()},
                                     'reward_vector': rewards, 'task_id': task['task_id'],
                                     'zero_variance_group': group['zero_variance_group']})
            print('D1G1RA_STEP_COMPLETE=' + json.dumps(results['steps'][-1]), flush=True)
            if group['zero_variance_group']:
                case_z = True
                gate(abs(metrics['pilot_a/grpo_policy_contribution']) < 1e-6,
                     'RESUME_PILOT_A_NUMERICAL_BLOCKED', 'GRPO policy contribution at zero variance')
            if step > 1 and group['population_std'] > 0 and advantage_nonzero_count > 0 and \
                    any(p['changed'] for p in after['parameters'].values()):
                case_v = True
            persist('zero_variance_audit', {'case_z': case_z, 'case_v': case_v,
                                            'steps': results['steps']})
            if step == 2:
                gate(results['steps'][-1]['parameter_changed'] and
                     metrics['pilot_a/entropy_tool_overlap'] == 0,
                     'RESUME_PILOT_A_PARAMETER_UPDATE_BLOCKED',
                     'repaired exact-turn two-GPU runtime gate failed')
                persist('exact_turn_runtime_gate', {
                    'status': 'A_EXACT_TURN_TWO_GPU_RUNTIME_PASS', 'global_steps': 2,
                    'physical_gpu_ids': [2, 3], 'sampled_alignment_pass': True,
                    'tool_loss_entropy_overlap': 0,
                    'actual_lr': expected_lr, 'Student_parameter_changed': True,
                    'grad_norm': metrics['actor/grad_norm']})
                print('PILOT_A_EXACT_TURN_RUNTIME_GATE=PASS', flush=True)
            if step in (8, 16, 32, 48):
                saved = save_a_checkpoint(self, results, seeds, step)
                results.setdefault('periodic_checkpoints', []).append(saved)
                persist('checkpoints', results['periodic_checkpoints'])
                print('PILOT_A_CHECKPOINT_COMPLETE=' + json.dumps(saved), flush=True)
        return finish_a_checkpoint(self, results, seeds)



def main():
    gate(sys.executable == PYTHON and os.environ.get('PYTHONPATH') == str(REPO),
         'RESUME_PILOT_A_RUNTIME_BLOCKED', 'wrong isolated Python/PYTHONPATH')
    gate(json.loads((REPORTS / 'd1g3_result.json').read_text())['status'] == 'D1_G3_PASS',
         'RESUME_PILOT_A_RUNTIME_BLOCKED', 'G3 must PASS before pilot')
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    gate(physical == '2,3', 'RESUME_PILOT_A_RUNTIME_BLOCKED',
         'only preflighted physical H100 GPUs 2 and 3 permitted')
    gate(MODE == 'SAVE' and os.environ.get('D1_SEED') == '20261006',
         'RESUME_PILOT_A_RUNTIME_BLOCKED', 'fresh A run/seed required')
    gpu_state = subprocess.run(
        ['nvidia-smi', '-i', '2,3',
         '--query-gpu=index,name,memory.free',
         '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    gpu_rows = [line.split(',') for line in gpu_state.stdout.splitlines()]
    free_values = [int(row[-1].strip()) for row in gpu_rows]
    gate(len(gpu_rows) == 2 and all('H100' in row[1] for row in gpu_rows)
         and all(x >= 76800 for x in free_values),
         'RESUME_PILOT_A_GPU_BLOCKED', 'two empty H100 GPUs required')
    persist('gpu_preflight', {'physical_gpu_ids': [2, 3],
                              'free_mib': free_values,
                              'raw': gpu_state.stdout})
    seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), use_fast=True, local_files_only=True)
    gate(type(tokenizer).__name__ == 'Qwen2TokenizerFast' and tokenizer.is_fast,
         'RESUME_PILOT_A_RUNTIME_BLOCKED', 'fast Qwen2 tokenizer unavailable')
    task = frozen_task(tokenizer, 1)
    slice_path = ensure_task_slice()
    cfg = config(slice_path)
    ray.init(num_gpus=2, include_dashboard=False,
             runtime_env={'env_vars': {'PYTHONNOUSERSITE': '1', 'PYTHONPATH': str(REPO),
                                      'TOKENIZERS_PARALLELISM': 'false', 'NCCL_DEBUG': 'WARN',
                                      'D1_SEED': '20261006', **RAY_RUN_ENV}})
    try:
        from verl.utils import hf_tokenizer, hf_processor
        model_tokenizer = hf_tokenizer(str(MODEL))
        processor = hf_processor(str(MODEL), use_fast=True)
        mapping = {Role.ActorRollout: ray.remote(PilotAWorker)}
        pool = ResourcePoolManager({'pilot_a_pool': [2]},
                                   {Role.ActorRollout: 'pilot_a_pool'})
        trainer = Trainer(cfg, model_tokenizer, mapping, pool, RayWorkerGroup,
                          processor=processor, reward_fn=None, val_reward_fn=None)
        print('D1G1RA_PHASE=init_workers_start', flush=True)
        trainer.init_workers()
        print('D1G1RA_PHASE=init_workers_complete', flush=True)
        return trainer.fit()
    finally:
        ray.shutdown()


if __name__ == '__main__':
    started = time.time()
    try:
        result = main()
    except Exception as exc:
        message = str(exc)
        recognized = ('RESUME_PILOT_A_MASK_BLOCKED', 'RESUME_PILOT_A_ZERO_VARIANCE_BLOCKED',
                      'RESUME_PILOT_A_MEMORY_BLOCKED', 'RESUME_PILOT_A_NUMERICAL_BLOCKED',
                      'RESUME_PILOT_A_PARAMETER_UPDATE_BLOCKED', 'RESUME_PILOT_A_RUNTIME_BLOCKED')
        status = next((x for x in recognized if x in message), None)
        if status is None:
            status = 'RESUME_PILOT_A_MEMORY_BLOCKED' if 'out of memory' in message.lower() or \
                'OutOfMemoryError' in traceback.format_exc() else 'RESUME_PILOT_A_RUNTIME_BLOCKED'
        result = {'status': status, 'error': message, 'traceback': traceback.format_exc()}
    result['duration_seconds'] = round(time.time() - started, 2)
    training_result = REPORTS / 'resume_pilot_training_A.json'
    if training_result.exists():
        raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: result output exists')
    training_result.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n')
    # D1-G2 final report is written after Method D comparison.
    print('D1G1RA_FINAL=' + json.dumps(result, ensure_ascii=False, default=str), flush=True)
    if result['status'] != 'RESUME_PILOT_A64_PASS':
        sys.exit(1)
