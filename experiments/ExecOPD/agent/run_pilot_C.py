#!/usr/bin/env python3
"""Method C 64-step Resume Pilot: frozen Code-only Agentic OPD protocol."""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, ResourcePoolManager, Role, reduce_metrics)

from experiments.ExecOPD.agent.pilot_turn_tensor import batch_pad, tensorize
from experiments.ExecOPD.agent.pilot_c_worker import PilotCWorker
from experiments.ExecOPD.agent.pilot_common import (
    ensure_task_slice, frozen_train_task, split)

REPO = REPO_ROOT
BASE = REPO / 'experiments/ExecOPD'
from experiments.ExecOPD.agent.pilot_paths import REPORTS, RAY_RUN_ENV
ARTIFACTS = REPORTS / 'resume_pilot_artifacts/C'
STUDENT = STUDENT_MODEL
TEACHER = TEACHER_MODEL
PYTHON = PYTHON_EXE
K = 16
SEED = 20261006


def gate(condition, detail):
    if not condition:
        raise RuntimeError('RESUME_PILOT_C_BLOCKED: ' + detail)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.write('\n')


def append_sync(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def rank_zero(values):
    gate(isinstance(values, list) and len(values) == 2,
         'two distributed rank outputs required')
    selected = [x for x in values if x['rank'] == 0]
    gate(len(selected) == 1, 'rank zero output missing')
    return selected[0]


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def gpu_preflight():
    gate(os.environ.get('CUDA_VISIBLE_DEVICES') == '2,3',
         'only physical GPUs 2,3 permitted')
    result = subprocess.run(
        ['nvidia-smi', '-i', '2,3',
         '--query-gpu=index,name,memory.free',
         '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    rows = [line.split(',') for line in result.stdout.splitlines()]
    gate(len(rows) == 2 and all('H100' in row[1] and
         int(row[-1].strip()) >= 76800 for row in rows),
         'two free H100 80GB GPUs required')
    return {'physical_gpu_ids': [2, 3],
            'initial_free_mib': [int(row[-1].strip()) for row in rows],
            'nvidia_smi': result.stdout}


def config(slice_path):
    cfg = OmegaConf.load(REPO / 'verl/trainer/config/ppo_trainer.yaml')
    values = {
        'data.train_files': str(slice_path),
        'data.val_files': str(BASE / 'c1/data/c1b_val.parquet'),
        'data.train_batch_size': 1,
        'data.max_prompt_length': 512,
        'data.max_response_length': 512,
        'data.shuffle': False,
        'actor_rollout_ref.model.path': str(STUDENT),
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
        'trainer.total_training_steps': 64,
        'trainer.n_gpus_per_node': 2,
        'trainer.nnodes': 1,
        'trainer.logger': ['console'],
        'trainer.save_freq': -1,
        'trainer.test_freq': -1,
        'trainer.resume_mode': 'disable',
        'trainer.project_name': 'ExecOPD',
        'trainer.experiment_name': 'resume_pilot_C64',
    }
    for key, value in values.items():
        OmegaConf.update(cfg, key, value)
    OmegaConf.update(cfg, 'trainer.val_before_train', False, force_add=True)
    gate(cfg.actor_rollout_ref.rollout.n == 4 and
         cfg.trainer.total_training_steps == 64 and
         cfg.algorithm.adv_estimator == 'token_reward_direct' and
         not cfg.actor_rollout_ref.actor.use_kl_loss,
         'frozen Method C objective differs')
    write_new(ARTIFACTS / 'effective_config.json',
              OmegaConf.to_container(cfg, resolve=True))
    return cfg


def run_offline(phase, step, source=None):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '2'
    cmd = [PYTHON, str(BASE / 'agent/pilot_c_offline.py'),
           '--phase', phase, '--step', str(step)]
    if phase == 'collect':
        cmd += ['--model', str(source)]
    log = ARTIFACTS / f'step{step}_{phase}.log'
    with log.open('xb') as stream:
        result = subprocess.run(cmd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    gate(result.returncode == 0,
         f'{phase} step{step} exited {result.returncode}; see {log}')


def load_evidence(tokenizer, step, task_id):
    folder = ARTIFACTS / f'step{step}'
    path = folder / 'trajectories.jsonl'
    student = json.loads((folder / 'student_topk.json').read_text())
    teacher = json.loads((folder / 'teacher_scores.json').read_text())
    gate(student['status'] == teacher['status'] == 'PASS' and
         teacher['teacher_parameter_unchanged'] and
         teacher['teacher_generation_count'] == 0 and
         student['k'] == teacher['k'] == K and
         student['trajectory_count'] == 4 and
         student['task_id'] == task_id and
         sha(path) == student['trajectories_sha256'] ==
         teacher['trajectories_sha256'],
         'Student/Teacher same-state evidence hash mismatch')
    trajectories = [json.loads(x) for x in path.read_text().splitlines()]
    gate(len(trajectories) == len(student['records']) ==
         len(teacher['records']) == 4, 'fresh trajectory count')
    for index, (tr, s, t) in enumerate(zip(
            trajectories, student['records'], teacher['records'])):
        gate(tr['global_step'] == s['global_step'] ==
             t['global_step'] == step and
             tr['trajectory_index'] == s['trajectory_index'] ==
             t['trajectory_index'] == index and tr['task_id'] == task_id,
             'trajectory/score order')
        item, detail = tensorize(tokenizer, tr)
        gate(s['input_ids_sha256'] == t['input_ids_sha256'] ==
             hashlib.sha256(item['input_ids'].numpy().tobytes()).hexdigest() and
             s['turn_inputs'] == t['turn_inputs'] == detail['turn_inputs'] and
             len(s['positions']) == len(t['positions']) ==
             detail['generated_count'] == t['teacher_scored_count'] and
             detail['code_count'] == t['teacher_scored_code_count'] and
             t['tool_target_count'] == t['noncode_target_count'] == 0,
             'sampled IDs/code mask/Teacher positions')
        for src, tgt in zip(s['positions'], t['positions']):
            position = src['position']
            gate(position == tgt['position'] and
                 src['student_topk_ids'] == tgt['gather_ids'] and
                 bool(item['generated_token_mask'][position]) and
                 not bool(item['execution_observation_mask'][position]) and
                 tgt['code_target'] == bool(item['code_token_mask'][position]),
                 'same-ID Teacher gather or tool overlap')
    return trajectories, student, teacher


def checkpoint(trainer, step_records):
    step = len(step_records)
    target = ARTIFACTS / f'checkpoint_step{step}'
    gate(not target.exists() and step in (8, 16, 32, 48, 64),
         'checkpoint target exists or incomplete steps')
    started = time.perf_counter()
    target.mkdir()
    ranks = trainer.actor_rollout_wg.save_checkpoint(str(target / 'actor'), step)
    gate(len(ranks) == 2 and all(x['global_step'] == step for x in ranks),
         'rank checkpoint save incomplete')
    driver = {'python_rng': random.getstate(),
              'numpy_rng': np.random.get_state(),
              'torch_cpu_rng': torch.get_rng_state(),
              'torch_cuda_rng': torch.cuda.get_rng_state_all(),
              'global_step': step, 'next_task_cursor': step+1}
    torch.save(driver, target / 'driver_state.pt')
    worker_state = trainer.actor_rollout_wg.pilot_c_state_audit()
    gate(len(worker_state) == 2 and all(x['step'] == step for x in worker_state),
         'worker checkpoint state differs')
    write_new(target / 'worker_state.json', worker_state)
    metadata = {'status': 'PREPARED', 'method': 'C', 'seed': SEED,
                'global_step': step, 'next_task_cursor': step+1,
                'split_sha256': sha(BASE / 'reports/d1_data_split.json'),
                'config_sha256': sha(ARTIFACTS / 'effective_config.json'),
                'student_config_sha256': sha(STUDENT / 'config.json'),
                'student_tokenizer_sha256': sha(STUDENT / 'tokenizer.json'),
                'world_size': 2, 'teacher_state_present': False}
    write_new(target / 'metadata.json', metadata)
    (target / 'frozen_effective_config.json').write_bytes((ARTIFACTS / 'effective_config.json').read_bytes())
    files = [{'path': str(path.relative_to(target)),
              'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(target.rglob('*')) if path.is_file()]
    for rank in (0, 1):
        for stem in ('model', 'optim', 'extra_state'):
            gate(any(x['path'] == f'actor/{stem}_world_size_2_rank_{rank}.pt'
                     for x in files), f'missing {stem} rank {rank}')
    manifest = {'files': files, 'total_bytes': sum(x['bytes'] for x in files),
                'write_seconds': time.perf_counter() - started,
                'rank_bytes': {str(rank): sum(x['bytes'] for x in files if
                               x['path'].endswith(f'_rank_{rank}.pt'))
                               for rank in (0, 1)}}
    write_new(target / 'manifest.json', manifest)
    marker = target / 'CHECKPOINT_COMPLETE.tmp'
    marker.write_text(sha(target / 'manifest.json') + '\n')
    os.replace(marker, target / 'CHECKPOINT_COMPLETE')
    return {'path': str(target), 'manifest': manifest,
            'worker_state': worker_state}


class Trainer(RayPPOTrainer):
    def __init__(self, config, *args, **kwargs):
        gate(config.algorithm.adv_estimator == 'token_reward_direct',
             'direct OPD estimator unavailable')
        config.algorithm.adv_estimator = 'grpo'
        try:
            super().__init__(config, *args, **kwargs)
        finally:
            config.algorithm.adv_estimator = 'token_reward_direct'

    def fit(self):
        gate(self.total_training_steps == 64 and not self.use_critic and
             not self.use_rm and not self.use_reference_policy and
             self.val_reward_fn is None, 'unexpected trainer roles')
        tokenizer = self.tokenizer
        gate(json.loads((REPORTS / 'd1g3_result.json').read_text())['status'] ==
             'D1_G3_PASS', 'D1-G3 gate must PASS first')
        gate(not (ARTIFACTS / 'step1').exists(),
             'C64 already started; no implicit reroll')
        initial = rank_zero(self.actor_rollout_wg.pilot_c_parameters('before_step1'))
        lr = self.actor_rollout_wg.pilot_c_lr_audit()
        gate(len(lr) == 2 and all(x['lrs'] == [0.0] for x in lr),
             'Step1 LR not zero')
        records = []
        source = STUDENT
        for step in range(1, 65):
            task = frozen_train_task(tokenizer, step)
            gate(task['task_id'] == split()['splits']['train'][step - 1]['task_key'],
                 'frozen task order differs')
            print(f'PILOT_C_PHASE=step{step}_collect', flush=True)
            run_offline('collect', step, source)
            print(f'PILOT_C_PHASE=step{step}_teacher', flush=True)
            run_offline('teacher', step)
            trajectories, student, teacher = load_evidence(
                tokenizer, step, task['task_id'])
            items, alignments = zip(*(tensorize(tokenizer, tr)
                                      for tr in trajectories))
            tensors, prompt_width, response_len = batch_pad(
                tokenizer, list(items), list(alignments))
            generated = tensors['generated_token_mask'][:, -response_len:]
            code = tensors['code_token_mask'][:, -response_len:]
            tool = tensors['execution_observation_mask'][:, -response_len:]
            gate(bool(torch.all(code <= generated)) and
                 not bool(torch.any(code & tool)) and
                 not bool(torch.any(generated & tool)),
                 'mask invariant or tool overlap')
            ids = torch.zeros((4, response_len, K), dtype=torch.long)
            student_logp = torch.zeros((4, response_len, K), dtype=torch.float32)
            signal = torch.zeros((4, response_len, K), dtype=torch.float32)
            for row, (detail, s, t) in enumerate(zip(
                    alignments, student['records'], teacher['records'])):
                left = prompt_width - detail['initial_prompt_length']
                for src, tgt in zip(s['positions'], t['positions']):
                    j = left + src['position'] - prompt_width
                    gate(0 <= j < response_len and generated[row, j] == 1 and
                         tool[row, j] == 0 and
                         tgt['code_target'] == bool(code[row, j]),
                         'OPD position maps outside Student code context')
                    ids[row, j] = torch.tensor(src['student_topk_ids'])
                    student_logp[row, j] = torch.tensor(src['student_logprobs'])
                    signal[row, j] = torch.tensor(tgt['opd_signal'])
            gate(bool(torch.isfinite(signal).all()) and
                 not bool(torch.any(signal * (1 - code.unsqueeze(-1)))),
                 'nonfinite or non-code OPD signal')
            advantage = signal * code.unsqueeze(-1)
            gate(not bool(torch.any(advantage * tool.unsqueeze(-1))) and
                 not bool(torch.any(advantage *
                                    (generated - code).unsqueeze(-1))),
                 'OPD advantage leaked into tool/noncode')
            tensors.update({
                'prompts': tensors['input_ids'][:, :prompt_width],
                'responses': tensors['input_ids'][:, prompt_width:],
                'response_mask': code,
                'student_top_k_ids': ids,
                'student_top_k_log_probs': student_logp,
                'old_log_probs': student_logp,
                'token_level_scores': signal,
                'token_level_rewards': signal,
                'advantages': advantage,
                'returns': advantage.clone(),
            })
            batch = DataProto(
                batch=TensorDict(tensors, batch_size=[4]),
                non_tensor_batch={'uid': np.array([task['task_id']] * 4,
                                                 dtype=object)},
                meta_info={'global_token_num': tensors['turn_attention_mask'].sum((-1, -2)).tolist(),
                           'pilot_c_step': step, 'temperature': 1.0,
                           'top_k_strategy': 'only_stu',
                           'reward_weight_mode': 'student_p',
                           'log_prob_top_k': K})
            actor_output = self.actor_rollout_wg.update_actor(batch)
            metrics = reduce_metrics(actor_output.meta_info['metrics'])
            gate(all(math.isfinite(float(v)) for v in metrics.values()) and
                 metrics['pilot_c/tool_loss_overlap'] == 0 and
                 metrics['pilot_c/noncode_assistant_loss_overlap'] == 0 and
                 metrics['pilot_c/entropy_tool_overlap'] == 0 and
                 metrics['pilot_c/entropy_mask_unweighted_generated'] == 1 and
                 metrics['pilot_c/optimizer_step_count'] == step and
                 metrics['pilot_c/actual_optimizer_lr'] ==
                     (0.0 if step == 1 else 5e-7),
                 'actor objective/LR/mask audit')
            if step > 1:
                gate(metrics['pilot_c/any_nonzero_gradient'] > 0 and
                     metrics['actor/grad_norm'] > 0,
                     'nonzero LR step had no gradient')
            lr = self.actor_rollout_wg.pilot_c_lr_audit()
            gate(len(lr) == 2 and all(x['optimizer_step_count'] == step and
                 x['lrs'] == [5e-7] for x in lr),
                 'scheduler progression differs')
            teacher_memory = json.loads(
                (ARTIFACTS / f'step{step}/teacher_memory.json').read_text())
            student_memory = json.loads(
                (ARTIFACTS / f'step{step}/student_memory.json').read_text())
            record = {
                'step': step, 'task_id': task['task_id'],
                'trajectory_count': 4,
                'terminal_status': [x['state']['terminal_reason']
                                    for x in trajectories],
                'generated_tokens': int(generated.sum()),
                'code_tokens': int(code.sum()),
                'tool_tokens': int(tensors['execution_observation_mask'].sum()),
                'teacher_forward_calls': teacher['teacher_forward_calls'],
                'teacher_scored_tokens': teacher['teacher_scored_tokens'],
                'teacher_scored_code_tokens':
                    teacher['teacher_scored_code_tokens'],
                'teacher_input_tokens': teacher['teacher_input_tokens'],
                'teacher_forward_wall_seconds':
                    teacher['teacher_forward_wall_seconds'],
                'opd_signal_abs_mean_on_code': (
                    float(signal[code.bool()].abs().mean())
                    if bool(code.any()) else 0.0),
                'opd_advantage_nonzero': int(torch.count_nonzero(advantage)),
                'tool_opd_count': int(torch.count_nonzero(
                    signal * tool.unsqueeze(-1))),
                'noncode_opd_count': int(torch.count_nonzero(
                    signal * (generated - code).unsqueeze(-1))),
                'zero_code_group': not bool(code.any()),
                'actor_loss': metrics['pilot_c/total_loss'],
                'entropy_term': metrics['actor/entropy_loss'],
                'grad_norm': metrics['actor/grad_norm'],
                'actual_lr': metrics['pilot_c/actual_optimizer_lr'],
                'actor_metrics': metrics,
                'student_memory': student_memory,
                'teacher_memory': teacher_memory,
            }
            records.append(record)
            append_sync(ARTIFACTS / 'step_records.jsonl', record)
            print('PILOT_C_STEP_COMPLETE=' + json.dumps({
                'step': step, 'task_id': task['task_id'],
                'code_tokens': record['code_tokens'],
                'loss': record['actor_loss'],
                'teacher_forward_seconds': record['teacher_forward_wall_seconds']}),
                flush=True)
            if step == 2:
                write_new(ARTIFACTS / 'exact_turn_runtime_gate.json', {
                    'status': 'C_EXACT_TURN_TWO_GPU_RUNTIME_PASS', 'global_steps': 2,
                    'same_state_same_ID_teacher_pass': True,
                    'tool_noncode_OPD_overlap': 0,
                    'actual_lr': record['actual_lr'],
                    'Student_parameter_changed': True,
                    'grad_norm': record['grad_norm']})
            if step >= 2:
                source = ARTIFACTS / f'export_after_step{step}'
                exported = rank_zero(self.actor_rollout_wg.pilot_c_export_hf(
                    str(source)))
                gate(exported['weight_file_bytes'] > 0,
                     'trained Student export empty')
            if step in (8, 16, 32, 48):
                saved = checkpoint(self, records)
                append_sync(ARTIFACTS / 'periodic_checkpoints.jsonl', {
                    'step': step, 'path': saved['path'],
                    'bytes': saved['manifest']['total_bytes'],
                    'write_seconds': saved['manifest']['write_seconds']})
                print(f'PILOT_C_CHECKPOINT_COMPLETE=step{step}', flush=True)
        final = rank_zero(self.actor_rollout_wg.pilot_c_parameters('after_step64'))
        gate(any(x['changed'] for x in final['parameters'].values()),
             'Student parameters unchanged after 64 steps')
        check = checkpoint(self, records)
        return {'status': 'RESUME_PILOT_C64_PASS', 'global_steps': 64,
                'fresh_trajectories': 256, 'start_model': str(STUDENT),
                'eval_model_path': str(ARTIFACTS / 'export_after_step64'),
                'initial_parameter_snapshot': initial,
                'final_parameter_snapshot': final,
                'checkpoint': {'path': check['path'],
                               'bytes': check['manifest']['total_bytes']},
                'teacher_forward_calls': sum(x['teacher_forward_calls'] for x in records),
                'teacher_scored_code_tokens': sum(
                    x['teacher_scored_code_tokens'] for x in records),
                'teacher_input_tokens': sum(x['teacher_input_tokens'] for x in records),
                'teacher_forward_wall_seconds': sum(
                    x['teacher_forward_wall_seconds'] for x in records),
                'tool_opd_count': sum(x['tool_opd_count'] for x in records),
                'noncode_opd_count': sum(x['noncode_opd_count'] for x in records),
                'zero_code_group_count': sum(x['zero_code_group'] for x in records),
                'steps': records}


def main():
    gate(sys.executable == PYTHON and os.environ.get('PYTHONPATH') == str(REPO)
         and os.environ.get('D1_SEED') == str(SEED),
         'isolated runtime/seed differs')
    gate(json.loads((REPORTS / 'd1g3_result.json').read_text())['status'] ==
         'D1_G3_PASS', 'G3 must PASS before pilot')
    gate(not ARTIFACTS.exists(), 'C pilot output exists; no overwrite')
    preflight = gpu_preflight()
    ARTIFACTS.mkdir(parents=True)
    write_new(ARTIFACTS / 'gpu_preflight.json', preflight)
    seed_everything()
    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT), use_fast=True,
                                              local_files_only=True)
    gate(type(tokenizer).__name__ == 'Qwen2TokenizerFast' and
         tokenizer.is_fast, 'Student tokenizer differs')
    frozen_train_task(tokenizer, 1)
    cfg = config(ensure_task_slice())
    ray.init(num_gpus=2, include_dashboard=False,
             runtime_env={'env_vars': {
                 'PYTHONNOUSERSITE': '1', 'PYTHONPATH': str(REPO),
                 'TOKENIZERS_PARALLELISM': 'false', 'NCCL_DEBUG': 'WARN',
                 'D1_SEED': str(SEED), **RAY_RUN_ENV}})
    try:
        from verl.utils import hf_tokenizer, hf_processor
        mapping = {Role.ActorRollout: ray.remote(PilotCWorker)}
        pool = ResourcePoolManager({'pilot_c_pool': [2]},
                                   {Role.ActorRollout: 'pilot_c_pool'})
        trainer = Trainer(cfg, hf_tokenizer(str(STUDENT)), mapping, pool,
                          RayWorkerGroup,
                          processor=hf_processor(str(STUDENT), use_fast=True),
                          reward_fn=None, val_reward_fn=None)
        trainer.init_workers()
        return trainer.fit()
    finally:
        ray.shutdown()


if __name__ == '__main__':
    started = time.time()
    try:
        result = main()
    except Exception as exc:
        result = {'status': 'RESUME_PILOT_C_BLOCKED', 'error': str(exc),
                  'traceback': traceback.format_exc()}
    result['duration_seconds'] = round(time.time() - started, 2)
    if not (REPORTS / 'resume_pilot_training_C.json').exists():
        write_new(REPORTS / 'resume_pilot_training_C.json', result)
    print('PILOT_C_FINAL=' + json.dumps({
        'status': result['status'], 'duration_seconds': result['duration_seconds']}),
        flush=True)
    if result['status'] != 'RESUME_PILOT_C64_PASS':
        sys.exit(1)
