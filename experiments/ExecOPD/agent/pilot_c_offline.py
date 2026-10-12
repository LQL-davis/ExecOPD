#!/usr/bin/env python3
"""One frozen-task, fresh Method C collection or same-state Teacher scoring.

The two phases run in separate processes. Neither phase trains a model.
"""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

REPO = REPO_ROOT
BASE = REPO / 'experiments/ExecOPD'
from experiments.ExecOPD.agent.pilot_paths import REPORTS
ARTIFACTS = REPORTS / 'resume_pilot_artifacts/C'
STUDENT = STUDENT_MODEL
TEACHER = TEACHER_MODEL
K = 16
SEED = 20261006
sys.path.insert(0, str(REPO))

from experiments.ExecOPD.agent.c2c_rollout import collect_one  # noqa: E402
from experiments.ExecOPD.agent.d1g1_c_offline import (  # noqa: E402
    digest_tensor, gpu, load_model)
from experiments.ExecOPD.agent.pilot_turn_tensor import tensorize as inspect_item
from experiments.ExecOPD.agent.pilot_turn_forward import forward_selected
from experiments.ExecOPD.agent.pilot_common import frozen_train_task  # noqa: E402
from experiments.ExecOPD.agent.secure_execution_tool import SecureExecutionTool  # noqa: E402


def gate(condition, detail):
    if not condition:
        raise RuntimeError('RESUME_PILOT_C_OFFLINE_BLOCKED: ' + detail)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.write('\n')


def append_sync(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def collect(step: int, source: Path):
    folder = ARTIFACTS / f'step{step}'
    gate(not folder.exists(), f'step{step} already exists; no reroll')
    folder.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT), use_fast=True,
                                              local_files_only=True)
    gate(type(tokenizer).__name__ == 'Qwen2TokenizerFast', 'wrong tokenizer')
    task = frozen_train_task(tokenizer, step)
    gate(source.is_dir() and (source / 'config.json').is_file(),
         'Student source export unavailable')
    for seed in (random.seed, np.random.seed, torch.manual_seed):
        seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started = time.perf_counter()
    memory = {'phase': 'student_collect_topk', 'step': step,
              'initial': gpu('initial'), 'source_model': str(source)}
    model = load_model(source)
    memory['after_load'] = gpu('after_load')
    memory['load_seconds'] = time.perf_counter() - started
    tool_events = []
    tool = SecureExecutionTool(log_event=tool_events.append)
    path = folder / 'trajectories.jsonl'
    raw_path = folder / 'raw_samples.jsonl'
    path.touch(exist_ok=False)
    raw_path.touch(exist_ok=False)
    records = []
    try:
        for index in range(4):
            turn_counter = 0

            def generate(prompt_ids):
                nonlocal turn_counter
                gate(0 < len(prompt_ids) <= 1536, 'prompt length')
                prompt = torch.tensor([prompt_ids], dtype=torch.long,
                                      device='cuda:0')
                with torch.no_grad():
                    output = model.generate(
                        input_ids=prompt, attention_mask=torch.ones_like(prompt),
                        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                        max_new_tokens=512, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id)
                sampled = output[0, prompt.size(1):].tolist()
                del prompt, output
                gate(0 < len(sampled) <= 512, 'sample length')
                finish = ('eos' if sampled[-1] == tokenizer.eos_token_id
                          else 'length_or_stop')
                append_sync(raw_path, {
                    'event': 'sampled', 'global_step': step,
                    'trajectory_index': index, 'turn_index': turn_counter,
                    'task_id': task['task_id'], 'prompt_token_ids': list(prompt_ids),
                    'raw_response_text': tokenizer.decode(sampled,
                                                          skip_special_tokens=True),
                    'raw_student_token_ids': sampled,
                    'eos_positions': [j for j, tok in enumerate(sampled)
                                      if tok == tokenizer.eos_token_id],
                    'finish_reason': finish})
                turn_counter += 1
                return {'sampled_ids': sampled, 'finish_reason': finish,
                        'prompt_length': len(prompt_ids)}

            trajectory = collect_one(tokenizer, generate, tool, task,
                                     step=step, sample_index=index)
            gate(trajectory['state']['terminal_reason'] != 'SECURE_EXECUTOR_ERROR',
                 'secure Docker failed')
            append_sync(raw_path, {'event': 'trajectory_snapshot',
                                   'global_step': step,
                                   'trajectory_index': index,
                                   'trajectory': trajectory})
            append_sync(path, trajectory)
            item, detail = inspect_item(tokenizer, trajectory)
            logits, positions = forward_selected(model, item)
            with torch.no_grad():
                values, ids = torch.topk(logits, K, dim=-1)
                logp = values - torch.logsumexp(logits, dim=-1, keepdim=True)
                gate(bool(torch.isfinite(logp).all()), 'Student Top-K nonfinite')
                ids_cpu, logp_cpu = ids.cpu().tolist(), logp.cpu().tolist()
            records.append({
                'global_step': step, 'trajectory_index': index,
                'task_id': task['task_id'],
                'input_ids_sha256': digest_tensor(item['input_ids']),
                'turn_inputs': detail['turn_inputs'],
                'carrier_is_model_input': False,
                'sequence_length': len(item['input_ids']),
                'generated_count': detail['generated_count'],
                'code_count': detail['code_count'],
                'tool_count': detail['tool_count'],
                'boundary_crossing_token_count': sum(
                    int(span.get('boundary_crossing', False))
                    for span in detail['code_spans']),
                'terminal_reason': trajectory['state']['terminal_reason'],
                'positions': [
                    {'position': int(pos),
                     'turn_id': int(item['turn_ids'][pos]),
                     'sampled_token_id': int(item['input_ids'][pos]),
                     'student_topk_ids': ids_cpu[j],
                     'student_logprobs': logp_cpu[j]}
                    for j, pos in enumerate(positions.tolist())]})
            print('PILOT_C_COLLECT=' + json.dumps({
                'step': step, 'trajectory_index': index,
                'generated': detail['generated_count'],
                'code': detail['code_count'],
                'turns': len(trajectory['turns']),
                'terminal': trajectory['state']['terminal_reason']}), flush=True)
            del item, logits, positions, values, ids, logp
            torch.cuda.empty_cache()
        gate(len(records) == 4, 'four fresh trajectories required')
        memory['after_collection'] = gpu('after_collection')
        memory['elapsed_seconds'] = time.perf_counter() - started
        write_new(folder / 'student_topk.json', {
            'status': 'PASS', 'step': step, 'task_id': task['task_id'],
            'k': K, 'model_path': str(source), 'trajectory_count': 4,
            'trajectories_sha256': sha(path),
            'raw_samples_sha256': sha(raw_path),
            'sampling_seed': SEED, 'secure_tool_calls': tool.call_count,
            'records': records})
        write_new(folder / 'student_memory.json', memory)
        write_new(folder / 'secure_tool_events.json', tool_events)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def teacher(step: int):
    folder = ARTIFACTS / f'step{step}'
    gate(not (folder / 'teacher_scores.json').exists(),
         'Teacher score exists; no rescore')
    student = json.loads((folder / 'student_topk.json').read_text())
    path = folder / 'trajectories.jsonl'
    gate(student['status'] == 'PASS' and student['k'] == K and
         student['step'] == step and sha(path) == student['trajectories_sha256'],
         'Student cache mismatch')
    trajectories = [json.loads(line) for line in path.read_text().splitlines()]
    gate(len(trajectories) == len(student['records']) == 4,
         'four same-state trajectories required')
    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT), use_fast=True,
                                              local_files_only=True)
    started = time.perf_counter()
    memory = {'phase': 'teacher_same_state', 'step': step,
              'initial': gpu('initial'), 'model_path': str(TEACHER)}
    model = load_model(TEACHER)
    memory['after_load'] = gpu('after_load')
    memory['load_seconds'] = time.perf_counter() - started
    watched = model.model.layers[14].self_attn.q_proj.weight
    before = hashlib.sha256(watched.detach().cpu().contiguous()
                            .view(torch.uint8).numpy().tobytes()).hexdigest()
    output_records = []
    forward_seconds = 0.0
    for trajectory, record in zip(trajectories, student['records']):
        gate(trajectory['global_step'] == record['global_step'] == step and
             trajectory['trajectory_index'] == record['trajectory_index'],
             'same-state trajectory order')
        item, detail = inspect_item(tokenizer, trajectory)
        gate(digest_tensor(item['input_ids']) == record['input_ids_sha256'] and
             detail['generated_count'] == len(record['positions']) and
             detail['turn_inputs'] == record['turn_inputs'],
             'same-state input IDs')
        torch.cuda.synchronize()
        start_forward = time.perf_counter()
        logits, positions = forward_selected(model, item)
        ids = torch.tensor([x['student_topk_ids'] for x in record['positions']],
                           dtype=torch.long, device='cuda:0')
        with torch.no_grad():
            gathered = logits.gather(1, ids)
            teacher_logp = gathered - torch.logsumexp(logits, dim=-1,
                                                      keepdim=True)
            gate(teacher_logp.shape == ids.shape and
                 bool(torch.isfinite(teacher_logp).all()),
                 'Teacher same-ID logprob nonfinite')
            teacher_values = teacher_logp.cpu().tolist()
        torch.cuda.synchronize()
        forward_seconds += time.perf_counter() - start_forward
        gate(positions.tolist() == [x['position'] for x in record['positions']],
             'Teacher scoring positions differ')
        positions_out = []
        for source, values in zip(record['positions'], teacher_values):
            student_lp = source['student_logprobs']
            exp = [math.exp(x - max(student_lp)) for x in student_lp]
            weights = [x / sum(exp) for x in exp]
            signal = [-(s - t) * w for s, t, w in
                      zip(student_lp, values, weights)]
            is_code = bool(item['code_token_mask'][source['position']])
            if not is_code:
                signal = [0.0] * K
            gate(all(math.isfinite(v) for v in signal), 'OPD nonfinite')
            positions_out.append({
                'position': source['position'], 'turn_id': source['turn_id'],
                'gather_ids': source['student_topk_ids'],
                'teacher_logprobs': values, 'code_target': is_code,
                'opd_signal': signal})
        output_records.append({
            'global_step': step, 'trajectory_index': record['trajectory_index'],
            'input_ids_sha256': record['input_ids_sha256'],
            'sequence_length': record['sequence_length'],
            'turn_inputs': detail['turn_inputs'],
            'teacher_forward_calls': len(detail['turn_inputs']),
            'teacher_input_tokens': sum(x['sequence_length'] for x in detail['turn_inputs']),
            'generated_count': detail['generated_count'],
            'code_count': detail['code_count'],
            'tool_count': detail['tool_count'],
            'teacher_scored_count': len(positions_out),
            'teacher_scored_code_count': sum(p['code_target'] for p in positions_out),
            'tool_target_count': 0, 'noncode_target_count': 0,
            'positions': positions_out})
        del item, logits, positions, ids, gathered, teacher_logp
        torch.cuda.empty_cache()
    after = hashlib.sha256(watched.detach().cpu().contiguous()
                           .view(torch.uint8).numpy().tobytes()).hexdigest()
    gate(before == after, 'Teacher parameter changed')
    memory['after_scoring'] = gpu('after_scoring')
    memory['teacher_forward_wall_seconds'] = forward_seconds
    memory['elapsed_seconds'] = time.perf_counter() - started
    memory['teacher_parameter_before_sha256'] = before
    memory['teacher_parameter_after_sha256'] = after
    write_new(folder / 'teacher_scores.json', {
        'status': 'PASS', 'step': step, 'k': K,
        'trajectories_sha256': student['trajectories_sha256'],
        'teacher_generation_count': 0,
        'teacher_parameter_unchanged': True,
        'teacher_forward_calls': sum(x['teacher_forward_calls'] for x in output_records),
        'teacher_scored_tokens': sum(len(x['positions']) for x in output_records),
        'teacher_scored_code_tokens': sum(x['teacher_scored_code_count']
                                          for x in output_records),
        'teacher_input_tokens': sum(x['teacher_input_tokens']
                                     for x in output_records),
        'teacher_forward_wall_seconds': forward_seconds,
        'records': output_records})
    write_new(folder / 'teacher_memory.json', memory)
    del model, watched
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('collect', 'teacher'), required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    gate(sys.executable == PYTHON_EXE and
         os.environ.get('PYTHONPATH') == str(REPO) and
         os.environ.get('CUDA_VISIBLE_DEVICES') == '2' and
         os.environ.get('D1_SEED') == str(SEED),
         'frozen runtime/GPU/seed')
    gate(1 <= args.step <= 64, 'frozen task cursor')
    if args.phase == 'collect':
        gate(args.model is not None, 'Student model path required')
        collect(args.step, args.model)
    else:
        teacher(args.step)


if __name__ == '__main__':
    main()
