#!/usr/bin/env python3
"""C4-B4 fresh Agent collection and sequential same-state Code-only scoring.

Run collect and teacher as separate processes. This module never trains.
"""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = REPO_ROOT
BASE = REPO / 'experiments/ExecOPD'
REPORTS = BASE / 'reports'
TMP = BASE / 'agent/tmp/d1g1_c'
STUDENT = STUDENT_MODEL
TEACHER = TEACHER_MODEL
DATA = TRAIN_PARQUET
K = 16

sys.path.insert(0, str(REPO))
from experiments.ExecOPD.agent.c2c_rollout import collect_one  # noqa: E402
from experiments.ExecOPD.agent.c2c_tensor import tensorize  # noqa: E402
from experiments.ExecOPD.agent.secure_execution_tool import SecureExecutionTool  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def read(path):
    return json.loads(path.read_text())


def digest_tensor(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def gpu(stage):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    return {'stage': stage, 'physical_gpu': int(os.environ['CUDA_VISIBLE_DEVICES']),
            'free_bytes': free, 'total_bytes': total,
            'allocated_bytes': torch.cuda.memory_allocated(0),
            'reserved_bytes': torch.cuda.memory_reserved(0),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(0)}


def gate_gpu():
    assert torch.cuda.device_count() == 1
    assert 'H100' in torch.cuda.get_device_name(0)
    before = gpu('initial')
    assert before['free_bytes'] >= 60 * 1024**3, 'D1_G1_C_GPU_BLOCKED: insufficient free GPU memory'
    return before


from experiments.ExecOPD.agent.d1g1_common import frozen_task


def task(tokenizer, step):
    return frozen_task(tokenizer, step)


def load_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, device_map='cuda:0',
        low_cpu_mem_usage=True, local_files_only=True, trust_remote_code=False,
        attn_implementation='sdpa')
    model.eval()
    assert model.config.vocab_size == 152064
    return model


def forward_selected(model, item):
    positions = torch.nonzero(item['generated_token_mask'], as_tuple=False).flatten()
    assert len(positions) > 0 and int(positions.min()) > 0
    assert not bool(torch.any(item['generated_token_mask'] & item['execution_observation_mask']))
    with torch.no_grad():
        out = model(input_ids=item['input_ids'][None].to('cuda:0'),
                    attention_mask=item['attention_mask'][None].to('cuda:0'),
                    position_ids=item['position_ids'][None].to('cuda:0'),
                    use_cache=False, return_dict=True)
        logits = out.logits[0].index_select(0, (positions - 1).to('cuda:0')).float()
        assert logits.shape == (len(positions), 152064)
        return logits, positions


def inspect_item(tokenizer, trajectory):
    item, detail = tensorize(tokenizer, trajectory)
    assert detail['checks']['generated_tool_overlap_zero']
    assert detail['checks']['assistant_le_generated']
    assert detail['checks']['code_le_assistant']
    assert detail['checks']['code_tool_overlap_zero']
    assert int(item['generated_token_mask'].sum()) == detail['generated_count']
    assert int(item['code_token_mask'].sum()) == detail['code_count']
    return item, detail


def collect():
    REPORTS.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    path = REPORTS / 'd1g1_c_trajectories.jsonl'
    raw_path = REPORTS / 'd1g1_c_raw_samples.jsonl'
    assert not path.exists() and not raw_path.exists(), \
        'D1_G1_C_RUNTIME_BLOCKED: C4-B4 collection output exists; no reroll or overwrite'

    def append_raw(value):
        with raw_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT), use_fast=True, local_files_only=True)
    assert tokenizer.is_fast and type(tokenizer).__name__ == 'Qwen2TokenizerFast'
    fixed_by_step = {step: task(tokenizer, step) for step in (1, 2)}
    memory = {'phase': 'student_collect_topk', 'gpu_name': torch.cuda.get_device_name(0),
              'initial': gate_gpu(), 'model_path': str(STUDENT)}
    start = time.perf_counter()
    model = load_model(STUDENT)
    memory['load_seconds'] = time.perf_counter() - start
    memory['after_load'] = gpu('after_load')
    # C4-B4 requires new trajectories, independent of C3-B's frozen sample.
    torch.manual_seed(20261006)
    tool_events = []
    tool = SecureExecutionTool(log_event=tool_events.append)
    records = []
    path.open('x', encoding='utf-8').close()
    raw_path.open('x', encoding='utf-8').close()

    try:
        for step in (1, 2):
            fixed = fixed_by_step[step]
            for index in range(4):
                turn_counter = 0

                def generate(prompt_ids):
                    nonlocal turn_counter
                    assert 0 < len(prompt_ids) <= 1536
                    prompt = torch.tensor([prompt_ids], dtype=torch.long, device='cuda:0')
                    with torch.no_grad():
                        output = model.generate(input_ids=prompt,
                                                attention_mask=torch.ones_like(prompt),
                                                do_sample=True, temperature=1.0,
                                                top_p=1.0, top_k=0,
                                                max_new_tokens=512,
                                                eos_token_id=tokenizer.eos_token_id,
                                                pad_token_id=tokenizer.pad_token_id)
                    sampled = output[0, prompt.size(1):].tolist()
                    del prompt, output
                    assert 0 < len(sampled) <= 512
                    finish = ('eos' if sampled[-1] == tokenizer.eos_token_id
                              else 'length_or_stop')
                    append_raw({'event': 'sampled', 'global_step': step,
                                'trajectory_index': index, 'turn_index': turn_counter,
                                'task_id': fixed['task_id'],
                                'prompt_token_ids': list(prompt_ids),
                                'raw_response_text': tokenizer.decode(
                                    sampled, skip_special_tokens=True),
                                'raw_student_token_ids': sampled,
                                'eos_positions': [i for i, value in enumerate(sampled)
                                                  if value == tokenizer.eos_token_id],
                                'finish_reason': finish})
                    turn_counter += 1
                    return {'sampled_ids': sampled, 'finish_reason': finish,
                            'prompt_length': len(prompt_ids)}

                trajectory = collect_one(tokenizer, generate, tool, fixed,
                                         step=step, sample_index=index)
                messages = trajectory['state']['messages']
                for turn in trajectory['turns']:
                    turn_id = turn['turn_id']
                    through = 2
                    for _ in range(turn_id + 1):
                        through += 1
                        if through < len(messages) and messages[through]['role'] == 'tool':
                            through += 1
                    append_raw({'event': 'turn_snapshot', 'global_step': step,
                                'trajectory_index': index, 'turn_index': turn_id,
                                'task_id': trajectory['task_id'],
                                'prompt_token_ids': turn['prompt_token_ids'],
                                'raw_response_text': turn['raw_student_response_text'],
                                'raw_student_token_ids': turn['raw_student_token_ids'],
                                'finish_reason': turn['finish_reason'],
                                'messages_through_turn': messages[:through],
                                'full_final_messages': messages,
                                'selected_code_artifact': turn['code_artifact'],
                                'tool_observation': turn['execution_observation'],
                                'turn_record': turn})
                append_raw({'event': 'trajectory_snapshot', 'global_step': step,
                            'trajectory_index': index, 'trajectory': trajectory})
                print('D1G1C_RAW_PERSISTED=' + json.dumps({'step': step,
                      'index': index, 'turns': len(trajectory['turns'])}), flush=True)
                item, detail = inspect_item(tokenizer, trajectory)
                logits, positions = forward_selected(model, item)
                with torch.no_grad():
                    values, ids = torch.topk(logits, K, dim=-1)
                    logp = values - torch.logsumexp(logits, dim=-1, keepdim=True)
                    assert torch.isfinite(logp).all()
                    ids_cpu, logp_cpu = ids.cpu().tolist(), logp.cpu().tolist()
                row = {'global_step': step, 'trajectory_index': index,
                       'task_id': fixed['task_id'],
                       'input_ids_sha256': digest_tensor(item['input_ids']),
                       'sequence_length': len(item['input_ids']),
                       'generated_count': detail['generated_count'],
                       'code_count': detail['code_count'],
                       'tool_count': detail['tool_count'],
                       'boundary_crossing_token_count': sum(
                           int(span.get('boundary_crossing', False))
                           for span in detail['code_spans']),
                       'assistant_turns': len(trajectory['turns']),
                       'terminal_reason': trajectory['state']['terminal_reason'],
                       'positions': [
                           {'position': int(pos), 'turn_id': int(item['turn_ids'][pos]),
                            'sampled_token_id': int(item['input_ids'][pos]),
                            'student_topk_ids': ids_cpu[i],
                            'student_logprobs': logp_cpu[i]}
                           for i, pos in enumerate(positions.tolist())]}
                records.append(row)
                with path.open('a') as stream:
                    stream.write(json.dumps(trajectory, ensure_ascii=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                print('D1G1C_COLLECT=' + json.dumps({'step': step, 'index': index,
                      'generated': detail['generated_count'], 'code': detail['code_count'],
                      'tool': detail['tool_count'],
                      'turns': len(trajectory['turns'])}), flush=True)
                del logits, values, ids, logp, item
                torch.cuda.empty_cache()
        assert len(records) == 8 and len(tool_events) > 0
        assert hashlib.sha256(path.read_bytes()).hexdigest() != hashlib.sha256(
            (REPORTS / 'c3b_trajectories.jsonl').read_bytes()).hexdigest(), \
            'D1_G1_C_RUNTIME_BLOCKED: C4-B4 trajectories duplicate C3-B'
        memory['after_collection'] = gpu('after_collection')
        write(REPORTS / 'd1g1_c_student_topk.json', {
            'status': 'PASS', 'k': K, 'task_ids': [x['task_id'] for x in fixed_by_step.values()],
            'model_path': str(STUDENT), 'trajectory_count': 8,
            'trajectories_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'raw_samples_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            'sampling_seed': 20261006,
            'new_rollouts': 8, 'records': records})
        memory['secure_executor_calls'] = len(tool_events)
        write(TMP / 'student_memory.json', memory)
        print('D1G1C_STUDENT_COLLECT_PASS trajectories=8', flush=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def describe(values):
    assert values and all(math.isfinite(x) for x in values)
    mean = statistics.mean(values)
    return {'count': len(values), 'mean': mean, 'std_population': statistics.pstdev(values),
            'min': min(values), 'max': max(values),
            'abs_mean': statistics.mean(abs(x) for x in values)}


def teacher():
    path = REPORTS / 'd1g1_c_trajectories.jsonl'
    student = read(REPORTS / 'd1g1_c_student_topk.json')
    assert student['k'] == K and len(student['records']) == 8
    assert hashlib.sha256(path.read_bytes()).hexdigest() == student['trajectories_sha256']
    trajectories = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(trajectories) == 8
    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT), use_fast=True, local_files_only=True)
    memory = {'phase': 'teacher_same_state', 'gpu_name': torch.cuda.get_device_name(0),
              'initial': gate_gpu(), 'model_path': str(TEACHER)}
    start = time.perf_counter()
    model = load_model(TEACHER)
    memory['load_seconds'] = time.perf_counter() - start
    memory['after_load'] = gpu('after_load')
    watched = model.model.layers[14].self_attn.q_proj.weight
    before = hashlib.sha256(watched.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    output_records = []
    for trajectory, record in zip(trajectories, student['records']):
        assert trajectory['global_step'] == record['global_step']
        assert trajectory['trajectory_index'] == record['trajectory_index']
        item, detail = inspect_item(tokenizer, trajectory)
        assert digest_tensor(item['input_ids']) == record['input_ids_sha256']
        assert detail['code_count'] == record['code_count']
        logits, positions = forward_selected(model, item)
        assert positions.tolist() == [x['position'] for x in record['positions']]
        ids = torch.tensor([x['student_topk_ids'] for x in record['positions']],
                           dtype=torch.long, device='cuda:0')
        with torch.no_grad():
            gathered = logits.gather(1, ids)
            teacher_logp = gathered - torch.logsumexp(logits, dim=-1, keepdim=True)
            assert teacher_logp.shape == ids.shape and torch.isfinite(teacher_logp).all()
            teacher_values = teacher_logp.cpu().tolist()
        positions_out = []
        for source, values in zip(record['positions'], teacher_values):
            student_lp = source['student_logprobs']
            exp = [math.exp(x - max(student_lp)) for x in student_lp]
            weights = [x / sum(exp) for x in exp]
            vanilla_signal = [-(s - t) * w for s, t, w in zip(student_lp, values, weights)]
            is_code = bool(item['code_token_mask'][source['position']])
            signal = vanilla_signal if is_code else [0.0] * K
            assert len(signal) == K and all(math.isfinite(x) for x in signal)
            positions_out.append({'position': source['position'], 'turn_id': source['turn_id'],
                                  'gather_ids': source['student_topk_ids'],
                                  'teacher_logprobs': values, 'weights': weights,
                                  'code_target': is_code,
                                  'opd_signal': signal})
        output_records.append({'global_step': record['global_step'],
                               'trajectory_index': record['trajectory_index'],
                               'input_ids_sha256': record['input_ids_sha256'],
                               'generated_count': detail['generated_count'],
                               'code_count': detail['code_count'],
                               'tool_count': detail['tool_count'],
                               'teacher_scored_count': len(positions_out),
                               'teacher_scored_code_count': sum(p['code_target'] for p in positions_out),
                               'tool_target_count': 0,
                               'noncode_target_count': 0,
                               'positions': positions_out})
        print('D1G1C_TEACHER=' + json.dumps({'step': record['global_step'],
              'index': record['trajectory_index'], 'scored': len(positions_out)}), flush=True)
        del logits, ids, gathered, teacher_logp, item
        torch.cuda.empty_cache()
    after = hashlib.sha256(watched.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert before == after
    memory['after_scoring'] = gpu('after_scoring')
    memory['teacher_parameter_before_sha256'] = before
    memory['teacher_parameter_after_sha256'] = after
    write(TMP / 'teacher_memory.json', memory)
    write(REPORTS / 'd1g1_c_teacher_scores.json', {
        'status': 'PASS', 'k': K, 'model_path': str(TEACHER),
        'trajectories_sha256': student['trajectories_sha256'],
        'teacher_generated_trajectories': 0, 'teacher_parameter_unchanged': True,
        'records': output_records})
    totals = []
    per_step = []
    for step in (1, 2):
        subset = [r for r in output_records if r['global_step'] == step]
        values = [v for r in subset for p in r['positions'] if p['code_target']
                  for v in p['opd_signal']]
        assert values and any(v != 0 for v in values)
        totals.extend(values)
        per_step.append({'step': step, 'statistics': describe(values),
                         'generated_positions': sum(len(r['positions']) for r in subset),
                         'code_positions': sum(r['teacher_scored_code_count'] for r in subset),
                         'tool_target_count': 0, 'noncode_target_count': 0})
    write(REPORTS / 'd1g1_c_opd_signal.json', {
        'status': 'PASS', 'k': K, 'top_k_strategy': 'only_stu',
        'reward_weight_mode': 'student_p',
        'formula': '-(student_logprob-teacher_logprob)*softmax(student_topk_logprob)',
        'target_mask': 'code_token_mask',
        'statistics_on_code': describe(totals), 'per_step': per_step,
        'teacher_score_file': 'd1g1_c_teacher_scores.json'})
    del model, watched
    gc.collect()
    torch.cuda.empty_cache()
    print('D1G1C_TEACHER_PASS code_positions=' + str(sum(x['code_positions'] for x in per_step)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=('collect', 'teacher'))
    args = parser.parse_args()
    assert sys.executable == PYTHON_EXE
    assert os.environ.get('PYTHONPATH') == str(REPO)
    try:
        (collect if args.phase == 'collect' else teacher)()
    except Exception:
        traceback.print_exc()
        raise
