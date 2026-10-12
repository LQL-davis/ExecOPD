#!/usr/bin/env python3
"""Frozen D1 validation evaluator for G3 and the 32-task Resume Pilot.

This module never opens the Code-R1 test parquet or resolves final_test rows.
"""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import argparse
import collections
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

REPO = REPO_ROOT
BASE = REPO / 'experiments/ExecOPD'
from experiments.ExecOPD.agent.pilot_paths import REPORTS, LOGS
TRAIN = TRAIN_PARQUET
SPLIT = BASE / 'reports/d1_data_split.json'
PROTOCOL = BASE / 'reports/d1_training_config.json'
BASE_MODEL = STUDENT_MODEL
EXISTING_CHECKPOINT = BASE / 'reports/d1g2_artifacts/R/checkpoint_step2'
PILOT_CHECKPOINTS = {
    'A64': REPORTS / 'resume_pilot_artifacts/A/checkpoint_step64',
    'C64': REPORTS / 'resume_pilot_artifacts/C/checkpoint_step64',
}

sys.path.insert(0, str(REPO))
from experiments.ExecOPD.agent.agent_controller import AGENT_SYSTEM  # noqa: E402
from experiments.ExecOPD.agent.c2c_rollout import (  # noqa: E402
    MAX_MODEL_LEN, MAX_NEW_TOKENS, MAX_TURNS, collect_one)
from experiments.ExecOPD.agent.secure_execution_tool import SecureExecutionTool  # noqa: E402
from experiments.ExecOPD.agent.trajectory_serializer import render_history  # noqa: E402


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, default=str).encode('utf-8')


def write_new(path: Path, value) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
        stream.write('\n')


def checkpoint_snapshot(checkpoint: Path) -> dict:
    # The trained state is read-only during evaluation. Its prior manifest
    # already contains per-file SHA256; stat/manifest fingerprints detect writes.
    if not (checkpoint / 'CHECKPOINT_COMPLETE').is_file():
        raise RuntimeError('D1_G3_CHECKPOINT_BLOCKED: prior checkpoint marker missing')
    rows = []
    for path in sorted(checkpoint.rglob('*')):
        if path.is_file():
            stat = path.stat()
            rows.append((str(path.relative_to(checkpoint)),
                         stat.st_size, stat.st_mtime_ns))
    marker = (checkpoint / 'CHECKPOINT_COMPLETE').read_bytes()
    manifest = (checkpoint / 'manifest.json').read_bytes()
    return {'path': str(checkpoint), 'file_count': len(rows),
            'stat_fingerprint_sha256': digest(canonical(rows)),
            'manifest_sha256': digest(manifest), 'marker_sha256': digest(marker)}


def frozen_validation(tokenizer, limit: int):
    split_bytes = SPLIT.read_bytes()
    protocol = json.loads(PROTOCOL.read_text())
    if digest(split_bytes) != protocol['data_split_file_sha256']:
        raise RuntimeError('D1_G3_SPLIT_BLOCKED: frozen split file changed')
    split = json.loads(split_bytes)
    if split['status'] != 'FROZEN' or len(split['splits']['validation']) != 128:
        raise RuntimeError('D1_G3_SPLIT_BLOCKED: frozen validation split unavailable')
    if digest(TRAIN.read_bytes()) != split['source_files']['train']['sha256']:
        raise RuntimeError('D1_G3_SPLIT_BLOCKED: train data source changed')
    # The only parquet opened here is train.parquet. Frozen validation rows all
    # refer to that source. final_test IDs in split metadata are never resolved.
    rows = pq.read_table(TRAIN).to_pylist()
    tasks = []
    for item in split['splits']['validation'][:limit]:
        if item['source_split'] != 'train':
            raise RuntimeError('D1_G3_SPLIT_BLOCKED: validation source is not train')
        row = rows[item['original_row_index']]
        if digest(canonical(row)) != item['source_row_sha256']:
            raise RuntimeError('D1_G3_SPLIT_BLOCKED: validation row hash mismatch')
        problem = next(m['content'] for m in row['prompt'] if m['role'] == 'user')
        if digest(problem.encode()) != item['problem_sha256']:
            raise RuntimeError('D1_G3_SPLIT_BLOCKED: validation problem hash mismatch')
        context = render_history(tokenizer, [
            {'role': 'system', 'content': AGENT_SYSTEM, 'turn_id': None},
            {'role': 'user', 'content': problem, 'turn_id': None}],
            add_generation_prompt=True, artifacts_by_turn={})
        if len(context['token_ids']) != item['initial_agent_prompt_tokens']:
            raise RuntimeError('D1_G3_SPLIT_BLOCKED: initial prompt changed')
        tasks.append({'task_id': item['task_key'],
                      'original_row_index': item['original_row_index'],
                      'problem': problem,
                      'ground_truth': row['reward_model']['ground_truth']})
    if len(tasks) != limit or len({t['task_id'] for t in tasks}) != limit:
        raise RuntimeError('D1_G3_SPLIT_BLOCKED: validation order/count changed')
    return tasks, digest(split_bytes), split['split_sha256']['validation']


def gpu_gate() -> dict:
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible.isdigit() or torch.cuda.device_count() != 1:
        raise RuntimeError('D1_G3_GPU_BLOCKED: require exactly one explicit physical GPU')
    if 'H100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('D1_G3_GPU_BLOCKED: selected GPU is not H100')
    result = subprocess.run([
        'nvidia-smi', '-i', visible, '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    free_mib = int(result.stdout.strip())
    if free_mib < 76800:
        raise RuntimeError(f'D1_G3_GPU_BLOCKED: GPU {visible} free={free_mib} MiB')
    return {'physical_gpu': int(visible), 'initial_free_MiB': free_mib,
            'name': torch.cuda.get_device_name(0)}


def model_digest(engine) -> dict:
    executor = engine.llm_engine.model_executor
    worker = getattr(executor, 'driver_worker', None)
    runner = getattr(worker, 'model_runner', None)
    model = getattr(runner, 'model', None)
    if model is None:
        raise RuntimeError('D1_G3_PARAMETER_AUDIT_BLOCKED: vLLM model unavailable')
    h = hashlib.sha256()
    count = 0
    tensor_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            value = param.detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(value.view(torch.uint8).numpy().tobytes())
            count += value.numel()
            tensor_count += 1
    return {'sha256': h.hexdigest(), 'parameter_count': count,
            'tensor_count': tensor_count}


def metrics_from(trajectories, calls) -> dict:
    if len(trajectories) != len(calls):
        raise RuntimeError('D1_G3_METRICS_BLOCKED: task/call count mismatch')
    terminal = collections.Counter()
    failure = collections.Counter()
    turns = []
    passes = 0
    recoveries = 0
    recovery_eligible = 0
    records = []
    for tr, tool_calls in zip(trajectories, calls):
        reason = tr['state']['terminal_reason']
        terminal[reason] += 1
        if reason != 'PASS':
            failure[reason] += 1
        passes += int(reason == 'PASS')
        turns.append(len(tr['turns']))
        statuses = [x['status'] for x in tr['state']['execution_history']]
        eligible = len(statuses) >= 2 and any(s != 'PASS' for s in statuses[:-1])
        recovery_eligible += int(eligible)
        recovered = eligible and reason == 'PASS'
        recoveries += int(recovered)
        records.append({'task_id': tr['task_id'], 'terminal_reason': reason,
                        'execution_turns': len(tr['turns']),
                        'execution_statuses': statuses,
                        'failure_to_pass_recovery': recovered,
                        'secure_tool_calls': tool_calls})
    n = len(trajectories)
    return {'task_count': n, 'pass_count': passes,
            'execution_success_rate': passes / n,
            'average_execution_turns': sum(turns) / n,
            'failure_to_pass_recovery_count': recoveries,
            'failure_to_pass_recovery_eligible_count': recovery_eligible,
            'failure_to_pass_recovery_rate':
                recoveries / recovery_eligible if recovery_eligible else None,
            'terminal_status_counts': dict(terminal),
            'terminal_failure_types': dict(failure),
            'secure_tool_calls': sum(calls),
            'secure_tool_calls_per_task': sum(calls) / n,
            'tasks': records}


def output_paths(mode: str):
    if mode == 'gate':
        return {'result': REPORTS / 'd1g3_result.json',
                'metrics': REPORTS / 'd1g3_metrics.json',
                'access': REPORTS / 'd1g3_final_test_access_audit.json',
                'report': REPORTS / 'D1G3_VALIDATION_EVALUATOR_GATE_REPORT.md',
                'trajectories': REPORTS / 'd1g3_trajectories.jsonl',
                'tool_log': LOGS / 'd1g3_secure_tool.jsonl'}
    return {'result': REPORTS / f'resume_pilot_{mode}_eval.json',
            'metrics': REPORTS / f'resume_pilot_{mode}_metrics.json',
            'access': REPORTS / f'resume_pilot_{mode}_final_test_access_audit.json',
            'trajectories': REPORTS / f'resume_pilot_{mode}_trajectories.jsonl',
            'tool_log': LOGS / f'resume_pilot_{mode}_secure_tool.jsonl'}


def run(mode: str, model_path: Path) -> dict:
    if mode != 'gate' and json.loads((REPORTS / 'd1g3_result.json').read_text())['status'] != 'D1_G3_PASS':
        raise RuntimeError('D1_G3_REQUIRED: pilot evaluation cannot precede G3 PASS')
    limit = 8 if mode == 'gate' else 32
    paths = output_paths(mode)
    if any(path.exists() for path in paths.values()):
        raise RuntimeError('VALIDATION_OUTPUT_EXISTS: no implicit rerun or overwrite')
    if not model_path.is_dir() or not (model_path / 'config.json').is_file():
        raise RuntimeError('VALIDATION_MODEL_BLOCKED: model path unavailable')
    audited_checkpoint = PILOT_CHECKPOINTS.get(mode, EXISTING_CHECKPOINT)
    before = checkpoint_snapshot(audited_checkpoint)
    gpu = gpu_gate()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path),
                                              use_fast=True, local_files_only=True)
    if tokenizer.__class__.__name__ != 'Qwen2TokenizerFast' or not tokenizer.is_fast:
        raise RuntimeError('VALIDATION_TOKENIZER_BLOCKED: incompatible tokenizer')
    tasks, split_file_hash, validation_hash = frozen_validation(tokenizer, limit)
    os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
    from vllm import LLM, SamplingParams
    started = time.monotonic()
    engine = LLM(model=str(model_path), tokenizer=str(model_path),
                 tensor_parallel_size=1, dtype='bfloat16',
                 max_model_len=MAX_MODEL_LEN, max_num_seqs=1,
                 gpu_memory_utilization=0.35, enforce_eager=True,
                 trust_remote_code=False)
    model_before = model_digest(engine)
    trajectories = []
    calls = []
    with paths['trajectories'].open('x', encoding='utf-8') as stream, \
         paths['tool_log'].open('x', encoding='utf-8') as tool_stream:
        def tool_event(event):
            tool_stream.write(json.dumps(event, ensure_ascii=False) + '\n')
            tool_stream.flush()
        tool = SecureExecutionTool(tool_event)
        def generate(prompt_ids):
            params = SamplingParams(n=1, temperature=0.0, top_p=1.0,
                                    top_k=-1, max_tokens=MAX_NEW_TOKENS,
                                    detokenize=False, ignore_eos=False,
                                    seed=20261006)
            with torch.no_grad():
                outputs = engine.generate(prompt_token_ids=[prompt_ids],
                                          sampling_params=params, use_tqdm=False)
            if len(outputs) != 1 or len(outputs[0].outputs) != 1 or \
                    list(outputs[0].prompt_token_ids) != list(prompt_ids):
                raise RuntimeError('VALIDATION_GENERATION_BLOCKED: prompt ID mismatch')
            sample = outputs[0].outputs[0]
            return {'sampled_ids': list(sample.token_ids),
                    'finish_reason': sample.finish_reason}
        for index, task in enumerate(tasks):
            previous_calls = tool.call_count
            tr = collect_one(tokenizer, generate, tool, task,
                             step=0, sample_index=index)
            if tr['state']['terminal_reason'] == 'SECURE_EXECUTOR_ERROR':
                raise RuntimeError('VALIDATION_SECURE_EXECUTOR_BLOCKED')
            trajectories.append(tr)
            calls.append(tool.call_count - previous_calls)
            stream.write(json.dumps(tr, ensure_ascii=False, default=str) + '\n')
            stream.flush()
            print('VALIDATION_TASK=' + json.dumps({
                'mode': mode, 'index': index, 'task_id': task['task_id'],
                'terminal': tr['state']['terminal_reason'],
                'turns': len(tr['turns']), 'tool_calls': calls[-1]}), flush=True)
    model_after = model_digest(engine)
    duration = time.monotonic() - started
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    after = checkpoint_snapshot(audited_checkpoint)
    metrics = metrics_from(trajectories, calls)
    if mode == 'gate' and metrics['secure_tool_calls'] <= 0:
        raise RuntimeError('D1_G3_SECURE_TOOL_BLOCKED: no real secure Docker call observed')
    if [tr['task_id'] for tr in trajectories] != [task['task_id'] for task in tasks]:
        raise RuntimeError('VALIDATION_TASK_ORDER_BLOCKED: frozen validation order changed')
    access = {'status': 'PASS', 'resolved_split': 'validation',
              'validation_task_content_access_count': limit,
              'final_test_task_content_access_count': 0,
              'test_parquet_open_count': 0,
              'only_parquet_opened': str(TRAIN),
              'task_ids_sha256': digest(canonical([t['task_id'] for t in tasks])),
              'frozen_split_file_sha256': split_file_hash,
              'frozen_validation_split_sha256': validation_hash}
    if model_before != model_after or before != after:
        raise RuntimeError('VALIDATION_STATE_CHANGED: model or checkpoint changed')
    result = {'status': 'D1_G3_PASS' if mode == 'gate' else 'RESUME_PILOT_EVAL_PASS',
              'mode': mode, 'validation_subset_size': limit,
              'student_model_path': str(model_path), 'gpu': gpu,
              'sampling': {'temperature': 0.0, 'top_p': 1.0, 'top_k': -1,
                           'trajectories_per_task': 1, 'max_turns': MAX_TURNS,
                           'max_response_tokens_per_turn': MAX_NEW_TOKENS,
                           'secure_docker': True},
              'student_parameter_sha256_before': model_before['sha256'],
              'student_parameter_sha256_after': model_after['sha256'],
              'student_parameter_count': model_before['parameter_count'],
              'checkpoint_state_before': before,
              'checkpoint_state_after': after,
              'no_backward': True, 'no_optimizer': True,
              'teacher_loaded': False, 'teacher_generation_count': 0,
              'elapsed_seconds': duration,
              'final_test_task_content_access_count': 0}
    write_new(paths['metrics'], metrics)
    write_new(paths['access'], access)
    write_new(paths['result'], result)
    if mode == 'gate':
        lines = ['# D1-G3 Validation Evaluator Gate', '',
                 '**Result: D1_G3_PASS**', '',
                 f"Frozen validation subset: first {limit} tasks; split SHA256 `{validation_hash}`.",
                 'One deterministic trajectory per task. Temperature 0, top_p 1, unrestricted top_k, at most two execution turns and 512 tokens per turn. Secure Docker was used.', '',
                 f"Execution success: {metrics['pass_count']}/{limit} ({metrics['execution_success_rate']:.1%}).",
                 f"Average execution turns: {metrics['average_execution_turns']:.2f}.",
                 f"Failure-to-PASS recovery: {metrics['failure_to_pass_recovery_count']}/{metrics['failure_to_pass_recovery_eligible_count']} eligible tasks.",
                 f"Terminal failure categories: `{json.dumps(metrics['terminal_failure_types'], sort_keys=True)}`.",
                 f"Secure tool calls: {metrics['secure_tool_calls']}.", '',
                 'No backward, optimizer or Teacher was used. Full in-memory Student parameter digest and pre-existing checkpoint state are identical before/after evaluation.',
                 'Final-test task content access count: **0**; the test parquet was never opened.', '',
                 'This is an evaluator gate on eight frozen validation tasks, not a benchmark score.', '']
        with paths['report'].open('x', encoding='utf-8') as stream:
            stream.write('\n'.join(lines))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['gate', 'base', 'A64', 'C64'])
    parser.add_argument('--model-path', type=Path)
    args = parser.parse_args()
    if os.environ.get('PYTHONPATH') != str(REPO):
        raise RuntimeError('VALIDATION_RUNTIME_BLOCKED: wrong PYTHONPATH')
    model = BASE_MODEL if args.mode in ('gate', 'base') else args.model_path
    if model is None:
        raise RuntimeError('VALIDATION_MODEL_BLOCKED: explicit trained model required')
    print('VALIDATION_FINAL=' + json.dumps(run(args.mode, model)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        if '--mode' in sys.argv:
            try:
                mode = sys.argv[sys.argv.index('--mode') + 1]
                if mode == 'gate':
                    status = 'D1_G3_BLOCKED'
                    result_path = REPORTS / 'd1g3_result.json'
                    report_path = REPORTS / 'D1G3_VALIDATION_EVALUATOR_GATE_REPORT.md'
                    if not result_path.exists():
                        write_new(result_path, {'status': status, 'error': error,
                                                'pilot_started': False,
                                                'final_test_task_content_access_count': 0})
                    if not report_path.exists():
                        report_path.write_text(
                            '# D1-G3 Validation Evaluator Gate\n\n'
                            '**Result: D1_G3_BLOCKED**\n\n'
                            'The gate did not meet PASS. Pilot training was not started.\n\n'
                            '```text\n' + error + '\n```\n', encoding='utf-8')
            except Exception:
                print('VALIDATION_FAILURE_REPORT_ERROR=' + traceback.format_exc(), flush=True)
        print('VALIDATION_FAILURE=' + error, flush=True)
        raise
