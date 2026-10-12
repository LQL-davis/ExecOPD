#!/usr/bin/env python3
"""G3-gated Resume Pilot runner. Run only after GPUs 2 and 3 are idle."""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = REPO_ROOT
BASE = REPO / 'experiments/ExecOPD'
AGENT = BASE / 'agent'
from experiments.ExecOPD.agent.pilot_paths import REPORTS, LOGS
PYTHON = PYTHON_EXE
STUDENT = STUDENT_MODEL
A_EXPORT = REPORTS / 'resume_pilot_artifacts/A/export_step64'
C_EXPORT = REPORTS / 'resume_pilot_artifacts/C/export_after_step64'
STATE = REPORTS / 'resume_pilot_workflow_state.json'
AUDIT = REPORTS / 'resume_pilot_gpu_audit.jsonl'


def read_status(name):
    path = REPORTS / name
    return json.loads(path.read_text())['status'] if path.exists() else None


def append(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_state(value):
    value['updated_unix'] = time.time()
    temporary = STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, STATE)


def gpu_gate(stage):
    query = subprocess.run(
        ['nvidia-smi', '-i', '2,3',
         '--query-gpu=index,uuid,name,memory.free,utilization.gpu',
         '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    rows = [line.split(',') for line in query.stdout.splitlines()]
    if len(rows) != 2:
        raise RuntimeError(f'{stage}: exactly two H100 GPUs required')
    gpu_info = []
    for row in rows:
        index, uuid, name, free, util = [part.strip() for part in row]
        gpu_info.append({'index': int(index), 'uuid': uuid, 'name': name,
                         'free_mib': int(free), 'utilization_percent': int(util)})
    apps = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
         '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    uuids = {x['uuid'] for x in gpu_info}
    relevant = [line for line in apps.stdout.splitlines()
                if line.split(',')[0].strip() in uuids]
    audit = {'stage': stage, 'time_unix': time.time(), 'gpus': gpu_info,
             'compute_apps': relevant}
    append(AUDIT, audit)
    if {x['index'] for x in gpu_info} != {2, 3} or \
            any('H100' not in x['name'] or x['free_mib'] < 76800 or
                x['utilization_percent'] != 0 for x in gpu_info) or relevant:
        raise RuntimeError(f'{stage}: GPUs 2/3 are not genuinely idle; '
                           f'see {AUDIT}')
    return audit


def run(stage, script, arguments, visible, result_file, expected_status):
    current = read_status(result_file)
    if current == expected_status:
        return {'stage': stage, 'status': 'ALREADY_PASS'}
    if current is not None:
        raise RuntimeError(f'{stage}: prior result is {current}; no reroll')
    gpu_gate(stage)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': visible,
                'PYTHONPATH': str(REPO), 'PYTHONNOUSERSITE': '1',
                'TOKENIZERS_PARALLELISM': 'false',
                'D1_SEED': '20261006', 'PILOT_A_MODE': 'SAVE'})
    log = LOGS / f'{stage}.log'
    if log.exists():
        raise RuntimeError(f'{stage}: log already exists; no rerun')
    command = [PYTHON, str(AGENT / script)] + arguments
    started = time.time()
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=REPO, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
        write_state({'status': 'RUNNING', 'stage': stage, 'child_pid': child.pid,
                     'command': command, 'log': str(log), 'started_unix': started})
        while child.poll() is None:
            sample = subprocess.run(['nvidia-smi', '-i', '2,3',
                '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            append(AUDIT, {'stage': stage, 'time_unix': time.time(),
                           'sampled_gpu_memory_csv': sample.stdout,
                           'query_exit': sample.returncode})
            time.sleep(5)
        exit_code = child.returncode
    actual = read_status(result_file)
    record = {'stage': stage, 'command': command,
              'visible_physical_gpus': visible,
              'exit_code': exit_code,
              'actual_status': actual,
              'expected_status': expected_status,
              'elapsed_seconds': time.time() - started,
              'log': str(log)}
    append(REPORTS / 'resume_pilot_stage_events.jsonl', record)
    if exit_code != 0 or actual != expected_status:
        raise RuntimeError(f'{stage} blocked: status={actual}, '
                           f'exit={exit_code}; see {log}')
    # Ray/vLLM subprocesses can take a few seconds to exit after the driver.
    # Wait for natural cleanup only; never kill another user's GPU process.
    for attempt in range(20):
        try:
            gpu_gate(stage + '_cleanup')
            break
        except RuntimeError:
            if attempt == 19:
                raise RuntimeError(stage + ': GPU cleanup not confirmed')
            time.sleep(3)
    write_state({'status': 'RUNNING', 'stage': stage + '_complete', 'last_stage': record})
    return record


def main():
    if not (REPORTS / 'resume_pilot_static_preflight.json').is_file():
        raise RuntimeError('CPU static preflight missing')
    static = json.loads((REPORTS / 'resume_pilot_static_preflight.json').read_text())
    if static['status'] != 'PILOT_STATIC_PREFLIGHT_PASS' or \
            static['final_test_task_content_access_count'] != 0:
        raise RuntimeError('CPU static preflight invalid')
    LOGS.mkdir(parents=True, exist_ok=True)
    gpu_gate('workflow_initial')
    stages = []
    stages.append(run('d1g3_validation_gate', 'run_validation_eval.py',
                      ['--mode', 'gate'], '2', 'd1g3_result.json',
                      'D1_G3_PASS'))
    # The PASS status is read again immediately before every pilot stage.
    if read_status('d1g3_result.json') != 'D1_G3_PASS':
        raise RuntimeError('D1-G3 gate did not pass; stop')
    stages.append(run('resume_pilot_A64_train', 'run_pilot_A.py', [],
                      '2,3', 'resume_pilot_training_A.json',
                      'RESUME_PILOT_A64_PASS'))
    stages.append(run('resume_pilot_C64_train', 'run_pilot_C.py', [],
                      '2,3', 'resume_pilot_training_C.json',
                      'RESUME_PILOT_C64_PASS'))
    stages.append(run('resume_pilot_base_eval', 'run_validation_eval.py',
                      ['--mode', 'base', '--model-path', str(STUDENT)],
                      '2', 'resume_pilot_base_eval.json',
                      'RESUME_PILOT_EVAL_PASS'))
    stages.append(run('resume_pilot_A64_eval', 'run_validation_eval.py',
                      ['--mode', 'A64', '--model-path', str(A_EXPORT)],
                      '2', 'resume_pilot_A64_eval.json',
                      'RESUME_PILOT_EVAL_PASS'))
    stages.append(run('resume_pilot_C64_eval', 'run_validation_eval.py',
                      ['--mode', 'C64', '--model-path', str(C_EXPORT)],
                      '2', 'resume_pilot_C64_eval.json',
                      'RESUME_PILOT_EVAL_PASS'))
    with (LOGS / 'resume_pilot_finalize.log').open('xb') as stream:
        final = subprocess.run([PYTHON, str(AGENT / 'finalize_resume_pilot.py')],
                               cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    if final.returncode != 0 or read_status('resume_pilot_summary.json') != \
            'RESUME_PILOT_64STEP_COMPLETE':
        raise RuntimeError('Resume Pilot finalization failed; see log')
    result = {'status': 'RESUME_PILOT_64STEP_COMPLETE',
              'stages': stages, 'final_test_task_content_access_count': 0}
    write_state(result)
    print('RESUME_PILOT_WORKFLOW_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        previous = json.loads(STATE.read_text()) if STATE.exists() else {}
        write_state({'status': 'BLOCKED', 'error': str(exc), 'previous_state': previous})
        print('RESUME_PILOT_WORKFLOW_BLOCKED=' + str(exc), flush=True)
        raise
