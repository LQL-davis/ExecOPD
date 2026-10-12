"""Read-only frozen D1 task resolver for the 64-step Resume Pilot.

Only train.parquet is opened. The final-test split is never resolved here.
"""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from experiments.ExecOPD.agent.agent_controller import AGENT_SYSTEM
from experiments.ExecOPD.agent.trajectory_serializer import render_history

REPO = REPO_ROOT
BASE = REPO / 'experiments/ExecOPD'
REPORTS = BASE / 'reports'
DATA = TRAIN_PARQUET
SPLIT = REPORTS / 'd1_data_split.json'
PROTOCOL = REPORTS / 'd1_training_config.json'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, default=str).encode('utf-8')


@lru_cache(maxsize=1)
def split():
    raw = SPLIT.read_bytes()
    protocol = json.loads(PROTOCOL.read_text())
    if sha(raw) != protocol['data_split_file_sha256']:
        raise RuntimeError('PILOT_SPLIT_BLOCKED: frozen split hash differs')
    obj = json.loads(raw)
    if obj['status'] != 'FROZEN' or len(obj['splits']['train']) != 512:
        raise RuntimeError('PILOT_SPLIT_BLOCKED: frozen train order unavailable')
    if sha(DATA.read_bytes()) != obj['source_files']['train']['sha256']:
        raise RuntimeError('PILOT_SPLIT_BLOCKED: train parquet hash differs')
    return obj


@lru_cache(maxsize=1)
def rows():
    return pq.read_table(DATA).to_pylist()


def frozen_train_task(tokenizer, step: int):
    if not 1 <= step <= 64:
        raise RuntimeError('PILOT_TASK_BLOCKED: step must be in frozen first 64')
    record = split()['splits']['train'][step - 1]
    if record['source_split'] != 'train':
        raise RuntimeError('PILOT_TASK_BLOCKED: unexpected source split')
    row = rows()[record['original_row_index']]
    if sha(canonical(row)) != record['source_row_sha256']:
        raise RuntimeError('PILOT_TASK_BLOCKED: row hash differs')
    problem = next(m['content'] for m in row['prompt'] if m['role'] == 'user')
    if sha(problem.encode('utf-8')) != record['problem_sha256']:
        raise RuntimeError('PILOT_TASK_BLOCKED: problem hash differs')
    history = [
        {'role': 'system', 'content': AGENT_SYSTEM, 'turn_id': None},
        {'role': 'user', 'content': problem, 'turn_id': None},
    ]
    context = render_history(tokenizer, history,
                             add_generation_prompt=True, artifacts_by_turn={})
    if len(context['token_ids']) != record['initial_agent_prompt_tokens'] or \
            len(context['token_ids']) > 512:
        raise RuntimeError('PILOT_TASK_BLOCKED: frozen prompt token count differs')
    return {'task_id': record['task_key'],
            'original_row_index': record['original_row_index'],
            'problem': problem,
            'ground_truth': row['reward_model']['ground_truth']}


def ensure_task_slice():
    """One-row veRL loader input; custom pilot loop uses frozen_train_task."""
    record = split()['splits']['train'][0]
    target = BASE / 'agent/data/resume_pilot_first_task.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        pd.read_parquet(DATA).iloc[[record['original_row_index']]].to_parquet(
            target, index=False)
    one = pq.read_table(target).to_pylist()
    if len(one) != 1 or sha(canonical(one[0])) != record['source_row_sha256']:
        raise RuntimeError('PILOT_TASK_BLOCKED: veRL loader slice mismatch')
    return target
