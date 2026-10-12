"""Read-only D1 task resolution for the four two-step objective gates."""
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


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, default=str).encode('utf-8')


@lru_cache(maxsize=1)
def split():
    obj = json.loads((REPORTS / 'd1_data_split.json').read_text())
    assert obj['status'] == 'FROZEN'
    assert sha(DATA.read_bytes()) == obj['source_files']['train']['sha256']
    assert len(obj['splits']['train']) == 512
    return obj


@lru_cache(maxsize=1)
def rows():
    return pq.read_table(DATA).to_pylist()


def frozen_task(tokenizer, step: int):
    assert step in (1, 2)
    record = split()['splits']['train'][step - 1]
    i = record['original_row_index']
    row = rows()[i]
    assert sha(canonical(row)) == record['source_row_sha256']
    problem = next(m['content'] for m in row['prompt'] if m['role'] == 'user')
    assert sha(problem.encode('utf-8')) == record['problem_sha256']
    context = render_history(tokenizer, [
        {'role':'system','content':AGENT_SYSTEM,'turn_id':None},
        {'role':'user','content':problem,'turn_id':None}],
        add_generation_prompt=True, artifacts_by_turn={})
    assert len(context['token_ids']) == record['initial_agent_prompt_tokens'] <= 512
    return {'task_id': record['task_key'], 'original_row_index': i,
            'problem': problem, 'ground_truth': row['reward_model']['ground_truth']}


def ensure_task_slice():
    record = split()['splits']['train'][0]
    target = BASE / 'agent/data/d1g1_first_task.parquet'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        pd.read_parquet(DATA).iloc[[record['original_row_index']]].to_parquet(target, index=False)
    assert len(pd.read_parquet(target)) == 1
    one = pq.read_table(target).to_pylist()[0]
    assert sha(canonical(one)) == record['source_row_sha256']
    return target
