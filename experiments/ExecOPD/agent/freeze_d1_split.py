"""Freeze ExecOPD D1 row identities and deterministic, leakage-checked splits.

This script reads dataset and tokenizer only. It does not execute code, generate,
score, evaluate, or train any model.
"""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

REPO = REPO_ROOT
BASE = REPO / 'experiments/ExecOPD'
REPORTS = BASE / 'reports'
SOURCE = TRAIN_PARQUET.parent
STUDENT = STUDENT_MODEL
SEED = 20261006
TRAIN_COUNT = 512
VALIDATION_COUNT = 128

sys.path.insert(0, str(REPO))
from experiments.ExecOPD.agent.agent_controller import AGENT_SYSTEM
from experiments.ExecOPD.agent.trajectory_serializer import render_history


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      default=str).encode('utf-8')


def eligible_rows(split: str, tokenizer):
    source = SOURCE / f'{split}.parquet'
    rows = pq.read_table(source).to_pylist()
    accepted = []
    rejects = Counter()
    for i, row in enumerate(rows):
        users = [m.get('content') for m in (row.get('prompt') or [])
                 if m.get('role') == 'user']
        if len(users) != 1 or not isinstance(users[0], str) or not users[0].strip():
            rejects['missing_or_ambiguous_user_prompt'] += 1
            continue
        problem = users[0]
        try:
            truth = json.loads(row['reward_model']['ground_truth'])
        except (TypeError, ValueError, KeyError):
            rejects['invalid_ground_truth_json'] += 1
            continue
        if not isinstance(truth, dict):
            rejects['invalid_ground_truth_schema'] += 1
            continue
        if 'functional' in truth or 'pytest' in truth:
            field = 'functional' if 'functional' in truth else 'pytest'
            if not isinstance(truth[field], str):
                rejects['invalid_ground_truth_schema'] += 1
                continue
        elif 'inputs' in truth and 'outputs' in truth:
            ins, outs = truth['inputs'], truth['outputs']
            if not isinstance(ins, list) or not isinstance(outs, list) or not ins or len(ins) != len(outs) or not all(isinstance(v, str) for v in ins + outs):
                rejects['invalid_ground_truth_schema'] += 1
                continue
        else:
            rejects['unsupported_ground_truth_schema'] += 1
            continue
        messages = [{'role': 'system', 'content': AGENT_SYSTEM, 'turn_id': None},
                    {'role': 'user', 'content': problem, 'turn_id': None}]
        tokens = render_history(tokenizer, messages, add_generation_prompt=True,
                                artifacts_by_turn={})['token_ids']
        if len(tokens) > 512:
            rejects['initial_prompt_over_512_tokens'] += 1
            continue
        problem_sha = sha(problem.encode('utf-8'))
        native_id = row.get('task_id')
        native_id = str(native_id) if native_id is not None and str(native_id).strip() else None
        accepted.append({
            'task_key': f'{split}:row:{i}:{native_id or problem_sha[:16]}',
            'source_split': split,
            'original_row_index': i,
            'native_task_id': native_id,
            'problem_sha256': problem_sha,
            'source_row_sha256': sha(canonical(row)),
            'initial_agent_prompt_tokens': len(tokens),
        })
    return rows, accepted, dict(rejects)


def unique_problem(records):
    seen = set()
    output = []
    dupes = 0
    for record in records:
        if record['problem_sha256'] in seen:
            dupes += 1
            continue
        seen.add(record['problem_sha256'])
        output.append(record)
    return output, dupes


def digest_records(records):
    return sha(canonical(records))


def main():
    target = REPORTS / 'd1_data_split.json'
    if target.exists():
        raise SystemExit('D1 split already exists; refusing to replace frozen output')
    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT), use_fast=True,
                                              local_files_only=True)
    assert tokenizer.is_fast and type(tokenizer).__name__ == 'Qwen2TokenizerFast'
    train_rows, train_candidates, train_rejects = eligible_rows('train', tokenizer)
    test_rows, test_candidates, test_rejects = eligible_rows('test', tokenizer)
    # Detect overlap against *every* original test prompt, including excluded rows.
    all_test_hashes = set()
    for row in test_rows:
        users = [m.get('content') for m in (row.get('prompt') or [])
                 if m.get('role') == 'user']
        if len(users) == 1 and isinstance(users[0], str):
            all_test_hashes.add(sha(users[0].encode('utf-8')))
    train_no_test = [r for r in train_candidates if r['problem_sha256'] not in all_test_hashes]
    train_unique, train_duplicate_count = unique_problem(train_no_test)
    final_test, test_duplicate_count = unique_problem(test_candidates)
    assert not ({r['problem_sha256'] for r in train_unique} &
                {r['problem_sha256'] for r in final_test})
    # Fixed seed controls row ordering. Validation comes first; the next 512
    # distinct tasks are the exact one-pass formal training schedule.
    ordered = sorted(train_unique, key=lambda r: (sha(f"{SEED}: {r['problem_sha256']}".encode()),
                                                 r['original_row_index']))
    assert len(ordered) >= TRAIN_COUNT + VALIDATION_COUNT
    validation = ordered[:VALIDATION_COUNT]
    train = ordered[VALIDATION_COUNT:VALIDATION_COUNT + TRAIN_COUNT]
    assert len(train) == 512 and len(validation) == 128 and final_test
    assert not ({r['problem_sha256'] for r in train} &
                {r['problem_sha256'] for r in validation})
    assert not ({r['problem_sha256'] for r in validation} &
                {r['problem_sha256'] for r in final_test})
    source_hashes = {s: sha((SOURCE / f'{s}.parquet').read_bytes()) for s in ('train', 'test')}
    result = {
        'status': 'FROZEN',
        'rule': 'Code-R1 train/test sources; secure-tool schema; one nonempty user prompt; initial Agent chat-template prompt <=512; exact user-prompt SHA256 dedup; remove train prompts appearing anywhere in original test; seed-20261006 SHA256 ordering; first 128 validation and next 512 one-pass train; all eligible unique test prompts final test',
        'seed': SEED,
        'source_files': {s: {'path': str(SOURCE / f'{s}.parquet'), 'sha256': source_hashes[s],
                             'row_count': len(train_rows if s == 'train' else test_rows)}
                         for s in ('train', 'test')},
        'student_tokenizer_path': str(STUDENT),
        'student_tokenizer_json_sha256': sha((STUDENT / 'tokenizer.json').read_bytes()),
        'agent_system_sha256': sha(AGENT_SYSTEM.encode('utf-8')),
        'eligibility': {'max_initial_agent_prompt_tokens': 512,
                        'train_eligible_rows': len(train_candidates),
                        'test_eligible_rows': len(test_candidates),
                        'train_rejected': train_rejects, 'test_rejected': test_rejects,
                        'train_rows_overlapping_any_original_test_prompt': len(train_candidates)-len(train_no_test),
                        'train_duplicate_problem_rows_after_purge': train_duplicate_count,
                        'test_duplicate_problem_rows': test_duplicate_count,
                        'unused_eligible_train_unique_tasks': len(ordered)-TRAIN_COUNT-VALIDATION_COUNT,
                        'full_original_test_covered': False},
        'splits': {'train': train, 'validation': validation, 'final_test': final_test},
        'split_sha256': {s: digest_records(r) for s, r in
                         [('train', train), ('validation', validation), ('final_test', final_test)]},
    }
    for name, records in result['splits'].items():
        assert len({r['task_key'] for r in records}) == len(records)
        assert len({r['problem_sha256'] for r in records}) == len(records)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'path': str(target), 'sha256': sha(target.read_bytes()),
                      'counts': {s: len(r) for s, r in result['splits'].items()},
                      'split_sha256': result['split_sha256'],
                      'eligibility': result['eligibility']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
