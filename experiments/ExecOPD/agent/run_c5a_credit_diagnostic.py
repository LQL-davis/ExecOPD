#!/usr/bin/env python3
"""Offline execution-aware turn credit on frozen C4-B4 Agent trajectories."""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import hashlib
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from experiments.ExecOPD.agent.c2c_tensor import tensorize

BASE = EXPERIMENT_ROOT
REPORTS = BASE / 'reports'
TRAJECTORIES = REPORTS / 'c4b4_trajectories.jsonl'
SCORES = REPORTS / 'c4b4_teacher_scores.json'
MODEL = str(STUDENT_MODEL)
EXPECTED = {
    'c4b4_trajectories.jsonl': 'cef5d398602fb4f4b458f9f8127c5edea290214a87ce165c5c5c698dddabc87f',
    'c4b4_teacher_scores.json': '11f59187fbf350366df5725812fd06a91f1ffa94e304f7605ea602a44d7d3173',
}
TIERS = (0.0, 0.25, 0.5, 1.0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(condition, message):
    if not condition:
        raise RuntimeError('C5_A_BLOCKED: ' + message)


def save(name, value):
    (REPORTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def describe(values):
    gate(values and all(math.isfinite(x) for x in values), 'nonfinite/empty signal')
    return {'count': len(values), 'mean': statistics.mean(values),
            'abs_mean': statistics.mean(abs(x) for x in values),
            'std_population': statistics.pstdev(values),
            'min': min(values), 'max': max(values)}


def effective_revision(previous, current):
    record = current['revision_from_previous_code']
    if record is None:
        return False
    return bool(record['code_changed_between_turns'] and
                record['normalized_code_changed'] and
                not record['exactly_identical'] and
                record['character_edit_size'] > 0 and
                previous['code_artifact']['normalized_code_sha256'] !=
                current['code_artifact']['normalized_code_sha256'] and
                previous['code_artifact']['selected_code'] !=
                current['code_artifact']['selected_code'])


def assign_credit(turns, turn_index, final_pass):
    current = turns[turn_index]
    if current['execution_observation']['status'] == 'PASS':
        return 1.0, 'direct_execution_PASS'
    if turn_index + 1 >= len(turns):
        return 0.0, 'failed_without_later_revision'
    if not effective_revision(current, turns[turn_index + 1]):
        return 0.0, 'failed_then_repeated_or_no_effective_revision'
    if final_pass:
        return 0.5, 'failed_then_effective_revision_final_PASS'
    return 0.25, 'failed_then_effective_revision_final_failure'


def main():
    gate(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only stage required')
    source_sha = {name: sha(REPORTS / name) for name in EXPECTED}
    gate(source_sha == EXPECTED, 'frozen trajectory or signal SHA256 differs')
    result = json.loads((REPORTS / 'c4b4_result.json').read_text())
    gate(result['status'] == 'C4_B4_PASS', 'C4-B4 source stage did not pass')
    for name in ('C5A_EXECUTION_AWARE_CREDIT_REPORT.md', 'c5a_result.json',
                 'c5a_turn_credit.json', 'c5a_credit_mask_audit.json',
                 'c5a_weighted_signal_summary.json'):
        gate(not (REPORTS / name).exists(), 'existing C5-A output: ' + name)
    trajectories = [json.loads(x) for x in TRAJECTORIES.read_text().splitlines()]
    score_file = json.loads(SCORES.read_text())
    scores = score_file['records']
    gate(len(trajectories) == len(scores) == 8 and
         score_file['trajectories_sha256'] == source_sha['c4b4_trajectories.jsonl'] and
         score_file['k'] == 16, 'score file/trajectory mismatch')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    gate(tokenizer.is_fast and type(tokenizer).__name__ == 'Qwen2TokenizerFast',
         'original fast tokenizer unavailable')
    turn_rows = []
    mask_rows = []
    per_trajectory = []
    all_vanilla_values = []
    all_weighted_values = []
    all_credit_weights = []
    active_tokens = 0
    total_code_tokens = 0
    tier_tokens = Counter()
    status_counts = Counter()
    for trajectory, score in zip(trajectories, scores):
        step, index = trajectory['global_step'], trajectory['trajectory_index']
        gate((score['global_step'], score['trajectory_index']) == (step, index),
             'trajectory/Teacher score order mismatch')
        turns = trajectory['turns']
        terminal = trajectory['state']['terminal_reason']
        final_pass = terminal == 'PASS'
        gate((turns[-1]['execution_observation']['status'] == 'PASS') == final_pass,
             'terminal reason differs from final execution status')
        item, alignment = tensorize(tokenizer, trajectory)
        gate(all(alignment['checks'].values()) and
             all(x['exact'] for x in alignment['assistant_spans']),
             'sampled-ID/template/mask reconstruction failed')
        ids_hash = hashlib.sha256(item['input_ids'].numpy().tobytes()).hexdigest()
        gate(ids_hash == score['input_ids_sha256'], 'same-state input IDs differ')
        code_mask = item['code_token_mask'].bool()
        generated_mask = item['generated_token_mask'].bool()
        attention_mask = item['attention_mask'].bool()
        tool_mask = item['execution_observation_mask'].bool()
        turn_ids = item['turn_ids']
        gate(bool(torch.all(code_mask <= generated_mask)) and
             bool(torch.all(generated_mask <= attention_mask)) and
             not bool(torch.any(code_mask & tool_mask)) and
             not bool(torch.any(generated_mask & tool_mask)), 'source mask invariant failed')
        credit_by_turn = {}
        for j, turn in enumerate(turns):
            gate(turn['turn_id'] == j and turn['code_artifact']['extraction_status'] == 'SELECTED',
                 'turn/code artifact missing')
            credit, reason = assign_credit(turns, j, final_pass)
            gate(credit in TIERS, 'undefined credit tier')
            credit_by_turn[j] = credit
            obs = turn['execution_observation']
            previous_change = effective_revision(turns[j - 1], turn) if j > 0 else False
            later_change = (effective_revision(turn, turns[j + 1])
                            if j + 1 < len(turns) else False)
            row = {'global_step': step, 'trajectory_index': index,
                   'task_id': trajectory['task_id'], 'turn_id': j,
                   'execution_status': obs['status'], 'execution_exit_code': obs['exit_code'],
                   'code_sha256': turn['code_artifact']['sha256'],
                   'normalized_code_sha256': turn['code_artifact']['normalized_code_sha256'],
                   'code_changed_from_previous_turn': previous_change,
                   'effective_revision_after_this_turn': later_change,
                   'final_trajectory_status': terminal,
                   'terminal_reason': terminal, 'turn_credit': credit,
                   'credit_reason': reason}
            turn_rows.append(row)
            status_counts[obs['status']] += 1
        n = len(item['input_ids'])
        weight = torch.zeros(n, dtype=torch.float32)
        code_positions = torch.nonzero(code_mask, as_tuple=False).flatten().tolist()
        for position in code_positions:
            j = int(turn_ids[position])
            gate(j in credit_by_turn, 'code token has no assistant turn')
            weight[position] = credit_by_turn[j]
        gate(bool(torch.all(weight[~code_mask] == 0)) and
             not bool(torch.any(weight[tool_mask] != 0)) and
             not bool(torch.any(weight[generated_mask & ~code_mask] != 0)),
             'credit leaked to tool or non-code assistant tokens')
        position_rows = score['positions']
        gate(len(position_rows) == int(generated_mask.sum()) and
             score['teacher_scored_code_count'] == int(code_mask.sum()),
             'Teacher scoring/code count mismatch')
        signal_by_turn = {j: [] for j in credit_by_turn}
        weighted_by_turn = {j: [] for j in credit_by_turn}
        signal_values = []
        weighted_values = []
        for record in position_rows:
            position = record['position']
            gate(bool(generated_mask[position]) and not bool(tool_mask[position]) and
                 record['turn_id'] == int(turn_ids[position]) and
                 record['code_target'] == bool(code_mask[position]),
                 'same-ID position/turn/code mask mismatch')
            values = record['opd_signal']
            gate(len(values) == 16 and all(math.isfinite(v) for v in values),
                 'nonfinite original OPD signal')
            if not bool(code_mask[position]):
                gate(all(v == 0 for v in values), 'original OPD signal outside code')
                continue
            j = record['turn_id']
            credit = credit_by_turn[j]
            signal_by_turn[j].extend(values)
            weighted = [credit * value for value in values]
            weighted_by_turn[j].extend(weighted)
            signal_values.extend(values)
            weighted_values.extend(weighted)
            all_credit_weights.append(credit)
            tier_tokens[credit] += 1
        gate(len(signal_values) == int(code_mask.sum()) * 16 and
             all(math.isfinite(v) for v in weighted_values),
             'code-token weighted signal coverage/finite check')
        for row in turn_rows[-len(turns):]:
            j = row['turn_id']
            count = int((code_mask & (turn_ids == j)).sum())
            gate(count * 16 == len(signal_by_turn[j]) and count > 0,
                 'turn code-token count/OPD coverage mismatch')
            row['code_token_count'] = count
            row['original_code_only_opd_signal_abs_mean'] = describe(signal_by_turn[j])['abs_mean']
            row['weighted_signal_mean'] = statistics.mean(weighted_by_turn[j])
        code_count = int(code_mask.sum())
        active = int(torch.count_nonzero(weight))
        total_code_tokens += code_count
        active_tokens += active
        all_vanilla_values.extend(signal_values)
        all_weighted_values.extend(weighted_values)
        mask_rows.append({'global_step': step, 'trajectory_index': index,
                          'input_ids_sha256': ids_hash,
                          'sequence_length': n,
                          'generated_token_count': int(generated_mask.sum()),
                          'code_token_count': code_count,
                          'tool_token_count': int(tool_mask.sum()),
                          'active_credit_token_count': active,
                          'credit_weight_per_token': weight.tolist(),
                          'code_token_positions': code_positions,
                          'tool_credit_count': int(torch.count_nonzero(weight[tool_mask])),
                          'noncode_assistant_credit_count': int(torch.count_nonzero(
                              weight[generated_mask & ~code_mask])),
                          'noncode_any_credit_count': int(torch.count_nonzero(weight[~code_mask])),
                          'code_le_generated_le_attention': True,
                          'generated_tool_overlap_count': int((generated_mask & tool_mask).sum()),
                          'code_tool_overlap_count': int((code_mask & tool_mask).sum())})
        per_trajectory.append({'global_step': step, 'trajectory_index': index,
                               'terminal_reason': terminal,
                               'code_tokens': code_count,
                               'vanilla_active_tokens': code_count,
                               'credit_aware_active_tokens': active,
                               'vanilla_signal_mean': statistics.mean(signal_values),
                               'credit_weighted_signal_mean_over_all_code_slots':
                                   statistics.mean(weighted_values),
                               'credit_weighted_signal_mean_over_weight_mass':
                                   (sum(weighted_values) / (sum(weight[code_mask].tolist()) * 16)
                                    if active else None)})
    gate(len(turn_rows) == 14 and len(mask_rows) == 8 and total_code_tokens == 2472,
         'frozen turn/trajectory/code count changed')
    gate(sum(tier_tokens.values()) == total_code_tokens and
         active_tokens == sum(count for tier, count in tier_tokens.items() if tier > 0),
         'credit tier token accounting mismatch')
    gate({name: sha(REPORTS / name) for name in EXPECTED} == source_sha,
         'frozen input changed during audit')
    tier_data = {str(tier): {'token_count': tier_tokens[tier],
                            'fraction_of_all_code_tokens': tier_tokens[tier] / total_code_tokens}
                 for tier in TIERS}
    mass = sum(all_credit_weights)
    summary = {'status': 'PASS', 'source_sha256': source_sha,
               'comparison': {
                   'original_code_only_opd': {
                       'code_token_weight': 1.0,
                       'active_code_token_count': total_code_tokens,
                       'signal': describe(all_vanilla_values)},
                   'credit_aware_opd_diagnostic': {
                       'active_code_token_count': active_tokens,
                       'inactive_code_token_count': total_code_tokens - active_tokens,
                       'weighted_signal_over_all_code_slots': describe(all_weighted_values),
                       'weighted_signal_mean_normalized_by_weight_mass':
                           sum(all_weighted_values) / (mass * 16),
                       'credit_weight_mass': mass}},
               'credit_tier_token_distribution': tier_data,
               'per_trajectory': per_trajectory}
    gate(math.isfinite(summary['comparison']['credit_aware_opd_diagnostic']
                       ['weighted_signal_mean_normalized_by_weight_mass']),
         'weighted signal mean nonfinite')
    turn_output = {'status': 'PASS', 'rule_scope': 'failed turn receives credit if a later effective revision occurs',
                   'turn_count': len(turn_rows), 'execution_status_counts': dict(status_counts),
                   'credit_tier_turn_counts': {str(x): sum(r['turn_credit'] == x for r in turn_rows)
                                               for x in TIERS},
                   'turns': turn_rows}
    mask_output = {'status': 'PASS', 'source_sha256': source_sha,
                   'credit_only_on_code_mask': True,
                   'tool_credit_count': sum(x['tool_credit_count'] for x in mask_rows),
                   'noncode_assistant_credit_count': sum(
                       x['noncode_assistant_credit_count'] for x in mask_rows),
                   'all_noncode_credit_count': sum(x['noncode_any_credit_count'] for x in mask_rows),
                   'trajectories': mask_rows}
    gate(mask_output['tool_credit_count'] ==
         mask_output['noncode_assistant_credit_count'] ==
         mask_output['all_noncode_credit_count'] == 0,
         'credit mask leakage')
    result_output = {'status': 'C5_A_PASS', 'trajectory_count': 8,
                     'assistant_turn_count': len(turn_rows),
                     'terminal_pass_trajectories': sum(
                         t['state']['terminal_reason'] == 'PASS' for t in trajectories),
                     'code_token_count': total_code_tokens,
                     'active_credit_token_count': active_tokens,
                     'tool_credit_count': 0,
                     'noncode_assistant_credit_count': 0,
                     'all_weighted_signal_finite': True,
                     'student_load_count': 0, 'teacher_load_count': 0,
                     'new_rollout_count': 0, 'backward_count': 0,
                     'optimizer_step_count': 0,
                     'frozen_source_sha256': source_sha}
    save('c5a_turn_credit.json', turn_output)
    save('c5a_credit_mask_audit.json', mask_output)
    save('c5a_weighted_signal_summary.json', summary)
    save('c5a_result.json', result_output)
    report = f'''# C5-A Execution-Aware Turn Credit Diagnostic

**Result: C5_A_PASS.** This was a CPU-only offline diagnostic on eight frozen C4-B4 trajectories and their saved OPD signals. No rollout, Student/Teacher model load, backward, optimizer step, or reward change occurred.

## Frozen inputs and rule interpretation

Trajectory SHA256: `{source_sha['c4b4_trajectories.jsonl']}`. Teacher same-state score artifact SHA256: `{source_sha['c4b4_teacher_scores.json']}`. The original terminal reward, `code_token_mask`, Teacher scores, and OPD formula were read-only.

Credit belongs to the **failed turn before a later revision**. A direct execution PASS gets 1.0. A failed turn followed by an effective code revision that eventually yields terminal PASS gets 0.5. A failed turn followed by effective revision but terminal failure gets 0.25. A failed turn with no later effective revision gets 0.0. An effective revision requires a changed selected code body, changed normalized SHA256, and positive recorded edit size. Each turn record separately identifies whether its own code changed from the previous turn and whether a later revision followed it. A final failed turn with no later revision therefore gets 0.0, even if its code differed from the previous turn.

The eight trajectories contain {result_output['terminal_pass_trajectories']} terminal PASS outcomes and {8-result_output['terminal_pass_trajectories']} terminal failures. All {len(turn_rows)} assistant execution turns received a credit. Turn-tier counts are {turn_output['credit_tier_turn_counts']}; execution statuses are {turn_output['execution_status_counts']}.

## Token projection and signal comparison

The original Qwen2TokenizerFast and C4-B3 tensorizer rebuilt each frozen trajectory. Exact sampled-ID/template/mask checks passed. Every code token inherited its execution turn's credit, and the full `credit_weight_per_token` array is saved per trajectory in `c5a_credit_mask_audit.json`. All tool, non-code assistant, and other non-code token credit counts are zero.

| Diagnostic | Original Code-only OPD | Credit-aware OPD |
| --- | ---: | ---: |
| Code tokens | {total_code_tokens} | {total_code_tokens} |
| Active code tokens | {total_code_tokens} | {active_tokens} |
| Signal mean over all code × Top-K slots | {statistics.mean(all_vanilla_values):.8f} | {statistics.mean(all_weighted_values):.8f} |
| Absolute signal mean over all code × Top-K slots | {statistics.mean(abs(v) for v in all_vanilla_values):.8f} | {statistics.mean(abs(v) for v in all_weighted_values):.8f} |

The credit-weighted signal mean normalized by the sum of token credit weights is `{summary['comparison']['credit_aware_opd_diagnostic']['weighted_signal_mean_normalized_by_weight_mass']:.8f}`. Credit-tier token fractions are `{tier_data}`. The weighted statistics are finite. Per-turn execution status, code modification, final status, credit, code token count, and original Code-only OPD absolute mean are in `c5a_turn_credit.json`.

This diagnostic only computes potential future training weights. It does not apply them to Actor loss or begin Credit Assignment training or Selective Teacher Query.
'''
    (REPORTS / 'C5A_EXECUTION_AWARE_CREDIT_REPORT.md').write_text(report)
    print(json.dumps(result_output, ensure_ascii=False))


if __name__ == '__main__':
    main()
