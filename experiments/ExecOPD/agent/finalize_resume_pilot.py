#!/usr/bin/env python3
"""Summarize only completed, frozen 32-task Resume Pilot evaluations."""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import csv
import json
import math
from pathlib import Path

BASE = EXPERIMENT_ROOT
from experiments.ExecOPD.agent.pilot_paths import REPORTS


def read(name):
    return json.loads((REPORTS / name).read_text())


def write_new(name, value):
    with (REPORTS / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def gate(condition, detail):
    if not condition:
        raise RuntimeError('RESUME_PILOT_SUMMARY_BLOCKED: ' + detail)


def main():
    gate(read('d1g3_result.json')['status'] == 'D1_G3_PASS',
         'G3 evaluator gate not PASS')
    training_a = read('resume_pilot_training_A.json')
    training_c = read('resume_pilot_training_C.json')
    gate(training_a['status'] == 'RESUME_PILOT_A64_PASS' and
         training_c['status'] == 'RESUME_PILOT_C64_PASS' and
         training_a['global_step'] == training_c['global_steps'] == 64,
         'A/C 64-step independent training incomplete')
    gate(training_a['teacher_loaded'] is False and
         training_c['teacher_forward_calls'] > 0 and
         training_c['tool_opd_count'] == 0 and
         training_c['noncode_opd_count'] == 0,
         'A/C method or OPD mask audit failed')
    methods = {}
    ids_sha = None
    for mode in ('base', 'A64', 'C64'):
        result = read(f'resume_pilot_{mode}_eval.json')
        metrics = read(f'resume_pilot_{mode}_metrics.json')
        access = read(f'resume_pilot_{mode}_final_test_access_audit.json')
        gate(result['status'] == 'RESUME_PILOT_EVAL_PASS' and
             result['validation_subset_size'] == 32 and
             metrics['task_count'] == 32 and
             len(metrics['tasks']) == 32 and
             result['final_test_task_content_access_count'] == 0 and
             access['final_test_task_content_access_count'] == 0 and
             access['test_parquet_open_count'] == 0,
             mode + ' validation/final-test audit')
        if ids_sha is None:
            ids_sha = access['task_ids_sha256']
        gate(ids_sha == access['task_ids_sha256'],
             'A/C/Base validation task IDs differ')
        for key in ('execution_success_rate', 'average_execution_turns',
                    'secure_tool_calls_per_task'):
            gate(math.isfinite(float(metrics[key])),
                 mode + ' nonfinite metric ' + key)
        gate(metrics['pass_count'] == sum(
            task['terminal_reason'] == 'PASS' for task in metrics['tasks']),
            mode + ' pass count mismatch')
        methods[mode] = {'result': result, 'metrics': metrics,
                         'final_test_access': access}
    gate(len(training_a['steps']) == len(training_c['steps']) == 64,
         'A/C training step records incomplete')
    result = {
        'status': 'RESUME_PILOT_64STEP_COMPLETE',
        'label': 'Resume Pilot / 32-task Validation Subset',
        'formal_benchmark': False,
        'final_test_evaluated': False,
        'final_test_task_content_access_count': 0,
        'validation_task_ids_sha256': ids_sha,
        'training': {
            'A': {'steps': 64, 'fresh_trajectories': 256,
                  'checkpoint_bytes': training_a['checkpoint_bytes'],
                  'initialization': 'original 7B Student checkpoint'},
            'C': {'steps': 64, 'fresh_trajectories': 256,
                  'checkpoint_bytes': training_c['checkpoint']['bytes'],
                  'initialization': 'original 7B Student checkpoint',
                  'teacher_forward_calls': training_c['teacher_forward_calls'],
                  'teacher_scored_code_tokens':
                      training_c['teacher_scored_code_tokens'],
                  'teacher_input_tokens': training_c['teacher_input_tokens'],
                  'teacher_forward_wall_seconds':
                      training_c['teacher_forward_wall_seconds']}},
        'validation': {
            mode: {key: item['metrics'][key] for key in (
                'pass_count', 'execution_success_rate',
                'average_execution_turns',
                'failure_to_pass_recovery_count',
                'failure_to_pass_recovery_eligible_count',
                'failure_to_pass_recovery_rate',
                'terminal_failure_types', 'secure_tool_calls')}
            for mode, item in methods.items()},
        'C_minus_A_success_rate': (
            methods['C64']['metrics']['execution_success_rate'] -
            methods['A64']['metrics']['execution_success_rate']),
    }
    write_new('resume_pilot_summary.json', result)
    rows = [
        ('Pass count / 32', lambda m: str(m['pass_count'])),
        ('Execution success rate',
         lambda m: f"{m['execution_success_rate']:.4f}"),
        ('Average execution turns',
         lambda m: f"{m['average_execution_turns']:.4f}"),
        ('Failure-to-PASS recoveries',
         lambda m: str(m['failure_to_pass_recovery_count'])),
        ('Secure tool calls', lambda m: str(m['secure_tool_calls'])),
    ]
    with (REPORTS / 'resume_pilot_comparison.csv').open('x',
                                                        newline='',
                                                        encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Metric', 'Base', 'A-64', 'C-64'])
        for label, convert in rows:
            writer.writerow([label] + [convert(methods[mode]['metrics'])
                                        for mode in ('base', 'A64', 'C64')])
    base = methods['base']['metrics']
    a = methods['A64']['metrics']
    c = methods['C64']['metrics']
    lines = [
        '# Resume Pilot: A vs C at 64 Steps', '',
        '**Resume Pilot / 32-task Validation Subset**', '',
        'This is a small, frozen validation subset. It is not a formal '
        'benchmark or final-test result.', '',
        '## Protocol', '',
        '- Base, A-64, and C-64 use identical first 32 frozen validation tasks '
        'with temperature 0, top_p 1, unrestricted top_k, one trajectory per '
        'task, at most two execution turns, and secure Docker.',
        '- A and C independently start from the original '
        'Qwen2.5-7B-Instruct-1M Student. Each uses the first 64 frozen train '
        'tasks, four fresh trajectories per task, and seed 20261006.',
        '- C uses the frozen Qwen2.5-14B-Instruct-1M Teacher for same-state '
        'same-ID scoring; OPD targets selected executable code tokens only.',
        '- Final-test task content access count: **0**.', '',
        '## Results', '',
        '| Metric | Base | A-64 | C-64 |',
        '|---|---:|---:|---:|',
        f"| Pass / 32 | {base['pass_count']} | {a['pass_count']} | {c['pass_count']} |",
        f"| Success rate | {base['execution_success_rate']:.1%} | "
        f"{a['execution_success_rate']:.1%} | {c['execution_success_rate']:.1%} |",
        f"| Avg. execution turns | {base['average_execution_turns']:.2f} | "
        f"{a['average_execution_turns']:.2f} | {c['average_execution_turns']:.2f} |",
        f"| Failure→PASS recoveries | {base['failure_to_pass_recovery_count']} | "
        f"{a['failure_to_pass_recovery_count']} | "
        f"{c['failure_to_pass_recovery_count']} |",
        f"| Secure tool calls | {base['secure_tool_calls']} | "
        f"{a['secure_tool_calls']} | {c['secure_tool_calls']} |", '',
        'Terminal failure categories:', '',
        f"- Base: `{json.dumps(base['terminal_failure_types'], sort_keys=True)}`",
        f"- A-64: `{json.dumps(a['terminal_failure_types'], sort_keys=True)}`",
        f"- C-64: `{json.dumps(c['terminal_failure_types'], sort_keys=True)}`", '',
        '## Training and Teacher cost', '',
        f"- A and C each completed 64 global steps and 256 fresh Agent trajectories.",
        f"- C Teacher forward calls: {training_c['teacher_forward_calls']}.",
        f"- C Teacher-scored code tokens: "
        f"{training_c['teacher_scored_code_tokens']}.",
        f"- C measured Teacher forward wall time: "
        f"{training_c['teacher_forward_wall_seconds']:.1f} seconds. "
        'This excludes model load and data transfer.',
        f"- A checkpoint: {training_a['checkpoint_bytes']:,} bytes; "
        f"C checkpoint: {training_c['checkpoint']['bytes']:,} bytes.", '',
        '## Resume-ready technical contribution drafts', '',
        '1. Built a secure execution-interactive coding agent that captures '
        'multi-turn Student trajectories with exact sampled-token and tool '
        'observation boundaries.',
        '2. Integrated code-only, same-state Teacher supervision into a '
        'two-GPU FSDP Agent training path with mask audits and full checkpoint '
        'state capture.',
        '3. Ran a controlled 64-step A/C pilot from independent original '
        'Student initializations and evaluated Base/A/C on the same frozen '
        '32-task validation subset; report the measured pass counts above.', '',
        '## Evidence and limits', '',
        '- **Verified implementation:** secure Agent loop, token/code/tool masks, '
        'A/C training objectives, Teacher same-state scoring, checkpointing, '
        'and deterministic evaluator gate.',
        '- **Pilot result:** the table above covers 32 frozen validation tasks '
        'after 64 steps per method. It supports only a small-sample descriptive '
        'comparison.',
        '- **Outstanding:** full 512-step experiments and final-test evaluation '
        'were not run.', '',
    ]
    with (REPORTS / 'RESUME_PILOT_64STEP_REPORT.md').open(
            'x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    print('RESUME_PILOT_FINALIZED=' + json.dumps({
        'status': result['status'], 'pass_counts': {
            mode: methods[mode]['metrics']['pass_count']
            for mode in ('base', 'A64', 'C64')}}))


if __name__ == '__main__':
    main()
