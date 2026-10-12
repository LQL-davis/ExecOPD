# ExecOPD

**Code-only online policy distillation for an execution-interactive coding agent**

ExecOPD turns one-shot code generation into a short agent loop: the Student writes Python, a restricted Docker tool executes it, and the Student can revise its code using the execution observation. During Method C training, a frozen Teacher scores the **same Student state and the same Student Top-K token IDs**. OPD supervision is restricted to tokens intersecting the selected executable code span; tool observations remain context only.

This repository is a **curated source release and aggregate pilot report**. It does not include model weights, datasets, raw trajectories, full checkpoints, the frozen task split, vendored veRL, or the pinned Docker image. The public path adjustments have passed static checks, but this copy has not been rerun as a new GPU experiment. See [Reproduction and limits](docs/REPRODUCTION.md).

## Resume Pilot / 32-task Validation Subset

Methods A and C each started independently from **Qwen2.5-7B-Instruct-1M** and trained for 64 global steps on the same first 64 frozen Code-R1 training tasks, with four fresh agent trajectories per task and at most two execution turns per trajectory. Method C used a frozen **Qwen2.5-14B-Instruct-1M** Teacher. The base Student and both trained models were evaluated on the same first 32 frozen validation tasks with one deterministic trajectory per task.

| Metric | Base Student | A: Agent GRPO, 64 steps | C: Code-only OPD, 64 steps |
|---|---:|---:|---:|
| Strict execution passes | 17 / 32 | 15 / 32 | 19 / 32 |
| Strict execution success rate | 53.1% | 46.9% | 59.4% |
| Average execution turns | 1.47 | 1.50 | 1.38 |

These are **pilot validation results**, not final-test or full Code-R1 benchmark results. C exceeded A by four passes (12.5 percentage points) on this subset; A did not improve over the base Student. No 512-step formal run or final-test evaluation is claimed. C used 409 measured Teacher forward calls to score 57,497 code tokens; its measured forward time was 63.9 seconds, excluding model loading and data transfer. The detailed aggregate counts are in [the pilot report](reports/RESUME_PILOT_64STEP_REPORT.md) and [machine-readable summary](reports/resume_pilot_summary.json).

## What is implemented

1. **Execution-interactive agent.** Message history, code artifact selection, bounded tool observations and turn termination are in [`agent_state.py`](experiments/ExecOPD/agent/agent_state.py), [`c2c_rollout.py`](experiments/ExecOPD/agent/c2c_rollout.py), [`code_artifact.py`](experiments/ExecOPD/agent/code_artifact.py) and [`secure_execution_tool.py`](experiments/ExecOPD/agent/secure_execution_tool.py). The pilot limits each trajectory to two execution turns and each response to 512 tokens.
2. **Exact token alignment.** [`pilot_turn_tensor.py`](experiments/ExecOPD/agent/pilot_turn_tensor.py) uses the actual sampled token IDs and the actual prompt IDs for each turn. It projects selected code character spans to generated tokens by nonempty span intersection. The [offline repair report](reports/RESUME_PILOT_ALIGNMENT_REPAIR_REPORT.md) records 124 frozen trajectories across 31 tasks, 199 exact turn alignments and zero code/tool overlap. This was an alignment regression, not a performance evaluation.
3. **Method A and C training.** [`run_pilot_A.py`](experiments/ExecOPD/agent/run_pilot_A.py) runs Student-only terminal-reward Agent GRPO. [`run_pilot_C.py`](experiments/ExecOPD/agent/run_pilot_C.py) and [`pilot_c_offline.py`](experiments/ExecOPD/agent/pilot_c_offline.py) implement Student Top-K=16, `only_stu`, `student_p`, Teacher same-state/same-ID scoring, code-token OPD masking and Actor update. OPD loss contributions from tool and non-code assistant positions are zero; the common entropy term still uses the generated-token mask.
4. **Stateful recovery and evaluation.** The pilot saves model, optimizer, scheduler, RNG and task cursor state; [`run_pilot_C_resume_latest_nvme.py`](experiments/ExecOPD/agent/run_pilot_C_resume_latest_nvme.py) contains the checked recovery path. [`run_validation_eval.py`](experiments/ExecOPD/agent/run_validation_eval.py) implements deterministic validation. Recovery scripts are specific to the recorded pilot run and require the matching checkpoint and run metadata.

The repository also includes a **separate two-step execution-aware credit prototype** (`run_c5a_credit_diagnostic.py`, `run_c5b_update.py`). Credit-aware Method D was **not** part of the 64-step comparison.

## Source layout

| Path | Contents |
|---|---|
| [`experiments/ExecOPD/agent`](experiments/ExecOPD/agent) | Agent loop, token masks, A/C pilot, credit smoke, validation and recovery |
| [`experiments/ExecOPD/sandbox`](experiments/ExecOPD/sandbox) | Fixed Docker execution policy and tests |
| [`reports`](reports) | Aggregate pilot result and alignment regression report |
| [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) | Dependencies, missing assets and verification scope |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | Upstream sources and release changes |

## Provenance

The agent experiment was built on [Code-R1](https://github.com/ganler/code-r1), with tasks from [Code-R1-12K](https://huggingface.co/datasets/ganler/code-r1-12k), and follows the Vanilla OPD setting investigated by [Rethinking-OPD](https://github.com/Thinking-Space/Rethinking-OPD). Model checkpoints are external Qwen assets. Upstream projects, datasets, models and their terms remain separate; this repository does not vendor them.

No license is included in this release.
