# Reproduction and limits

This release makes the custom ExecOPD source and aggregate evidence inspectable. It is **not a turnkey reproduction bundle** for the recorded Resume Pilot. The reports describe runs performed on the original server source. The public source copy changes path resolution and documentation only; no new training or validation result is attributed to those edits.

## External assets needed

- A compatible [Code-R1](https://github.com/ganler/code-r1) checkout with its veRL training tree. This release does not include that upstream tree or any local upstream patches.
- The [Code-R1-12K](https://huggingface.co/datasets/ganler/code-r1-12k) `train.parquet` and `test.parquet` files. The task schema, source hashes and frozen row identities must match the original experiment to compare exact task-level results.
- Qwen2.5-7B-Instruct-1M Student and Qwen2.5-14B-Instruct-1M Teacher checkpoints. Full token-to-ID mapping, special IDs and chat template compatibility were checked for the recorded assets; alternate revisions need a new compatibility audit.
- The pinned local Docker image in `sandbox/secure_docker_exec_pytest.py`, with digest `sha256:670c4859fb73abd3d2ca933302c73656ea85aef1c514650a9efce34af155c39b`. No Dockerfile or image artifact is included. The executor intentionally fails closed if the fixed image or security policy is unavailable.
- The frozen D1 split/config, validation gate result, run identity and (for continuation) complete checkpoint metadata. Those per-task and state artifacts are not published. `freeze_d1_split.py` documents the split construction, but regenerating a split from different inputs does not reproduce the historical split hash.
- For the recorded pilot configuration, two idle H100 80GB cards exposed as physical GPU 2 and 3, secure Docker, and sufficient checkpoint space. The scripts retain the observed hardware and admission gates rather than silently changing the experiment.

The original environment used Python 3.10, PyTorch 2.4.0, Transformers 4.45.0, vLLM 0.6.3, Ray 2.10.0, NumPy 1.26.4, pandas 2.2.3, PyArrow 15.0.0 and OmegaConf 2.3.0. Package versions alone do not reproduce the original custom veRL source or CUDA build.

## How the source is arranged

`experiments/ExecOPD` is designed to sit inside the matching Code-R1 checkout so imports such as `experiments.ExecOPD.agent` and `verl` resolve together. `agent/release_paths.py` replaces private machine paths with repository-relative paths and optional environment variables:

| Variable | Default relative to Code-R1 root |
|---|---|
| `EXECOPD_CODE_R1_ROOT` | Repository containing `verl/` and `experiments/` |
| `EXECOPD_STUDENT_MODEL` | `models/Qwen2.5-7B-Instruct-1M` |
| `EXECOPD_TEACHER_MODEL` | `models/Qwen2.5-14B-Instruct-1M` |
| `EXECOPD_TRAIN_PARQUET` | `data/code-r1-12k/train.parquet` |
| `EXECOPD_PYTHON` | Current Python executable |
| `EXECOPD_SCRATCH_DIR` | `experiments/ExecOPD/runs/tmp` |

The two training methods independently reload the original Student; there is no A-to-C warm start. The recorded pilot uses seed `20261006`, 64 distinct frozen train tasks, four fresh trajectories per task, at most two execution turns, no outcome-driven reroll, a 512-token response budget per turn, and K=16 Student candidates for Method C. Its evaluator uses temperature 0, `top_p=1`, unrestricted `top_k`, one trajectory per frozen validation task and the same secure executor. The frozen 32-task validation ID list has SHA256 `84adaf2363f127d1cd1c1320eb6ae4399f8ac662a54f6ed1ad3dc32217092513` in the aggregate report; the ID list itself is not part of this source release.

## Verification of this release

The public Python modules were compiled and scanned for private machine paths and credential-like strings. A small standard-library code-artifact check was run locally. The full Agent tests require the external tokenizer and Docker image; GPU training, validation and checkpoint resume were **not rerun from this sanitized public copy**. Historical run claims come from the bundled aggregate reports.

The 31-task/124-trajectory alignment repair was a frozen-data regression. Its report predates the completed A/C pilot; its then-pending GPU gate is historical context, while `RESUME_PILOT_64STEP_REPORT.md` contains the later completed pilot result.
