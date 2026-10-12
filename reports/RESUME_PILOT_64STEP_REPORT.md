# Resume Pilot: A vs C at 64 Steps

**Resume Pilot / 32-task Validation Subset**

This is a small, frozen validation subset. It is not a formal benchmark or final-test result.

## Protocol

- Base, A-64, and C-64 use identical first 32 frozen validation tasks with temperature 0, top_p 1, unrestricted top_k, one trajectory per task, at most two execution turns, and secure Docker.
- A and C independently start from the original Qwen2.5-7B-Instruct-1M Student. Each uses the first 64 frozen train tasks, four fresh trajectories per task, and seed 20261006.
- C uses the frozen Qwen2.5-14B-Instruct-1M Teacher for same-state same-ID scoring; OPD targets selected executable code tokens only.
- Final-test task content access count: **0**.

## Results

| Metric | Base | A-64 | C-64 |
|---|---:|---:|---:|
| Pass / 32 | 17 | 15 | 19 |
| Success rate | 53.1% | 46.9% | 59.4% |
| Avg. execution turns | 1.47 | 1.50 | 1.38 |
| Failure→PASS recoveries | 1 | 0 | 0 |
| Secure tool calls | 100 | 93 | 95 |

Terminal failure categories:

- Base: `{"ACTION_FORMAT_ERROR": 1, "MAX_TURNS": 14}`
- A-64: `{"ACTION_FORMAT_ERROR": 1, "MAX_TURNS": 16}`
- C-64: `{"ACTION_FORMAT_ERROR": 1, "MAX_TURNS": 12}`

## Training and Teacher cost

- A and C each completed 64 global steps and 256 fresh Agent trajectories.
- C Teacher forward calls: 409.
- C Teacher-scored code tokens: 57497.
- C measured Teacher forward wall time: 63.9 seconds. This excludes model load and data transfer.
- A checkpoint: 91,403,771,993 bytes; C checkpoint: 91,403,772,002 bytes.

## Resume-ready technical contribution drafts

1. Built a secure execution-interactive coding agent that captures multi-turn Student trajectories with exact sampled-token and tool observation boundaries.
2. Integrated code-only, same-state Teacher supervision into a two-GPU FSDP Agent training path with mask audits and full checkpoint state capture.
3. Ran a controlled 64-step A/C pilot from independent original Student initializations and evaluated Base/A/C on the same frozen 32-task validation subset; report the measured pass counts above.

## Evidence and limits

- **Verified implementation:** secure Agent loop, token/code/tool masks, A/C training objectives, Teacher same-state scoring, checkpointing, and deterministic evaluator gate.
- **Pilot result:** the table above covers 32 frozen validation tasks after 64 steps per method. It supports only a small-sample descriptive comparison.
- **Outstanding:** full 512-step experiments and final-test evaluation were not run.
