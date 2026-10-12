# Resume Pilot Token Alignment Repair

Date: 2026-10-08

## Conclusion

**RESUME_PILOT_ALIGNMENT_REPAIR_CPU_PASS**

The two recorded failing samples are now accepted without changing their sampled IDs, decoded text, prompt IDs, code artifacts, or frozen trajectories. The repaired A/C entrypoints have been deployed. CPU-only regression and actual actor-method tests passed. Real two-GPU FSDP/vLLM runtime and long training after this repair have not been run; this report is not an A64/C64 training PASS.

## Stopped run

A completed 30 optimizer steps. All four step31 trajectories were saved, but tensorization failed before the step31 update. C had not started. The original failure reports, raw trajectories and blocked workflow remain intact.

There is no complete step30 model/optimizer/scheduler/RNG/task-cursor checkpoint. Selected parameter diagnostic snapshots are insufficient to resume. A new independent attempt would have to start from the original Student; the old attempt must remain identified as interrupted, rather than silently relabeled as resumed. The existing no-reroll guard still blocks restarting the old workflow. No fresh rollout was initiated during this repair.

## Root cause and token evidence

The tokenizer can decode more than one valid BPE segmentation to the same text. Re-encoding decoded text does not guarantee reproduction of sampled IDs.

- Step31 trajectory0, turn0, first mismatch sampled index11 / original packed position500.
- Step31 trajectory1, turn1, first mismatch sampled index217 / original packed position1060.
- Sampled `[717, 327]` (`_m` + `ex`) becomes canonical `[59930]` (`_mex`) after re-encoding.
- Saved raw response, sampled decode, messages, canonical history and actual prompt reconstruction were internally consistent.

This failure was not an EOS handling failure or the earlier code-span boundary crossing failure. The old tensorizer's equality assertion rejected a valid sampled segmentation.

## Repair design

1. Each turn's model input is the **actual saved prompt IDs + actual sampled IDs**.
2. A packed carrier stores IDs/masks and scoring slots only. It is explicitly marked `carrier_is_model_input=false` and never passed to Actor or Teacher as the multi-turn forward context.
3. Subsequent prompts retain exactly their original rollout encoding. Simply splicing the earlier sampled IDs into a later turn's context would change the state and is not used.
4. Code boundaries are projected from sampled vocabulary bytes to UTF-8 character offsets. A token is code if its character span intersects the selected executable code span, then intersected with generated mask. Crossing tokens are explicitly recorded.
5. A old/current sampled-token logprob, C Student Top-K and Teacher same-ID scores all use the same exact per-turn inputs. Logits at position `t-1` score the actual token at position `t`.
6. Per-turn loss weights preserve the original per-trajectory masked mean: policy uses its original supervision mask counts; entropy uses generated-token counts. C remains code-only OPD, while entropy remains generated-token entropy.
7. Every FSDP rank makes two forward/backward calls per trajectory. An absent second turn uses an attached zero-gradient dummy graph so ranks with different trajectory lengths do not diverge in collective counts. This was verified with toy CPU actor methods, not a real distributed FSDP run.
8. Teacher cost accounting counts actual turn forwards and actual prompt+sample input tokens, including repeated history. Forward wall timing uses CUDA synchronization.

No changes were made to sampling, task order, seed, GRPO/OPD equations, K=16, `only_stu`, `student_p`, optimizer equations, LR, execution protocol or frozen data split. Vendored veRL and the shared earlier `c2c_tensor.py` were not edited.

## Frozen-data regression

| Check | Result |
|---|---:|
| Frozen trajectories | 124/124 PASS |
| Actual sampled assistant IDs | 199/199 exact |
| Actual rollout prompt IDs | 199/199 exact |
| Selected executable code coverage | 197/197 PASS |
| Tool message spans | 197/197 exact |
| Four-trajectory padding batches | 31/31 PASS |
| Canonical trajectories identical to legacy IDs and masks | 122/122 |
| Recorded noncanonical failures | 2/2 fixed |
| Generated/tool overlap | 0 |
| Code/tool overlap | 0 |
| Maximum actual turn input length | 1408 tokens |
| Maximum storage carrier length | 1540 tokens |
| Boundary-crossing tokens explicitly recorded | 6 |
| Split UTF-8 byte offsets | PASS |
| Earlier token271 span [9,11), code_start10 case | PASS |

Frozen input SHA256:
`6f47d8f27ab8eca3d7e57ff355983e9a1c4dc420c03207f508b3b40152a56304`

The SHA256 was checked before and after validation. The 2048-token actual model context limit remains enforced. Unknown decoder types and invalid sampled UTF-8 fail explicitly.

## CPU scoring and gradient checks

Toy causal models tested causal scoring shift, sampled IDs, same-ID Top-K gather, exact next-turn context, inactive-turn zero-gradient backward and objective equivalence. These tests do not load Student or Teacher checkpoints.

The actual repaired `PilotAActor.compute_log_prob`, `PilotAActor.update_policy` and `PilotCActor.update_policy` were also exercised with toy CPU models and a mixture of one/two-turn trajectories and a zero-code trajectory. The device and collective APIs were mocked for these integration tests.

- Method A: PASS; original-objective gradient maximum absolute error `1.4901161193847656e-08`; 8 forward/backward calls across four mixed-length trajectories; toy optimizer update verified.
- Method C: PASS; original-objective gradient maximum absolute error `2.7939677238464355e-09`; 8 forward/backward calls across four mixed-length trajectories; toy optimizer update verified.

Tool policy-loss/OPD/entropy overlaps were zero. All metrics were finite. C non-code OPD contribution remained zero; a zero-code trajectory still received only the unchanged generated-token entropy objective.

## Artifacts and backups

- `reports/resume_pilot_alignment_repair_result.json`
- `reports/resume_pilot_alignment_repair_audit.json`
- `reports/resume_pilot_alignment_repair_actor_check.json`
- `reports/resume_pilot_alignment_repair_manifest.json`
- Old pilot scripts: `agent/repair_backups/alignment_20261008/` with SHA256 manifest.
- Reproducible CPU validators: `agent/validate_pilot_turn_repair.py` and `agent/validate_pilot_actor_repair.py`.
- Independent repaired modules: `agent/pilot_turn_tensor.py`, `agent/pilot_turn_forward.py`.

Validators create reports with exclusive file creation and will refuse to overwrite earlier evidence. Use a new report namespace for any additional validation run.

## Remaining runtime gate / continuation

No real Student/Teacher forward, fresh rollout, secure generated-code execution, or GPU optimizer step occurred during this repair. Final-test task content access count was zero.

Before starting a new independently named A/C pilot attempt, verify two GPUs are idle and run a bounded two-GPU mechanics check of the repaired exact-turn path, including mixed one/two-turn FSDP behavior and memory. CPU tests do not prove distributed runtime/memory success.

A cannot resume from step31 because step30 had no complete checkpoint. A restart requires a new run identity and explicit treatment of the original no-reroll constraint; existing artifacts must be retained. This repair did not start that restart or clear its guard.

Last read-only GPU inspection:

```text
2, 17 MiB, 81064 MiB, 0 %
3, 17 MiB, 81064 MiB, 0 %
```
