# Source provenance

This repository contains selected custom experiment files from the ExecOPD work directory. The public selection includes the execution agent, exact-turn tensorization, pilot training/evaluation, checkpoint recovery and the two-step credit prototype. It excludes model files, raw samples, per-task traces, private run directories, frozen split/config artifacts, and the vendored upstream project.

The source archive used to prepare the release contained only `agent/*.py` and `sandbox/*.py` from the experiment directory. Its SHA256 was:

`6b481eb113fc9ffc0e065aaefa0ab52a03b11996b07c88d9d98a30f164667e96`

Before publication, private absolute paths in selected files were replaced with values from `agent/release_paths.py`; train-data references now use its `TRAIN_PARQUET`. Package markers and documentation were added, and the Method A pilot docstring was corrected. These edits were made to the **publication copy**, so its file hashes differ from the exact server source snapshots used in the reported run. The experiment algorithms, fixed Docker policy, K=16, masks, seed, sampling settings, reward and optimizer behavior were not intentionally changed in this release. The public copy has only static verification; its training has not been rerun.

Related external sources:

- [Code-R1 repository](https://github.com/ganler/code-r1) and [Code-R1-12K dataset](https://huggingface.co/datasets/ganler/code-r1-12k): base trainer and task/test data.
- [Rethinking-OPD](https://github.com/Thinking-Space/Rethinking-OPD): prior Vanilla OPD setting.
- Qwen2.5-7B-Instruct-1M and Qwen2.5-14B-Instruct-1M: external Student/Teacher checkpoints.

No upstream repository or model weights are copied here. This publication currently has **no license**; visibility of source does not grant reuse rights.
