"""Local paths for the public source release.

The historical run used fixed machine paths. Environment variables let readers
place the same source in their own Code-R1 checkout without changing the
algorithm or the recorded experiment artifacts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(
    os.environ.get("EXECOPD_CODE_R1_ROOT", Path(__file__).resolve().parents[3])
).resolve()
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "ExecOPD"
STUDENT_MODEL = Path(
    os.environ.get(
        "EXECOPD_STUDENT_MODEL",
        REPO_ROOT / "models" / "Qwen2.5-7B-Instruct-1M",
    )
).resolve()
TEACHER_MODEL = Path(
    os.environ.get(
        "EXECOPD_TEACHER_MODEL",
        REPO_ROOT / "models" / "Qwen2.5-14B-Instruct-1M",
    )
).resolve()
TRAIN_PARQUET = Path(
    os.environ.get(
        "EXECOPD_TRAIN_PARQUET", REPO_ROOT / "data" / "code-r1-12k" / "train.parquet"
    )
).resolve()
PYTHON_EXE = os.environ.get("EXECOPD_PYTHON", sys.executable)
SCRATCH_DIR = Path(
    os.environ.get("EXECOPD_SCRATCH_DIR", EXPERIMENT_ROOT / "runs" / "tmp")
).resolve()
