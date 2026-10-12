"""Separate run outputs from immutable protocol and previous pilot evidence."""
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)
import json
import os
from pathlib import Path

BASE = EXPERIMENT_ROOT
value = os.environ.get('EXECOPD_PILOT_RUN_ROOT')
OUTPUT = Path(value).resolve() if value else BASE
if value:
    if OUTPUT.parent != (BASE/'runs').resolve():
        raise RuntimeError('PILOT_RUN_BLOCKED: run must be a direct child of ExecOPD/runs')
    identity = json.loads((OUTPUT/'RUN_IDENTITY.json').read_text())
    if identity['run_root'] != str(OUTPUT) or identity['restart_from_original_student'] is not True:
        raise RuntimeError('PILOT_RUN_BLOCKED: isolated independent run identity unavailable')
REPORTS = OUTPUT/'reports'
LOGS = OUTPUT/'logs'
RAY_RUN_ENV = {'EXECOPD_PILOT_RUN_ROOT': str(OUTPUT)} if value else {}
