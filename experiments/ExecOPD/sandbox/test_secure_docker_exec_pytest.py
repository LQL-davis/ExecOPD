#!/usr/bin/env python3
"""Fixed harmless C0.6 validation cases for secure_docker_exec."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from secure_docker_exec_pytest import CONTAINER_PREFIX, execute_code, execute_pytest


def require(label, condition, result):
    if not condition:
        raise AssertionError(f"{label} failed: {json.dumps(result, ensure_ascii=False)}")


def main() -> int:
    cases = {}

    cases["print"] = execute_code("print('EXECOPD_OK')\n")
    require("print", cases["print"]["status"] == "PASS" and cases["print"]["stdout"].strip() == "EXECOPD_OK", cases["print"])
    required_fields = {
        "ok", "status", "exit_code", "stdout", "stderr", "timed_out",
        "duration_ms", "executor", "image_digest", "limits",
        "stdout_truncated", "stderr_truncated",
    }
    require("result_schema", required_fields <= set(cases["print"]), cases["print"])
    require("executor_identity", cases["print"]["executor"] == "secure_docker", cases["print"])
    require("duration_type", isinstance(cases["print"]["duration_ms"], float), cases["print"])

    cases["stdin_echo"] = execute_code("import sys\nprint(sys.stdin.read())\n", "hello")
    require("stdin_echo", cases["stdin_echo"]["status"] == "PASS" and "hello" in cases["stdin_echo"]["stdout"], cases["stdin_echo"])

    cases["exit_3"] = execute_code("import sys\nsys.exit(3)\n")
    require("exit_3", cases["exit_3"]["status"] == "PROCESS_ERROR" and cases["exit_3"]["exit_code"] == 3, cases["exit_3"])

    cases["timeout"] = execute_code("while True:\n    pass\n")
    require("timeout", cases["timeout"]["status"] == "TIMEOUT" and cases["timeout"]["timed_out"] is True, cases["timeout"])

    cases["pytest"] = execute_pytest(
        "def add(a, b):\n    return a + b\n",
        "from solution import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    )
    pytest_pending = (
        cases["pytest"]["status"] == "PROCESS_ERROR"
        and "No module named pytest" in cases["pytest"]["stderr"]
    )
    require("pytest", cases["pytest"]["status"] == "PASS" or pytest_pending, cases["pytest"])

    listed = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_PREFIX}", "--format", "{{.Names}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require("cleanup_query", listed.returncode == 0, {"stdout": listed.stdout, "stderr": listed.stderr})
    require("no_leftovers", not listed.stdout.strip(), {"leftovers": listed.stdout.splitlines()})

    summary = {
        "status": "SECURE_EXECUTOR_BASE_PASS",
        "pytest_status": "PYTEST_NOT_AVAILABLE_IN_IMAGE" if pytest_pending else "SECURE_EXECUTOR_PYTEST_PASS",
        "cases": {
            key: {
                "status": value["status"],
                "exit_code": value["exit_code"],
                "timed_out": value["timed_out"],
                "duration_ms": value["duration_ms"],
                "stdout_truncated": value["stdout_truncated"],
                "stderr_truncated": value["stderr_truncated"],
            }
            for key, value in cases.items()
        },
        "leftover_containers": [],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
