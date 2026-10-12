"""Read-only Agent adapter around the existing fixed secure Docker executor.

Hidden tests are sent only to the container. Their source, inputs, expected
outputs, and raw test streams never enter a Student-facing observation.
"""

import json
from typing import Callable, Optional

from experiments.ExecOPD.agent.execution_observation import ExecutionObservation, bounded_text
from experiments.ExecOPD.sandbox import secure_docker_exec_pytest as secure


PINNED_IMAGE = "execopd-pytest@sha256:670c4859fb73abd3d2ca933302c73656ea85aef1c514650a9efce34af155c39b"
assert secure.IMAGE_DIGEST == PINNED_IMAGE


class SecureExecutorFatal(RuntimeError):
    pass


class SecureExecutionTool:
    def __init__(self, log_event: Optional[Callable[[dict], None]] = None):
        self.log_event = log_event or (lambda event: None)
        self.call_count = 0
        secure._validate_fixed_policy()

    def _run(self, turn_id: int, purpose: str, kind: str, code: str, *, stdin: str = "", test_code: str = "") -> dict:
        self.call_count += 1
        if kind == "pytest":
            result = secure.execute_pytest(code, test_code)
        else:
            result = secure.execute_code(code, stdin_text=stdin)
        self.log_event({
            "turn_id": turn_id, "call_index": self.call_count, "purpose": purpose,
            "status": result.get("status"), "duration_ms": result.get("duration_ms"),
            "executor": result.get("executor"), "image_digest": result.get("image_digest"),
        })
        if result.get("executor") != "secure_docker" or result.get("image_digest") != PINNED_IMAGE or result.get("limits") != secure.LIMITS:
            raise SecureExecutorFatal("secure executor identity, image, or policy mismatch")
        if result.get("status") == "EXECUTOR_ERROR":
            raise SecureExecutorFatal("secure Docker execution unavailable")
        if result.get("status") not in {"PASS", "PROCESS_ERROR", "TIMEOUT"}:
            raise SecureExecutorFatal("unknown secure Docker status")
        return result

    @staticmethod
    def _observation(turn_id: int, status: str, result: dict, *, duration_ms: float,
                     passed_tests: Optional[int], total_tests: Optional[int],
                     withhold_test_stream: bool = False) -> ExecutionObservation:
        stdout, stdout_cut = bounded_text("" if withhold_test_stream else str(result.get("stdout") or ""))
        stderr, stderr_cut = bounded_text("" if withhold_test_stream else str(result.get("stderr") or ""))
        return ExecutionObservation(
            turn_id=turn_id, status=status, exit_code=result.get("exit_code"),
            stdout=stdout, stderr=stderr, timed_out=bool(result.get("timed_out")),
            duration_ms=round(duration_ms, 3), passed_tests=passed_tests,
            total_tests=total_tests, executor_identity="secure_docker",
            stdout_truncated=bool(result.get("stdout_truncated")) or stdout_cut,
            stderr_truncated=bool(result.get("stderr_truncated")) or stderr_cut,
            test_output_withheld=withhold_test_stream,
        )

    def execute(self, turn_id: int, code: str, ground_truth: str) -> ExecutionObservation:
        tests = json.loads(ground_truth)
        if not isinstance(tests, dict):
            raise SecureExecutorFatal("ground truth schema is not an object")
        elapsed = 0.0
        if "functional" in tests or "pytest" in tests:
            # Isolated candidate run gives diagnostic stderr without exposing test source.
            candidate = self._run(turn_id, "candidate_diagnostic", "python", code)
            elapsed += candidate["duration_ms"]
            if candidate["status"] != "PASS":
                return self._observation(turn_id, candidate["status"], candidate,
                                         duration_ms=elapsed, passed_tests=None, total_tests=None)
            if "functional" in tests:
                if not isinstance(tests["functional"], str):
                    raise SecureExecutorFatal("functional test schema invalid")
                test_result = self._run(turn_id, "hidden_functional_tests", "python",
                                        code + "\n" + tests["functional"])
            else:
                if not isinstance(tests["pytest"], str):
                    raise SecureExecutorFatal("pytest test schema invalid")
                test_result = self._run(turn_id, "hidden_pytest_tests", "pytest", code,
                                        test_code=tests["pytest"])
            elapsed += test_result["duration_ms"]
            return self._observation(turn_id, test_result["status"], candidate,
                                     duration_ms=elapsed, passed_tests=None, total_tests=None,
                                     withhold_test_stream=False) if test_result["status"] == "PASS" else \
                self._observation(turn_id, test_result["status"], test_result,
                                  duration_ms=elapsed, passed_tests=None, total_tests=None,
                                  withhold_test_stream=True)

        if "inputs" in tests and "outputs" in tests:
            inputs, outputs = tests["inputs"], tests["outputs"]
            if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs) or \
                    not all(isinstance(item, str) for item in inputs + outputs):
                raise SecureExecutorFatal("stdin test schema invalid")
            passed = 0
            for stdin, expected in zip(inputs, outputs):
                result = self._run(turn_id, "hidden_stdin_case", "python", code, stdin=stdin)
                elapsed += result["duration_ms"]
                if result["status"] != "PASS":
                    return self._observation(turn_id, result["status"], result,
                                             duration_ms=elapsed, passed_tests=passed,
                                             total_tests=len(inputs), withhold_test_stream=True)
                if result["stdout"].strip() != expected.strip():
                    return self._observation(turn_id, "OUTPUT_MISMATCH", result,
                                             duration_ms=elapsed, passed_tests=passed,
                                             total_tests=len(inputs), withhold_test_stream=True)
                passed += 1
            return self._observation(turn_id, "PASS", result, duration_ms=elapsed,
                                     passed_tests=passed, total_tests=len(inputs),
                                     withhold_test_stream=True)
        raise SecureExecutorFatal("unsupported Code-R1 ground truth schema")
