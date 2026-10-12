#!/usr/bin/env python3
"""Experimental fixed-policy Docker executor for ExecOPD C0.6.

This module deliberately exposes no security-policy overrides.  Source bytes are
streamed over ``docker exec -i`` into the running container's ``/work`` tmpfs;
host paths are never mounted or exposed to the container.
"""

from __future__ import annotations

import json
import secrets
import shutil
import subprocess
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple


IMAGE_DIGEST = (
    "execopd-pytest@sha256:"
    "670c4859fb73abd3d2ca933302c73656ea85aef1c514650a9efce34af155c39b"
)
CONTAINER_PREFIX = "execopd-secure-"
WALL_TIMEOUT_SECONDS = 30
OUTPUT_LIMIT_BYTES = 64 * 1024
INPUT_LIMIT_BYTES = 1024 * 1024

LIMITS = {
    "network": "none",
    "read_only_root": True,
    "no_new_privileges": True,
    "cap_drop": ["ALL"],
    "pids": 32,
    "memory_bytes": 512 * 1024 * 1024,
    "cpus": 1.0,
    "nofile": 64,
    "fsize_bytes": 2 * 1024 * 1024,
    "wall_timeout_seconds": WALL_TIMEOUT_SECONDS,
    "stdout_limit_bytes": OUTPUT_LIMIT_BYTES,
    "stderr_limit_bytes": OUTPUT_LIMIT_BYTES,
    "tmpfs": {
        "/work": "rw,noexec,nosuid,size=64m",
        "/tmp": "rw,noexec,nosuid,size=64m",
    },
    "bind_mounts": 0,
    "docker_socket": False,
    "privileged": False,
    "gpu": False,
}


class ExecutorSetupError(RuntimeError):
    pass


def _validate_fixed_policy() -> None:
    """Fail closed if a future edit weakens the fixed C0.6 policy."""
    assert "@sha256:" in IMAGE_DIGEST and ":latest" not in IMAGE_DIGEST
    assert LIMITS["network"] == "none"
    assert LIMITS["read_only_root"] is True
    assert LIMITS["no_new_privileges"] is True
    assert LIMITS["cap_drop"] == ["ALL"]
    assert LIMITS["pids"] == 32
    assert LIMITS["memory_bytes"] == 536870912
    assert LIMITS["cpus"] == 1.0
    assert LIMITS["nofile"] == 64
    assert LIMITS["fsize_bytes"] == 2097152
    assert LIMITS["wall_timeout_seconds"] == 30
    assert LIMITS["bind_mounts"] == 0
    assert LIMITS["docker_socket"] is False
    assert LIMITS["privileged"] is False
    assert LIMITS["gpu"] is False
    assert set(LIMITS["tmpfs"]) == {"/work", "/tmp"}


def _docker_base() -> List[str]:
    docker = shutil.which("docker")
    if not docker:
        raise ExecutorSetupError("docker CLI not found")
    return [docker]


def _control(args: Iterable[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        _docker_base() + list(args),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def _create_command(name: str) -> List[str]:
    cmd = [
        "create",
        "--pull", "never",
        "--name", name,
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges=true",
        "--pids-limit", "32",
        "--memory", "512m",
        "--cpus", "1.0",
        "--ulimit", "nofile=64:64",
        "--ulimit", "fsize=2097152:2097152",
        "--tmpfs", "/work:rw,noexec,nosuid,size=64m",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--workdir", "/work",
        "--entrypoint", "python",
        IMAGE_DIGEST,
        "-c", "import time; time.sleep(3600)",
    ]
    joined = " ".join(cmd)
    forbidden = ("--privileged", "--gpus", "--runtime=nvidia", "-v ", "--volume", "--mount")
    if any(token in joined for token in forbidden):
        raise ExecutorSetupError("forbidden Docker option in fixed policy")
    return cmd


def _inspect_and_assert(name: str) -> None:
    inspected = _control(["inspect", name], timeout=10)
    if inspected.returncode != 0:
        raise ExecutorSetupError(inspected.stderr.decode("utf-8", "replace"))
    payload = json.loads(inspected.stdout)[0]
    host = payload["HostConfig"]
    config = payload["Config"]
    ulimits = {x["Name"]: (x["Soft"], x["Hard"]) for x in host.get("Ulimits") or []}
    checks = {
        "image_digest": config.get("Image") == IMAGE_DIGEST,
        "network_none": host.get("NetworkMode") == "none",
        "read_only": host.get("ReadonlyRootfs") is True,
        "cap_drop_all": "ALL" in (host.get("CapDrop") or []),
        "no_new_privileges": any(
            str(x).lower().startswith("no-new-privileges")
            for x in (host.get("SecurityOpt") or [])
        ),
        "pids": host.get("PidsLimit") == 32,
        "memory": host.get("Memory") == 536870912,
        "cpu": host.get("NanoCpus") == 1000000000,
        "nofile": ulimits.get("nofile") == (64, 64),
        "fsize": ulimits.get("fsize") == (2097152, 2097152),
        "tmpfs": set((host.get("Tmpfs") or {}).keys()) == {"/work", "/tmp"},
        "no_binds": not (host.get("Binds") or []),
        "not_privileged": host.get("Privileged") is False,
        "no_devices": not (host.get("Devices") or []),
        "no_gpu_requests": not (host.get("DeviceRequests") or []),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        raise ExecutorSetupError("container policy inspection failed: " + ", ".join(failed))


def _drain(pipe, retained: bytearray, state: Dict[str, bool]) -> None:
    while True:
        chunk = pipe.read(8192)
        if not chunk:
            break
        remaining = OUTPUT_LIMIT_BYTES - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > remaining:
            state["truncated"] = True


def _run_bounded(name: str, command: List[str], stdin_bytes: bytes) -> Tuple[int, bytes, bytes, bool, bool, bool]:
    if len(stdin_bytes) > INPUT_LIMIT_BYTES:
        raise ExecutorSetupError("stdin exceeds fixed 1 MiB limit")
    proc = subprocess.Popen(
        _docker_base() + ["exec", "-i", name] + command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = bytearray(), bytearray()
    out_state, err_state = {"truncated": False}, {"truncated": False}
    out_thread = threading.Thread(target=_drain, args=(proc.stdout, out, out_state), daemon=True)
    err_thread = threading.Thread(target=_drain, args=(proc.stderr, err, err_state), daemon=True)
    out_thread.start()
    err_thread.start()

    def writer() -> None:
        try:
            if proc.stdin:
                proc.stdin.write(stdin_bytes)
                proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    input_thread = threading.Thread(target=writer, daemon=True)
    input_thread.start()
    timed_out = False
    try:
        proc.wait(timeout=WALL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        _control(["rm", "-f", name], timeout=15)
        try:
            proc.kill()
        except OSError:
            pass
        proc.wait(timeout=10)
    input_thread.join(timeout=1)
    out_thread.join(timeout=5)
    err_thread.join(timeout=5)
    return (
        proc.returncode if proc.returncode is not None else -1,
        bytes(out),
        bytes(err),
        timed_out,
        out_state["truncated"],
        err_state["truncated"],
    )


def _stage_bytes(name: str, filename: str, content: str) -> None:
    """Copy bytes over docker exec stdin into the already-mounted /work tmpfs."""
    if filename not in {"solution.py", "test_solution.py"}:
        raise ExecutorSetupError("unsupported staging filename")
    payload = content.encode("utf-8")
    if len(payload) > INPUT_LIMIT_BYTES:
        raise ExecutorSetupError("staged source exceeds fixed 1 MiB limit")
    writer = (
        "import pathlib,sys; "
        f"pathlib.Path('/work/{filename}').write_bytes(sys.stdin.buffer.read())"
    )
    copied = subprocess.run(
        _docker_base() + ["exec", "-i", name, "python", "-c", writer],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )
    if copied.returncode != 0:
        raise ExecutorSetupError(copied.stderr.decode("utf-8", "replace"))


def _result(
    *, status: str, exit_code: Optional[int], stdout: bytes = b"", stderr: bytes = b"",
    timed_out: bool, started: float, stdout_truncated: bool = False,
    stderr_truncated: bool = False,
) -> Dict[str, object]:
    return {
        "ok": status == "PASS",
        "status": status,
        "exit_code": exit_code,
        "stdout": stdout.decode("utf-8", "replace"),
        "stderr": stderr.decode("utf-8", "replace"),
        "timed_out": timed_out,
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "executor": "secure_docker",
        "image_digest": IMAGE_DIGEST,
        "limits": LIMITS,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
    }


def _execute(files: Dict[str, str], command: List[str], stdin_text: str = "") -> Dict[str, object]:
    _validate_fixed_policy()
    started = time.monotonic()
    name = CONTAINER_PREFIX + secrets.token_hex(8)
    created = False
    try:
        made = _control(_create_command(name), timeout=20)
        if made.returncode != 0:
            raise ExecutorSetupError(made.stderr.decode("utf-8", "replace"))
        created = True
        started_container = _control(["start", name], timeout=20)
        if started_container.returncode != 0:
            raise ExecutorSetupError(started_container.stderr.decode("utf-8", "replace"))
        _inspect_and_assert(name)
        for filename, content in files.items():
            _stage_bytes(name, filename, content)
        code, out, err, timed_out, out_trunc, err_trunc = _run_bounded(
            name, command, stdin_text.encode("utf-8")
        )
        status = "TIMEOUT" if timed_out else ("PASS" if code == 0 else "PROCESS_ERROR")
        return _result(
            status=status,
            exit_code=None if timed_out else code,
            stdout=out,
            stderr=err,
            timed_out=timed_out,
            started=started,
            stdout_truncated=out_trunc,
            stderr_truncated=err_trunc,
        )
    except (ExecutorSetupError, subprocess.SubprocessError, OSError, ValueError, json.JSONDecodeError) as exc:
        return _result(
            status="EXECUTOR_ERROR",
            exit_code=None,
            stderr=str(exc).encode("utf-8", "replace"),
            timed_out=False,
            started=started,
        )
    finally:
        if created:
            _control(["rm", "-f", name], timeout=20)


def execute_code(code: str, stdin_text: str = "") -> Dict[str, object]:
    """Execute one Python file under the non-configurable C0.6 policy."""
    return _execute({"solution.py": code}, ["python", "/work/solution.py"], stdin_text)


def execute_pytest(solution_code: str, test_code: str) -> Dict[str, object]:
    """Run a fixed solution/test pair under the same policy."""
    return _execute(
        {"solution.py": solution_code, "test_solution.py": test_code},
        ["python", "-m", "pytest", "-q", "/work/test_solution.py"],
    )


if __name__ == "__main__":
    print(json.dumps(execute_code("print('EXECOPD_OK')\n"), indent=2, sort_keys=True))
