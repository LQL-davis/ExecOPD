"""Isolated Docker control-plane repair; fixed untrusted execution policy is unchanged."""
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

from experiments.ExecOPD.sandbox import secure_docker_exec_pytest as secure

ORIGINAL_CONTROL = secure._control
ORIGINAL_EXECUTE_SOURCE = inspect.getsource(secure._execute)
FIXED_LIMITS = copy.deepcopy(secure.LIMITS)
FIXED_IMAGE = secure.IMAGE_DIGEST
AUDIT_PATH = None
INSTALLED = False
MANAGEMENT_TIMEOUTS = {'create': 120, 'start': 120, 'inspect': 30, 'rm': 120}


def audit(value):
    if AUDIT_PATH is None:
        return
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'time_unix': time.time(), **value}) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def control(args, timeout=30):
    args = list(args)
    effective = MANAGEMENT_TIMEOUTS.get(args[0], timeout)
    started = time.monotonic()
    try:
        result = ORIGINAL_CONTROL(args, timeout=effective)
    except Exception as exc:
        audit({'event': 'docker_control', 'operation': args[0],
               'container': args[args.index('--name')+1] if '--name' in args else args[-1],
               'timeout_seconds': effective, 'elapsed_seconds': time.monotonic()-started,
               'exception': type(exc).__name__, 'error': str(exc)[:4096]})
        raise
    audit({'event': 'docker_control', 'operation': args[0],
           'container': args[args.index('--name')+1] if '--name' in args else args[-1],
           'timeout_seconds': effective, 'elapsed_seconds': time.monotonic()-started,
           'returncode': result.returncode,
           'stderr': result.stderr.decode('utf-8', 'replace')[:4096]})
    return result


def remove_owned(name):
    """Remove only the random name allocated by this invocation, including late creates."""
    assert name.startswith(secure.CONTAINER_PREFIX)
    result = control(['rm', '-f', name])
    if result.returncode == 0 or b'No such container' in result.stderr:
        return
    raise secure.ExecutorSetupError('owned container cleanup failed: '+
                                    result.stderr.decode('utf-8', 'replace'))


def execute(files, command, stdin_text=''):
    secure._validate_fixed_policy()
    assert secure.LIMITS == FIXED_LIMITS and secure.IMAGE_DIGEST == FIXED_IMAGE
    started = time.monotonic()
    name = secure.CONTAINER_PREFIX + secure.secrets.token_hex(8)
    result = None
    create_timed_out = False
    try:
        try:
            made = control(secure._create_command(name))
        except subprocess.TimeoutExpired:
            create_timed_out = True
            raise
        if made.returncode != 0:
            raise secure.ExecutorSetupError(made.stderr.decode('utf-8', 'replace'))
        launched = control(['start', name])
        if launched.returncode != 0:
            raise secure.ExecutorSetupError(launched.stderr.decode('utf-8', 'replace'))
        secure._inspect_and_assert(name)
        for filename, content in files.items():
            secure._stage_bytes(name, filename, content)
        code, out, err, timed_out, out_trunc, err_trunc = secure._run_bounded(
            name, command, stdin_text.encode('utf-8'))
        status = 'TIMEOUT' if timed_out else ('PASS' if code == 0 else 'PROCESS_ERROR')
        result = secure._result(status=status, exit_code=None if timed_out else code,
                                stdout=out, stderr=err, timed_out=timed_out,
                                started=started, stdout_truncated=out_trunc,
                                stderr_truncated=err_trunc)
    except (secure.ExecutorSetupError, subprocess.SubprocessError, OSError,
            ValueError, json.JSONDecodeError) as exc:
        audit({'event': 'executor_infrastructure_error', 'container': name,
               'error': str(exc)[:4096]})
        result = secure._result(status='EXECUTOR_ERROR', exit_code=None,
                                stderr=str(exc).encode('utf-8', 'replace'),
                                timed_out=False, started=started)
    finally:
        try:
            if create_timed_out:
                # A terminated CLI can leave an in-flight daemon create. Wait boundedly
                # for this exact name, then remove it. Never repeat Student code.
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    inspected = control(['inspect', name])
                    if inspected.returncode == 0:
                        break
                    if b'No such' not in inspected.stderr:
                        raise secure.ExecutorSetupError('late create inspection failed')
                    time.sleep(2)
            remove_owned(name)
        except Exception as exc:
            audit({'event': 'cleanup_error', 'container': name, 'error': str(exc)[:4096]})
            result = secure._result(status='EXECUTOR_ERROR', exit_code=None,
                                    stderr=('cleanup failed: '+str(exc)).encode(),
                                    timed_out=False, started=started)
    return result


def install(audit_path):
    global AUDIT_PATH, INSTALLED
    AUDIT_PATH = Path(audit_path)
    if not INSTALLED:
        secure._validate_fixed_policy()
        secure._control = control
        secure._execute = execute
        INSTALLED = True
    assert secure.WALL_TIMEOUT_SECONDS == 30 and secure.LIMITS == FIXED_LIMITS
    audit({'event': 'repair_installed', 'management_timeouts': MANAGEMENT_TIMEOUTS,
           'execution_wall_seconds': 30, 'fixed_limits': secure.LIMITS,
           'image_digest': secure.IMAGE_DIGEST,
           'student_code_retry_count': 0})
