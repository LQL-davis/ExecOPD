"""Resume the validated full checkpoint after a /tmp ENOSPC infrastructure failure."""
import json
from experiments.ExecOPD.agent import run_resume_pilot_workflow as workflow
ORIGINAL_RUN = workflow.run

def repaired_run(stage, script, arguments, visible, result_file, expected_status):
    if stage == 'resume_pilot_C64_train':
        script = 'run_pilot_C_resume_latest_nvme.py'
    elif script == 'run_validation_eval.py':
        script = 'run_validation_eval_docker_repair.py'
    return ORIGINAL_RUN(stage, script, arguments, visible, result_file, expected_status)

if __name__ == '__main__':
    workflow.run = repaired_run
    try:
        workflow.main()
    except Exception as exc:
        previous = json.loads(workflow.STATE.read_text()) if workflow.STATE.exists() else {}
        workflow.write_state({'status': 'BLOCKED', 'error': str(exc), 'previous_state': previous})
        print('C_NVME_RESUME_WORKFLOW_BLOCKED='+str(exc), flush=True)
        raise
