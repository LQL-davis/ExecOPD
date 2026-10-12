"""Identical frozen evaluation with repaired Docker management only."""
from experiments.ExecOPD.agent import run_validation_eval as evaluation
from experiments.ExecOPD.agent.pilot_paths import REPORTS
from experiments.ExecOPD.agent.pilot_docker_control_repair import install

if __name__ == '__main__':
    install(REPORTS/'validation_docker_control_audit.jsonl')
    evaluation.main()
