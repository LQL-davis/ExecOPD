"""Full-state continuation with private NVMe temporary/spill storage.

All task, sampling, tensorization and Actor objective code is cloned unchanged
from the frozen pilot. Only recovery, durability and process/storage admission
are inserted. No original source file is edited.
"""
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)
import ast
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap
import time
import traceback

import numpy as np
import torch
from experiments.ExecOPD.agent import run_pilot_C as pilot

PREPARED = json.loads((pilot.ARTIFACTS/'resume_prepared.json').read_text())
RESUME_STEP = PREPARED['resume_checkpoint_step']
ARCHIVE = Path(PREPARED['archive'])
NVME = SCRATCH_DIR
pilot.gate(type(RESUME_STEP) is int and 2 <= RESUME_STEP < 64 and
           PREPARED['full_checkpoint_sha256_verified'] is True and
           PREPARED['automatic_fault_recovery_authorized'] is True,
           'explicit full-state recovery preparation missing')
pilot.gate(ARCHIVE.resolve().parent == pilot.ARTIFACTS.resolve().parent,
           'archive must remain inside this run')
ACTIVE = None

# Reuse the already checked ownership function without importing the old
# resume8 module, whose import mutates Trainer.fit.
helper_path = pilot.BASE/'agent/run_pilot_C_resume8.py'
helper_text = helper_path.read_text()
helper_tree = ast.parse(helper_text)
node = next(n for n in helper_tree.body if isinstance(n, ast.FunctionDef)
            and n.name == 'check_owned_gpu_processes')
exec(ast.get_source_segment(helper_text, node), globals())


def restore_full_state(trainer):
    global ACTIVE
    ACTIVE = trainer
    cp = pilot.ARTIFACTS/f'checkpoint_step{RESUME_STEP}'
    metadata = json.loads((cp/'metadata.json').read_text())
    pilot.gate(metadata['global_step'] == RESUME_STEP and
               metadata['next_task_cursor'] == RESUME_STEP+1 and
               metadata['seed'] == pilot.SEED and metadata['world_size'] == 2 and
               metadata['config_sha256'] == pilot.sha(pilot.ARTIFACTS/'effective_config.json') and
               metadata['split_sha256'] == pilot.sha(pilot.BASE/'reports/d1_data_split.json') and
               metadata['student_config_sha256'] == pilot.sha(pilot.STUDENT/'config.json') and
               metadata['student_tokenizer_sha256'] == pilot.sha(pilot.STUDENT/'tokenizer.json'),
               'checkpoint identity/cursor/config differs')
    pilot.gate((cp/'CHECKPOINT_COMPLETE').read_text().strip() == pilot.sha(cp/'manifest.json'),
               'complete checkpoint manifest marker differs')
    restored = trainer.actor_rollout_wg.load_checkpoint(str(cp/'actor'), RESUME_STEP)
    pilot.gate(len(restored) == 2 and all(x['step'] == RESUME_STEP for x in restored),
               'worker restore cursor differs')
    expected = {x['rank']: x for x in json.loads((cp/'worker_state.json').read_text())}
    actual = trainer.actor_rollout_wg.pilot_c_state_audit()
    for rank in actual:
        before = expected[rank['rank']]
        for key in ('model_local_sha256', 'optimizer_sha256', 'scheduler_sha256',
                    'rng_sha256', 'optimizer_state_entries', 'optimizer_lr', 'step'):
            pilot.gate(rank[key] == before[key], 'restored rank state mismatch: '+key)
    state = torch.load(cp/'driver_state.pt', map_location='cpu', weights_only=False)
    pilot.gate(state['global_step'] == RESUME_STEP and
               state['next_task_cursor'] == RESUME_STEP+1, 'driver cursor differs')
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_cpu_rng'])
    torch.cuda.set_rng_state_all(state['torch_cuda_rng'])
    trainer.global_steps = RESUME_STEP
    pilot.write_new(pilot.ARTIFACTS/'resume_state_audit.json', {
        'status': 'C_FULL_STATE_RESTORED_EXACT', 'checkpoint': str(cp),
        'worker_state': actual, 'driver_global_step': RESUME_STEP,
        'next_task_cursor': RESUME_STEP+1,
        'model_optimizer_scheduler_rng_exact_on_both_ranks': True,
        'driver_rng_restored': True})
    print(f'PILOT_C_CHECKPOINT_RESTORE_PASS=step{RESUME_STEP}_next_task{RESUME_STEP+1}', flush=True)


checkpoint_source = inspect.getsource(pilot.checkpoint)
old = 'step in (8, 16, 32, 48, 64)'
pilot.gate(checkpoint_source.count(old) == 1, 'checkpoint insertion point differs')
checkpoint_namespace = dict(vars(pilot))
exec(checkpoint_source.replace(old, '2 <= step <= 64'), checkpoint_namespace)
SAVE_CHECKPOINT = checkpoint_namespace['checkpoint']


def save_completed(trainer, reason):
    records = [json.loads(x) for x in (pilot.ARTIFACTS/'step_records.jsonl').read_text().splitlines()]
    step = len(records)
    pilot.gate(step >= RESUME_STEP and
               [r['step'] for r in records] == list(range(1, step+1)),
               'committed record cursor differs')
    target = pilot.ARTIFACTS/f'checkpoint_step{step}'
    if (target/'CHECKPOINT_COMPLETE').is_file():
        metadata = json.loads((target/'metadata.json').read_text())
        pilot.gate(metadata['global_step'] == step and metadata['next_task_cursor'] == step+1 and
                   (target/'CHECKPOINT_COMPLETE').read_text().strip() == pilot.sha(target/'manifest.json'),
                   'existing checkpoint identity differs')
        return {'step': step, 'path': str(target), 'existing': True}
    states = trainer.actor_rollout_wg.pilot_c_lr_audit()
    pilot.gate(len(states) == 2 and all(s['optimizer_step_count'] == step for s in states),
               'worker/committed-step disagreement; unsafe emergency checkpoint')
    saved = SAVE_CHECKPOINT(trainer, records)
    value = {'step': step, 'path': saved['path'], 'bytes': saved['manifest']['total_bytes'],
             'write_seconds': saved['manifest']['write_seconds'], 'reason': reason}
    pilot.append_sync(pilot.ARTIFACTS/'periodic_checkpoints.jsonl', value)
    print('PILOT_C_DURABLE_CHECKPOINT_COMPLETE='+json.dumps(value), flush=True)
    return value


offline_source = inspect.getsource(pilot.run_offline)
pilot.gate(offline_source.count('agent/pilot_c_offline.py') == 1, 'offline launcher differs')
offline_namespace = dict(vars(pilot))
exec(offline_source.replace('agent/pilot_c_offline.py',
                           'agent/pilot_c_offline_docker_repair.py'), offline_namespace)


def fresh_offline(phase, step, source=None):
    check_owned_gpu_processes(f'nvme_fresh_step{step}_{phase}')
    if phase == 'collect' and ((step-1) % 4 == 0 or step == 15):
        save_completed(ACTIVE, 'periodic_before_next_collection')
    return offline_namespace['run_offline'](phase, step, source)


fit_source = inspect.getsource(pilot.Trainer.fit)
old_prefix = """        gate(not (ARTIFACTS / 'step1').exists(),
             'C64 already started; no implicit reroll')
        initial = rank_zero(self.actor_rollout_wg.pilot_c_parameters('before_step1'))
        lr = self.actor_rollout_wg.pilot_c_lr_audit()
        gate(len(lr) == 2 and all(x['lrs'] == [0.0] for x in lr),
             'Step1 LR not zero')
        records = []
        source = STUDENT
        for step in range(1, 65):"""
new_prefix = """        initial = rank_zero(self.actor_rollout_wg.pilot_c_parameters('before_step1'))
        restore_full_state(self)
        records = [json.loads(x) for x in (ARTIFACTS/'step_records.jsonl').read_text().splitlines()]
        gate(len(records) == RESUME_STEP, 'inherited record cursor differs')
        source = ARCHIVE/f'export_after_step{RESUME_STEP}'
        for step in range(RESUME_STEP+1, 65):"""
pilot.gate(fit_source.count(old_prefix) == 1, 'frozen trainer recovery insertion differs')
fit_namespace = dict(vars(pilot))
fit_namespace.update({'restore_full_state': restore_full_state, 'ARCHIVE': ARCHIVE,
                      'RESUME_STEP': RESUME_STEP, 'run_offline': fresh_offline})
exec(textwrap.dedent(fit_source.replace(old_prefix, new_prefix)), fit_namespace)
pilot.Trainer.fit = fit_namespace['fit']


class DurableTrainer(pilot.Trainer):
    def fit(self):
        self._actor_update_inflight = False
        update = self.actor_rollout_wg.update_actor
        def owned_update(batch):
            check_owned_gpu_processes('nvme_actor_step'+str(batch.meta_info['pilot_c_step']))
            self._actor_update_inflight = True
            output = update(batch)
            self._actor_update_inflight = False
            return output
        self.actor_rollout_wg.update_actor = owned_update
        try:
            return super().fit()
        except Exception:
            original_error = traceback.format_exc()
            try:
                pilot.gate(not self._actor_update_inflight,
                           'actor RPC failed in flight; partial update cannot be saved as complete')
                saved = save_completed(self, 'emergency_before_ray_shutdown')
                audit = {'status': 'EMERGENCY_CHECKPOINT_PASS', 'checkpoint': saved,
                         'original_error': original_error}
            except Exception:
                audit = {'status': 'EMERGENCY_CHECKPOINT_BLOCKED',
                         'checkpoint_error': traceback.format_exc(), 'original_error': original_error}
            pilot.write_new(pilot.ARTIFACTS/'emergency_checkpoint_audit.json', audit)
            raise


main_source = inspect.getsource(pilot.main)
old_prepare = """    gate(not ARTIFACTS.exists(), 'C pilot output exists; no overwrite')
    preflight = gpu_preflight()
    ARTIFACTS.mkdir(parents=True)"""
new_prepare = """    gate((ARTIFACTS/'resume_prepared.json').is_file(), 'explicit resume preparation missing')
    preflight = gpu_preflight()"""
pilot.gate(main_source.count(old_prepare) == 1, 'main recovery insertion differs')
main_source = main_source.replace(old_prepare, new_prepare)
old_ray = 'ray.init(num_gpus=2, include_dashboard=False,'
new_ray = "ray.init(num_gpus=2, include_dashboard=False, _temp_dir=str(NVME/'r'), object_store_memory=8*1024**3, _system_config={'object_spilling_config': json.dumps({'type': 'filesystem', 'params': {'directory_path': str(NVME/'s')}})},"
pilot.gate(main_source.count(old_ray) == 1, 'Ray storage insertion differs')
main_source = main_source.replace(old_ray, new_ray)
old_env = "'D1_SEED': str(SEED), **RAY_RUN_ENV"
new_env = "'D1_SEED': str(SEED), 'TMPDIR': str(NVME/'t'), 'TMP': str(NVME/'t'), 'TEMP': str(NVME/'t'), 'RAY_TMPDIR': str(NVME), **RAY_RUN_ENV"
pilot.gate(main_source.count(old_env) == 1, 'worker tempdir insertion differs')
main_namespace = dict(vars(pilot))
main_namespace.update({'Trainer': DurableTrainer, 'NVME': NVME})
exec(main_source.replace(old_env, new_env), main_namespace)
RESUME_MAIN = main_namespace['main']


if __name__ == '__main__':
    started = time.time()
    try:
        pilot.gate(os.environ.get('TMPDIR') == str(NVME/'t'), 'private NVMe TMPDIR required')
        result = RESUME_MAIN()
        result.update({'resume_checkpoint_step': RESUME_STEP,
                       'inherited_steps': list(range(1,RESUME_STEP+1)),
                       'fresh_continuation_steps': list(range(RESUME_STEP+1,65)),
                       'new_trajectories_this_attempt': (64-RESUME_STEP)*4,
                       'total_completed_run_trajectories': 256,
                       'previous_incomplete_attempt_samples_excluded': True,
                       'automatic_fault_recovery_authorization': PREPARED,
                       'no_outcome_based_reroll_within_continuation': True})
    except Exception as exc:
        result = {'status': 'RESUME_PILOT_C_BLOCKED', 'error': str(exc),
                  'traceback': traceback.format_exc()}
    result['duration_seconds'] = round(time.time()-started, 2)
    pilot.write_new(pilot.REPORTS/'resume_pilot_training_C.json', result)
    print('PILOT_C_FINAL='+json.dumps({'status': result['status'],
                                     'duration_seconds': result['duration_seconds']}), flush=True)
    if result['status'] != 'RESUME_PILOT_C64_PASS':
        sys.exit(1)
