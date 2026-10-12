"""Restore full step8 state and recover cached steps9-14 without reroll."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import textwrap
import time
import traceback

import numpy as np
import torch
from safetensors import safe_open
from experiments.ExecOPD.agent import run_pilot_C as pilot

ARCHIVE = pilot.REPORTS/'resume_pilot_artifacts/C_oom_after_step13_20261008'
PLAN = json.loads((pilot.REPORTS/'C_resume8_plan.json').read_text())
ORIGINAL_OFFLINE = pilot.run_offline
ACTIVE = None
REPLAY_EXPORTS = []


def check_owned_gpu_processes(stage):
    query = subprocess.run(['nvidia-smi', '-i', '2,3', '--query-gpu=index,uuid',
                            '--format=csv,noheader,nounits'],
                           check=True, capture_output=True, text=True).stdout
    uuids = {line.split(',')[1].strip() for line in query.splitlines()}
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                           '--format=csv,noheader,nounits'],
                          check=True, capture_output=True, text=True).stdout
    unexpected = []
    def owned(pid):
        visited = set()
        while pid > 1 and pid not in visited:
            if pid == os.getpid():
                return True
            visited.add(pid)
            try:
                values = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                return True  # process exited while the snapshot was read
            pid = int(values[1])
        return False
    relevant = []
    for line in apps.splitlines():
        uuid, pid, memory = [x.strip() for x in line.split(',')]
        if uuid in uuids:
            item = {'uuid': uuid, 'pid': int(pid), 'used_mib': memory}
            relevant.append(item)
            if not owned(int(pid)):
                unexpected.append(item)
    pilot.append_sync(pilot.ARTIFACTS/'runtime_gpu_ownership.jsonl', {
        'stage': stage, 'time_unix': time.time(), 'apps': relevant,
        'unexpected_apps': unexpected})
    pilot.gate(not unexpected, 'new external GPU process detected: '+str(unexpected))


def restore_step8(trainer):
    global ACTIVE
    ACTIVE = trainer
    cp = pilot.ARTIFACTS/'checkpoint_step8'
    metadata = json.loads((cp/'metadata.json').read_text())
    pilot.gate(metadata['global_step'] == 8 and metadata['next_task_cursor'] == 9 and
               metadata['seed'] == pilot.SEED and metadata['world_size'] == 2 and
               metadata['config_sha256'] == pilot.sha(pilot.ARTIFACTS/'effective_config.json'),
               'checkpoint cursor/config identity differs')
    restored = trainer.actor_rollout_wg.load_checkpoint(str(cp/'actor'), 8)
    pilot.gate(len(restored) == 2 and all(x['step'] == 8 for x in restored),
               'worker checkpoint restore step differs')
    expected = {x['rank']: x for x in json.loads((cp/'worker_state.json').read_text())}
    actual = trainer.actor_rollout_wg.pilot_c_state_audit()
    for rank in actual:
        before = expected[rank['rank']]
        for key in ('model_local_sha256', 'optimizer_sha256', 'scheduler_sha256',
                    'rng_sha256', 'optimizer_state_entries', 'optimizer_lr', 'step'):
            pilot.gate(rank[key] == before[key], 'restored rank state mismatch: '+key)
    state = torch.load(cp/'driver_state.pt', map_location='cpu', weights_only=False)
    pilot.gate(state['global_step'] == 8 and state['next_task_cursor'] == 9,
               'driver checkpoint cursor differs')
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_cpu_rng'])
    torch.cuda.set_rng_state_all(state['torch_cuda_rng'])
    trainer.global_steps = 8
    pilot.write_new(pilot.ARTIFACTS/'resume8_state_audit.json', {
        'status': 'C_STEP8_FULL_STATE_RESTORED_EXACT', 'checkpoint': str(cp),
        'worker_state': actual, 'driver_global_step': 8, 'next_task_cursor': 9,
        'model_optimizer_scheduler_rng_exact_on_both_ranks': True,
        'driver_rng_restored': True})
    print('PILOT_C_CHECKPOINT_RESTORE_PASS=step8_next_task9', flush=True)


def compare_replayed_export(step):
    if step > 13:
        return
    new = pilot.ARTIFACTS/f'export_after_step{step}/model.safetensors'
    old = ARCHIVE/f'export_after_step{step}/model.safetensors'
    new_hash = pilot.sha(new)
    exact_file = new_hash == PLAN['export_sha256'][str(step)]
    tensor_exact = True
    bf16_exact = True
    max_abs = 0.0
    if not exact_file:
        with safe_open(str(new), framework='pt', device='cpu') as left, \
                safe_open(str(old), framework='pt', device='cpu') as right:
            pilot.gate(set(left.keys()) == set(right.keys()), 'replayed model keys differ')
            for key in left.keys():
                a, b = left.get_tensor(key), right.get_tensor(key)
                pilot.gate(a.shape == b.shape and a.dtype == b.dtype,
                           'replayed model tensor metadata differs: '+key)
                same = torch.equal(a, b)
                tensor_exact = tensor_exact and same
                if not same:
                    max_abs = max(max_abs, float((a.float()-b.float()).abs().max()))
                    bf16_exact = bf16_exact and torch.equal(a.to(torch.bfloat16), b.to(torch.bfloat16))
                del a, b
    audit = {'step': step, 'file_sha256_exact': exact_file,
             'tensor_values_exact': tensor_exact, 'BF16_sampling_weights_exact': bf16_exact,
             'max_fp32_abs_difference': max_abs,
             'no_new_rollout': True,
             'cached_next_state_allowed': tensor_exact or bf16_exact}
    pilot.write_new(pilot.ARTIFACTS/f'replay_step{step}_model_audit.json', audit)
    pilot.gate(audit['cached_next_state_allowed'],
               'replayed Student sampling weights differ; stop before consuming next cached state')
    REPLAY_EXPORTS.append(audit)
    print('PILOT_C_CACHED_STATE_MODEL_MATCH=step'+str(step), flush=True)


checkpoint_source = inspect.getsource(pilot.checkpoint)
pilot.gate(checkpoint_source.count('step in (8, 16, 32, 48, 64)') == 1,
           'checkpoint function interface differs')
checkpoint_namespace = dict(vars(pilot))
exec(checkpoint_source.replace('step in (8, 16, 32, 48, 64)',
                               'step in (8, 14, 16, 32, 48, 64)'), checkpoint_namespace)
RECOVERY_CHECKPOINT = checkpoint_namespace['checkpoint']


def cached_or_fresh(phase, step, source=None):
    check_owned_gpu_processes(f'step{step}_{phase}')
    if phase == 'collect' and 10 <= step <= 14:
        compare_replayed_export(step-1)
    if phase == 'collect' and step == 15:
        records = [json.loads(x) for x in (pilot.ARTIFACTS/'step_records.jsonl').read_text().splitlines()]
        pilot.gate(len(records) == 14, 'step14 recovery checkpoint cursor differs')
        saved = RECOVERY_CHECKPOINT(ACTIVE, records)
        pilot.append_sync(pilot.ARTIFACTS/'periodic_checkpoints.jsonl', {
            'step': 14, 'path': saved['path'], 'bytes': saved['manifest']['total_bytes'],
            'write_seconds': saved['manifest']['write_seconds']})
        print('PILOT_C_RECOVERY_CHECKPOINT_COMPLETE=step14', flush=True)
    if 9 <= step <= 14:
        cached = ARCHIVE/f'step{step}'
        if phase == 'collect':
            destination = pilot.ARTIFACTS/f'step{step}'
            pilot.gate(not destination.exists(), 'cached destination exists')
            shutil.copytree(cached, destination)
        for name, checksum in PLAN['cached_files'][str(step)].items():
            pilot.gate(pilot.sha(pilot.ARTIFACTS/f'step{step}'/name) == checksum,
                       'cached step hash differs: '+name)
        pilot.append_sync(pilot.ARTIFACTS/'cache_reuse_audit.jsonl', {
            'step': step, 'phase': phase, 'source': str(cached),
            'new_sampling_calls': 0, 'new_teacher_forward_calls': 0,
            'on_policy_model_state_verified': True})
        print(f'PILOT_C_CACHED_EVIDENCE_REUSED=step{step}_{phase}', flush=True)
        return
    return ORIGINAL_OFFLINE(phase, step, source)


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
        restore_step8(self)
        records = [json.loads(x) for x in (ARTIFACTS/'step_records.jsonl').read_text().splitlines()]
        gate(len(records) == 8, 'inherited step record cursor differs')
        source = ARCHIVE/'export_after_step8'
        for step in range(9, 65):"""
pilot.gate(fit_source.count(old_prefix) == 1, 'trainer resume insertion point differs')
fit_source = fit_source.replace(old_prefix, new_prefix)
fit_namespace = dict(vars(pilot))
fit_namespace.update({'restore_step8': restore_step8, 'ARCHIVE': ARCHIVE,
                      'run_offline': cached_or_fresh})
exec(textwrap.dedent(fit_source), fit_namespace)
pilot.Trainer.fit = fit_namespace['fit']

main_source = inspect.getsource(pilot.main)
old_prepare = """    gate(not ARTIFACTS.exists(), 'C pilot output exists; no overwrite')
    preflight = gpu_preflight()
    ARTIFACTS.mkdir(parents=True)"""
new_prepare = """    gate((ARTIFACTS/'resume8_prepared.json').is_file(), 'explicit resume plan missing')
    preflight = gpu_preflight()"""
pilot.gate(main_source.count(old_prepare) == 1, 'main resume insertion point differs')
main_namespace = dict(vars(pilot))
exec(main_source.replace(old_prepare, new_prepare), main_namespace)
RESUME_MAIN = main_namespace['main']


if __name__ == '__main__':
    started = time.time()
    original_update_binding = None
    # A small Trainer override wraps only resource admission; the actor math is untouched.
    OriginalTrainer = pilot.Trainer
    class OwnershipCheckedTrainer(OriginalTrainer):
        def fit(self):
            update = self.actor_rollout_wg.update_actor
            def owned_update(batch):
                check_owned_gpu_processes('step'+str(batch.meta_info['pilot_c_step'])+'_actor_update')
                return update(batch)
            self.actor_rollout_wg.update_actor = owned_update
            return super().fit()
    main_namespace['Trainer'] = OwnershipCheckedTrainer
    try:
        result = RESUME_MAIN()
        result['resume_checkpoint_step'] = 8
        result['cached_steps_replayed'] = [9, 10, 11, 12, 13, 14]
        result['replay_model_audits'] = REPLAY_EXPORTS
        result['new_trajectories_this_attempt'] = 200
        result['unique_total_trajectories'] = 256
    except Exception as exc:
        result = {'status': 'RESUME_PILOT_C_BLOCKED', 'error': str(exc),
                  'traceback': traceback.format_exc(), 'replay_model_audits': REPLAY_EXPORTS}
    result['duration_seconds'] = round(time.time()-started, 2)
    pilot.write_new(pilot.REPORTS/'resume_pilot_training_C.json', result)
    print('PILOT_C_FINAL='+json.dumps({'status': result['status'],
                                     'duration_seconds': result['duration_seconds']}), flush=True)
    if result['status'] != 'RESUME_PILOT_C64_PASS':
        sys.exit(1)
