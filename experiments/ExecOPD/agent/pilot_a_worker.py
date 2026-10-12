"""Student-only GRPO worker with the same FSDP checkpoint wrapper as Method D."""
import hashlib
import torch
import torch.distributed as dist
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.base import Worker
from experiments.ExecOPD.agent.pilot_a_inner import _Inner


def digest(value):
    h = hashlib.sha256()
    def visit(obj):
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(obj, dict):
            for key in sorted(obj, key=str):
                h.update(str(key).encode())
                visit(obj[key])
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                visit(item)
        else:
            h.update(repr(obj).encode())
    visit(value)
    return h.hexdigest()


class PilotAWorker(Worker):
    def __init__(self, config, role):
        self.inner = _Inner(config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        return self.inner.init_model()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_generate(self, prompt_ids):
        return self.inner.pilot_a_generate(prompt_ids)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data):
        return self.inner.compute_log_prob(data)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data):
        return self.inner.compute_ref_log_prob(data)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data):
        return self.inner.update_actor(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_lr_audit(self):
        return self.inner.pilot_a_lr_audit()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_seed_audit(self):
        return self.inner.pilot_a_seed_audit()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_parameters(self, phase):
        return self.inner.pilot_a_parameters(phase)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_export_hf(self, destination):
        return self.inner.pilot_a_export_hf(destination)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path, global_step):
        self.inner.save_checkpoint(path, global_step=global_step,
                                   remove_previous_ckpt=False)
        return {'rank': dist.get_rank(), 'global_step': global_step}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, global_step, generation_call):
        self.inner.load_checkpoint(path, del_local_after_load=False)
        self.inner.actor.pilot_a_step_count = global_step
        self.inner.pilot_a_generation_call = generation_call
        return {'rank': dist.get_rank(), 'global_step': global_step,
                'generation_call': generation_call}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_state_audit(self):
        inner = self.inner
        optim = inner.actor_optimizer.state_dict()
        sched = inner.actor_lr_scheduler.state_dict()
        rng = inner.checkpoint_manager.get_rng_state()
        return {'rank': dist.get_rank(),
                'global_step': int(inner.actor.pilot_a_step_count),
                'generation_call': int(inner.pilot_a_generation_call),
                'model_local_sha256': digest([p.detach().cpu().contiguous()
                                              for p in inner.actor_module_fsdp.parameters()]),
                'optimizer_sha256': digest(optim),
                'scheduler_sha256': digest(sched),
                'scheduler_state': sched,
                'rng_sha256': {key: digest(value) for key, value in rng.items()},
                'optimizer_state_entries': len(optim['state']),
                'optimizer_lr': [float(x['lr']) for x in inner.actor_optimizer.param_groups]}
