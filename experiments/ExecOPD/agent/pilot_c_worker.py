"""C5-B Actor worker: Vanilla OPD equations with code-only target mask.

The only change to the vendored mechanics is explicit Agent generated-token
masking and audit instrumentation. No vendored veRL file is modified.
"""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import torch
import numpy as np
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.py_functional import append_to_dict
import verl.utils.torch_functional as verl_F
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from safetensors.torch import save_file
from experiments.ExecOPD.agent.pilot_turn_tensor import TURN_KEYS, MAX_TURNS
from experiments.ExecOPD.agent.pilot_turn_forward import turn_forward

from experiments.ExecOPD.agent.pilot_paths import OUTPUT
TMP = OUTPUT / 'agent/tmp/pilot_c/resume_pilot'
ORIGINAL = STUDENT_MODEL
PARAMS = ('model.layers.14.self_attn.q_proj.weight', 'model.norm.weight')
K = 16


class PilotCActor(DataParallelPPOActor):
    def _agent_topk_forward(self, sample, temperature, turn):
        """Code-R1 FSDP forward with Rethinking-OPD same-ID Top-K gather."""
        return turn_forward(self.actor_module, sample, turn, temperature, topk=True)

    def _vanilla_opd_ppo_loss(self, current, old, advantages, mask):
        """Exact 3-D only_stu Vanilla branch from Rethinking-OPD core_algos.py.

        Baseline source SHA256 d7a2796319d9de5e0672138ae131e458b82ae0e54e9e121e44d415ece9a24bc3.
        Token-mean aggregation equals masked_mean after summing the K dimension.
        """
        cliprange = float(self.config.clip_ratio)
        clip_ratio_c = float(self.config.get('clip_ratio_c', 3.0))
        delta = torch.clamp(current - old, min=-20.0, max=20.0)
        ratio = torch.exp(delta)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.minimum(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1).sum(dim=-1)
        pg_loss = verl_F.masked_mean(pg_losses, mask)
        with torch.no_grad():
            ppo_kl = verl_F.masked_mean(-delta.sum(dim=-1), mask)
            pg_clipfrac = verl_F.masked_mean((pg_losses2 > pg_losses1).float(), mask.unsqueeze(-1))
        return pg_loss, {'actor/pg_clipfrac': float(pg_clipfrac.item()),
                         'actor/ppo_kl': float(ppo_kl.item()),
                         'actor/pg_clipfrac_lower': 0.0}

    def update_policy(self, data: DataProto):
        step = self.pilot_c_step_count + 1
        if not 1 <= step <= 64 or self.config.ppo_epochs != 1 or self.config.use_dynamic_bsz:
            raise RuntimeError('RESUME_PILOT_C_RUNTIME_BLOCKED: unexpected actor update configuration')
        if self.config.use_kl_loss:
            raise RuntimeError('RESUME_PILOT_C_RUNTIME_BLOCKED: original Vanilla OPD has KL loss disabled')
        self.actor_module.train()
        temperature = data.meta_info['temperature']
        keys = ['responses', 'input_ids', 'attention_mask', 'position_ids',
                'generated_token_mask', 'code_token_mask', 'execution_observation_mask',
                'student_top_k_ids', 'student_top_k_log_probs', 'advantages'] + list(TURN_KEYS)
        batch = data.select(batch_keys=keys).batch
        minis = batch.split(self.config.ppo_mini_batch_size)
        if len(minis) != 1:
            raise RuntimeError('RESUME_PILOT_C_RUNTIME_BLOCKED: expected one four-trajectory PPO minibatch')
        metrics = {}
        self.actor_optimizer.zero_grad()
        for micro in minis[0].split(self.config.ppo_micro_batch_size_per_gpu):
            sample = micro.cuda()
            response_len = sample['responses'].size(1)
            generated = sample['generated_token_mask'][:, -response_len:]
            mask = sample['code_token_mask'][:, -response_len:]
            tool = sample['execution_observation_mask'][:, -response_len:]
            noncode = generated.bool() & ~mask.bool()
            advantage = sample['advantages']
            if (advantage.shape != (*mask.shape, K) or not bool(generated.any()) or
                    not bool(torch.all(mask <= generated)) or
                    bool((mask & tool).any()) or bool((generated & tool).any())):
                raise RuntimeError('RESUME_PILOT_C_MASK_BLOCKED: OPD advantage or generated/tool mask invalid')
            if bool(torch.any(advantage * (1 - mask.unsqueeze(-1)))):
                raise RuntimeError('RESUME_PILOT_C_MASK_BLOCKED: OPD advantage leaked to tool or padding')
            gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            totals = dict(pg=0.0, entropy=0.0, loss=0.0)
            pg_metrics = {'actor/pg_clipfrac': 0.0, 'actor/ppo_kl': 0.0,
                          'actor/pg_clipfrac_lower': 0.0}
            covered = torch.zeros_like(generated)
            for turn in range(MAX_TURNS):
                entropy, current, turn_generated = self._agent_topk_forward(sample, temperature, turn)
                covered += turn_generated
                turn_code = mask * turn_generated
                code_weight = turn_code.sum() / mask.sum().clamp_min(1)
                generated_weight = turn_generated.sum() / generated.sum()
                if current.shape != advantage.shape:
                    raise RuntimeError('RESUME_PILOT_C_NUMERICAL_BLOCKED: actor Top-K logprob shape mismatch')
                # Preserve the baseline ppo_epochs=1 on-policy ratio branch.
                if bool(turn_code.any()):
                    pg, stats = self._vanilla_opd_ppo_loss(
                        current, current.detach(), advantage, turn_code)
                    for key in pg_metrics:
                        pg_metrics[key] += stats[key] * float(code_weight)
                else:
                    pg = current.sum() * 0.0
                if bool(turn_generated.any()) and self.config.entropy_coeff != 0:
                    ent = verl_F.masked_mean(entropy, turn_generated)
                else:
                    ent = entropy.sum() * 0.0
                pg, ent = pg * code_weight, ent * generated_weight
                total = pg - ent * self.config.entropy_coeff
                loss = total / gradient_accumulation
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('RESUME_PILOT_C_NUMERICAL_BLOCKED: nonfinite actor loss')
                loss.backward()
                totals['pg'] += float(pg.detach())
                totals['entropy'] += float(ent.detach())
                totals['loss'] += float(total.detach())
                del entropy, current, pg, ent, total, loss
            if not torch.equal(covered, generated):
                raise RuntimeError('RESUME_PILOT_C_MASK_BLOCKED: turn target coverage')
            append_to_dict(metrics, {
                'actor/pg_loss': totals['pg'] / gradient_accumulation,
                'actor/entropy_loss': totals['entropy'] / gradient_accumulation,
                'pilot_c/entropy_mask_generated_tokens': int(generated.sum().item()),
                'pilot_c/entropy_tool_overlap': int((generated & tool).sum().item()),
                'pilot_c/entropy_noncode_generated_tokens': int(noncode.sum().item()),
                'pilot_c/total_loss': totals['loss'] / gradient_accumulation,
                'pilot_c/exact_turn_forward_backward_count': MAX_TURNS,
                'pilot_c/loss_mask_tokens': int(mask.sum().item()),
                'pilot_c/zero_code_minibatch': int(not bool(mask.any())),
                'pilot_c/tool_loss_overlap': int((mask & tool).sum().item()),
                'pilot_c/noncode_assistant_loss_overlap': int((mask & noncode).sum().item()),
                'pilot_c/entropy_mask_unweighted_generated': 1,
                'pilot_c/topk_advantage_nonzero': int(torch.count_nonzero(advantage).item()),
            })
            append_to_dict(metrics, pg_metrics)
        grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError('RESUME_PILOT_C_NUMERICAL_BLOCKED: nonfinite gradient norm')
        nonzero = torch.tensor(int(any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                                        for p in self.actor_module.parameters())),
                               device=torch.cuda.current_device())
        dist.all_reduce(nonzero, op=dist.ReduceOp.MAX)
        lr = [float(group['lr']) for group in self.actor_optimizer.param_groups]
        expected = 0.0 if step == 1 else 5e-7
        if lr != [expected]:
            raise RuntimeError(f'RESUME_PILOT_C_RUNTIME_BLOCKED: optimizer LR {lr} differs from {expected}')
        # Code-R1 AdamW defaults to foreach=True here. The foreach sqrt creates
        # a transient tensor that exhausted the H100 after gradients and Adam
        # state were already allocated. The per-tensor AdamW path keeps the
        # same optimizer equations without that bulk temporary allocation.
        for group in self.actor_optimizer.param_groups:
            group['foreach'] = False
        torch.cuda.empty_cache()
        self.actor_optimizer.step()
        self.pilot_c_step_count = step
        self.actor_optimizer.zero_grad()
        append_to_dict(metrics, {'actor/grad_norm': float(grad_norm.item()),
                                 'pilot_c/any_nonzero_gradient': int(nonzero.item()),
                                 'pilot_c/actual_optimizer_lr': lr[0],
                                 'pilot_c/optimizer_step_count': step,
                                 'pilot_c/optimizer_foreach_disabled': 1,
                                 'pilot_c/actor_peak_allocated_gib':
                                     torch.cuda.max_memory_allocated() / 1024**3})
        return metrics


class _Inner(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result = super().init_model()
        seed = 20261006
        if os.environ.get('D1_SEED') != str(seed):
            raise RuntimeError('RESUME_PILOT_C_RUNTIME_BLOCKED: Ray worker seed environment')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self._is_actor:
            self.actor = PilotCActor(config=self.config.actor,
                                  actor_module=self.actor_module_fsdp,
                                  actor_optimizer=self.actor_optimizer)
            self.actor.pilot_c_step_count = 0
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_c_lr_audit(self):
        return {'rank': dist.get_rank(),
                'lrs': [float(g['lr']) for g in self.actor_optimizer.param_groups],
                'scheduler_last_epoch': int(self.actor_lr_scheduler.last_epoch),
                'optimizer_step_count': int(self.actor.pilot_c_step_count)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_c_set_step(self, step):
        if not 0 <= step <= 64:
            raise RuntimeError('RESUME_PILOT_C_RUNTIME_BLOCKED: invalid restored step')
        self.actor.pilot_c_step_count = step
        return {'rank': dist.get_rank(), 'step': step}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_c_state_audit(self):
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
                    for one in obj:
                        visit(one)
                else:
                    h.update(repr(obj).encode())
            visit(value)
            return h.hexdigest()
        rng = self.checkpoint_manager.get_rng_state()
        optim = self.actor_optimizer.state_dict()
        sched = self.actor_lr_scheduler.state_dict()
        model_parts = [p.detach().cpu().contiguous() for p in self.actor_module_fsdp.parameters()]
        return {'rank': dist.get_rank(), 'step': int(self.actor.pilot_c_step_count),
                'model_local_sha256': digest(model_parts),
                'optimizer_sha256': digest(optim), 'scheduler_sha256': digest(sched),
                'scheduler_state': sched,
                'rng_sha256': {key: digest(val) for key, val in rng.items()},
                'optimizer_state_entries': len(optim['state']),
                'optimizer_lr': [float(x['lr']) for x in self.actor_optimizer.param_groups]}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_c_export_hf(self, destination):
        target = Path(destination)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.actor_module_fsdp, StateDictType.FULL_STATE_DICT, cfg):
            state = self.actor_module_fsdp.state_dict()
        result = {'rank': dist.get_rank(), 'destination': str(target), 'weight_count': len(state)}
        if dist.get_rank() == 0:
            target.mkdir(parents=True, exist_ok=False)
            small = ('config.json', 'generation_config.json', 'tokenizer.json',
                     'tokenizer_config.json', 'vocab.json', 'merges.txt')
            for name in small:
                shutil.copy2(ORIGINAL / name, target / name)
            weight_path = target / 'model.safetensors'
            save_file({key: value.detach().cpu().contiguous() for key, value in state.items()},
                      str(weight_path), metadata={'format': 'pt'})
            result['weight_file_bytes'] = weight_path.stat().st_size
            checksum = hashlib.sha256()
            with weight_path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    checksum.update(block)
            result['weight_file_sha256'] = checksum.hexdigest()
            result['first_keys'] = list(state)[:5]
        del state
        dist.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        torch.cuda.empty_cache()
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_c_parameters(self, phase):
        if phase != 'before_step1' and phase not in tuple(f'after_step{i}' for i in range(1, 65)):
            raise RuntimeError('RESUME_PILOT_C_PARAMETER_BLOCKED: invalid parameter snapshot phase')
        module = self.actor_module_fsdp
        if self._is_offload_param:
            load_fsdp_model_to_gpu(module)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, cfg):
            state = module.state_dict()
        result = {'rank': dist.get_rank(), 'phase': phase, 'parameters': {}}
        if dist.get_rank() == 0:
            TMP.mkdir(parents=True, exist_ok=True)
            for wanted in PARAMS:
                matches = [key for key in state if key.endswith(wanted)]
                if len(matches) != 1:
                    raise RuntimeError('RESUME_PILOT_C_PARAMETER_BLOCKED: parameter lookup ' + wanted)
                tensor = state[matches[0]].detach().cpu().contiguous()
                cache = TMP / (wanted.replace('.', '_') + '.pt')
                rec = {'sha256': hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
                       'l2_norm': float(torch.linalg.vector_norm(tensor.float()))}
                if phase == 'before_step1':
                    torch.save(tensor, cache)
                else:
                    prior = torch.load(cache, map_location='cpu', weights_only=True)
                    delta = tensor.float() - prior.float()
                    rec.update({'changed': bool(torch.any(delta != 0)),
                                'l2_delta': float(torch.linalg.vector_norm(delta))})
                    torch.save(tensor, cache)
                result['parameters'][wanted] = rec
        del state
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(module)
        torch.cuda.empty_cache()
        return result


class PilotCWorker(Worker):
    def __init__(self, config, role):
        self.inner = _Inner(config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        return self.inner.init_model()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, global_step):
        self.inner.save_checkpoint(local_path, global_step=global_step,
                                   remove_previous_ckpt=False)
        return {'rank': dist.get_rank(), 'global_step': global_step}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, global_step):
        self.inner.load_checkpoint(path, del_local_after_load=False)
        return self.inner.pilot_c_set_step(global_step)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data):
        return self.inner.update_actor(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_c_lr_audit(self):
        return self.inner.pilot_c_lr_audit()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_c_parameters(self, phase):
        return self.inner.pilot_c_parameters(phase)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_c_state_audit(self):
        return self.inner.pilot_c_state_audit()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_c_export_hf(self, destination):
        return self.inner.pilot_c_export_hf(destination)
