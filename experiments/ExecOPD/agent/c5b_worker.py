"""C5-B Actor worker: Vanilla OPD equations with code-only target mask.

The only change to the vendored mechanics is explicit Agent generated-token
masking and audit instrumentation. No vendored veRL file is modified.
"""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import hashlib
from pathlib import Path

import torch
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

TMP = (EXPERIMENT_ROOT / 'agent/tmp/c5b')
PARAMS = ('model.layers.14.self_attn.q_proj.weight', 'model.norm.weight')
K = 16


class C5BActor(DataParallelPPOActor):
    def _agent_topk_forward(self, sample, temperature):
        """Code-R1 FSDP forward with Rethinking-OPD same-ID Top-K gather."""
        response_len = sample['responses'].size(1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.actor_module(input_ids=sample['input_ids'],
                                       attention_mask=sample['attention_mask'],
                                       position_ids=sample['position_ids'], use_cache=False)
            logits = output.logits[:, -response_len - 1:-1, :] / temperature
            ids = sample['student_top_k_ids']
            selected = logits.gather(-1, ids)
            topk_logp = selected.float() - torch.logsumexp(logits.float(), dim=-1, keepdim=True)
            entropy = verl_F.entropy_from_logits(logits)
            return entropy, topk_logp

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
        step = self.c5b_step_count + 1
        if step not in (1, 2) or self.config.ppo_epochs != 1 or self.config.use_dynamic_bsz:
            raise RuntimeError('C5_B_RUNTIME_BLOCKED: unexpected actor update configuration')
        if self.config.use_kl_loss:
            raise RuntimeError('C5_B_RUNTIME_BLOCKED: original Vanilla OPD has KL loss disabled')
        self.actor_module.train()
        temperature = data.meta_info['temperature']
        keys = ['responses', 'input_ids', 'attention_mask', 'position_ids',
                'generated_token_mask', 'code_token_mask', 'execution_observation_mask',
                'credit_weight_per_token', 'student_top_k_ids',
                'student_top_k_log_probs', 'advantages']
        batch = data.select(batch_keys=keys).batch
        minis = batch.split(self.config.ppo_mini_batch_size)
        if len(minis) != 1:
            raise RuntimeError('C5_B_RUNTIME_BLOCKED: expected one four-trajectory PPO minibatch')
        metrics = {}
        self.actor_optimizer.zero_grad()
        for micro in minis[0].split(self.config.ppo_micro_batch_size_per_gpu):
            sample = micro.cuda()
            response_len = sample['responses'].size(1)
            generated = sample['generated_token_mask'][:, -response_len:]
            mask = sample['code_token_mask'][:, -response_len:]
            tool = sample['execution_observation_mask'][:, -response_len:]
            credit = sample['credit_weight_per_token'][:, -response_len:]
            noncode = generated.bool() & ~mask.bool()
            advantage = sample['advantages']
            if (advantage.shape != (*mask.shape, K) or not bool(mask.any()) or
                    not bool(torch.all(mask <= generated)) or
                    not bool(torch.all(credit[~mask.bool()] == 0)) or
                    not bool(torch.all((credit >= 0) & (credit <= 1))) or
                    bool((mask & tool).any()) or bool((generated & tool).any())):
                raise RuntimeError('C5_B_MASK_BLOCKED: OPD advantage or generated/tool mask invalid')
            if bool(torch.any(advantage * (1 - mask.unsqueeze(-1)))):
                raise RuntimeError('C5_B_MASK_BLOCKED: OPD advantage leaked to tool or padding')
            if bool(torch.any(advantage[mask.bool() & (credit == 0)] != 0)):
                raise RuntimeError('C5_B_MASK_BLOCKED: zero-credit code token has PPO advantage')
            entropy, current_topk_log_probs = self._agent_topk_forward(sample, temperature)
            if current_topk_log_probs.shape != advantage.shape:
                raise RuntimeError('C5_B_NUMERICAL_BLOCKED: actor Top-K logprob shape mismatch')
            # Original dp_actor.py on-policy ppo_epochs=1 branch detaches the
            # current policy as old_log_prob for the 3-D Vanilla PPO ratio.
            old_log_prob = current_topk_log_probs.detach()
            pg_loss, pg_metrics = self._vanilla_opd_ppo_loss(
                current_topk_log_probs, old_log_prob, advantage, mask)
            policy_loss = pg_loss
            if self.config.entropy_coeff != 0:
                entropy_loss = verl_F.masked_mean(entropy, mask)
                policy_loss = policy_loss - entropy_loss * self.config.entropy_coeff
            else:
                entropy_loss = torch.zeros((), device=policy_loss.device)
            gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            loss = policy_loss / gradient_accumulation
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('C5_B_NUMERICAL_BLOCKED: nonfinite actor loss')
            loss.backward()
            append_to_dict(metrics, {
                'actor/pg_loss': float(pg_loss.detach().item()) / gradient_accumulation,
                'actor/entropy_loss': float(entropy_loss.detach().item()) / gradient_accumulation,
                'c5b/total_loss': float(policy_loss.detach().item()) / gradient_accumulation,
                'c5b/loss_mask_tokens': int(mask.sum().item()),
                'c5b/tool_loss_overlap': int((mask & tool).sum().item()),
                'c5b/noncode_assistant_loss_overlap': int((mask & noncode).sum().item()),
                'c5b/zero_credit_code_advantage_nonzero': int(torch.count_nonzero(
                    advantage[mask.bool() & (credit == 0)]).item()),
                'c5b/zero_credit_code_entropy_mask_tokens': int((mask.bool() &
                    (credit == 0)).sum().item()),
                'c5b/entropy_mask_unweighted_code_only': 1,
                'c5b/topk_advantage_nonzero': int(torch.count_nonzero(advantage).item()),
            })
            append_to_dict(metrics, pg_metrics)
        grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError('C5_B_NUMERICAL_BLOCKED: nonfinite gradient norm')
        nonzero = torch.tensor(int(any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                                        for p in self.actor_module.parameters())),
                               device=torch.cuda.current_device())
        dist.all_reduce(nonzero, op=dist.ReduceOp.MAX)
        lr = [float(group['lr']) for group in self.actor_optimizer.param_groups]
        expected = 0.0 if step == 1 else 5e-7
        if lr != [expected]:
            raise RuntimeError(f'C5_B_RUNTIME_BLOCKED: optimizer LR {lr} differs from {expected}')
        # Code-R1 AdamW defaults to foreach=True here. The foreach sqrt creates
        # a transient tensor that exhausted the H100 after gradients and Adam
        # state were already allocated. The per-tensor AdamW path keeps the
        # same optimizer equations without that bulk temporary allocation.
        for group in self.actor_optimizer.param_groups:
            group['foreach'] = False
        torch.cuda.empty_cache()
        self.actor_optimizer.step()
        self.c5b_step_count = step
        self.actor_optimizer.zero_grad()
        append_to_dict(metrics, {'actor/grad_norm': float(grad_norm.item()),
                                 'c5b/any_nonzero_gradient': int(nonzero.item()),
                                 'c5b/actual_optimizer_lr': lr[0],
                                 'c5b/optimizer_step_count': step,
                                 'c5b/optimizer_foreach_disabled': 1,
                                 'c5b/actor_peak_allocated_gib':
                                     torch.cuda.max_memory_allocated() / 1024**3})
        return metrics


class _Inner(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result = super().init_model()
        if self._is_actor:
            self.actor = C5BActor(config=self.config.actor,
                                  actor_module=self.actor_module_fsdp,
                                  actor_optimizer=self.actor_optimizer)
            self.actor.c5b_step_count = 0
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def c5b_lr_audit(self):
        return {'rank': dist.get_rank(),
                'lrs': [float(g['lr']) for g in self.actor_optimizer.param_groups],
                'scheduler_last_epoch': int(self.actor_lr_scheduler.last_epoch),
                'optimizer_step_count': int(self.actor.c5b_step_count)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def c5b_parameters(self, phase):
        if phase not in ('before_step1', 'after_step1', 'after_step2'):
            raise RuntimeError('C5_B_PARAMETER_BLOCKED: invalid parameter snapshot phase')
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
                    raise RuntimeError('C5_B_PARAMETER_BLOCKED: parameter lookup ' + wanted)
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


class C5BWorker(Worker):
    def __init__(self, config, role):
        self.inner = _Inner(config, role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        return self.inner.init_model()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data):
        return self.inner.update_actor(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def c5b_lr_audit(self):
        return self.inner.c5b_lr_audit()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def c5b_parameters(self, phase):
        return self.inner.c5b_parameters(phase)
