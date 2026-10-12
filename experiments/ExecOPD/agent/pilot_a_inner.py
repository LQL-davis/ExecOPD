"""C2-C experiment worker: hybrid Student rollout and generated-token-only PPO."""
from __future__ import annotations
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)

import hashlib
import math
import os
import random
import shutil
from pathlib import Path

import torch
import numpy as np
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from safetensors.torch import save_file
from experiments.ExecOPD.agent.pilot_turn_tensor import TURN_KEYS, MAX_TURNS
from experiments.ExecOPD.agent.pilot_turn_forward import turn_forward

from experiments.ExecOPD.agent.pilot_paths import OUTPUT
TMP = OUTPUT / 'agent/tmp/pilot_a'
ORIGINAL = STUDENT_MODEL
PARAMS = ('model.layers.14.self_attn.q_proj.weight', 'model.norm.weight')


class PilotAActor(DataParallelPPOActor):
    def compute_log_prob(self, data: DataProto):
        if data.meta_info['use_dynamic_bsz'] or data.meta_info['micro_batch_size'] != 1:
            raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: exact-turn logprob requires microbatch 1')
        self.actor_module.eval()
        keys = ['responses', 'input_ids', 'attention_mask', 'position_ids',
                'generated_token_mask', 'execution_observation_mask'] + list(TURN_KEYS)
        outputs = []
        with torch.no_grad():
            for micro in data.select(batch_keys=keys).batch.split(1):
                sample = micro.cuda()
                combined = torch.zeros_like(sample['responses'], dtype=torch.float32)
                target = torch.zeros_like(sample['responses'])
                for turn in range(MAX_TURNS):
                    _, scores, mask = turn_forward(self.actor_module, sample, turn,
                                                    data.meta_info['temperature'])
                    combined += scores
                    target += mask
                if not torch.equal(target, sample['generated_token_mask'][:, -target.shape[1]:]):
                    raise RuntimeError('RESUME_PILOT_A_MASK_BLOCKED: old logprob target coverage')
                outputs.append(combined)
        return torch.cat(outputs)

    def update_policy(self, data: DataProto):
        step = self.pilot_a_step_count + 1
        if not 1 <= step <= 64 or self.config.ppo_epochs != 1 or self.config.use_dynamic_bsz:
            raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: unexpected actor update configuration')
        if self.config.use_kl_loss:
            raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: Reference/KL must be disabled')
        self.actor_module.train()
        temperature = data.meta_info['temperature']
        keys = ['responses', 'input_ids', 'attention_mask', 'position_ids',
                'generated_token_mask', 'execution_observation_mask',
                'old_log_probs', 'advantages'] + list(TURN_KEYS)
        batch = data.select(batch_keys=keys).batch
        minis = batch.split(self.config.ppo_mini_batch_size)
        if len(minis) != 1:
            raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: expected one four-trajectory PPO minibatch')
        metrics = {}
        self.actor_optimizer.zero_grad()
        for micro in minis[0].split(self.config.ppo_micro_batch_size_per_gpu):
            sample = micro.cuda()
            response_len = sample['responses'].size(1)
            mask = sample['generated_token_mask'][:, -response_len:]
            tool = sample['execution_observation_mask'][:, -response_len:]
            if mask.shape != sample['advantages'].shape or not bool(mask.any()) or bool((mask & tool).any()):
                raise RuntimeError('RESUME_PILOT_A_MASK_BLOCKED: actor loss mask or overlap invalid')
            if data.meta_info['zero_variance_group'] and bool(torch.any(sample['advantages'] != 0)):
                raise RuntimeError('RESUME_PILOT_A_NUMERICAL_BLOCKED: zero variance has nonzero advantage')
            totals = dict(pg=0.0, entropy=0.0, clip=0.0, kl=0.0, loss=0.0)
            covered = torch.zeros_like(mask)
            accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            for turn in range(MAX_TURNS):
                entropy, log_prob, turn_mask = turn_forward(self.actor_module, sample, turn, temperature)
                covered += turn_mask
                weight = turn_mask.sum() / mask.sum()
                if bool(turn_mask.any()):
                    pg, clip, kl = core_algos.compute_policy_loss(
                        old_log_prob=sample['old_log_probs'], log_prob=log_prob,
                        advantages=sample['advantages'], eos_mask=turn_mask,
                        cliprange=self.config.clip_ratio)
                    ent = masked_mean(entropy, turn_mask)
                else:
                    pg = log_prob.sum() * 0.0
                    ent = entropy.sum() * 0.0
                    clip = kl = torch.zeros_like(pg)
                # Splitting the same trajectory masked mean across turns
                # preserves its original loss and gradient normalization.
                pg, ent = pg * weight, ent * weight
                total = pg - ent * self.config.entropy_coeff
                loss = total / accumulation
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('RESUME_PILOT_A_NUMERICAL_BLOCKED: nonfinite actor loss')
                loss.backward()
                totals['pg'] += float(pg.detach())
                totals['entropy'] += float(ent.detach())
                totals['clip'] += float((clip * weight).detach())
                totals['kl'] += float((kl * weight).detach())
                totals['loss'] += float(total.detach())
                del entropy, log_prob, pg, ent, total, loss, clip, kl
            if not torch.equal(covered, mask):
                raise RuntimeError('RESUME_PILOT_A_MASK_BLOCKED: turn target coverage')
            append_to_dict(metrics, {
                'actor/pg_loss': totals['pg'],
                'actor/pg_clipfrac': totals['clip'],
                'actor/ppo_kl': totals['kl'],
                'actor/entropy_loss': totals['entropy'],
                'actor/kl_loss': 0.0,
                'pilot_a/entropy_tool_overlap': int((mask & tool).sum().item()),
                'pilot_a/total_loss': totals['loss'],
                'pilot_a/grpo_policy_contribution': totals['pg'],
                'pilot_a/entropy_contribution': -totals['entropy'] * self.config.entropy_coeff,
                'pilot_a/exact_turn_forward_backward_count': MAX_TURNS,
                'pilot_a/loss_mask_tokens': int(mask.sum().item()),
                'pilot_a/tool_loss_overlap': int((mask & tool).sum().item()),
            })
        grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError('RESUME_PILOT_A_NUMERICAL_BLOCKED: nonfinite gradient norm')
        nonzero = torch.tensor(int(any(p.grad is not None and bool(torch.count_nonzero(p.grad))
                                        for p in self.actor_module.parameters())),
                               device=torch.cuda.current_device())
        dist.all_reduce(nonzero, op=dist.ReduceOp.MAX)
        lr = [float(group['lr']) for group in self.actor_optimizer.param_groups]
        expected = 0.0 if step == 1 else 5e-7
        if lr != [expected]:
            raise RuntimeError(f'RESUME_PILOT_A_RUNTIME_BLOCKED: optimizer LR {lr} differs from {expected}')
        for group in self.actor_optimizer.param_groups:
            group['foreach'] = False
        torch.cuda.empty_cache()
        self.actor_optimizer.step()
        self.pilot_a_step_count = step
        self.actor_optimizer.zero_grad()
        append_to_dict(metrics, {'actor/grad_norm': float(grad_norm.item()),
                                 'pilot_a/any_nonzero_gradient': int(nonzero.item()),
                                 'pilot_a/actual_optimizer_lr': lr[0],
                                 'pilot_a/optimizer_step_count': step})
        return metrics


class _Inner(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        result = super().init_model()
        seed = 20261006
        if os.environ.get('D1_SEED') != str(seed):
            raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: Ray worker seed environment mismatch')
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.pilot_a_seed = seed
        self.pilot_a_generation_call = 0
        if self.rollout_sharding_manager.gen_random_states is not None:
            self.rollout_sharding_manager.gen_random_states = torch.cuda.get_rng_state().clone()
        if self._is_actor:
            self.actor = PilotAActor(config=self.config.actor,
                                  actor_module=self.actor_module_fsdp,
                                  actor_optimizer=self.actor_optimizer)
            self.actor.pilot_a_step_count = 0
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_generate(self, prompt_ids):
        """One exact Student sample on each TP rank, synchronized from current Actor."""
        if not self._is_rollout or self.config.rollout.tensor_model_parallel_size != 2 or \
                self.config.rollout.n != 4 or not 0 < len(prompt_ids) <= 1536:
            raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: invalid interactive rollout state')
        self.pilot_a_generation_call += 1
        request_seed = self.pilot_a_seed + self.pilot_a_generation_call - 1
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        with self.rollout_sharding_manager:
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                from verl.utils.fsdp_utils import offload_fsdp_optimizer
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            if self.config.rollout.free_cache_engine:
                self.rollout.inference_engine.init_cache_engine()
            try:
                with self.rollout.update_sampling_params(n=1, temperature=1.0, top_p=1.0,
                                                         top_k=-1, max_tokens=512, detokenize=False,
                                                         ignore_eos=False, seed=request_seed):
                    output, _ = self.rollout.inference_engine.generate(
                        prompts=None, prompt_token_ids=[list(prompt_ids)],
                        sampling_params=self.rollout.sampling_params, use_tqdm=False)
                if output.shape[0] != 1 or not 0 < output.shape[1] <= 512:
                    raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: invalid exact sample tensor')
                sampled = output[0].tolist()
            finally:
                if self.config.rollout.free_cache_engine:
                    self.rollout.inference_engine.free_cache_engine()
        torch.cuda.empty_cache()
        return {'rank': dist.get_rank(), 'sampled_ids': sampled,
                'base_seed': self.pilot_a_seed, 'request_seed': request_seed,
                'generation_call': self.pilot_a_generation_call,
                'finish_reason': 'eos' if sampled[-1] == self.tokenizer.eos_token_id else 'length_or_stop',
                'prompt_length': len(prompt_ids)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_lr_audit(self):
        return {'rank': dist.get_rank(), 'lrs': [float(g['lr']) for g in self.actor_optimizer.param_groups],
                'scheduler_last_epoch': int(self.actor_lr_scheduler.last_epoch),
                'optimizer_step_count': int(self.actor.pilot_a_step_count)}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_seed_audit(self):
        state = self.rollout_sharding_manager.gen_random_states
        return {'rank': dist.get_rank(), 'python_seed': self.pilot_a_seed,
                'numpy_seed': self.pilot_a_seed, 'torch_cpu_initial_seed': torch.initial_seed(),
                'torch_cuda_seed': self.pilot_a_seed,
                'vllm_rng_state_sha256': hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
                if state is not None else None,
                'sampling_seed_rule': '20261006 + generation_call - 1'}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_export_hf(self, destination):
        target = Path(destination)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.actor_module_fsdp,
                                  StateDictType.FULL_STATE_DICT, cfg):
            state = self.actor_module_fsdp.state_dict()
        result = {'rank': dist.get_rank(), 'destination': str(target),
                  'weight_count': len(state)}
        if dist.get_rank() == 0:
            target.mkdir(parents=True, exist_ok=False)
            for name in ('config.json', 'generation_config.json',
                         'tokenizer.json', 'tokenizer_config.json',
                         'vocab.json', 'merges.txt'):
                shutil.copy2(ORIGINAL / name, target / name)
            weight_path = target / 'model.safetensors'
            save_file({key: value.detach().cpu().contiguous()
                       for key, value in state.items()}, str(weight_path),
                      metadata={'format': 'pt'})
            result['weight_file_bytes'] = weight_path.stat().st_size
            checksum = hashlib.sha256()
            with weight_path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    checksum.update(block)
            result['weight_file_sha256'] = checksum.hexdigest()
        del state
        dist.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        torch.cuda.empty_cache()
        return result

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def pilot_a_parameters(self, phase):
        if phase != 'before_step1' and phase not in tuple(f'after_step{i}' for i in range(1, 65)):
            raise RuntimeError('RESUME_PILOT_A_RUNTIME_BLOCKED: bad parameter audit phase')
        module = self.actor_module_fsdp
        if self._is_offload_param:
            load_fsdp_model_to_gpu(module)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT, cfg):
            state = module.state_dict()
        result = {'rank': dist.get_rank(), 'role': 'actor',
                  'phase': phase, 'parameters': {}}
        if dist.get_rank() == 0:
            TMP.mkdir(parents=True, exist_ok=True)
            for wanted in PARAMS:
                matches = [key for key in state if key.endswith(wanted)]
                if len(matches) != 1:
                    raise RuntimeError('RESUME_PILOT_A_PARAMETER_UPDATE_BLOCKED: parameter lookup ' + wanted)
                tensor = state[matches[0]].detach().cpu().contiguous()
                cache = TMP / ('actor_' + wanted.replace('.', '_') + '.pt')
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
