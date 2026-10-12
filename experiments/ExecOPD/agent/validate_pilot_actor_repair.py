"""Exercise the actual repaired actor methods with a toy CPU causal model."""
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.path.insert(0,str(REPO_ROOT))
import copy,json,hashlib,math,py_compile
from pathlib import Path
from unittest.mock import patch
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.torch_functional import masked_mean
from experiments.ExecOPD.agent.pilot_a_inner import PilotAActor
from experiments.ExecOPD.agent.pilot_c_worker import PilotCActor
from experiments.ExecOPD.agent.validate_pilot_turn_repair import ToyCausal,toy_item
from experiments.ExecOPD.agent.pilot_turn_tensor import TURN_KEYS
from experiments.ExecOPD.agent import run_pilot_A,run_pilot_C,pilot_c_offline

def gate(value,message):
    if not value: raise RuntimeError(message)

class ClipToy(ToyCausal):
    def clip_grad_norm_(self,max_norm):
        self.gradient_before_clip=torch.cat([p.grad.flatten().clone() for p in self.parameters()])
        return torch.nn.utils.clip_grad_norm_(self.parameters(),max_norm)

base=EXPERIMENT_ROOT
files=['pilot_turn_tensor.py','pilot_turn_forward.py','pilot_a_inner.py','pilot_c_worker.py',
       'pilot_c_offline.py','run_pilot_A.py','run_pilot_C.py','validate_pilot_turn_repair.py']
for name in files: py_compile.compile(str(base/'agent'/name),doraise=True)
gate(run_pilot_A.tensorize.__module__.endswith('pilot_turn_tensor') and
     run_pilot_C.tensorize.__module__.endswith('pilot_turn_tensor') and
     pilot_c_offline.inspect_item.__module__.endswith('pilot_turn_tensor') and
     pilot_c_offline.forward_selected.__module__.endswith('pilot_turn_forward'),'entry point wiring')

items=[]
for n in range(4):
    item=toy_item()
    if n in (1,3):
        for key in ('generated_token_mask','code_token_mask'): item[key][8:]=0
        for key in TURN_KEYS: item[key][1]=-1 if key=='turn_output_indices' else 0
    if n==3: # Zero-code turn still has generated-token entropy.
        item['code_token_mask'].zero_();item['turn_code_token_mask'].zero_()
    items.append(item)
packed={key:torch.stack([i[key] for i in items]) for key in items[0]}
packed['responses']=packed['input_ids'][:,3:]
checks={}
for method,cls in [('A',PilotAActor),('C',PilotCActor)]:
    torch.manual_seed(91)
    model=ClipToy()
    reference=copy.deepcopy(model)
    actor=cls.__new__(cls)
    actor.actor_module=model
    actor.config=OmegaConf.create(dict(ppo_epochs=1,use_dynamic_bsz=False,use_kl_loss=False,
        ppo_mini_batch_size=4,ppo_micro_batch_size_per_gpu=1,clip_ratio=0.2,
        entropy_coeff=0.001,grad_clip=1000,clip_ratio_c=3.0))
    actor.actor_optimizer=torch.optim.AdamW(model.parameters(),lr=5e-7,foreach=False)
    setattr(actor,'pilot_'+method.lower()+'_step_count',1)
    batch={k:v.clone() for k,v in packed.items()}
    g=batch['generated_token_mask'][:,3:]
    c=batch['code_token_mask'][:,3:]
    logits=reference(batch['input_ids'],batch['attention_mask'],batch['position_ids']).logits[:,2:-1].float()
    norm=logits.logsumexp(-1,keepdim=True)
    ent=-(logits.softmax(-1)*(logits-norm)).sum(-1)
    if method=='A':
        lp=(logits.gather(-1,batch['responses'][...,None])-norm).squeeze(-1)
        old=lp.detach()*g
        adv=torch.tensor([-1.,1.,-0.5,0.5])[:,None]*g
        batch.update(old_log_probs=old,advantages=adv)
        objective=sum(core_algos.compute_policy_loss(old[i:i+1],lp[i:i+1],adv[i:i+1],g[i:i+1],0.2)[0]
            -0.001*masked_mean(ent[i:i+1],g[i:i+1]) for i in range(4))/4
    else:
        ids=logits.topk(16,-1).indices
        lp=logits.gather(-1,ids)-norm
        adv=torch.linspace(-0.2,0.3,4*7*16).reshape(4,7,16)*c[...,None]
        batch.update(student_top_k_ids=ids,student_top_k_log_probs=lp.detach(),advantages=adv)
        terms=[]
        for i in range(4):
            pg=(actor._vanilla_opd_ppo_loss(lp[i:i+1],lp[i:i+1].detach(),adv[i:i+1],c[i:i+1])[0]
                if bool(c[i].any()) else lp[i].sum()*0.)
            terms.append(pg-0.001*masked_mean(ent[i:i+1],g[i:i+1]))
        objective=sum(terms)/4
    objective.backward()
    reference_gradient=torch.cat([p.grad.flatten() for p in reference.parameters()])
    proto=DataProto(batch=TensorDict(batch,batch_size=[4]),meta_info=dict(
        temperature=1.0,zero_variance_group=False,use_dynamic_bsz=False,micro_batch_size=1))
    before=torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    with patch.object(TensorDict,'cuda',lambda self,*a,**k:self), \
         patch('torch.cuda.current_device',lambda:'cpu'), \
         patch('torch.cuda.empty_cache',lambda:None), \
         patch('torch.cuda.max_memory_allocated',lambda:0), \
         patch('torch.distributed.all_reduce',lambda *a,**k:None):
        if method=='A':
            scored=actor.compute_log_prob(proto)
            gate(torch.allclose(scored,old,atol=1e-6,rtol=1e-6),'actual old logprob scorer')
        model.calls.clear()
        metrics=actor.update_policy(proto)
    after=torch.cat([p.detach().flatten() for p in model.parameters()])
    error=float((reference_gradient-model.gradient_before_clip).abs().max())
    gate(error<1e-6,method+' actual actor gradient differs from original objective')
    gate(len(model.calls)==8,method+' mixed-turn FSDP collective count')
    gate(bool((before!=after).any()),method+' toy optimizer did not update')
    gate(all(math.isfinite(float(v)) for values in metrics.values() for v in values),method+' nonfinite metrics')
    checks[method]={'actual_actor_update_method_pass':True,'gradient_max_abs_error':error,
                   'mixed_1_2_turn_forward_backward_calls':len(model.calls),
                   'toy_parameter_changed':True,'real_GPU_training':False,'metrics':metrics}

result={'status':'REPAIRED_ACTOR_CPU_INTEGRATION_PASS','checks':checks,
        'entrypoints_wired':True,'real_checkpoint_loads':0,'Teacher_loads':0,
        'new_rollouts':0,'final_test_task_content_access_count':0,
        'FSDP_distributed_runtime':'NOT_RUN; mock collective/device checks only',
        'file_sha256':{n:hashlib.sha256((base/'agent'/n).read_bytes()).hexdigest() for n in files}}
path=base/'reports/resume_pilot_alignment_repair_actor_check.json'
with path.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
