"""CPU-only regression of frozen samples and exact-turn scoring/loss mechanics."""
from experiments.ExecOPD.agent.release_paths import (EXPERIMENT_ROOT, PYTHON_EXE, REPO_ROOT, SCRATCH_DIR, STUDENT_MODEL, TEACHER_MODEL)
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from transformers import AutoTokenizer

from experiments.ExecOPD.agent.pilot_turn_tensor import (
    tensorize, batch_pad, sampled_offsets, TURN_KEYS)
from experiments.ExecOPD.agent.pilot_turn_forward import turn_forward, forward_selected
from experiments.ExecOPD.agent.c2c_tensor import tensorize as legacy_tensorize

BASE = EXPERIMENT_ROOT
SOURCE = BASE / 'reports/resume_pilot_A_trajectories.jsonl'
EXPECTED = '6f47d8f27ab8eca3d7e57ff355983e9a1c4dc420c03207f508b3b40152a56304'


def gate(value, detail):
    if not value:
        raise RuntimeError('ALIGNMENT_REPAIR_VALIDATION_BLOCKED: ' + detail)


class ToyCausal(torch.nn.Module):
    """CPU causal model with history dependence; never loads a real checkpoint."""
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 8)
        self.head = torch.nn.Linear(8, 32)
        self.calls = []

    def forward(self, input_ids, attention_mask, position_ids, use_cache=False):
        self.calls.append(input_ids.detach().clone())
        hidden = (self.embed(input_ids) * attention_mask[..., None]).cumsum(1)
        hidden = hidden / (position_ids[..., None]+1).clamp_min(1)
        return SimpleNamespace(logits=self.head(hidden))


def toy_item():
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    gen = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    code = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    tool = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    return dict(input_ids=ids, attention_mask=torch.ones(10, dtype=torch.long),
        position_ids=torch.arange(10), generated_token_mask=gen, code_token_mask=code,
        execution_observation_mask=tool,
        turn_input_ids=torch.stack((torch.cat((ids[:5], torch.zeros(5, dtype=torch.long))), ids)),
        turn_attention_mask=torch.tensor([[1]*5+[0]*5, [1]*10]),
        turn_position_ids=torch.stack((torch.cat((torch.arange(5), torch.zeros(5, dtype=torch.long))), torch.arange(10))),
        turn_generated_token_mask=torch.tensor([[0, 0, 0, 1, 1]+[0]*5, [0]*8+[1, 1]]),
        turn_code_token_mask=torch.tensor([[0]*4+[1]+[0]*5, [0]*9+[1]]),
        turn_execution_observation_mask=torch.tensor([[0]*10, [0]*5+[1, 1]+[0]*3]),
        turn_output_indices=torch.tensor([[-1]*3+[3, 4]+[-1]*5, [-1]*8+[8, 9]]))


def masked_mean(values, mask):
    return (values*mask).sum()/mask.sum().clamp_min(1)


def toy_checks():
    torch.manual_seed(23)
    model = ToyCausal()
    item = toy_item()
    sample = {k: v[None].clone() for k, v in item.items()}
    sample['responses'] = sample['input_ids'][:, 3:]
    g = sample['generated_token_mask'][:, 3:]
    c = sample['code_token_mask'][:, 3:]
    full = model(sample['input_ids'], sample['attention_mask'], sample['position_ids']).logits[:, 2:-1].float()
    normalizer = full.logsumexp(-1, keepdim=True)
    expected_logp = (full.gather(-1, sample['responses'][..., None])-normalizer).squeeze(-1)*g
    expected_entropy = -(full.softmax(-1)*(full-normalizer)).sum(-1)*g
    combined_logp, combined_ent = torch.zeros_like(g, dtype=torch.float32), torch.zeros_like(g, dtype=torch.float32)
    for turn in range(2):
        ent, logp, mask = turn_forward(model, sample, turn, 1.0)
        combined_logp += logp
        combined_ent += ent
    gate(torch.allclose(combined_logp, expected_logp, atol=1e-6, rtol=1e-6), 'causal shift sampled logprob')
    gate(torch.allclose(combined_ent, expected_entropy, atol=1e-6, rtol=1e-6), 'generated entropy scoring')
    ids = full.topk(16, -1).indices
    sample['student_top_k_ids'] = ids
    expected_topk = (full.gather(-1, ids)-normalizer)*g[..., None]
    combined = torch.zeros_like(expected_topk)
    for turn in range(2):
        _, scores, _ = turn_forward(model, sample, turn, 1.0, topk=True)
        combined += scores
    gate(torch.allclose(combined, expected_topk, atol=1e-6, rtol=1e-6), 'same-ID Top-K gather')
    offline_logits, positions = forward_selected(model, item)
    gate(positions.tolist() == [3, 4, 8, 9], 'offline scoring positions')
    gate(torch.allclose(offline_logits, full[0][g[0].bool()], atol=1e-6, rtol=1e-6), 'offline same-state logits')
    # A packed concatenation is deliberately a different state in this case.
    # Actual saved next-turn context must win, not the carrier's earlier IDs.
    changed = copy.deepcopy(item)
    changed['turn_input_ids'][1, 3] = 11
    model.calls.clear()
    forward_selected(model, changed)
    gate(torch.equal(model.calls[1][0], changed['turn_input_ids'][1]), 'exact next-turn input used')
    gate(not torch.equal(model.calls[1][0], changed['input_ids']), 'carrier never used as model context')
    gradient_checks = {}
    for method in ('A', 'C'):
        reference, perturn = copy.deepcopy(model), copy.deepcopy(model)
        reference.calls.clear(); perturn.calls.clear()
        logits = reference(sample['input_ids'], sample['attention_mask'], sample['position_ids']).logits[:, 2:-1].float()
        norm = logits.logsumexp(-1, keepdim=True)
        ent = -(logits.softmax(-1)*(logits-norm)).sum(-1)
        advantage = torch.linspace(-0.3, 0.6, 7)[None]
        if method == 'A':
            logp = (logits.gather(-1, sample['responses'][..., None])-norm).squeeze(-1)
            old = logp.detach()+0.04
            reference_loss = masked_mean(torch.maximum(-advantage*torch.exp(logp-old),
                -advantage*torch.exp(logp-old).clamp(0.8, 1.2)), g)-0.001*masked_mean(ent, g)
        else:
            logp = logits.gather(-1, ids)-norm
            advantage = torch.linspace(-0.2, 0.3, 7*16).reshape(1, 7, 16)*c[..., None]
            old = logp.detach()
            ratio = (logp-old).clamp(-20,20).exp()
            upper = torch.maximum(-advantage*ratio, -advantage*ratio.clamp(0.8,1.2))
            loss_k = torch.where(advantage<0, torch.minimum(-advantage*3, upper), upper).sum(-1)
            reference_loss = masked_mean(loss_k,c)-0.001*masked_mean(ent,g)
        reference_loss.backward()
        perturn_loss = 0.0
        for turn in range(2):
            ent, logp, gm = turn_forward(perturn, sample, turn, 1.0, topk=method=='C')
            if method=='A':
                policy = masked_mean(torch.maximum(-advantage*torch.exp(logp-old),
                    -advantage*torch.exp(logp-old).clamp(0.8,1.2)),gm)*gm.sum()/g.sum()
            else:
                cm = c*gm
                ratio = (logp-logp.detach()).clamp(-20,20).exp()
                upper = torch.maximum(-advantage*ratio, -advantage*ratio.clamp(0.8,1.2))
                loss_k = torch.where(advantage<0, torch.minimum(-advantage*3,upper),upper).sum(-1)
                policy = masked_mean(loss_k,cm)*cm.sum()/c.sum()
            loss = policy-0.001*masked_mean(ent,gm)*gm.sum()/g.sum()
            perturn_loss += float(loss.detach())
            loss.backward()
        gate(abs(perturn_loss-float(reference_loss.detach())) < 1e-6, method+' loss equivalence')
        max_grad_error = max(float((a.grad-b.grad).abs().max()) for a,b in zip(reference.parameters(), perturn.parameters()))
        gate(max_grad_error<1e-6, method+' gradient equivalence')
        gradient_checks[method] = {'loss_abs_error':abs(perturn_loss-float(reference_loss.detach())),
                                   'gradient_max_abs_error':max_grad_error}
    dummy = copy.deepcopy(sample)
    for k in TURN_KEYS:
        dummy[k][:,1] = -1 if k=='turn_output_indices' else 0
    inactive = ToyCausal()
    _, scores, gm = turn_forward(inactive, dummy, 1, 1.0)
    scores.sum().backward()
    gate(not bool(gm.any()) and all(p.grad is not None and not bool(p.grad.any()) for p in inactive.parameters()),
         'inactive turn attached zero-gradient backward')
    return {'causal_shift':True,'same_ID_topk':True,'same_state_offline':True,
            'saved_prompt_not_carrier':True,'inactive_rank_backward':True,
            'objective_gradient_equivalence':gradient_checks,
            'real_checkpoint_loads':0, 'GPU_model_forwards':0,
            'note':'Toy CPU backward checks implementation only; no Student/Teacher training.'}


def main():
    raw = SOURCE.read_bytes()
    gate(hashlib.sha256(raw).hexdigest()==EXPECTED, 'frozen input SHA256')
    tokenizer = AutoTokenizer.from_pretrained(str(STUDENT_MODEL),
                                               use_fast=True, local_files_only=True)
    rows = [json.loads(x) for x in raw.decode().splitlines()]
    items, alignments, noncanonical, legacy_equal = [], [], [], 0
    mask_keys = ('input_ids', 'attention_mask', 'position_ids','generated_token_mask',
                 'assistant_token_mask','code_token_mask','execution_observation_mask','turn_ids')
    for row in rows:
        item, info = tensorize(tokenizer,row)
        items.append(item);alignments.append(info)
        altered = [x for x in info['turn_inputs'] if not x['canonical_reencode_equal']]
        if altered:
            noncanonical.append({'step':row['global_step'],'trajectory':row['trajectory_index'],
                                 'turns':[x['turn_id'] for x in altered]})
        else:
            old, _ = legacy_tensorize(tokenizer,row)
            gate(all(torch.equal(item[k],old[k]) for k in mask_keys),'legacy canonical masks changed')
            legacy_equal += 1
    gate(len(noncanonical)==2,'failure cases not reproduced')
    for j in range(0,len(rows),4):
        batch_pad(tokenizer,items[j:j+4],alignments[j:j+4])
    # Byte offsets must remain exact even with split UTF-8 characters.
    euro_ids = [tokenizer.convert_tokens_to_ids(c) for c in ('â','Ĥ','¬')]
    gate(sampled_offsets(tokenizer,euro_ids,'€')==[(0,1)]*3,'split UTF-8 byte character spans')
    raw_case='```python\n\nx=1\n```'
    sample=tokenizer.encode('```python',add_special_tokens=False)+[271]+tokenizer.encode('x=1\n```',add_special_tokens=False)
    spans=sampled_offsets(tokenizer,sample,raw_case)
    cross=[j for j,(a,b) in enumerate(spans) if a<10<b]
    gate(len(cross)==1 and sample[cross[0]]==271 and spans[cross[0]]==(9,11),'boundary crossing 271')
    mechanics=toy_checks()
    gate(hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED,'frozen source mutated')
    result={'status':'RESUME_PILOT_ALIGNMENT_REPAIR_CPU_PASS','source_sha256':EXPECTED,
        'trajectories':len(rows),'assistant_turns_exact':sum(len(r['turns']) for r in rows),
        'actual_prompt_turns_exact':sum(len(r['turns']) for r in rows),
        'code_artifacts_covered':sum(int(c['span'] is not None) for a in alignments for c in a['code_spans']),
        'tool_observations_exact':sum(len(a['tool_spans']) for a in alignments),
        'batch_padding_groups_pass':len(rows)//4,'canonical_legacy_identical':legacy_equal,
        'noncanonical_cases_pass':noncanonical,'generated_tool_overlap':0,'code_tool_overlap':0,
        'max_actual_turn_context':max(t['sequence_length'] for a in alignments for t in a['turn_inputs']),
        'max_packed_carrier_length':max(a['sequence_length'] for a in alignments),
        'boundary_crossing_tokens':sum(len(c['crossing_tokens']) for a in alignments for c in a['code_spans']),
        'split_UTF8_byte_offsets_pass':True,'boundary_crossing_271_pass':True,
        'mechanics_checks':mechanics,'real_training_started':False,'new_rollouts':0,
        'two_GPU_runtime_gate':'NOT_RUN','restart_state':'prior step30 has no complete resumable checkpoint'}
    for name,value in [('resume_pilot_alignment_repair_result.json',result),
                       ('resume_pilot_alignment_repair_audit.json',alignments)]:
        with (BASE/'reports'/name).open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
