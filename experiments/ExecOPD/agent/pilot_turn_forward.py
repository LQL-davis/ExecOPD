"""Score actual rollout inputs, never the re-encoded packed mask carrier.

The two per-turn backward calls must be made on every FSDP rank, including an
inactive turn. Loss weights preserve the original per-trajectory masked mean.
"""
from contextlib import nullcontext

import torch

from experiments.ExecOPD.agent.pilot_turn_tensor import MAX_TURNS, check


def selected_logits(model, ids, attention, positions, selected):
    context = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if ids.is_cuda else nullcontext())
    with context:
        output = model(input_ids=ids, attention_mask=attention,
                       position_ids=positions, use_cache=False)
    # Loss at y_t uses logits at the previous position. Never score y_t from
    # logits at its own input position.
    rows, columns = selected.nonzero(as_tuple=True)
    check(bool(torch.all(columns > 0)), 'causal target has no predecessor')
    logits = output.logits[rows, columns-1, :].float()
    # An inactive FSDP rank still needs an attached zero-loss graph.
    anchor = output.logits[:, :1, :1].sum() * 0.0
    return logits, rows, columns, anchor


def turn_forward(model, sample, turn, temperature, *, topk=False):
    """One real/dummy forward; return scores scattered to packed response slots."""
    check(sample['input_ids'].shape[0] == 1, 'one trajectory per actor microbatch')
    check(temperature > 0, 'nonpositive scoring temperature')
    response_width = sample['responses'].shape[1]
    prefix_width = sample['input_ids'].shape[1] - response_width
    attention = sample['turn_attention_mask'][:, turn]
    length = int(attention.sum().item())
    if length:
        ids = sample['turn_input_ids'][:, turn, :length]
        attention = attention[:, :length]
        positions = sample['turn_position_ids'][:, turn, :length]
        generated = sample['turn_generated_token_mask'][:, turn, :length].bool()
        outputs = sample['turn_output_indices'][:, turn, :length]
        check(not bool((generated & sample['turn_execution_observation_mask'][:, turn, :length].bool()).any()),
              'local tool target')
    else:
        # All ranks execute the same number of forward/backward collectives.
        ids = sample['input_ids'][:, :1]
        attention = torch.ones_like(ids)
        positions = torch.zeros_like(ids)
        generated = torch.zeros_like(ids, dtype=torch.bool)
        outputs = torch.full_like(ids, -1)
    logits, rows, columns, anchor = selected_logits(model, ids, attention, positions, generated)
    logits = logits / temperature
    carrier = outputs[rows, columns]
    targets = carrier - prefix_width
    check(bool(torch.all((targets >= 0) & (targets < response_width))), 'response target mapping')
    check(torch.equal(ids[rows, columns], sample['input_ids'][rows, carrier]), 'scoring sampled IDs')
    check(not bool(sample['execution_observation_mask'][rows, carrier].any()), 'scoring tool overlap')
    normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
    entropy_values = -(logits.softmax(-1) * (logits-normalizer)).sum(-1)
    entropy = torch.zeros((1, response_width), device=ids.device) + anchor
    entropy = entropy.index_put((rows, targets), entropy_values)
    generated_mask = torch.zeros_like(entropy, dtype=torch.long)
    generated_mask[rows, targets] = 1
    if topk:
        candidate_ids = sample['student_top_k_ids'][rows, targets]
        values = logits.gather(-1, candidate_ids) - normalizer
        scores = torch.zeros((1, response_width, sample['student_top_k_ids'].shape[-1]), device=ids.device) + anchor
    else:
        values = (logits.gather(-1, ids[rows, columns, None])-normalizer).squeeze(-1)
        scores = torch.zeros((1, response_width), device=ids.device) + anchor
    scores = scores.index_put((rows, targets), values)
    return entropy, scores, generated_mask


@torch.no_grad()
def forward_selected(model, item):
    """Offline Student/Teacher forward on each turn's exact saved input."""
    device = next(model.parameters()).device
    values, outputs = [], []
    for turn in range(MAX_TURNS):
        length = int(item['turn_attention_mask'][turn].sum())
        if not length:
            continue
        ids = item['turn_input_ids'][turn, :length].to(device)[None]
        attention = item['turn_attention_mask'][turn, :length].to(device)[None]
        positions = item['turn_position_ids'][turn, :length].to(device)[None]
        generated = item['turn_generated_token_mask'][turn, :length].to(device)[None].bool()
        logits, rows, columns, _ = selected_logits(model, ids, attention, positions, generated)
        carrier = item['turn_output_indices'][turn, :length][columns.cpu()]
        check(torch.equal(ids[rows, columns].cpu(), item['input_ids'][carrier]), 'offline sampled IDs')
        check(not bool(item['execution_observation_mask'][carrier].any()), 'offline tool target')
        values.append(logits)
        outputs.append(carrier)
    logits, indices = torch.cat(values), torch.cat(outputs).to(device)
    check(indices.cpu().tolist() == item['generated_token_mask'].nonzero().flatten().tolist(),
          'all generated scoring positions')
    return logits, indices
