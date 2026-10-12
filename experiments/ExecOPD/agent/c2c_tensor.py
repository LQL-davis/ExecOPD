"""C2-B alignment rules applied to fresh C2-C Agent histories."""
from __future__ import annotations

import hashlib
import json

import torch


def chat_ids(tokenizer, messages, *, generation_prompt=False):
    """Same C2-B chat-template rule, without importing its CPU-only entrypoint."""
    plain = [{'role': message['role'], 'content': message['content']} for message in messages]
    return list(tokenizer.apply_chat_template(plain, tokenize=True,
                                              add_generation_prompt=generation_prompt))


def checked(condition, detail):
    if not condition:
        raise RuntimeError('C2_C_MASK_BLOCKED: ' + detail)


def tensorize(tokenizer, trajectory):
    messages, turns = trajectory['state']['messages'], trajectory['turns']
    checked(len(turns) in (1, 2), 'wrong number of assistant turns')
    checked([m['role'] for m in messages[:2]] == ['system', 'user'], 'initial role order')
    full = chat_ids(tokenizer, messages)
    checked(full == trajectory['final_history_token_ids'], 'final saved IDs differ')
    initial = chat_ids(tokenizer, messages[:2], generation_prompt=True)
    checked(initial == turns[0]['prompt_token_ids'], 'first prompt IDs differ')
    n = len(full)
    item = {'input_ids': torch.tensor(full, dtype=torch.long),
            'attention_mask': torch.ones(n, dtype=torch.long),
            'position_ids': torch.arange(n, dtype=torch.long),
            'generated_token_mask': torch.zeros(n, dtype=torch.long),
            'assistant_token_mask': torch.zeros(n, dtype=torch.long),
            'code_token_mask': torch.zeros(n, dtype=torch.long),
            'execution_observation_mask': torch.zeros(n, dtype=torch.long),
            'turn_ids': torch.full((n,), -1, dtype=torch.long)}
    alignment = {'global_step': trajectory['global_step'],
                 'trajectory_index': trajectory['trajectory_index'],
                 'assistant_spans': [], 'tool_spans': [], 'code_spans': [],
                 'initial_prompt_length': len(initial), 'sequence_length': n}
    cursor = 2
    for index, turn in enumerate(turns):
        checked(cursor < len(messages) and messages[cursor]['role'] == 'assistant',
                'assistant message absent')
        prefix = chat_ids(tokenizer, messages[:cursor], generation_prompt=True)
        checked(prefix == turn['prompt_token_ids'], 'turn prompt sampled IDs differ')
        sample = list(turn['raw_student_token_ids'])
        a0, a1 = len(prefix), len(prefix) + len(sample)
        checked(sample and full[a0:a1] == sample, 'sampled Student IDs fail exact reconstruction')
        raw = turn['raw_student_response_text']
        checked(raw == tokenizer.decode(sample, skip_special_tokens=True) == messages[cursor]['content'],
                'decoded assistant content differs')
        encoded = tokenizer(raw, add_special_tokens=False, return_offsets_mapping=True)
        content_ids = list(encoded['input_ids'])
        eos = bool(sample[-1] == tokenizer.eos_token_id)
        checked(content_ids == (sample[:-1] if eos else sample),
                'assistant content does not map to sampled token IDs')
        assistant_end = chat_ids(tokenizer, messages[:cursor + 1])
        checked(full[:len(assistant_end)] == assistant_end and len(assistant_end) >= a1,
                'assistant template boundary differs')
        item['generated_token_mask'][a0:a1] = 1
        item['assistant_token_mask'][a0:a1] = 1
        item['turn_ids'][a0:a1] = index
        alignment['assistant_spans'].append({'turn_id': index, 'span': [a0, a1],
            'sampled_ids_sha256': hashlib.sha256(json.dumps(sample, separators=(',', ':')).encode()).hexdigest(),
            'exact': True, 'sampled_eos': eos})
        artifact = turn['code_artifact']
        if artifact['extraction_status'] == 'SELECTED':
            c0, c1 = artifact['char_start'], artifact['char_end']
            checked(isinstance(c0, int) and isinstance(c1, int) and 0 <= c0 < c1 <= len(raw)
                    and raw[c0:c1] == artifact['selected_code'], 'selected code char span differs')
            offsets = [tuple(pair) for pair in encoded['offset_mapping']]
            touched = [i for i, (left, right) in enumerate(offsets) if right > c0 and left < c1]
            checked(touched and touched == list(range(touched[0], touched[-1] + 1)),
                    'selected code has no contiguous token coverage')
            first, last = touched[0], touched[-1] + 1
            checked(full[a0 + first:a0 + last] == sample[first:last], 'selected code IDs differ')
            left_boundary_crossing = offsets[first][0] < c0
            right_boundary_crossing = offsets[last - 1][1] > c1
            # A BPE token can straddle the selected code's character edge.
            # Include every sampled assistant token with nonempty overlap,
            # while preserving explicit boundary-crossing evidence below.
            item['code_token_mask'][a0 + first:a0 + last] = 1
            item['code_token_mask'] &= item['generated_token_mask']
            alignment['code_spans'].append({'turn_id': index,
                'span': [a0 + first, a0 + last], 'char_span': [c0, c1],
                'exact': not (left_boundary_crossing or right_boundary_crossing),
                'projection_rule': 'nonempty_character_overlap_and_generated_mask',
                'left_boundary_crossing': left_boundary_crossing,
                'right_boundary_crossing': right_boundary_crossing,
                'boundary_crossing': left_boundary_crossing or right_boundary_crossing})
        else:
            alignment['code_spans'].append({'turn_id': index, 'span': None,
                                             'status': artifact['extraction_status'], 'exact': True})
        cursor += 1
        if cursor < len(messages) and messages[cursor]['role'] == 'tool':
            through_tool = chat_ids(tokenizer, messages[:cursor + 1])
            t0, t1 = len(assistant_end), len(through_tool)
            checked(t0 < t1 and full[:t1] == through_tool, 'tool template boundary differs')
            item['execution_observation_mask'][t0:t1] = 1
            alignment['tool_spans'].append({'turn_id': index, 'span': [t0, t1], 'exact': True})
            cursor += 1
        else:
            checked(index == len(turns) - 1, 'missing intermediate tool observation')
        if index + 1 < len(turns):
            checked(chat_ids(tokenizer, messages[:cursor], generation_prompt=True) ==
                    turns[index + 1]['prompt_token_ids'], 'next prompt lacks tool observation')
    checked(cursor == len(messages), 'extra messages after final assistant/tool')
    a, g, s, c, t = (item[name] for name in ('attention_mask', 'generated_token_mask',
        'assistant_token_mask', 'code_token_mask', 'execution_observation_mask'))
    checks = {'code_le_assistant': bool(torch.all(c <= s)),
              'assistant_le_generated': bool(torch.all(s <= g)),
              'generated_le_attention': bool(torch.all(g <= a)),
              'generated_tool_overlap_zero': not bool(torch.any(g & t)),
              'code_tool_overlap_zero': not bool(torch.any(c & t)),
              'sampled_count_exact': int(g.sum()) == sum(len(x['raw_student_token_ids']) for x in turns),
              'tool_count_exact': len(alignment['tool_spans']) == sum(
                  x['execution_observation'] is not None for x in turns)}
    checked(all(checks.values()), 'mask invariant failed: ' + repr(checks))
    alignment['checks'] = checks
    alignment['generated_count'] = int(g.sum())
    alignment['tool_count'] = int(t.sum())
    alignment['code_count'] = int(c.sum())
    return item, alignment


def batch_pad(tokenizer, items, alignments):
    checked(len(items) == 4, 'expected exactly four trajectories')
    prompt = max(x['initial_prompt_length'] for x in alignments)
    continuation = max(x['sequence_length'] - x['initial_prompt_length'] for x in alignments)
    out = {key: [] for key in items[0]}
    for item, info in zip(items, alignments):
        left = prompt - info['initial_prompt_length']
        right = continuation - (info['sequence_length'] - info['initial_prompt_length'])
        for key, tensor in item.items():
            if key == 'position_ids':
                padded = torch.cat((torch.zeros(left, dtype=torch.long), tensor,
                                    torch.arange(len(tensor), len(tensor) + right)))
            else:
                fill = tokenizer.pad_token_id if key == 'input_ids' else (-1 if key == 'turn_ids' else 0)
                padded = torch.cat((torch.full((left,), fill, dtype=torch.long), tensor,
                                    torch.full((right,), fill, dtype=torch.long)))
            out[key].append(padded)
    out = {key: torch.stack(value) for key, value in out.items()}
    for row, (item, info) in enumerate(zip(items, alignments)):
        valid = out['attention_mask'][row].bool()
        for key in item:
            checked(torch.equal(out[key][row][valid], item[key]), f'batch reconstruction {key}')
        invalid = ~valid
        for key in ('attention_mask', 'generated_token_mask', 'assistant_token_mask',
                    'code_token_mask', 'execution_observation_mask'):
            checked(not bool(torch.any(out[key][row][invalid])), 'nonzero padding mask ' + key)
        checked(bool(torch.all(out['turn_ids'][row][invalid] == -1)), 'padding turn IDs')
    checked(out['input_ids'].shape[1] <= 2048, 'batch exceeds 2048 token context')
    return out, prompt, continuation
