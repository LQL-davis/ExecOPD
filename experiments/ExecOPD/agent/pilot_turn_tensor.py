"""Exact per-turn model inputs; a packed carrier stores masks, never model context.

Do not re-encode a sampled assistant response to construct a training target.
Every forward uses the prompt IDs actually submitted for that turn plus its
actual sampled IDs. Earlier turns inside that prompt retain their actual
text-template encoding, even when it differs from their earlier sampled IDs.
"""
from __future__ import annotations

import bisect
import hashlib
import json

import torch

TURN_KEYS = ('turn_input_ids', 'turn_attention_mask', 'turn_position_ids',
             'turn_generated_token_mask', 'turn_code_token_mask',
             'turn_execution_observation_mask', 'turn_output_indices')
MAX_TURNS = 2
MAX_CONTEXT = 2048


def check(value, reason):
    if not value:
        raise RuntimeError('PILOT_EXACT_TURN_ALIGNMENT_BLOCKED: ' + reason)


def chat_ids(tokenizer, messages, generation=False):
    return list(tokenizer.apply_chat_template(
        [{'role': m['role'], 'content': m['content']} for m in messages],
        tokenize=True, add_generation_prompt=generation))


def sha_ids(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def sampled_offsets(tokenizer, sample, raw):
    """Project original ByteLevel vocabulary pieces to decoded character spans.

    Byte spans are built from the sampled IDs, not canonical re-tokenization.
    UTF-8 character overlaps also cover tokens containing only part of a char.
    Unknown decoder/invalid UTF-8 fails explicitly rather than guessing offsets.
    """
    decoder = json.loads(tokenizer.backend_tokenizer.decoder.__getstate__())
    check(decoder['type'] == 'ByteLevel', 'unsupported tokenizer decoder')
    # Build the standard ByteLevel inverse without relying on encode(raw).
    base = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    inverse = {chr(b): b for b in base}
    for n, b in enumerate(b for b in range(256) if b not in base):
        inverse[chr(256 + n)] = b
    special = set(tokenizer.all_special_ids)
    data = bytearray()
    spans = []
    for tok in sample:
        start = len(data)
        if tok not in special:
            piece = tokenizer.convert_ids_to_tokens(int(tok))
            check(piece is not None and all(c in inverse for c in piece),
                  'sampled token has unsupported byte vocabulary piece')
            data.extend(inverse[c] for c in piece)
        spans.append((start, len(data)))
    try:
        decoded = bytes(data).decode('utf-8', errors='strict')
    except UnicodeDecodeError as exc:
        raise RuntimeError('PILOT_EXACT_TURN_ALIGNMENT_BLOCKED: invalid sampled UTF-8') from exc
    check(decoded == raw == tokenizer.decode(sample, skip_special_tokens=True),
          'sampled byte decoding differs from saved response')
    bounds = [0]
    for char in raw:
        bounds.append(bounds[-1] + len(char.encode('utf-8')))
    return [(bisect.bisect_right(bounds, a) - 1, bisect.bisect_left(bounds, b))
            if b > a else (bisect.bisect_left(bounds, a),) * 2 for a, b in spans]


def tensorize(tokenizer, trajectory):
    messages, turns = trajectory['state']['messages'], trajectory['turns']
    check(1 <= len(turns) <= MAX_TURNS, 'turn count')
    check([m['role'] for m in messages[:2]] == ['system', 'user'], 'initial roles')
    canonical = chat_ids(tokenizer, messages)
    check(canonical == trajectory['final_history_token_ids'], 'saved canonical history changed')
    initial = chat_ids(tokenizer, messages[:2], True)
    packed = initial.copy()
    generated, code, tool, turn_ids = ([0] * len(packed), [0] * len(packed),
                                     [0] * len(packed), [-1] * len(packed))
    parts = []
    info = {'representation': 'exact_turn_inputs_with_packed_mask_carrier',
            'carrier_is_model_input': False, 'global_step': trajectory['global_step'],
            'trajectory_index': trajectory['trajectory_index'],
            'initial_prompt_length': len(initial), 'assistant_spans': [],
            'tool_spans': [], 'code_spans': [], 'turn_inputs': []}
    cursor = 2

    def append(ids, generated_value=0, code_values=None, tool_value=0, turn=-1):
        packed.extend(ids)
        generated.extend([generated_value] * len(ids))
        code.extend(code_values if code_values is not None else [0] * len(ids))
        tool.extend([tool_value] * len(ids))
        turn_ids.extend([turn] * len(ids))

    closure = tokenizer.encode('<|im_end|>\n', add_special_tokens=False)
    check(closure == [tokenizer.eos_token_id, 198], 'unrecognized frozen chat closure')
    for index, turn in enumerate(turns):
        check(cursor < len(messages) and messages[cursor]['role'] == 'assistant', 'assistant role')
        prompt = chat_ids(tokenizer, messages[:cursor], True)
        check(prompt == turn['prompt_token_ids'], 'actual prompt reconstruction')
        sample = list(turn['raw_student_token_ids'])
        check(0 < len(sample) <= 512 and len(prompt) + len(sample) <= MAX_CONTEXT,
              'actual turn context budget')
        raw = turn['raw_student_response_text']
        check(raw == messages[cursor]['content'], 'saved assistant message differs')
        offsets = sampled_offsets(tokenizer, sample, raw)
        artifact = turn['code_artifact']
        cmask = [0] * len(sample)
        crossings = []
        selected = artifact['extraction_status'] == 'SELECTED'
        if selected:
            s, e = artifact['char_start'], artifact['char_end']
            check(0 <= s < e <= len(raw) and raw[s:e] == artifact['selected_code'],
                  'selected executable code changed')
            touched = [i for i, (a, b) in enumerate(offsets) if b > s and a < e]
            check(touched and touched == list(range(touched[0], touched[-1] + 1)), 'code coverage')
            for i in touched:
                cmask[i] = 1
                a, b = offsets[i]
                if a < s or b > e:
                    crossings.append({'sampled_index': i, 'token_id': sample[i],
                                      'decoded_piece': tokenizer.decode([sample[i]]),
                                      'char_span': [a, b], 'left': a < s, 'right': b > e,
                                      'non_code_characters': max(0, s-a) + max(0, b-e)})
            check(all(any(a <= j < b for i, (a, b) in enumerate(offsets) if cmask[i])
                      for j in range(s, e)), 'code characters not fully covered')
        start = len(packed)
        append(sample, generated_value=1, code_values=cmask, turn=index)
        end = len(packed)
        eos = sample[-1] == tokenizer.eos_token_id
        append(closure[1:] if eos else closure)
        info['assistant_spans'].append({'turn_id': index, 'span': [start, end],
            'exact': True, 'sampled_eos': eos, 'sampled_ids_sha256': sha_ids(sample)})
        info['code_spans'].append({'turn_id': index,
            'span': [start + touched[0], start + touched[-1] + 1] if selected else None,
            'char_span': [s, e] if selected else None, 'exact': not crossings,
            'status': artifact['extraction_status'], 'boundary_crossing': bool(crossings),
            'crossing_tokens': crossings,
            'projection_rule': 'sampled_byte_character_overlap_and_generated_mask'})
        local_ids = prompt + sample
        local_generated = [0] * len(prompt) + [1] * len(sample)
        local_code = [0] * len(prompt) + cmask
        local_tool = [0] * len(local_ids)
        # Prior observations are context, never a current-turn target.
        for m in range(2, cursor):
            if messages[m]['role'] == 'tool':
                before, through = chat_ids(tokenizer, messages[:m]), chat_ids(tokenizer, messages[:m+1])
                check(prompt[:len(through)] == through and through[:len(before)] == before,
                      'tool context boundary')
                local_tool[len(before):len(through)] = [1] * (len(through)-len(before))
        parts.append({'turn_input_ids': local_ids, 'turn_attention_mask': [1] * len(local_ids),
                      'turn_position_ids': list(range(len(local_ids))),
                      'turn_generated_token_mask': local_generated,
                      'turn_code_token_mask': local_code,
                      'turn_execution_observation_mask': local_tool,
                      'turn_output_indices': [-1] * len(prompt) + list(range(start, end))})
        info['turn_inputs'].append({'turn_id': index, 'prompt_length': len(prompt),
            'sampled_length': len(sample), 'sequence_length': len(local_ids),
            'prompt_ids_sha256': sha_ids(prompt), 'sampled_ids_sha256': sha_ids(sample),
            'full_input_ids_sha256': sha_ids(local_ids), 'prompt_exact': True, 'sampled_exact': True,
            'canonical_reencode_equal': tokenizer.encode(raw, add_special_tokens=False) ==
                (sample[:-1] if eos else sample)})
        cursor += 1
        if cursor < len(messages) and messages[cursor]['role'] == 'tool':
            before = chat_ids(tokenizer, messages[:cursor])
            through = chat_ids(tokenizer, messages[:cursor+1])
            check(through[:len(before)] == before, 'tool template prefix')
            t0 = len(packed)
            append(through[len(before):], tool_value=1)
            info['tool_spans'].append({'turn_id': index, 'span': [t0, len(packed)], 'exact': True})
            cursor += 1
        else:
            check(index == len(turns)-1, 'missing execution feedback')
        if index + 1 < len(turns):
            prior = chat_ids(tokenizer, messages[:cursor])
            next_prompt = chat_ids(tokenizer, messages[:cursor], True)
            check(next_prompt[:len(prior)] == prior and next_prompt == turns[index+1]['prompt_token_ids'],
                  'next actual prompt changed')
            append(next_prompt[len(prior):])
    check(cursor == len(messages), 'unaccounted messages')
    n = len(packed)
    item = {'input_ids': torch.tensor(packed), 'attention_mask': torch.ones(n, dtype=torch.long),
            'position_ids': torch.arange(n), 'generated_token_mask': torch.tensor(generated),
            'assistant_token_mask': torch.tensor(generated), 'code_token_mask': torch.tensor(code),
            'execution_observation_mask': torch.tensor(tool), 'turn_ids': torch.tensor(turn_ids)}
    width = max(len(p['turn_input_ids']) for p in parts)
    for key in TURN_KEYS:
        fill = tokenizer.pad_token_id if key == 'turn_input_ids' else (-1 if key == 'turn_output_indices' else 0)
        rows = [p[key] + [fill] * (width-len(p[key])) for p in parts]
        rows += [[fill] * width for _ in range(MAX_TURNS-len(parts))]
        item[key] = torch.tensor(rows, dtype=torch.long)
    g, c, t = item['generated_token_mask'], item['code_token_mask'], item['execution_observation_mask']
    checks = {'code_le_assistant': bool(torch.all(c <= g)), 'assistant_le_generated': True,
              'generated_le_attention': bool(torch.all(g <= item['attention_mask'])),
              'generated_tool_overlap_zero': not bool((g & t).any()),
              'code_tool_overlap_zero': not bool((c & t).any()),
              'sampled_count_exact': int(g.sum()) == sum(len(x['raw_student_token_ids']) for x in turns),
              'tool_count_exact': len(info['tool_spans']) == sum(x['execution_observation'] is not None for x in turns),
              'actual_prompts_exact': True,
              'local_generated_tool_overlap_zero': not bool((item['turn_generated_token_mask'] &
                                                            item['turn_execution_observation_mask']).any())}
    check(all(checks.values()), 'mask invariant')
    info.update({'checks': checks, 'sequence_length': n, 'generated_count': int(g.sum()),
                 'code_count': int(c.sum()), 'tool_count': int(t.sum())})
    return item, info


def batch_pad(tokenizer, items, alignments):
    check(len(items) == 4, 'exactly four trajectories required')
    prompt = max(i['initial_prompt_length'] for i in alignments)
    continuation = max(i['sequence_length']-i['initial_prompt_length'] for i in alignments)
    turn_width = max(x['turn_input_ids'].shape[1] for x in items)
    out = {key: [] for key in items[0]}
    for item, info in zip(items, alignments):
        left = prompt-info['initial_prompt_length']
        right = continuation-(info['sequence_length']-info['initial_prompt_length'])
        for key, tensor in item.items():
            fill = tokenizer.pad_token_id if key in ('input_ids', 'turn_input_ids') else (
                -1 if key in ('turn_ids', 'turn_output_indices') else 0)
            if key in TURN_KEYS:
                value = tensor.clone()
                if key == 'turn_output_indices':
                    value[value >= 0] += left
                value = torch.cat((value, torch.full((MAX_TURNS, turn_width-value.shape[1]), fill)), dim=1)
            else:
                value = torch.cat((torch.full((left,), fill), tensor, torch.full((right,), fill)))
                if key == 'position_ids':
                    value[:left] = 0
                    value[left+len(tensor):] = torch.arange(len(tensor), len(tensor)+right)
            out[key].append(value)
    out = {key: torch.stack(rows) for key, rows in out.items()}
    for row, (item, info) in enumerate(zip(items, alignments)):
        valid = out['attention_mask'][row].bool()
        for key, tensor in item.items():
            if key not in TURN_KEYS:
                check(torch.equal(out[key][row][valid], tensor), 'packed batch reconstruction: '+key)
        invalid = ~valid
        for key in ('attention_mask', 'generated_token_mask', 'assistant_token_mask',
                    'code_token_mask', 'execution_observation_mask'):
            check(not bool(out[key][row][invalid].any()), 'padding mask: '+key)
        check(bool(torch.all(out['turn_ids'][row][invalid] == -1)), 'padding turn IDs')
        outputs = out['turn_output_indices'][row]
        active = outputs >= 0
        check(torch.equal(out['input_ids'][row][outputs[active]], out['turn_input_ids'][row][active]),
              'turn-to-packed sampled IDs')
        check(bool(torch.all(out['generated_token_mask'][row][outputs[active]] == 1)), 'turn target mapping')
        check(int(active.sum()) == int(valid.logical_and(out['generated_token_mask'][row].bool()).sum()),
              'turn target count')
    return out, prompt, continuation
