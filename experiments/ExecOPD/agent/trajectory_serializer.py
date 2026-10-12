"""Role-preserving history serialization and descriptive token-span metadata.

No training masks are constructed in C2-A. Raw sampled Student token IDs are
stored separately from the tokenization of the multi-message chat history.
"""

import re
from typing import Any, Dict, List, Optional


class SerializationError(RuntimeError):
    pass


ASSISTANT_ROLE_MARKERS = ("<|im_start|>", "<|im_end|>")


def safe_assistant_context(raw: str) -> tuple[str, bool]:
    escaped = raw
    for marker in ASSISTANT_ROLE_MARKERS:
        escaped = escaped.replace(marker, marker.replace("<", r"\u003c").replace(">", r"\u003e"))
    return escaped, escaped != raw


def _assistant_regions(content: str, artifact: Optional[dict]) -> List[tuple[int, int, str]]:
    special = []
    if artifact and artifact.get("extraction_status") == "SELECTED":
        start, end = artifact["char_start"], artifact["char_end"]
        if not (0 <= start < end <= len(content)):
            raise SerializationError("invalid code artifact character span")
        special.append((start, end, "assistant_code"))
    for match in re.finditer(r"<think>.*?</think>", content, re.DOTALL):
        special.append((match.start(), match.end(), "assistant_reasoning"))
    boundaries = sorted({0, len(content)} | {x for start, end, _ in special for x in (start, end)})
    regions = []
    for start, end in zip(boundaries, boundaries[1:]):
        if start == end:
            continue
        kind = "assistant_other"
        for left, right, label in special:
            if left <= start and end <= right and label == "assistant_reasoning":
                kind = label
        for left, right, label in special:
            if left <= start and end <= right and label == "assistant_code":
                kind = label
        if regions and regions[-1][1] == start and regions[-1][2] == kind:
            regions[-1] = (regions[-1][0], end, kind)
        else:
            regions.append((start, end, kind))
    return regions


def render_history(tokenizer, messages: List[Dict[str, Any]], *, add_generation_prompt: bool,
                   artifacts_by_turn: Optional[Dict[int, dict]] = None) -> dict:
    plain = [{"role": m["role"], "content": m["content"]} for m in messages]
    if any(m["role"] not in {"system", "user", "assistant", "tool"} for m in plain):
        raise SerializationError("unsupported chat role")
    rendered = tokenizer.apply_chat_template(plain, tokenize=False,
                                              add_generation_prompt=add_generation_prompt)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    template_ids = tokenizer.apply_chat_template(plain, tokenize=True,
                                                  add_generation_prompt=add_generation_prompt)
    if ids != list(template_ids):
        raise SerializationError("chat-template token IDs disagree with offset tokenizer")

    cursor = 0
    segments = []
    for index, message in enumerate(messages):
        content = message["content"]
        if not content:
            raise SerializationError("empty message content has no span")
        base = rendered.find(content, cursor)
        if base < 0:
            raise SerializationError("message content absent from rendered chat template")
        cursor = base + len(content)
        role = message["role"]
        turn_id = message.get("turn_id")
        if role == "assistant":
            artifact = (artifacts_by_turn or {}).get(turn_id)
            regions = _assistant_regions(content, artifact)
        else:
            label = {"system": "system", "user": "user_problem", "tool": "tool_observation"}[role]
            regions = [(0, len(content), label)]
        for local_start, local_end, kind in regions:
            start, end = base + local_start, base + local_end
            touched = [i for i, (left, right) in enumerate(offsets)
                       if right > start and left < end]
            if not touched:
                raise SerializationError("nonempty content has no token overlap")
            first, last = touched[0], touched[-1]
            segments.append({
                "message_index": index, "role": role, "turn_id": turn_id,
                "segment_type": kind, "char_start": start, "char_end": end,
                "token_start": first, "token_end": last + 1,
                "token_boundary_exact": offsets[first][0] == start and offsets[last][1] == end,
                "origin": "student" if role == "assistant" else
                          ("environment" if role == "tool" else "prompt"),
            })
    if any(s["role"] == "tool" and s["origin"] == "student" for s in segments):
        raise SerializationError("tool observation marked as Student output")
    return {"rendered_text": rendered, "token_ids": ids, "segments": segments}
