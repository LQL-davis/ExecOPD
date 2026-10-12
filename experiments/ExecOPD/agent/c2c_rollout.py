"""Fresh on-policy, two-turn Agent collection using the C2-A protocol and tool."""
from __future__ import annotations

import hashlib

from experiments.ExecOPD.agent.agent_controller import AGENT_SYSTEM, revision_metadata, normalized_code_sha256
from experiments.ExecOPD.agent.agent_state import AgentState
from experiments.ExecOPD.agent.code_artifact import CodeArtifact, select_code_artifact
from experiments.ExecOPD.agent.execution_observation import ExecutionObservation, bounded_text
from experiments.ExecOPD.agent.secure_execution_tool import SecureExecutorFatal
from experiments.ExecOPD.agent.trajectory_serializer import render_history, safe_assistant_context

MAX_TURNS = 2
MAX_NEW_TOKENS = 512
MAX_MODEL_LEN = 2048


def collect_one(tokenizer, generator, execution_tool, task, *, step, sample_index):
    state = AgentState(problem=task['problem'], messages=[
        {'role': 'system', 'content': AGENT_SYSTEM, 'turn_id': None},
        {'role': 'user', 'content': task['problem'], 'turn_id': None}],
        max_turns=MAX_TURNS)
    turns = []
    artifacts = {}
    for turn_id in range(MAX_TURNS):
        state.turn_index = turn_id
        context = render_history(tokenizer, state.messages, add_generation_prompt=True,
                                 artifacts_by_turn=artifacts)
        prompt_ids = context['token_ids']
        if len(prompt_ids) + MAX_NEW_TOKENS > MAX_MODEL_LEN:
            raise RuntimeError('C2_C_RUNTIME_BLOCKED: next turn exceeds fixed 2048-token model context')
        prior_tool = [s for s in context['segments'] if s['role'] == 'tool']
        if turn_id > 0 and (len(prior_tool) != turn_id or state.messages[-1]['role'] != 'tool'):
            raise RuntimeError('C2_C_RUNTIME_BLOCKED: execution observation omitted from next prompt')
        output = generator(prompt_ids)
        sampled_ids = list(output['sampled_ids'])
        if not sampled_ids or len(sampled_ids) > MAX_NEW_TOKENS:
            raise RuntimeError('C2_C_RUNTIME_BLOCKED: sampled token budget invalid')
        raw = tokenizer.decode(sampled_ids, skip_special_tokens=True)
        safe, escaped = safe_assistant_context(raw)
        state.add_assistant(turn_id, safe)
        artifact = select_code_artifact(raw, turn_id)
        if escaped:
            artifact = CodeArtifact(turn_id, 'ACTION_FORMAT_ERROR', None, None, None, 0,
                                    artifact.block_count)
        artifact_data = artifact.to_dict()
        if artifact.extraction_status == 'SELECTED':
            artifact_data['sha256'] = hashlib.sha256(artifact.selected_code.encode()).hexdigest()
            artifact_data['normalized_code_sha256'] = normalized_code_sha256(artifact.selected_code)
            state.current_code = artifact.selected_code
        state.code_history.append(artifact_data)
        artifacts[turn_id] = artifact_data
        turn = {
            'turn_id': turn_id, 'input_context_tokens': len(prompt_ids),
            'max_new_tokens': MAX_NEW_TOKENS,
            'prompt_token_ids': prompt_ids, 'prompt_segments': context['segments'],
            'prior_tool_observation_count': len(prior_tool),
            'conditioned_on_execution_observation': turn_id > 0,
            'raw_student_response_text': raw,
            'decoded_with_special_tokens': tokenizer.decode(sampled_ids, skip_special_tokens=False),
            'raw_student_token_ids': sampled_ids, 'finish_reason': output['finish_reason'],
            'assistant_chat_marker_escaped_for_context': escaped,
            'code_artifact': artifact_data, 'execution_observation': None,
            'revision_from_previous_code': None,
        }
        turns.append(turn)
        if artifact.extraction_status != 'SELECTED':
            state.finish(artifact.extraction_status)
            break
        prior_codes = [x['selected_code'] for x in state.code_history[:-1]
                       if x['extraction_status'] == 'SELECTED']
        if prior_codes:
            turn['revision_from_previous_code'] = revision_metadata(prior_codes[-1], artifact.selected_code)
        try:
            observation = execution_tool.execute(turn_id, artifact.selected_code,
                                                 task['ground_truth'])
        except SecureExecutorFatal as exc:
            message, cut = bounded_text(str(exc))
            observation = ExecutionObservation(
                turn_id=turn_id, status='EXECUTOR_ERROR', exit_code=None, stdout='',
                stderr=message, timed_out=False, duration_ms=0.0, passed_tests=None,
                total_tests=None, executor_identity='secure_docker',
                stdout_truncated=False, stderr_truncated=cut)
        info = observation.to_dict()
        state.execution_history.append(info)
        turn['execution_observation'] = info
        state.add_tool(turn_id, observation.for_student())
        if observation.status == 'EXECUTOR_ERROR':
            state.finish('SECURE_EXECUTOR_ERROR')
            break
        if observation.status == 'PASS':
            state.finish('PASS')
            break
        if turn_id == MAX_TURNS - 1:
            state.finish('MAX_TURNS')
            break
    if not state.terminal:
        raise RuntimeError('C2_C_RUNTIME_BLOCKED: Agent loop without terminal')
    history = render_history(tokenizer, state.messages, add_generation_prompt=False,
                             artifacts_by_turn=artifacts)
    return {
        'global_step': step, 'trajectory_index': sample_index,
        'task_id': task['task_id'], 'original_row_index': task['original_row_index'],
        'state': {
            'problem': state.problem, 'messages': state.messages,
            'turn_index': state.turn_index, 'max_turns': state.max_turns,
            'current_code': state.current_code, 'code_history': state.code_history,
            'execution_history': state.execution_history, 'terminal': state.terminal,
            'terminal_reason': state.terminal_reason,
            'context_token_count': state.context_token_count},
        'turns': turns,
        'final_rendered_history': history['rendered_text'],
        'final_history_token_ids': history['token_ids'],
        'final_history_segments': history['segments'],
    }
