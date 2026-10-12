"""State for Student-only execution interaction, without training signals."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    problem: str
    messages: List[Dict[str, Any]]
    turn_index: int = 0
    max_turns: int = 3
    current_code: Optional[str] = None
    code_history: List[Dict[str, Any]] = field(default_factory=list)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    terminal: bool = False
    terminal_reason: Optional[str] = None
    context_token_count: int = 0

    def add_assistant(self, turn_id: int, response: str) -> None:
        if self.terminal or turn_id != self.turn_index:
            raise RuntimeError("assistant turn out of order")
        self.messages.append({"role": "assistant", "content": response, "turn_id": turn_id})

    def add_tool(self, turn_id: int, observation_text: str) -> None:
        if self.terminal or turn_id != self.turn_index:
            raise RuntimeError("tool turn out of order")
        if not self.messages or self.messages[-1]["role"] != "assistant":
            raise RuntimeError("tool observation must follow assistant action")
        self.messages.append({"role": "tool", "content": observation_text, "turn_id": turn_id})

    def finish(self, reason: str) -> None:
        self.terminal = True
        self.terminal_reason = reason
