"""
Minimal per-conversation memory. Each session_id gets its own message list.
No global state, no cross-session sharing that's what keeps one
customer's conversation from leaking into another's.
"""

from __future__ import annotations
from dataclasses import dataclass, field

MAX_TURNS_KEPT = 12  # cap history so old context doesn't drag on forever


@dataclass
class Session:
    session_id: str
    messages: list[dict] = field(default_factory=list)

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > MAX_TURNS_KEPT:
            self.messages = self.messages[-MAX_TURNS_KEPT:]

    def history(self) -> list[dict]:
        return list(self.messages)


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]

    def reset(self, session_id: str):
        self._sessions.pop(session_id, None)