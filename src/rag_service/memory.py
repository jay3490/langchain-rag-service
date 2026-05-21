"""Per-session conversation history kept in process memory."""

from __future__ import annotations

import threading
from collections import OrderedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages


class SessionStore:
    """Chat histories keyed by session id.

    Each history keeps at most ``max_messages`` messages, trimmed from the oldest end so
    that it always starts with a user turn. When more than ``max_sessions`` sessions
    exist, the least recently used one is dropped. State lives in this process only.
    """

    def __init__(self, max_messages: int = 10, max_sessions: int = 1000) -> None:
        self.max_messages = max_messages
        self.max_sessions = max_sessions
        self._histories: OrderedDict[str, list[BaseMessage]] = OrderedDict()
        self._lock = threading.Lock()

    def _trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        return trim_messages(
            messages,
            max_tokens=self.max_messages,
            token_counter=len,
            strategy="last",
            start_on="human",
        )

    def get(self, session_id: str) -> list[BaseMessage]:
        """Return a copy of the session's recent messages (empty for unknown sessions)."""
        with self._lock:
            history = self._histories.get(session_id)
            if history is None:
                return []
            self._histories.move_to_end(session_id)
            return list(history)

    def append_turn(self, session_id: str, question: str, answer: str) -> None:
        """Record one user question and the assistant's answer."""
        if self.max_messages == 0:
            return
        with self._lock:
            history = [
                *self._histories.get(session_id, []),
                HumanMessage(content=question),
                AIMessage(content=answer),
            ]
            self._histories[session_id] = self._trim(history)
            self._histories.move_to_end(session_id)
            while len(self._histories) > self.max_sessions:
                self._histories.popitem(last=False)

    def clear(self, session_id: str) -> bool:
        """Forget a session; returns whether it existed."""
        with self._lock:
            return self._histories.pop(session_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._histories)
