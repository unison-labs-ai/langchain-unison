"""UnisonChatMessageHistory — BaseChatMessageHistory backed by the Unison Brain."""
from __future__ import annotations

import uuid
from typing import List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_to_dict

from ._client import UnisonClient


class UnisonChatMessageHistory(BaseChatMessageHistory):
    """Stores chat messages in an in-memory buffer and ingests them to Unison.

    Each call to :meth:`add_message` immediately ships the turn to
    ``POST /v1/brain/ingest`` so the conversation is searchable in the brain.
    The ``messages`` property returns the local in-memory list — Unison is an
    *append* sink, not a remote source of truth for replay.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        *,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        visibility: str = "private",
    ) -> None:
        self.session_id: str = session_id or str(uuid.uuid4())
        self.visibility = visibility
        self._client = UnisonClient(token=token, base_url=base_url)
        self._messages: List[BaseMessage] = []

    # ------------------------------------------------------------------
    # BaseChatMessageHistory contract
    # ------------------------------------------------------------------

    @property
    def messages(self) -> List[BaseMessage]:
        return list(self._messages)

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        # Fire-and-forget ingest to Unison
        role = self._role_for(message)
        turn = {"role": role, "content": str(message.content)}
        self._client.ingest_conversation(
            turns=[turn],
            source_ref=self.session_id,
            visibility=self.visibility,
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        for msg in messages:
            self.add_message(msg)

    def clear(self) -> None:
        """Clear the local buffer only; Unison history is immutable."""
        self._messages.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _role_for(message: BaseMessage) -> str:
        type_name = message.type
        if type_name == "human":
            return "user"
        if type_name == "ai":
            return "assistant"
        return "system"
