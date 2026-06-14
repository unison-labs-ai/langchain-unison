"""UnisonMemory — LangChain-compatible BaseMemory backed by the Unison Brain recall API.

``langchain_core`` ≥0.3 dropped ``BaseMemory``.  We provide a lightweight ABC
that is drop-in compatible with the classic LangChain memory interface so that
``langchain-unison`` only needs ``langchain-core`` as a dependency.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ._client import UnisonClient


# ---------------------------------------------------------------------------
# Minimal BaseMemory shim (langchain_core ≥0.3 removed it)
# ---------------------------------------------------------------------------

class BaseMemory(ABC):
    """Minimal abstract base replicating the classic LangChain BaseMemory API."""

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]: ...

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...


# ---------------------------------------------------------------------------
# UnisonMemory
# ---------------------------------------------------------------------------

class UnisonMemory(BaseMemory):
    """LangChain memory that recalls context from the Unison Brain.

    On every chain call:
    - ``load_memory_variables`` hits ``GET /v1/brain/context`` and returns
      the ``contextMd`` under ``memory_key`` (empty string when
      ``weakEvidence`` is True).
    - ``save_context`` ships the user + assistant turns to
      ``POST /v1/brain/ingest``.

    Usage::

        from langchain_unison import UnisonMemory
        memory = UnisonMemory()
        # pass to any chain that accepts a `memory=` argument
    """

    def __init__(
        self,
        *,
        memory_key: str = "history",
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        session_id: Optional[str] = None,
        k: int = 5,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._memory_key = memory_key
        self._input_key = input_key
        self._output_key = output_key
        self._session_id = session_id or str(uuid.uuid4())
        self._k = k
        self._client = UnisonClient(token=token, base_url=base_url)

    # ------------------------------------------------------------------
    # BaseMemory contract
    # ------------------------------------------------------------------

    @property
    def memory_key(self) -> str:
        return self._memory_key

    @property
    def memory_variables(self) -> List[str]:
        return [self._memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Recall context from Unison for the current query."""
        question = self._extract_input(inputs)
        if not question:
            return {self._memory_key: ""}

        result = self._client.recall(query=question, k=self._k)
        if result is None or result.weak_evidence:
            return {self._memory_key: ""}
        return {self._memory_key: result.context_md}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Ingest the user + assistant turns into Unison."""
        user_text = self._extract_input(inputs)
        assistant_text = self._extract_output(outputs)
        if not user_text and not assistant_text:
            return

        turns = []
        if user_text:
            turns.append({"role": "user", "content": user_text})
        if assistant_text:
            turns.append({"role": "assistant", "content": assistant_text})

        self._client.ingest_conversation(
            turns=turns,
            source_ref=self._session_id,
        )

    def clear(self) -> None:
        """No-op: Unison history is immutable; local state is stateless."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_input(self, inputs: Dict[str, Any]) -> str:
        if self._input_key:
            return str(inputs.get(self._input_key, ""))
        for key in ("input", "question", "human_input", "query"):
            if key in inputs:
                return str(inputs[key])
        for v in inputs.values():
            if isinstance(v, str):
                return v
        return ""

    def _extract_output(self, outputs: Dict[str, Any]) -> str:
        if self._output_key:
            return str(outputs.get(self._output_key, ""))
        for key in ("output", "response", "answer", "text"):
            if key in outputs:
                return str(outputs[key])
        for v in outputs.values():
            if isinstance(v, str):
                return v
        return ""
