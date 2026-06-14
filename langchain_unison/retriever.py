"""UnisonRetriever — BaseRetriever backed by the Unison Brain search API."""
from __future__ import annotations

from typing import List, Optional

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ._client import UnisonClient


class UnisonRetriever(BaseRetriever):
    """Retrieve documents from the Unison Brain via ``GET /v1/brain/search``.

    Each hit is mapped to a LangChain :class:`Document` where:
    - ``page_content`` = the ``highlight`` snippet (or ``bodyMd`` if available)
    - ``metadata``     = ``{"path": ..., "title": ..., "score": ...}``

    Usage::

        from langchain_unison import UnisonRetriever
        retriever = UnisonRetriever(k=6)
        docs = retriever.invoke("What did we decide about pricing?")
    """

    k: int = 5

    # Private — not part of the Pydantic schema
    _client: Optional[UnisonClient] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        *,
        k: int = 5,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(k=k)
        object.__setattr__(self, "_client", UnisonClient(token=token, base_url=base_url))

    # ------------------------------------------------------------------
    # BaseRetriever contract
    # ------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        client: UnisonClient = object.__getattribute__(self, "_client")
        hits = client.search(query=query, k=self.k)
        docs: List[Document] = []
        for hit in hits:
            content = hit.body_md or hit.highlight or ""
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "path": hit.path,
                        "title": hit.title,
                        "score": hit.score,
                    },
                )
            )
        return docs
