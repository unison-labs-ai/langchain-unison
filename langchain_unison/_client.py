"""Low-level HTTP client for the Unison Brain API."""
from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

_DEFAULT_BASE_URL = "https://brain.unisonlabs.ai"
_TIMEOUT = 15.0


class IngestResult:
    __slots__ = ("job_id",)

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id


class Hit:
    __slots__ = ("path", "title", "body_md", "score", "highlight")

    def __init__(
        self,
        path: str,
        title: str,
        score: float,
        highlight: str,
        body_md: Optional[str] = None,
    ) -> None:
        self.path = path
        self.title = title
        self.score = score
        self.highlight = highlight
        self.body_md = body_md


class RecallResult:
    __slots__ = ("context_md", "weak_evidence", "hits")

    def __init__(
        self,
        context_md: str,
        weak_evidence: bool,
        hits: list[Hit],
    ) -> None:
        self.context_md = context_md
        self.weak_evidence = weak_evidence
        self.hits = hits


class UnisonClient:
    """Thin typed wrapper around the Unison Brain HTTP API."""

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = _TIMEOUT,
    ) -> None:
        resolved_token = token or os.environ.get("UNISON_TOKEN", "")
        resolved_base = (base_url or os.environ.get("UNISON_API_URL", _DEFAULT_BASE_URL)).rstrip("/")
        self._base = resolved_base
        self._headers = {
            "Authorization": f"Bearer {resolved_token}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_conversation(
        self,
        turns: list[dict[str, str]],
        source_ref: str,
        visibility: str = "private",
    ) -> Optional[IngestResult]:
        """POST /v1/brain/ingest with a conversation item."""
        payload: dict[str, Any] = {
            "items": [
                {
                    "type": "conversation",
                    "turns": turns,
                    "sourceRef": source_ref,
                    "visibility": visibility,
                }
            ]
        }
        try:
            resp = self._client.post(
                f"{self._base}/v1/brain/ingest",
                json=payload,
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()
            items = data.get("items", [])
            if items:
                return IngestResult(job_id=items[0].get("jobId", ""))
            return IngestResult(job_id="")
        except Exception:
            return None

    def recall(self, query: str, k: int = 5, mode: str = "auto") -> Optional[RecallResult]:
        """GET /v1/brain/context — hybrid recall."""
        params = urlencode({"q": query, "k": k, "mode": mode})
        try:
            resp = self._client.get(
                f"{self._base}/v1/brain/context?{params}",
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()
            # /v1/brain/context hits are flat (path/title/snippet at top level),
            # unlike /v1/brain/search hits which nest under `doc` with `highlight`.
            hits = [
                Hit(
                    path=h.get("path", ""),
                    title=h.get("title", ""),
                    score=float(h.get("score", 0.0)),
                    highlight=h.get("snippet", ""),
                    body_md=None,
                )
                for h in data.get("hits", [])
            ]
            return RecallResult(
                context_md=data.get("contextMd", ""),
                weak_evidence=bool(data.get("weakEvidence", False)),
                hits=hits,
            )
        except Exception:
            return None

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """GET /v1/brain/search — keyword/semantic search returning hits."""
        params = urlencode({"q": query, "k": k})
        try:
            resp = self._client.get(
                f"{self._base}/v1/brain/search?{params}",
                headers=self._headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                Hit(
                    path=r.get("doc", {}).get("path", ""),
                    title=r.get("doc", {}).get("title", ""),
                    score=float(r.get("score", 0.0)),
                    highlight=r.get("highlight", ""),
                    body_md=r.get("doc", {}).get("bodyMd"),
                )
                for r in data.get("results", [])
            ]
        except Exception:
            return []

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "UnisonClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
