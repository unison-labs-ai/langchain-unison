"""Smoke tests for langchain-unison — no network required (httpx monkeypatched)."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to build fake httpx responses
# ---------------------------------------------------------------------------

def _fake_response(status_code: int, body: Any) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

INGEST_RESPONSE = {"items": [{"type": "conversation", "jobId": "job-abc-123"}]}

# /v1/brain/context hits are flat with a `snippet` field (search hits nest under `doc`).
CONTEXT_RESPONSE = {
    "weakEvidence": False,
    "hits": [
        {
            "path": "/private/kb/decisions/pricing.md",
            "title": "Pricing decision",
            "score": 0.92,
            "snippet": "We decided on a freemium model with a $49/mo pro tier.",
        }
    ],
    "contextMd": "## Pricing\nWe decided on a freemium model with a $49/mo pro tier.",
}

SEARCH_RESPONSE = {
    "results": [
        {
            "doc": {
                "path": "/private/kb/decisions/pricing.md",
                "title": "Pricing decision",
                "bodyMd": "Full body of the pricing document.",
            },
            "score": 0.88,
            "highlight": "freemium with $49/mo",
        }
    ]
}

WEAK_CONTEXT_RESPONSE = {
    "weakEvidence": True,
    "hits": [],
    "contextMd": "",
}


# ---------------------------------------------------------------------------
# UnisonMemory tests
# ---------------------------------------------------------------------------

class TestUnisonMemory:
    def test_save_context_posts_to_ingest(self):
        """save_context must POST to /v1/brain/ingest with user+assistant turns."""
        from langchain_unison.memory import UnisonMemory

        memory = UnisonMemory(token="usk_live_test", base_url="http://localhost:9999")

        with patch.object(memory._client._client, "post", return_value=_fake_response(200, INGEST_RESPONSE)) as mock_post:
            memory.save_context(
                inputs={"input": "Hello Unison"},
                outputs={"output": "Hello back"},
            )
            mock_post.assert_called_once()
            url = mock_post.call_args[0][0]
            assert "/v1/brain/ingest" in url

            payload = mock_post.call_args[1]["json"]
            items = payload["items"]
            assert len(items) == 1
            turns = items[0]["turns"]
            assert any(t["role"] == "user" for t in turns)
            assert any(t["role"] == "assistant" for t in turns)

    def test_load_memory_variables_gets_context(self):
        """load_memory_variables must GET /v1/brain/context and return contextMd."""
        from langchain_unison.memory import UnisonMemory

        memory = UnisonMemory(token="usk_live_test", base_url="http://localhost:9999")

        with patch.object(memory._client._client, "get", return_value=_fake_response(200, CONTEXT_RESPONSE)) as mock_get:
            result = memory.load_memory_variables({"input": "What pricing model?"})
            mock_get.assert_called_once()
            url = mock_get.call_args[0][0]
            assert "/v1/brain/context" in url
            assert "q=" in url

        assert "history" in result
        assert "freemium" in result["history"]

    def test_load_memory_variables_weak_evidence_returns_empty(self):
        """When weakEvidence=True, load_memory_variables returns empty string."""
        from langchain_unison.memory import UnisonMemory

        memory = UnisonMemory(token="usk_live_test", base_url="http://localhost:9999")

        with patch.object(memory._client._client, "get", return_value=_fake_response(200, WEAK_CONTEXT_RESPONSE)):
            result = memory.load_memory_variables({"input": "Unknown topic"})

        assert result["history"] == ""

    def test_memory_variables_property(self):
        from langchain_unison.memory import UnisonMemory

        memory = UnisonMemory(token="t", base_url="http://localhost:9999")
        assert memory.memory_variables == ["history"]

    def test_custom_memory_key(self):
        from langchain_unison.memory import UnisonMemory

        memory = UnisonMemory(memory_key="brain_context", token="t", base_url="http://localhost:9999")
        assert memory.memory_variables == ["brain_context"]


# ---------------------------------------------------------------------------
# UnisonChatMessageHistory tests
# ---------------------------------------------------------------------------

class TestUnisonChatMessageHistory:
    def test_add_message_posts_to_ingest(self):
        """add_message must POST to /v1/brain/ingest."""
        from langchain_core.messages import HumanMessage
        from langchain_unison.chat_history import UnisonChatMessageHistory

        history = UnisonChatMessageHistory(
            session_id="test-session",
            token="usk_live_test",
            base_url="http://localhost:9999",
        )

        with patch.object(history._client._client, "post", return_value=_fake_response(200, INGEST_RESPONSE)) as mock_post:
            history.add_message(HumanMessage(content="Hello"))
            mock_post.assert_called_once()
            url = mock_post.call_args[0][0]
            assert "/v1/brain/ingest" in url

            payload = mock_post.call_args[1]["json"]
            turns = payload["items"][0]["turns"]
            assert turns[0]["role"] == "user"
            assert turns[0]["content"] == "Hello"

    def test_messages_returns_local_buffer(self):
        """messages property returns in-memory list, not network state."""
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_unison.chat_history import UnisonChatMessageHistory

        history = UnisonChatMessageHistory(
            session_id="test-session",
            token="t",
            base_url="http://localhost:9999",
        )

        with patch.object(history._client._client, "post", return_value=_fake_response(200, INGEST_RESPONSE)):
            history.add_message(HumanMessage(content="First"))
            history.add_message(AIMessage(content="Reply"))

        assert len(history.messages) == 2
        assert history.messages[0].content == "First"
        assert history.messages[1].content == "Reply"

    def test_clear_empties_buffer(self):
        from langchain_core.messages import HumanMessage
        from langchain_unison.chat_history import UnisonChatMessageHistory

        history = UnisonChatMessageHistory(token="t", base_url="http://localhost:9999")

        with patch.object(history._client._client, "post", return_value=_fake_response(200, INGEST_RESPONSE)):
            history.add_message(HumanMessage(content="msg"))

        assert len(history.messages) == 1
        history.clear()
        assert len(history.messages) == 0

    def test_ai_message_role_is_assistant(self):
        from langchain_core.messages import AIMessage
        from langchain_unison.chat_history import UnisonChatMessageHistory

        history = UnisonChatMessageHistory(token="t", base_url="http://localhost:9999")

        with patch.object(history._client._client, "post", return_value=_fake_response(200, INGEST_RESPONSE)) as mock_post:
            history.add_message(AIMessage(content="I am the assistant"))
            payload = mock_post.call_args[1]["json"]
            turns = payload["items"][0]["turns"]
            assert turns[0]["role"] == "assistant"


# ---------------------------------------------------------------------------
# UnisonRetriever tests
# ---------------------------------------------------------------------------

class TestUnisonRetriever:
    def test_get_relevant_documents_calls_search(self):
        """_get_relevant_documents must GET /v1/brain/search and return Documents."""
        from langchain_unison.retriever import UnisonRetriever

        retriever = UnisonRetriever(k=3, token="usk_live_test", base_url="http://localhost:9999")
        client = object.__getattribute__(retriever, "_client")

        with patch.object(client._client, "get", return_value=_fake_response(200, SEARCH_RESPONSE)) as mock_get:
            docs = retriever.invoke("What is the pricing decision?")
            mock_get.assert_called_once()
            url = mock_get.call_args[0][0]
            assert "/v1/brain/search" in url
            assert "q=" in url

        assert len(docs) == 1
        doc = docs[0]
        # bodyMd takes priority over highlight when present
        assert doc.page_content == "Full body of the pricing document."
        assert doc.metadata["path"] == "/private/kb/decisions/pricing.md"
        assert doc.metadata["title"] == "Pricing decision"
        assert abs(doc.metadata["score"] - 0.88) < 1e-6

    def test_get_relevant_documents_falls_back_to_highlight(self):
        """When bodyMd is absent, page_content is the highlight snippet."""
        from langchain_unison.retriever import UnisonRetriever

        no_body_response = {
            "results": [
                {
                    "doc": {"path": "/p", "title": "T"},
                    "score": 0.5,
                    "highlight": "just the highlight",
                }
            ]
        }
        retriever = UnisonRetriever(k=1, token="t", base_url="http://localhost:9999")
        client = object.__getattribute__(retriever, "_client")

        with patch.object(client._client, "get", return_value=_fake_response(200, no_body_response)):
            docs = retriever.invoke("anything")

        assert docs[0].page_content == "just the highlight"

    def test_network_error_returns_empty_list(self):
        """A network failure must not propagate — returns empty list."""
        from langchain_unison.retriever import UnisonRetriever

        retriever = UnisonRetriever(token="t", base_url="http://localhost:9999")
        client = object.__getattribute__(retriever, "_client")

        with patch.object(client._client, "get", side_effect=Exception("network down")):
            docs = retriever.invoke("query")

        assert docs == []
