# Contributing to langchain-unison

Thanks for helping improve the LangChain memory integration for Unison.

## Repo layout

A single-package Python project:

- `langchain_unison/_client.py` — low-level HTTP client for the Unison Brain API
- `langchain_unison/memory.py` — `UnisonMemory` (`BaseMemory` implementation)
- `langchain_unison/chat_history.py` — `UnisonChatMessageHistory` (`BaseChatMessageHistory` implementation)
- `langchain_unison/retriever.py` — `UnisonRetriever` (`BaseRetriever` implementation)
- `langchain_unison/__init__.py` — package exports
- `tests/test_smoke.py` — unit tests
- `pyproject.toml` — package metadata and dependencies

## Development

```bash
pip install -e . pytest httpx langchain-core
pytest -q
```

If a `[dev]` extra is defined:

```bash
pip install -e .[dev]
pytest -q
```

## Before opening a PR

1. `pytest -q` must pass.
2. Keep changes scoped — one logical change per PR.
3. Add or update a test for every new behavior.
4. Do not commit `.env` or any real credentials.

## Conventions

- Python 3.9+, typed with `from __future__ import annotations`.
- No additional runtime dependencies beyond `httpx` and `langchain-core` — keep the install footprint minimal.
- All three classes must degrade gracefully: if the brain is unreachable or the token is missing, return empty results or a no-op — never raise in a LangChain chain.
- The client enforces nothing — the Unison backend is the only security boundary. Do not add client-side scope or path checks.

## Reporting bugs / proposing features

Use the issue templates. For security issues, see [`SECURITY.md`](./SECURITY.md) — do **not** open a public issue.
