# AGENTS.md

Guidance for AI agents. This file covers two jobs — jump to yours:

- **Use langchain-unison** — you're an agent helping someone wire Unison memory into a LangChain application
- **Contribute to this repo** — you're changing this integration's code

Follows the [AGENTS.md](https://agents.md/) convention. Human contributors: see [`CONTRIBUTING.md`](./CONTRIBUTING.md).

---

## Use langchain-unison

### What it does

`langchain-unison` gives LangChain applications persistent memory backed by the
Unison brain. Three classes, one pattern — **recall before reasoning, persist after**.

| Class | LangChain base | What it does |
|---|---|---|
| `UnisonMemory` | `BaseMemory` | Recalls relevant context via `/v1/brain/context` before each chain call; saves the new turn after |
| `UnisonChatMessageHistory` | `BaseChatMessageHistory` | Appends every message to the brain in real time, scoped by `session_id` |
| `UnisonRetriever` | `BaseRetriever` | Queries `/v1/brain/search` and returns LangChain `Document` objects |

### Install

```bash
pip install langchain-unison
```

### Authenticate

Set the `UNISON_TOKEN` environment variable before running your application:

```bash
export UNISON_TOKEN="usk_live_..."
```

**Provision an account (headless / CI):**

```bash
curl -X POST https://brain.unisonlabs.ai/v1/auth/provision \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com"}'
# Returns: {"apiKey":"usk_live_...","workspaceId":"..."}
export UNISON_TOKEN="usk_live_..."
```

**Override the API base URL** (e.g. for a self-hosted brain):

```bash
export UNISON_API_URL="http://localhost:4001"
export UNISON_TOKEN="usk_live_..."
```

You can also pass `token=` and `base_url=` directly to any class constructor.

### The recall-before / persist-after pattern

The recommended usage loop for any class:

1. **Before the LLM call** — load context from the brain (what's relevant to this prompt).
2. **After the LLM call** — save the new exchange back so the brain stays current.

Both steps are automatic when you use `UnisonMemory`. With `UnisonChatMessageHistory`
and `UnisonRetriever`, you control the loop explicitly.

### UnisonMemory — drop-in chain memory

```python
from langchain_unison import UnisonMemory
from langchain_core.prompts import PromptTemplate

memory = UnisonMemory()  # reads UNISON_TOKEN from env

template = PromptTemplate.from_template(
    "Context from your knowledge base:\n{history}\n\nQuestion: {input}\nAnswer:"
)

# 1. Recall before the chain call
inputs = {"input": "What pricing model did we decide on?"}
inputs.update(memory.load_memory_variables(inputs))

# ... run your LLM chain with `inputs` ...
outputs = {"output": "We went with a freemium tier at $49/mo."}

# 2. Persist after
memory.save_context(inputs, outputs)
```

### UnisonChatMessageHistory — session-scoped message log

```python
from langchain_unison import UnisonChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

history = UnisonChatMessageHistory(session_id="my-session-001")

history.add_message(HumanMessage(content="What is the refund policy?"))
history.add_message(AIMessage(content="Refunds are available within 30 days."))

print(history.messages)
# [HumanMessage(content='What is the refund policy?'),
#  AIMessage(content='Refunds are available within 30 days.')]
```

### UnisonRetriever — semantic search as a retriever

```python
from langchain_unison import UnisonRetriever

retriever = UnisonRetriever(k=6)

docs = retriever.invoke("What did we decide about the enterprise tier?")
for doc in docs:
    print(doc.metadata["title"], "—", doc.page_content[:120])
```

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `UNISON_TOKEN` | *(required)* | Bearer token from [unisonlabs.ai](https://unisonlabs.ai) |
| `UNISON_API_URL` | `https://brain.unisonlabs.ai` | Override for self-hosted or staging deployments |

---

## Contributing to this repo

Single-package Python project. Source in `langchain_unison/`, tests in `tests/`.

### Build, test

```bash
pip install -e .[dev]   # install package + dev deps
pytest                  # run all tests
```

Or, if the package doesn't define a `[dev]` extra:

```bash
pip install -e . pytest httpx langchain-core
pytest -q
```

CI runs `pytest -q`. All tests must pass before merging.

### Key conventions

- No additional runtime dependencies beyond `httpx` and `langchain-core`. Keep the install footprint minimal.
- All three classes must be tolerant: if the brain is unreachable or the token is missing, degrade gracefully — never crash a LangChain chain.
- The client enforces nothing. The Unison backend is the only security boundary. Do not add client-side scope checks or path allow-lists.

### PRs

One logical change per PR. Add or update a test for every new behavior. Run
`pytest -q` before pushing. Security issues: see [`SECURITY.md`](./SECURITY.md).
