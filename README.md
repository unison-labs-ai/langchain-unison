# langchain-unison

![CI](https://github.com/unison-labs-ai/langchain-unison/actions/workflows/ci.yml/badge.svg)

LangChain memory, chat history, and retriever backed by the [Unison brain](https://unisonlabs.ai) — a personal knowledge graph that stores and recalls context across every session, tool, and agent.

Powered by the Unison brain.

## Install

```bash
pip install langchain-unison
```

## Quick start

### UnisonMemory

Drop-in memory for any LangChain chain. On every call it recalls relevant context from your brain and ingests the new turn when done.

```python
from langchain_unison import UnisonMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

memory = UnisonMemory()  # reads UNISON_TOKEN from env

template = PromptTemplate.from_template(
    "Context from your knowledge base:\n{history}\n\nQuestion: {input}\nAnswer:"
)

# Load memory before the chain call
inputs = {"input": "What pricing model did we decide on?"}
inputs.update(memory.load_memory_variables(inputs))

# ... run your LLM chain with `inputs` ...
outputs = {"output": "We went with a freemium tier at $49/mo."}

# Save the exchange back to the brain
memory.save_context(inputs, outputs)
```

### UnisonChatMessageHistory

`BaseChatMessageHistory` that appends every message to the Unison brain in real time.

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

### UnisonRetriever

`BaseRetriever` that queries `GET /v1/brain/search` and returns LangChain `Document` objects.

```python
from langchain_unison import UnisonRetriever

retriever = UnisonRetriever(k=6)

docs = retriever.invoke("What did we decide about the enterprise tier?")
for doc in docs:
    print(doc.metadata["title"], "—", doc.page_content[:120])
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `UNISON_TOKEN` | *(required)* | Bearer token from [unisonlabs.ai](https://unisonlabs.ai) |
| `UNISON_API_URL` | `https://brain.unisonlabs.ai` | Override for self-hosted or staging deployments |

You can also pass `token=` and `base_url=` directly to any class constructor.

## Links

- Homepage: https://unisonlabs.ai
- Docs: https://unisonlabs.ai/docs
- Unison brain repo: https://github.com/unison-labs-ai/unison-brain
- Issues: https://github.com/unison-labs-ai/langchain-unison/issues

## Releasing

This package publishes to PyPI via GitHub Actions on any `v*` tag.

**Preferred: PyPI Trusted Publishing (no long-lived secret)**

1. Go to https://pypi.org/manage/account/publishing/ and add a new publisher:
   - PyPI project name: `langchain-unison`
   - GitHub owner: `unison-labs-ai`
   - Repository: `langchain-unison`
   - Workflow: `release.yml`
   - Environment: *(leave blank)*
2. Push a tag: `git tag v0.1.0 && git push origin v0.1.0`

**Fallback: API token**

If Trusted Publishing is not configured, set the repository secret `PYPI_API_TOKEN` and uncomment the `password` field in `.github/workflows/release.yml`.

## License

MIT — see [LICENSE](LICENSE).
