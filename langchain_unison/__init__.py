"""langchain-unison — Unison Brain integration for LangChain.

Exports:
- UnisonChatMessageHistory: BaseChatMessageHistory that syncs to Unison
- UnisonMemory: BaseMemory that recalls context via /v1/brain/context
- UnisonRetriever: BaseRetriever that queries /v1/brain/search
"""
from .chat_history import UnisonChatMessageHistory
from .memory import UnisonMemory
from .retriever import UnisonRetriever

__all__ = [
    "UnisonChatMessageHistory",
    "UnisonMemory",
    "UnisonRetriever",
]

__version__ = "0.1.0"
