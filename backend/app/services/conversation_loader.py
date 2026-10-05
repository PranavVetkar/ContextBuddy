import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseConversationLoader(ABC):
    """Abstract base class for conversation loaders to enable extensible source types."""

    @abstractmethod
    def load(self, source: str) -> str:
        """Loads and returns raw conversation text from the given source."""
        pass


class TextConversationLoader(BaseConversationLoader):
    """Loads raw conversation text directly from strings."""

    def load(self, source: str) -> str:
        if not source or not source.strip():
            raise ValueError("Conversation content cannot be empty.")
        return source.strip()


class ConversationLoaderRegistry:
    """Registry allowing future loaders (file, URL, shared link, etc.) to be registered."""

    def __init__(self):
        self._loaders = {
            "text": TextConversationLoader(),
        }

    def register(self, source_type: str, loader: BaseConversationLoader) -> None:
        self._loaders[source_type.lower()] = loader

    def load(self, source_type: str, source: str) -> str:
        source_type_normalized = (source_type or "text").strip().lower()
        loader = self._loaders.get(source_type_normalized)
        if not loader:
            # Fallback to text loader if unknown
            logger.warning("Unknown source type '%s'. Falling back to 'text' loader.", source_type)
            loader = self._loaders["text"]
        return loader.load(source)


# Global loader registry instance
loader_registry = ConversationLoaderRegistry()


def load_conversation(source_type: str, content: str) -> str:
    """Convenience helper to load raw conversation text using the registry."""
    return loader_registry.load(source_type, content)
