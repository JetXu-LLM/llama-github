"""Public package surface with heavyweight RAG imports loaded on demand."""

from .logger import configure_logging
from .version import __version__

__all__ = ["GithubRAG", "__version__", "configure_logging"]


def __getattr__(name: str):
    """Preserve ``from llama_github import GithubRAG`` without eager ML imports."""
    if name == "GithubRAG":
        from .github_rag import GithubRAG

        return GithubRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Expose lazy public attributes to introspection tools."""
    return sorted(set(globals()) | set(__all__))
