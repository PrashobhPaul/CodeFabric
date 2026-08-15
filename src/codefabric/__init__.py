"""CodeFabric: structure-aware RAG knowledge fabric for codebases."""
from .indexer import Indexer, IndexConfig
from .search import SearchEngine

__version__ = "0.1.0"
__all__ = ["Indexer", "IndexConfig", "SearchEngine", "__version__"]
