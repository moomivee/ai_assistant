"""Persistent, versioned documents and local ingestion for the knowledge base."""

from .store import KnowledgeStore, get_store

__all__ = ["KnowledgeStore", "get_store"]
