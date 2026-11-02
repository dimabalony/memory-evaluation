"""Compose explicit host history metadata with memory-owned tables."""

from sqlalchemy import MetaData, Table

from kai_memory.storage import compose_metadata


def metadata(history_table: Table) -> MetaData:
    """Evaluation owns its history adapter; storage never imports Kai models."""
    target = compose_metadata(MetaData())
    history_table.to_metadata(target)
    return target
