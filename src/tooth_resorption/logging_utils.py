"""Centralised logging configuration.

Every module in :mod:`tooth_resorption` obtains its logger via
:func:`get_logger` so the whole pipeline writes through one consistent
formatter and respects the ``TRD_LOG_LEVEL`` environment variable
(defaults to ``INFO``).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
_CONFIGURED: bool = False


def _configure_root() -> None:
    """Attach a single stream handler to the root logger (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("TRD_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``.

    Args:
        name: Module ``__name__`` (e.g. ``tooth_resorption.training.train``).

    Returns:
        A :class:`logging.Logger` instance using the project's formatter.
    """
    _configure_root()
    return logging.getLogger(name)


__all__ = ["get_logger"]
