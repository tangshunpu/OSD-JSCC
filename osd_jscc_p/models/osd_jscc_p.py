"""Compatibility import for the shared OSD-JSCC-P implementation."""

from models import osd_jscc_p as _implementation
from models.osd_jscc_p import *  # noqa: F403


def __getattr__(name):
    return getattr(_implementation, name)
