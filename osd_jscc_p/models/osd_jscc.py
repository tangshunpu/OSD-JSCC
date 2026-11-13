"""Compatibility import for the shared OSD-JSCC implementation."""

from models import osd_jscc as _implementation
from models.osd_jscc import OSDJSCC, OSDJSCCOutput


def __getattr__(name):
    return getattr(_implementation, name)


__all__ = ["OSDJSCC", "OSDJSCCOutput"]
