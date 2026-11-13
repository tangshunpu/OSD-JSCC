"""Compatibility imports for the former multirate module name."""

from . import osd_jscc_p as _implementation
from .osd_jscc_p import *  # noqa: F403


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))
