"""Compatibility imports for the former OSDV3 module name."""

from . import osd_jscc as _implementation
from .osd_jscc import OSDJSCC, OSDJSCCOutput, OSDV3, OSDV3Output


def __getattr__(name):
    return getattr(_implementation, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_implementation)))


__all__ = ["OSDJSCC", "OSDJSCCOutput", "OSDV3", "OSDV3Output"]
