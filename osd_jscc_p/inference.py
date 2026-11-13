"""Compatibility entry point for OSD-JSCC-P inference.

The shared interface is ``python inference.py --model osd-jscc-p``.
"""

from pathlib import Path

import torch

import inference as shared
from models import OSDJSCCP


def parse_args():
    return shared.parse_args(default_model="osd-jscc-p")


def load_model(checkpoint: Path, device: torch.device, sd_model: str | None) -> OSDJSCCP:
    return shared.load_model(checkpoint, device, device.type == "cuda", sd_model,
                             model_type="osd-jscc-p")


def transmit(model: OSDJSCCP, image: torch.Tensor, snr_db: float,
             rates: list[int]) -> dict[int, torch.Tensor]:
    return shared.transmit_progressive(model, image, snr_db, rates)


def main() -> None:
    shared.main(default_model="osd-jscc-p")


if __name__ == "__main__":
    main()
