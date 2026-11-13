"""Transmit images over a simulated AWGN channel with OSD-JSCC or OSD-JSCC-P.

Examples:
    uv run python inference.py --input datasets/Kodak --snr-db 0 10 20
    uv run python inference.py --input my_photos/ --snr-db 5 --output-dir outputs/mine
    uv run python inference.py --model osd-jscc-p --input photo.png --rates 96 48
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
import torch
import torch.nn.functional as F
from torchvision import transforms

from models import OSDJSCC, OSDJSCCP

LOCAL_SD_TURBO = Path("checkpoints/sd-turbo")
MODEL_NAMES = ("osd-jscc", "osd-jscc-p")
RATES = (768, 384, 192, 96, 64, 48)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TO_TENSOR = transforms.Compose(
    (transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
)


def parse_args(default_model: str = "osd-jscc") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="An image, or a folder searched recursively for images.")
    parser.add_argument("--model", choices=MODEL_NAMES, default=default_model,
                        help="Model to use (default: %(default)s).")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rates", type=int, nargs="+", choices=RATES,
                        help="OSD-JSCC-P compression ratios (default: all six).")
    parser.add_argument("--sd-model",
                        help="SD-Turbo as a local folder (with unet/, vae/, scheduler/) or a "
                             "Hugging Face id. Default: checkpoints/sd-turbo if it exists, "
                             "otherwise stabilityai/sd-turbo from the Hub cache.")
    parser.add_argument("--snr-db", type=float, nargs="+", default=[10.0],
                        help="Channel SNRs in dB; the model was trained on 0-20 dB.")
    parser.add_argument("--metrics", nargs="*", default=["psnr"],
                        choices=("psnr", "lpips", "dists"))
    parser.add_argument("--tile-size", type=int, default=2048,
                        help="VAE tile size in pixels for large images. A 3072x2048 image "
                             "needs about 15 GiB of GPU memory at 2048 and 4 GiB at 768. "
                             "Images whose sides are all within it run in one piece. "
                             "0 disables tiling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.model == "osd-jscc" and args.rates is not None:
        parser.error("--rates requires --model osd-jscc-p")
    if args.rates and len(set(args.rates)) != len(args.rates):
        parser.error("Each compression ratio may be selected only once")
    if args.checkpoint is None:
        args.checkpoint = Path("checkpoints/osd_jscc_p_awgn.safetensors" if args.model == "osd-jscc-p"
                               else "checkpoints/osd_jscc_cr96_awgn.pt")
    if args.output_dir is None:
        args.output_dir = Path("outputs/osd_jscc_p" if args.model == "osd-jscc-p" else "outputs")
    return args


def load_model(
    checkpoint: Path, device: torch.device, amp: bool, sd_model: str | None = None,
    *, model_type: str = "osd-jscc",
) -> OSDJSCC | OSDJSCCP:
    if model_type == "osd-jscc-p":
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if metadata.get("format") != "osd_jscc_p_weights_v1":
            raise ValueError(f"Unsupported OSD-JSCC-P checkpoint: {checkpoint}")
        config = json.loads(metadata["model_config"])
        if sd_model is None:
            sd_model = str(LOCAL_SD_TURBO) if LOCAL_SD_TURBO.is_dir() else "stabilityai/sd-turbo"
        print(f"Loading SD-Turbo U-Net, VAE and scheduler from {sd_model}")
        model = OSDJSCCP.from_config(
            config, sd_model_id=sd_model, torch_dtype=torch.float16 if amp else torch.float32)
        # Keep trained modules in FP32, matching OSD-JSCC-P evaluation.
        for name, parameter in model.named_parameters():
            if parameter.requires_grad or name.startswith("rate_embedding."):
                parameter.data = parameter.data.float()
        result = model.load_state_dict(load_file(checkpoint, device="cpu"), strict=False)
        missing = sorted(model.persistent_keys().intersection(result.missing_keys))
        if missing or result.unexpected_keys:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, "
                               f"unexpected={result.unexpected_keys}")
        return model.to(device).eval()
    if model_type != "osd-jscc":
        raise ValueError(f"Unknown model: {model_type}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["model_config"]
    if sd_model is None:
        sd_model = (str(LOCAL_SD_TURBO) if LOCAL_SD_TURBO.is_dir()
                    else payload["train_config"]["sd_model_id"])
    print(f"Loading SD-Turbo U-Net, VAE and scheduler from {sd_model}")
    # SD-Turbo provides the frozen modules; the checkpoint then overwrites the
    # trained parts (U-Net LoRA, conv_in/conv_out, VAE decoder, JSCC modules).
    model = OSDJSCC.from_pretrained(
        sd_model_id=sd_model,
        torch_dtype=torch.float16 if amp else torch.float32,
        compression_ratio=config["compression_ratio"],
        channel=config["channel"],
        jscc_width=config["jscc_width"],
        lora_rank=config["lora_rank"],
        semantic_tokens=config["semantic_tokens"],
        snr_embedding_dim=config["snr_embedding_dim"],
        high_snr_threshold=config["high_snr_threshold"],
    ).to(device)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    missing = trainable.intersection(incompatible.missing_keys)
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model.eval()


def find_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    images = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images found in {path}")
    return images


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_image(model: OSDJSCC, image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Pad an image to the model's spatial alignment and return its original size."""
    height, width = image.shape[-2:]
    pad_h, pad_w = -height % model.downsample_factor, -width % model.downsample_factor
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    return image, height, width


@torch.inference_mode()
def transmit(model: OSDJSCC, image: torch.Tensor, snr_db: float, amp: bool) -> torch.Tensor:
    """Send one [1,3,H,W] image in [-1, 1] through OSD-JSCC."""
    image, height, width = pad_image(model, image)
    with torch.autocast(image.device.type, dtype=torch.float16, enabled=amp):
        reconstruction = model(image, snr_db)
    return reconstruction[..., :height, :width]


@torch.inference_mode()
def transmit_progressive(model: OSDJSCCP, image: torch.Tensor, snr_db: float,
                         rates: list[int] | None = None) -> dict[int, torch.Tensor]:
    """Reconstruct selected prefixes of one progressive transmission."""
    image, height, width = pad_image(model, image)
    with torch.autocast(image.device.type, dtype=torch.float16, enabled=image.device.type == "cuda"):
        outputs = model.progressive(image, snr_db, rates=rates)
    return {rate: output[..., :height, :width] for rate, output in outputs.items()}


def to_uint8(image: torch.Tensor) -> torch.Tensor:
    return image.float().clamp(-1, 1).add(1).mul(127.5).round().byte()


class Metrics:
    """Full-reference metrics on uint8-quantized images, as saved to PNG."""

    def __init__(self, names: list[str], device: torch.device) -> None:
        self.names = names
        self.models = {}
        if any(name != "psnr" for name in names):
            import pyiqa

            self.models = {
                name: pyiqa.create_metric(name, device=device).eval()
                for name in names if name != "psnr"
            }

    @torch.inference_mode()
    def __call__(self, output: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
        output = to_uint8(output).float().div(255)
        reference = to_uint8(reference).float().div(255)
        values = {}
        for name in self.names:
            if name == "psnr":
                mse = (output - reference).square().mean().item()
                values[name] = math.inf if mse == 0 else 10 * math.log10(1 / mse)
            else:
                values[name] = self.models[name](output, reference).mean().item()
        return values


def main(default_model: str = "osd-jscc") -> None:
    args = parse_args(default_model)
    if args.tile_size < 0 or args.tile_size % 8:
        raise ValueError("--tile-size must be 0 or a positive multiple of 8.")
    device = torch.device(args.device)
    amp = device.type == "cuda"
    images = find_images(args.input)
    model = load_model(args.checkpoint, device, amp, args.sd_model, model_type=args.model)
    if args.tile_size:
        # Only the VAE is tiled (overlapping tiles, blended by diffusers); the JSCC
        # codeword and the U-Net step always cover the whole image.
        model.vae.enable_tiling()
        model.vae.tile_sample_min_size = args.tile_size
        model.vae.tile_latent_min_size = args.tile_size // 8
    metrics = Metrics(args.metrics, device)
    root = args.input if args.input.is_dir() else args.input.parent

    progressive = args.model == "osd-jscc-p"
    rates = args.rates or (list(model.rates) if progressive else [])
    rows = []
    for snr_db in args.snr_db:
        print(f"SNR {snr_db:g} dB: {len(images)} images")
        for index, path in enumerate(images):
            # Seeding per image keeps the channel noise matched across SNRs.
            seed_everything(args.seed + index)
            with Image.open(path) as opened:
                image = TO_TENSOR(opened.convert("RGB")).unsqueeze(0).to(device)
            relative = path.relative_to(root).with_suffix(".png")
            outputs = (transmit_progressive(model, image, snr_db, rates) if progressive
                       else {None: transmit(model, image, snr_db, amp)})
            for rate, output in outputs.items():
                directory = args.output_dir / f"snr_{snr_db:g}dB"
                if rate is not None:
                    directory /= f"cr_{rate}"
                destination = directory / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(to_uint8(output[0]).permute(1, 2, 0).cpu().numpy()).save(destination)
                values = metrics(output, image)
                row = {"snr_db": snr_db, "image": str(relative), **values}
                if rate is not None:
                    row["rate"] = rate
                rows.append(row)
                label = f"CR {rate}  " if rate is not None else ""
                print(f"  {label}{relative}  " + "  ".join(f"{k}={v:.4f}" for k, v in values.items()))
        for rate in rates if progressive else [None]:
            current = [row for row in rows if row["snr_db"] == snr_db
                       and (rate is None or row["rate"] == rate)]
            means = {name: np.mean([row[name] for row in current]) for name in args.metrics}
            label = f"CR {rate} " if rate is not None else ""
            print(f"  {label}mean  " + "  ".join(f"{k}={v:.4f}" for k, v in means.items()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["snr_db", *(["rate"] if progressive else []),
                                                     "image", *args.metrics])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved reconstructions and metrics.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
