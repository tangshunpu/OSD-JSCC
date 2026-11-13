"""Run the journal OSD-JSCC-P model from the repository root.

Examples:
    uv run python -m osd_jscc_p.inference --input photo.png --snr-db 10 --rates 96
    uv run python -m osd_jscc_p.inference --input datasets/Kodak --snr-db 0 10 20

All selected rates share one progressive transmission per image and SNR.
The default checkpoint contains the trained AWGN model; SD-Turbo supplies the
remaining pretrained weights, as in the conference version.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from inference import LOCAL_SD_TURBO, Metrics, TO_TENSOR, find_images, seed_everything, to_uint8
from .models import MultirateOSDV3

RATES = (768, 384, 192, 96, 64, 48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="An image, or a folder searched recursively for images.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/osd_jscc_p"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/osd_jscc_p_awgn.safetensors"))
    parser.add_argument("--sd-model", help="SD-Turbo local folder or Hugging Face model ID.")
    parser.add_argument("--snr-db", type=float, nargs="+", default=[10.0],
                        help="Channel SNRs in dB; training used 0–20 dB.")
    parser.add_argument("--rates", type=int, nargs="+", choices=RATES, default=list(RATES),
                        help="Compression ratios to reconstruct (default: all six).")
    parser.add_argument("--metrics", nargs="*", default=["psnr"],
                        choices=("psnr", "lpips", "dists"))
    parser.add_argument("--tile-size", type=int, default=2048,
                        help="VAE tile size in pixels; 0 disables tiling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint: Path, device: torch.device, sd_model: str | None) -> MultirateOSDV3:
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != "osd_jscc_p_weights_v1":
        raise ValueError(f"Unsupported OSD-JSCC-P checkpoint: {checkpoint}")
    config = json.loads(metadata["model_config"])
    if sd_model is None:
        sd_model = str(LOCAL_SD_TURBO) if LOCAL_SD_TURBO.is_dir() else "stabilityai/sd-turbo"
    print(f"Loading SD-Turbo U-Net, VAE and scheduler from {sd_model}")
    model = MultirateOSDV3.from_config(
        config, sd_model_id=sd_model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    # Trainable layers run in FP32, matching the journal evaluation and demo.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad or name.startswith("rate_embedding."):
            parameter.data = parameter.data.float()
    result = model.load_state_dict(load_file(checkpoint, device="cpu"), strict=False)
    missing = sorted(model.persistent_keys().intersection(result.missing_keys))
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, "
                           f"unexpected={result.unexpected_keys}")
    return model.to(device).eval()


@torch.inference_mode()
def transmit(model: MultirateOSDV3, image: torch.Tensor, snr_db: float,
             rates: list[int]) -> dict[int, torch.Tensor]:
    height, width = image.shape[-2:]
    pad_h, pad_w = -height % model.downsample_factor, -width % model.downsample_factor
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    with torch.autocast(image.device.type, dtype=torch.float16, enabled=image.device.type == "cuda"):
        outputs = model.progressive(image, snr_db, rates=rates)
    return {rate: output[..., :height, :width] for rate, output in outputs.items()}


def main() -> None:
    args = parse_args()
    if args.tile_size < 0 or args.tile_size % 8:
        raise ValueError("--tile-size must be 0 or a positive multiple of 8.")
    if len(set(args.rates)) != len(args.rates):
        raise ValueError("Each compression ratio may be selected only once.")
    device = torch.device(args.device)
    images = find_images(args.input)
    model = load_model(args.checkpoint, device, args.sd_model)
    if args.tile_size:
        model.vae.enable_tiling()
        model.vae.tile_sample_min_size = args.tile_size
        model.vae.tile_latent_min_size = args.tile_size // 8
    metrics = Metrics(args.metrics, device)
    root = args.input if args.input.is_dir() else args.input.parent
    rows = []
    for snr_db in args.snr_db:
        print(f"SNR {snr_db:g} dB: {len(images)} images")
        for index, path in enumerate(images):
            seed_everything(args.seed + index)
            with Image.open(path) as opened:
                image = TO_TENSOR(opened.convert("RGB")).unsqueeze(0).to(device)
            outputs = transmit(model, image, snr_db, args.rates)
            relative = path.relative_to(root).with_suffix(".png")
            for rate, output in outputs.items():
                destination = args.output_dir / f"snr_{snr_db:g}dB" / f"cr_{rate}" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(to_uint8(output[0]).permute(1, 2, 0).cpu().numpy()).save(destination)
                values = metrics(output, image)
                rows.append({"snr_db": snr_db, "rate": rate, "image": str(relative), **values})
                print(f"  CR {rate}  {relative}  " +
                      "  ".join(f"{key}={value:.4f}" for key, value in values.items()))
        for rate in args.rates:
            current = [row for row in rows if row["snr_db"] == snr_db and row["rate"] == rate]
            means = {name: np.mean([row[name] for row in current]) for name in args.metrics}
            print(f"  CR {rate} mean  " +
                  "  ".join(f"{key}={value:.4f}" for key, value in means.items()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["snr_db", "rate", "image", *args.metrics])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved reconstructions and metrics.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
