"""OSD-JSCC-P: one OSD-JSCC model for six rates, with progressive transmission.

Inference-only copy of ``experiments/mulirate_succ/model.py`` (architecture
``mulirate_succ_v2``); training helpers and ablation variants are removed.

The encoder is rate-agnostic: it produces ``c_max`` real channels once, and a
transmission at compression ratio ``CR`` sends a fixed prefix of them.  Layer
widths are counted in half channels (``CR = 3072 / units``; 16 channels =
32 units = CR 96).  Each layer is power-normalized and sent through the channel
independently, so a receiver that stops after any layer sees exactly what a
rate-``CR`` transmission would produce.  Unsent values are zero-filled before
the decoder stem.  Only the receiver (JSCC decoder, UNet context token,
timestep routing) is conditioned on the received rate, and the decoder outputs
a deterministic latent.
"""

from __future__ import annotations

import torch
from torch import nn

from .osd_jscc import OSDJSCC, _module_dtype

DEFAULT_LAYERS = (4, 4, 8, 16, 16, 16)  # CR 768, 384, 192, 96, 64, 48
REFERENCE_CHANNELS = 16
REFERENCE_RATE = 96
UNIT_CR = 3072
ARCHITECTURE = "mulirate_succ_v2"

# Every parameter a training stage changed; checkpoints store exactly these.
PERSISTENT_PREFIXES = (
    "latent_adapter.", "jscc_encoder.", "jscc_decoder.", "snr_embedding.",
    "rate_embedding.", "snr_context.", "unet.conv_in.", "unet.conv_out.",
    "vae.decoder.", "vae.post_quant_conv.",
)


def layer_boundaries(layers) -> tuple[int, ...]:
    boundaries, total = [], 0
    for width in layers:
        if width <= 0:
            raise ValueError("Every layer must contain a positive unit count.")
        total += width
        if UNIT_CR % total:
            raise ValueError(f"A prefix of {total} units has a non-integer CR.")
        boundaries.append(total)
    if total % 2:
        raise ValueError("The layers must add up to whole channels.")
    return tuple(boundaries)


def unit_mask(start: int, end: int, channels: int, height: int, width: int,
              device=None) -> torch.Tensor:
    """Boolean [channels, height, width] mask of units ``start .. end-1``."""
    rows = torch.arange(height, device=device)[:, None]
    cols = torch.arange(width, device=device)[None, :]
    parity = (rows + cols) % 2
    mask = torch.zeros(channels, height, width, dtype=torch.bool, device=device)
    for unit in range(start, end):
        mask[unit // 2] |= parity == unit % 2
    return mask


class RateEmbedding(nn.Module):
    """Embedding of log2(96 / CR)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, rate: torch.Tensor) -> torch.Tensor:
        x = torch.log2(REFERENCE_RATE / rate.float())[:, None]
        return self.net(x.to(self.net[0].weight.dtype))


class OSDJSCCP(OSDJSCC):
    def __init__(self, unet: nn.Module, vae: nn.Module, scheduler, *,
                 layers=DEFAULT_LAYERS, power_norm: str = "layer",
                 rate_thresholds: dict | None = None, channel: str = "awgn",
                 jscc_width: int = 256, lora_rank: int = 32, semantic_tokens: int = 4,
                 snr_embedding_dim: int = 256, high_snr_threshold: float = 10.0,
                 snr_modulation: str = "adagn") -> None:
        layers = tuple(int(width) for width in layers)
        boundaries = layer_boundaries(layers)
        c_max = boundaries[-1] // 2
        if power_norm != "layer":
            raise ValueError("This release supports power_norm='layer' only.")
        super().__init__(
            unet, vae, scheduler,
            compression_ratio=1536 // c_max, channel=channel,
            jscc_width=jscc_width, lora_rank=lora_rank,
            semantic_tokens=semantic_tokens, snr_embedding_dim=snr_embedding_dim,
            high_snr_threshold=high_snr_threshold, snr_modulation=snr_modulation,
        )
        self.layers = layers
        self.boundaries = boundaries
        self.c_max = c_max
        self.power_norm = power_norm
        # Rates in transmission order (CR descending = payload ascending).
        self.rates = tuple(UNIT_CR // units for units in boundaries)
        self.rate_units = dict(zip(self.rates, boundaries))
        self.jscc_decoder.logvar_head = None
        self.rate_embedding = RateEmbedding(snr_embedding_dim)
        thresholds = rate_thresholds or {}
        self.rate_thresholds = {
            rate: float(thresholds.get(rate, thresholds.get(str(rate), high_snr_threshold)))
            for rate in self.rates
        }

    @classmethod
    def from_config(cls, config: dict, *, sd_model_id: str, torch_dtype=None):
        if config.get("architecture") != ARCHITECTURE:
            raise ValueError(f"Unsupported architecture {config.get('architecture')}.")
        if config.get("rate_conditioning", "receiver_only") != "receiver_only" \
                or not config.get("snr_token", True):
            raise ValueError("Ablation variants are not supported by this release.")
        return cls.from_pretrained(
            sd_model_id=sd_model_id, torch_dtype=torch_dtype,
            layers=tuple(config["layers"]), power_norm=config["power_norm"],
            rate_thresholds=config.get("rate_thresholds"),
            channel=config["channel"], jscc_width=int(config["jscc_width"]),
            lora_rank=int(config["lora_rank"]),
            semantic_tokens=int(config["semantic_tokens"]),
            snr_embedding_dim=int(config["snr_embedding_dim"]),
            high_snr_threshold=float(config.get("high_snr_threshold", 10.0)),
            snr_modulation=str(config.get("snr_modulation", "adagn")),
        )

    def persistent_keys(self) -> set[str]:
        return {name for name in self.state_dict()
                if name.startswith(PERSISTENT_PREFIXES) or "lora_A" in name
                or "lora_B" in name}

    def _check_rate(self, rate: int) -> int:
        rate = int(rate)
        if rate not in self.rate_units:
            raise ValueError(f"CR {rate} is not one of {self.rates}.")
        return rate

    def sent_channels(self, rate: int) -> int:
        return -(-self.rate_units[self._check_rate(rate)] // 2)

    def prefix_mask(self, rate: int, height: int, width: int, device=None) -> torch.Tensor:
        return unit_mask(0, self.rate_units[self._check_rate(rate)],
                         self.sent_channels(rate), height, width, device)

    def symbols(self, rate: int, height: int, width: int) -> int:
        """Complex channel symbols for a latent of the given spatial size."""
        return int(self.prefix_mask(rate, height, width).sum()) // 2

    def _norm_masks(self, rate: int, height: int, width: int, device) -> list[torch.Tensor]:
        units = self.rate_units[self._check_rate(rate)]
        channels = self.sent_channels(rate)
        starts = (0, *self.boundaries)
        return [unit_mask(start, end, channels, height, width, device)
                for start, end in zip(starts, self.boundaries) if end <= units]

    # -------------------------------------------------------------- transmitter
    def encode_source(self, image: torch.Tensor, snr_db):
        """Return (VAE latent, codeword with all c_max channels, SNR tensor)."""
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected image shaped [B,3,H,W].")
        if image.shape[-2] % 128 or image.shape[-1] % 128:
            raise ValueError("Image dimensions must be divisible by 128.")
        vae_latent = self.ground_truth_vae_latent(image)
        adapter_dtype = _module_dtype(self.latent_adapter, vae_latent.dtype)
        source = self.latent_adapter(vae_latent.to(dtype=adapter_dtype))
        snr = self._snr(snr_db, image)
        codeword = self.jscc_encoder(source, self.snr_embedding(snr))
        return vae_latent, codeword, snr

    def transmit(self, codeword: torch.Tensor, snr: torch.Tensor, rate: int):
        """Send each layer of rate ``rate`` separately; unsent values are zero."""
        channels = self.sent_channels(rate)
        prefix = codeword[:, :channels]
        received = torch.zeros_like(prefix)
        for mask in self._norm_masks(rate, *prefix.shape[-2:], prefix.device):
            received[:, mask] = self.channel(prefix[:, mask], snr).to(received.dtype)
        return received

    # ----------------------------------------------------------------- receiver
    def receiver_condition(self, snr: torch.Tensor, rate: int) -> torch.Tensor:
        condition = self.snr_embedding(snr)
        rate_tensor = torch.full_like(snr, float(rate))
        return condition + self.rate_embedding(rate_tensor).to(snr.dtype)

    def decode_latent(self, received: torch.Tensor, condition: torch.Tensor):
        """Zero-fill to c_max channels and run the decoder (mean + context)."""
        missing = self.c_max - received.shape[1]
        if missing:
            received = torch.cat(
                (received, received.new_zeros(
                    received.shape[0], missing, *received.shape[2:])), dim=1)
        dec = self.jscc_decoder
        x = dec.stem(received)
        for block in dec.low_attention_blocks:
            x = block(x, condition)
        for block in dec.low_residual_blocks:
            x = block(x, condition)
        context = dec.context_head(x.mean(dim=(-2, -1))).reshape(
            x.shape[0], dec.semantic_tokens, dec.context_dim)
        x = dec.upsample(x)
        for block in dec.high_blocks:
            x = block(x, condition)
        return dec.mean_head(x), context

    def route(self, snr: torch.Tensor, rate: int) -> torch.Tensor:
        threshold = self.rate_thresholds[self._check_rate(rate)]
        return torch.where(
            snr >= threshold,
            torch.full_like(snr, 249, dtype=torch.long),
            torch.full_like(snr, 499, dtype=torch.long),
        )

    def receive(self, received: torch.Tensor, snr: torch.Tensor, rate: int) -> torch.Tensor:
        rate = self._check_rate(rate)
        condition = self.receiver_condition(snr, rate)
        mean, context = self.decode_latent(received, condition)
        latent = mean.float().clamp(-8.0, 8.0)
        conditioning = torch.cat((context, self.snr_context(condition)[:, None]), dim=1)
        timestep = self.route(snr, rate)
        refined = torch.empty_like(latent)
        for value in timestep.unique(sorted=True).tolist():
            index = torch.nonzero(timestep == value, as_tuple=True)[0]
            refined[index] = self._refine_group_multistep(
                latent[index], conditioning[index], value, self.inference_steps)
        scale = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        shift = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
        vae_input = (refined / scale + shift).to(
            dtype=_module_dtype(self.vae, refined.dtype))
        return self.vae.decode(vae_input, return_dict=False)[0].clamp(-1, 1)

    @torch.no_grad()
    def progressive(self, image: torch.Tensor, snr_db, rates=None):
        """One transmission of every layer; decode each requested rate's prefix."""
        _, codeword, snr = self.encode_source(image, snr_db)
        received = self.transmit(codeword, snr, self.rates[-1])
        outputs = {}
        for rate in (self.rates if rates is None else rates):
            channels = self.sent_channels(rate)
            mask = self.prefix_mask(rate, *received.shape[-2:], received.device)
            outputs[rate] = self.receive(received[:, :channels] * mask, snr, rate)
        return outputs


# Backward-compatible name used by earlier releases.
MultirateOSDV3 = OSDJSCCP
