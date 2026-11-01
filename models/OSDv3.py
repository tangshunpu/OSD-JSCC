"""OSDV3: conditioned ConvNeXt/window-attention latent JSCC for SD-Turbo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .channel import WirelessChannel


SD_TURBO_MODEL_ID = "stabilityai/sd-turbo"


def _freeze(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    module.eval()
    return module


def _module_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    return next(module.parameters(), torch.empty((), dtype=fallback)).dtype


class SNREmbedding(nn.Module):
    """Continuous sinusoidal SNR embedding."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("SNR embedding dimension must be even.")
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(inplace=True), nn.Linear(dim, dim)
        )

    def forward(self, snr_db: torch.Tensor) -> torch.Tensor:
        snr_db = snr_db.to(dtype=self.mlp[0].weight.dtype)
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=snr_db.device, dtype=snr_db.dtype)
            * (-torch.log(snr_db.new_tensor(10_000.0)) / max(half - 1, 1))
        )
        angles = snr_db[:, None] * frequencies[None]
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=1))


class AdaGN(nn.Module):
    """Group normalization with per-block SNR-dependent affine modulation."""

    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels, affine=False)
        self.affine = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(condition).chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


class UnconditionalGroupNorm(nn.Module):
    """Ordinary GroupNorm with the conditional-call interface used by AdaGN."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels, affine=True)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        del condition
        return self.norm(x)


def _conditional_norm(
    channels: int, condition_dim: int, snr_modulation: str
) -> nn.Module:
    if snr_modulation == "adagn":
        return AdaGN(channels, condition_dim)
    if snr_modulation == "none":
        return UnconditionalGroupNorm(channels)
    raise ValueError("snr_modulation must be 'adagn' or 'none'.")


class ConditionalConvNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_dim: int,
        expansion: int = 4,
        snr_modulation: str = "adagn",
    ) -> None:
        super().__init__()
        self.adagn = _conditional_norm(channels, condition_dim, snr_modulation)
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.pointwise1 = nn.Conv2d(channels, channels * expansion, 1)
        self.pointwise2 = nn.Conv2d(channels * expansion, channels, 1)
        self.act = nn.GELU()
        self.layer_scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.adagn(x, condition)
        residual = self.depthwise(residual)
        residual = self.pointwise2(self.act(self.pointwise1(residual)))
        return x + residual * self.layer_scale[None, :, None, None]


class ConditionalResidualBlock(nn.Module):
    def __init__(
        self, channels: int, condition_dim: int, snr_modulation: str = "adagn"
    ) -> None:
        super().__init__()
        self.norm1 = _conditional_norm(channels, condition_dim, snr_modulation)
        self.norm2 = _conditional_norm(channels, condition_dim, snr_modulation)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.SiLU(inplace=True)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(self.act(self.norm1(x, condition)))
        residual = self.conv2(self.act(self.norm2(residual, condition)))
        return x + residual


class ConditionalWindowAttentionBlock(nn.Module):
    """Non-overlapping local attention with AdaGN conditioning."""

    def __init__(
        self, channels: int, condition_dim: int, window_size: int = 8,
        heads: int = 8, snr_modulation: str = "adagn",
    ) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("Window-attention channels must be divisible by heads.")
        self.window_size = window_size
        self.norm1 = _conditional_norm(channels, condition_dim, snr_modulation)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm2 = _conditional_norm(channels, condition_dim, snr_modulation)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1),
        )
        self.attn_scale = nn.Parameter(torch.tensor(1e-6))
        self.mlp_scale = nn.Parameter(torch.tensor(1e-6))

    def _windows(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        b, c, h, w = x.shape
        ws = self.window_size
        if h % ws or w % ws:
            raise ValueError(f"Feature dimensions {h}x{w} must be divisible by {ws}.")
        windows = x.reshape(b, c, h // ws, ws, w // ws, ws)
        windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)
        return windows, (b, c, h, w)

    def _unwindows(self, windows: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        b, c, h, w = shape
        ws = self.window_size
        x = windows.reshape(b, h // ws, w // ws, ws, ws, c)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        windows, shape = self._windows(self.norm1(x, condition))
        attended = self.attention(windows, windows, windows, need_weights=False)[0]
        x = x + self.attn_scale * self._unwindows(attended, shape)
        return x + self.mlp_scale * self.mlp(self.norm2(x, condition))


class IdentityLatentAdapter(nn.Module):
    """StableCodec-inspired depthwise residual adapter, initialized as identity."""

    def __init__(self, channels: int = 4, width: int = 64) -> None:
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 5, padding=2, groups=width),
            nn.GELU(),
            nn.Conv2d(width, channels, 1),
        )
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return latent + self.branch(latent)


class V3JSCCEncoder(nn.Module):
    def __init__(
        self, latent_channels: int, condition_dim: int, snr_modulation: str = "adagn"
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(4, 192, 3, padding=1)
        self.high_blocks = nn.ModuleList(
            ConditionalConvNeXtBlock(192, condition_dim, snr_modulation=snr_modulation)
            for _ in range(2)
        )
        self.downsample = nn.Conv2d(192, 256, 3, stride=2, padding=1)
        self.low_conv_blocks = nn.ModuleList(
            ConditionalConvNeXtBlock(256, condition_dim, snr_modulation=snr_modulation)
            for _ in range(2)
        )
        self.low_attention_blocks = nn.ModuleList(
            ConditionalWindowAttentionBlock(
                256, condition_dim, snr_modulation=snr_modulation
            )
            for _ in range(2)
        )
        self.symbol_head = nn.Conv2d(256, latent_channels, 1)

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = self.stem(latent)
        for block in self.high_blocks:
            x = block(x, condition)
        x = self.downsample(x)
        for block in self.low_conv_blocks:
            x = block(x, condition)
        for block in self.low_attention_blocks:
            x = block(x, condition)
        return self.symbol_head(x)


class V3JSCCDecoder(nn.Module):
    def __init__(
        self, latent_channels: int, condition_dim: int, context_dim: int,
        semantic_tokens: int, snr_modulation: str = "adagn",
    ) -> None:
        super().__init__()
        self.semantic_tokens = semantic_tokens
        self.context_dim = context_dim
        self.stem = nn.Conv2d(latent_channels, 256, 1)
        self.low_attention_blocks = nn.ModuleList(
            ConditionalWindowAttentionBlock(
                256, condition_dim, snr_modulation=snr_modulation
            )
            for _ in range(2)
        )
        self.low_residual_blocks = nn.ModuleList(
            ConditionalResidualBlock(256, condition_dim, snr_modulation)
            for _ in range(2)
        )
        self.context_head = (
            nn.Sequential(
                nn.Linear(256, 512), nn.SiLU(inplace=True),
                nn.Linear(512, semantic_tokens * context_dim),
            )
            if semantic_tokens
            else None
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(256, 192 * 4, 3, padding=1), nn.PixelShuffle(2)
        )
        self.high_blocks = nn.ModuleList(
            ConditionalResidualBlock(192, condition_dim, snr_modulation)
            for _ in range(2)
        )
        self.mean_head = nn.Conv2d(192, 4, 3, padding=1)
        self.logvar_head = nn.Conv2d(192, 4, 3, padding=1)

    def forward(
        self, received: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(received)
        for block in self.low_attention_blocks:
            x = block(x, condition)
        for block in self.low_residual_blocks:
            x = block(x, condition)
        context = (
            self.context_head(x.mean(dim=(-2, -1))).reshape(
                x.shape[0], self.semantic_tokens, self.context_dim
            )
            if self.context_head is not None
            else x.new_empty((x.shape[0], 0, self.context_dim))
        )
        x = self.upsample(x)
        for block in self.high_blocks:
            x = block(x, condition)
        return self.mean_head(x), self.logvar_head(x), context


@dataclass
class OSDV3Output:
    reconstruction: torch.Tensor
    y: torch.Tensor
    refined_latent: torch.Tensor
    codeword: torch.Tensor
    received: torch.Tensor
    semantic_tokens: torch.Tensor
    timestep: torch.Tensor
    latent_mean: torch.Tensor
    latent_logvar: torch.Tensor


class OSDV3(nn.Module):
    """Conditioned latent JSCC followed by one-step SD-Turbo refinement."""

    downsample_factor = 128

    def __init__(
        self,
        unet: nn.Module,
        vae: nn.Module,
        scheduler: Any,
        *,
        compression_ratio: int = 96,
        channel: str = "awgn",
        jscc_width: int = 256,
        lora_rank: int = 32,
        semantic_tokens: int = 4,
        snr_embedding_dim: int = 256,
        high_snr_threshold: float = 10.0,
        snr_modulation: str = "adagn",
    ) -> None:
        super().__init__()
        if compression_ratio <= 0 or 1536 % compression_ratio:
            raise ValueError("compression_ratio must be a positive divisor of 1536.")
        latent_channels = 1536 // compression_ratio
        if latent_channels < 2 or latent_channels % 2:
            raise ValueError("compression_ratio must produce an even latent width.")
        if semantic_tokens < 0:
            raise ValueError("semantic_tokens cannot be negative.")
        if snr_modulation not in ("adagn", "none"):
            raise ValueError("snr_modulation must be 'adagn' or 'none'.")

        self.compression_ratio = compression_ratio
        self.latent_channels = latent_channels
        self.channel_type = channel.lower()
        self.jscc_width = jscc_width
        self.lora_rank = lora_rank
        self.semantic_tokens = semantic_tokens
        self.snr_embedding_dim = snr_embedding_dim
        self.high_snr_threshold = float(high_snr_threshold)
        self.snr_modulation = snr_modulation
        self.vae = _freeze(vae)
        self.unet = _freeze(unet)
        self.scheduler = scheduler
        spacing = getattr(getattr(scheduler, "config", None), "timestep_spacing", None)
        if spacing != "trailing":
            raise ValueError("OSDV3 requires scheduler timestep_spacing='trailing'.")
        self.snr_embedding = SNREmbedding(snr_embedding_dim)
        self.channel = WirelessChannel(self.channel_type)
        self._install_lora(lora_rank)

        context_dim = self.unet.config.cross_attention_dim
        if isinstance(context_dim, (tuple, list)):
            context_dim = context_dim[0]
        self.latent_adapter = IdentityLatentAdapter()
        self.jscc_encoder = V3JSCCEncoder(
            self.latent_channels, self.snr_embedding_dim, self.snr_modulation
        )
        self.jscc_decoder = V3JSCCDecoder(
            self.latent_channels, self.snr_embedding_dim, int(context_dim),
            self.semantic_tokens, self.snr_modulation,
        )
        self.snr_context = nn.Linear(snr_embedding_dim, int(context_dim))
        # Evaluation-only controls. Training and legacy evaluation retain the
        # original adaptive one-step behavior unless explicitly overridden.
        self.inference_steps = 1
        self.timestep_policy = "adaptive"

    def _install_lora(self, rank: int) -> None:
        if rank <= 0:
            raise ValueError("lora_rank must be positive.")
        try:
            from peft import LoraConfig
        except ImportError as exc:  # pragma: no cover
            raise ImportError("OSDV3 requires peft; run `uv sync`.") from exc
        self.unet.add_adapter(LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=[
                "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1",
                "conv2", "conv_shortcut", "proj_in", "proj_out",
                "ff.net.2", "ff.net.0.proj",
            ],
        ))
        self.unet.conv_in.requires_grad_(True)
        self.unet.conv_out.requires_grad_(True)

    @classmethod
    def from_pretrained(
        cls,
        *,
        sd_model_id: str = SD_TURBO_MODEL_ID,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> "OSDV3":
        try:
            from diffusers import (
                AutoencoderKL,
                EulerDiscreteScheduler,
                UNet2DConditionModel,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OSDV3 requires diffusers, accelerate and peft; run `uv sync`."
            ) from exc
        dtype_args = {} if torch_dtype is None else {"torch_dtype": torch_dtype}
        unet = UNet2DConditionModel.from_pretrained(
            sd_model_id, subfolder="unet", **dtype_args
        )
        vae = AutoencoderKL.from_pretrained(
            sd_model_id, subfolder="vae", **dtype_args
        )
        scheduler = EulerDiscreteScheduler.from_pretrained(
            sd_model_id, subfolder="scheduler", timestep_spacing="trailing"
        )
        return cls(unet, vae, scheduler, **kwargs)

    @property
    def config(self) -> dict[str, int | float | str]:
        return {
            "compression_ratio": self.compression_ratio,
            "channel": self.channel_type,
            "jscc_width": self.jscc_width,
            "lora_rank": self.lora_rank,
            "semantic_tokens": self.semantic_tokens,
            "snr_embedding_dim": self.snr_embedding_dim,
            "high_snr_threshold": self.high_snr_threshold,
            "snr_modulation": self.snr_modulation,
            "architecture": "convnext_window_v3",
        }

    def train(self, mode: bool = True) -> "OSDV3":
        super().train(mode)
        self.vae.eval()
        return self

    @staticmethod
    def _snr(snr_db: float | torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        snr = torch.as_tensor(snr_db, device=image.device, dtype=image.dtype).flatten()
        if snr.numel() == 1:
            return snr.expand(image.shape[0])
        if snr.numel() != image.shape[0]:
            raise ValueError("SNR must be scalar or contain one value per image.")
        return snr

    @torch.no_grad()
    def ground_truth_vae_latent(self, image: torch.Tensor) -> torch.Tensor:
        vae_dtype = _module_dtype(self.vae, image.dtype)
        posterior = self.vae.encode(image.to(dtype=vae_dtype), return_dict=False)[0]
        scale = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        shift = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
        return (posterior.mode() - shift) * scale

    def _refine_group(
        self, latent: torch.Tensor, conditioning: torch.Tensor, timestep: int
    ) -> torch.Tensor:
        self.scheduler.set_timesteps(timesteps=[timestep], device=latent.device)
        scheduler_timestep = self.scheduler.timesteps[0]
        model_input = self.scheduler.scale_model_input(latent, scheduler_timestep)
        unet_dtype = _module_dtype(self.unet, latent.dtype)
        prediction = self.unet(
            model_input.to(dtype=unet_dtype),
            scheduler_timestep,
            encoder_hidden_states=conditioning.to(dtype=unet_dtype),
            return_dict=False,
        )[0]
        return self.scheduler.step(
            prediction.to(dtype=latent.dtype),
            scheduler_timestep,
            latent,
            return_dict=True,
        ).pred_original_sample

    @staticmethod
    def _inference_schedule(start_timestep: int, steps: int) -> list[int]:
        if steps <= 0:
            raise ValueError("Diffusion inference steps must be positive.")
        schedule = [
            round(start_timestep * remaining / steps)
            for remaining in range(steps, 0, -1)
        ]
        if any(current <= following for current, following in zip(schedule, schedule[1:])):
            raise ValueError(
                f"Inference schedule is not strictly descending: {schedule}."
            )
        return schedule

    def _refine_group_multistep(
        self,
        latent: torch.Tensor,
        conditioning: torch.Tensor,
        start_timestep: int,
        steps: int,
    ) -> torch.Tensor:
        if steps == 1:
            return self._refine_group(latent, conditioning, start_timestep)
        schedule = self._inference_schedule(start_timestep, steps)
        self.scheduler.set_timesteps(timesteps=schedule, device=latent.device)
        sample = latent
        unet_dtype = _module_dtype(self.unet, latent.dtype)
        for scheduler_timestep in self.scheduler.timesteps:
            model_input = self.scheduler.scale_model_input(sample, scheduler_timestep)
            prediction = self.unet(
                model_input.to(dtype=unet_dtype),
                scheduler_timestep,
                encoder_hidden_states=conditioning.to(dtype=unet_dtype),
                return_dict=False,
            )[0]
            sample = self.scheduler.step(
                prediction.to(dtype=sample.dtype),
                scheduler_timestep,
                sample,
                return_dict=True,
            ).prev_sample
        return sample

    def configure_inference(
        self, *, diffusion_steps: int = 1, timestep_policy: str = "adaptive"
    ) -> None:
        if diffusion_steps < 0:
            raise ValueError("diffusion_steps cannot be negative.")
        if timestep_policy not in ("adaptive", "fixed249", "fixed499"):
            raise ValueError(
                "timestep_policy must be 'adaptive', 'fixed249', or 'fixed499'."
            )
        self.inference_steps = diffusion_steps
        self.timestep_policy = timestep_policy

    def _inference_start_timestep(self, snr: torch.Tensor) -> torch.Tensor:
        if self.timestep_policy == "fixed249":
            return torch.full_like(snr, 249, dtype=torch.long)
        if self.timestep_policy == "fixed499":
            return torch.full_like(snr, 499, dtype=torch.long)
        return torch.where(
            snr >= self.high_snr_threshold,
            torch.full_like(snr, 249, dtype=torch.long),
            torch.full_like(snr, 499, dtype=torch.long),
        )

    def encode(
        self, image: torch.Tensor, snr_db: float | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Expected image shaped [B,3,H,W].")
        if image.shape[-2] % 128 or image.shape[-1] % 128:
            raise ValueError("OSDV3 image dimensions must be divisible by 128.")
        source_latent = self.ground_truth_vae_latent(image)
        adapter_dtype = _module_dtype(self.latent_adapter, source_latent.dtype)
        source_latent = self.latent_adapter(source_latent.to(dtype=adapter_dtype))
        snr = self._snr(snr_db, image)
        condition = self.snr_embedding(snr)
        codeword = self.jscc_encoder(source_latent, condition)
        return source_latent, codeword, self.channel(codeword, snr), condition

    def forward(
        self, image: torch.Tensor, snr_db: float | torch.Tensor, *,
        return_dict: bool = False, generator: torch.Generator | None = None,
    ) -> torch.Tensor | OSDV3Output:
        _, codeword, received, condition = self.encode(image, snr_db)
        raw_mean, raw_logvar, semantic_tokens = self.jscc_decoder(received, condition)
        # The variance branch has no KL/latent loss, so an unconstrained
        # log-variance can grow until sampled latents overflow the FP16
        # UNet/VAE. Keep the Gaussian expressive but within the SD latent's
        # useful numerical range: mean in [-8, 8], std in [exp(-3), 1].
        # Hard bounds preserve every already-valid output from an existing
        # checkpoint instead of remapping the entire learned distribution.
        mean = raw_mean.float().clamp(-8.0, 8.0)
        logvar = raw_logvar.float().clamp(-6.0, 0.0)
        epsilon = torch.randn(
            mean.shape, device=mean.device, dtype=mean.dtype, generator=generator
        )
        y = (mean + torch.exp(0.5 * logvar) * epsilon).clamp(-10.0, 10.0)
        conditioning = torch.cat(
            (semantic_tokens, self.snr_context(condition)[:, None]), dim=1
        )
        snr = self._snr(snr_db, image)
        batch_timestep = self._inference_start_timestep(snr)
        if self.inference_steps == 0:
            refined = y
        else:
            refined = torch.empty_like(y)
            for timestep in batch_timestep.unique(sorted=True).tolist():
                indices = torch.nonzero(
                    batch_timestep == timestep, as_tuple=True
                )[0]
                refined[indices] = self._refine_group_multistep(
                    y[indices], conditioning[indices], timestep,
                    self.inference_steps,
                )
        scale = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        shift = float(getattr(self.vae.config, "shift_factor", 0.0) or 0.0)
        vae_input = (refined / scale + shift).to(
            dtype=_module_dtype(self.vae, refined.dtype)
        )
        reconstruction = self.vae.decode(vae_input, return_dict=False)[0].clamp(-1, 1)
        if return_dict:
            return OSDV3Output(
                reconstruction, y, refined, codeword, received, conditioning,
                batch_timestep, mean, logvar,
            )
        return reconstruction


__all__ = ["OSDV3", "OSDV3Output"]
