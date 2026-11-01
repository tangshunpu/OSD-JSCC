"""Differentiable wireless channels for real-valued JSCC codewords."""

from __future__ import annotations

import math

import torch
from torch import nn


class WirelessChannel(nn.Module):
    """Unit-power AWGN or slow Rayleigh channel with perfect-CSI equalization.

    The real codeword is flattened and split into equal real/imaginary halves.
    Power is normalized independently for every image so the resulting complex
    symbols have unit average power.
    """

    SUPPORTED = ("awgn", "rayleigh", "none")

    def __init__(self, channel_type: str = "awgn", equalizer_eps: float = 1e-8):
        super().__init__()
        channel_type = channel_type.lower()
        if channel_type not in self.SUPPORTED:
            raise ValueError(
                f"Unknown channel {channel_type!r}; choose from {self.SUPPORTED}."
            )
        self.channel_type = channel_type
        self.equalizer_eps = equalizer_eps

    @staticmethod
    def _snr_vector(
        snr_db: float | torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        snr = torch.as_tensor(snr_db, dtype=torch.float32, device=device).flatten()
        if snr.numel() == 1:
            return snr.expand(batch_size)
        if snr.numel() != batch_size:
            raise ValueError(
                f"SNR must be scalar or have {batch_size} values; got {snr.numel()}."
            )
        return snr

    @staticmethod
    def _normalize(codeword: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(1, codeword.ndim))
        # Two real values form one complex symbol. Setting E[x_real^2] to 1/2
        # therefore sets E[|x_complex|^2] to one.
        power = 2.0 * codeword.square().mean(dim=dims, keepdim=True)
        return codeword / power.clamp_min(1e-12).sqrt()

    def forward(
        self, codeword: torch.Tensor, snr_db: float | torch.Tensor
    ) -> torch.Tensor:
        if codeword.ndim < 2:
            raise ValueError("The codeword must include batch and feature dimensions.")

        original_dtype = codeword.dtype
        working_dtype = (
            torch.float64 if original_dtype == torch.float64 else torch.float32
        )
        signal = self._normalize(codeword.to(working_dtype))
        flat = signal.flatten(start_dim=1)
        if flat.shape[1] % 2:
            raise ValueError(
                "Each codeword must contain an even number of real values so it can "
                "be represented as complex symbols."
            )

        half = flat.shape[1] // 2
        real, imag = flat[:, :half], flat[:, half:]
        snr = self._snr_vector(snr_db, flat.shape[0], flat.device)

        if self.channel_type == "none":
            out_real, out_imag = real, imag
        else:
            # For unit-power complex symbols, each real noise component has
            # variance 1 / (2 * SNR_linear).
            sigma = torch.sqrt(0.5 * torch.pow(10.0, -snr / 10.0)).unsqueeze(1)
            noise_real = torch.randn_like(real) * sigma
            noise_imag = torch.randn_like(imag) * sigma

            if self.channel_type == "awgn":
                out_real = real + noise_real
                out_imag = imag + noise_imag
            else:
                batch_size = flat.shape[0]
                h_real = torch.randn(
                    batch_size, 1, device=flat.device, dtype=flat.dtype
                ) / math.sqrt(2.0)
                h_imag = torch.randn(
                    batch_size, 1, device=flat.device, dtype=flat.dtype
                ) / math.sqrt(2.0)

                faded_real = h_real * real - h_imag * imag + noise_real
                faded_imag = h_imag * real + h_real * imag + noise_imag

                # Zero-forcing equalization: y * conj(h) / |h|^2.
                gain = (h_real.square() + h_imag.square()).clamp_min(self.equalizer_eps)
                out_real = (h_real * faded_real + h_imag * faded_imag) / gain
                out_imag = (h_real * faded_imag - h_imag * faded_real) / gain

        received = torch.cat((out_real, out_imag), dim=1).view_as(signal)
        return received.to(original_dtype)
