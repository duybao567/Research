
"""
Dual-level adaptive mask generation for ACE.

This module generates two nested binary masks:

    noisy_mask:
        Larger editable region for the noisy diffusion trajectory.

    clean_mask:
        Smaller editable region for the clean trajectory.

Invariant:
    clean_mask <= noisy_mask

Expected saliency shape:
    [B, 1, H, W]

Mask convention:
    1.0 = editable
    0.0 = preserve

The caller is responsible for converting mask conventions if
the downstream sampler expects 1.0 = preserve.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F


def _validate_saliency(saliency: torch.Tensor) -> torch.Tensor:
    """Validate and sanitize a saliency tensor."""

    if not isinstance(saliency, torch.Tensor):
        raise TypeError("saliency must be a torch.Tensor.")

    if saliency.ndim != 4:
        raise ValueError(
            "Expected saliency shape [B, 1, H, W], "
            f"got {tuple(saliency.shape)}."
        )

    if saliency.shape[1] != 1:
        raise ValueError(
            "Saliency must have exactly one channel, "
            f"got {saliency.shape[1]}."
        )

    if saliency.shape[0] < 1:
        raise ValueError("Batch size must be positive.")

    if saliency.shape[2] < 1 or saliency.shape[3] < 1:
        raise ValueError("Spatial dimensions must be positive.")

    if not torch.is_floating_point(saliency):
        saliency = saliency.float()

    # Remove NaN and Inf values.
    saliency = torch.nan_to_num(
        saliency,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Saliency represents non-negative importance.
    saliency = saliency.clamp_min(0.0)

    return saliency


def _topk_binary_mask(
    saliency: torch.Tensor,
    coverage: float,
) -> torch.Tensor:
    """
    Select the highest-scoring pixels globally per image.

    Args:
        saliency: [B, 1, H, W].
        coverage: Fraction of selected pixels, in (0, 1].

    Returns:
        Binary mask [B, 1, H, W].
    """

    if not 0.0 < coverage <= 1.0:
        raise ValueError(
            f"coverage must be in (0, 1], got {coverage}."
        )

    B, _, H, W = saliency.shape
    N = H * W

    k = max(1, min(N, int(round(N * coverage))))

    flat = saliency.reshape(B, N)

    indices = torch.topk(
        flat,
        k=k,
        dim=1,
        largest=True,
        sorted=False,
    ).indices

    mask = torch.zeros_like(flat)
    mask.scatter_(1, indices, 1.0)

    return mask.reshape(B, 1, H, W)


def _dilate_binary_mask(
    mask: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    Apply morphological dilation using max pooling.

    Args:
        mask: Binary mask [B, 1, H, W].
        kernel_size: Odd positive integer.
                     1 disables dilation.

    Returns:
        Dilated binary mask.
    """

    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1.")

    if kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be odd, "
            f"got {kernel_size}."
        )

    if kernel_size == 1:
        return mask

    padding = kernel_size // 2

    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    return (dilated > 0).to(mask.dtype)


def generate_dual_mask(
    saliency: torch.Tensor,
    noisy_topk: float = 0.15,
    clean_ratio: float = 0.5,
    dilation_kernel: int = 5,
    return_info: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, Dict],
]:
    """
    Generate nested noisy-level and clean-level masks.

    Args:
        saliency:
            Saliency tensor [B, 1, H, W].

        noisy_topk:
            Fraction of pixels selected for noisy_mask.

        clean_ratio:
            Fraction of noisy_mask pixels retained for clean_mask.

            Example:
                noisy_topk=0.15
                clean_ratio=0.5

            Before dilation:
                noisy coverage = 15%
                clean coverage = 7.5%

        dilation_kernel:
            Odd max-pooling kernel size.
            Use 1 to disable dilation.

        return_info:
            If True, also return coverage statistics.

    Returns:
        noisy_mask:
            [B, 1, H, W], binary editable mask.

        clean_mask:
            [B, 1, H, W], binary editable mask.

        info (optional):
            Coverage and configuration statistics.

    Important:
        clean_mask is guaranteed to be a subset of noisy_mask,
        including after dilation.
    """

    # --------------------------------------------------
    # 1. Validate arguments
    # --------------------------------------------------

    if not 0.0 < noisy_topk <= 1.0:
        raise ValueError(
            "noisy_topk must be in (0, 1], "
            f"got {noisy_topk}."
        )

    if not 0.0 < clean_ratio <= 1.0:
        raise ValueError(
            "clean_ratio must be in (0, 1], "
            f"got {clean_ratio}."
        )

    if dilation_kernel < 1 or dilation_kernel % 2 == 0:
        raise ValueError(
            "dilation_kernel must be a positive odd integer."
        )

    saliency = _validate_saliency(saliency)

    B, _, H, W = saliency.shape
    N = H * W

    # --------------------------------------------------
    # 2. Generate noisy-level mask
    # --------------------------------------------------

    noisy_mask = _topk_binary_mask(
        saliency=saliency,
        coverage=noisy_topk,
    )

    # --------------------------------------------------
    # 3. Generate clean-level mask
    # --------------------------------------------------

    # Clean mask must be selected only from noisy-mask pixels.
    flat_saliency = saliency.reshape(B, N)
    flat_noisy_mask = noisy_mask.reshape(B, N)

    noisy_pixel_count = flat_noisy_mask.sum(
        dim=1,
        keepdim=True,
    ).long()

    clean_pixel_count = torch.clamp(
        (noisy_pixel_count.float() * clean_ratio).round().long(),
        min=1,
    )

    # Exclude pixels outside noisy mask.
    masked_saliency = flat_saliency.masked_fill(
        flat_noisy_mask == 0,
        float("-inf"),
    )

    # Batch-safe Top-K: use the largest clean count in the batch,
    # then remove excess selected pixels per individual image.
    max_clean_k = int(clean_pixel_count.max().item())

    clean_indices = torch.topk(
        masked_saliency,
        k=max_clean_k,
        dim=1,
        largest=True,
        sorted=True,
    ).indices

    rank_positions = torch.arange(
        max_clean_k,
        device=saliency.device,
    ).unsqueeze(0)

    valid_positions = rank_positions < clean_pixel_count

    clean_values = torch.zeros(
        B,
        max_clean_k,
        device=saliency.device,
        dtype=saliency.dtype,
    )

    clean_values[valid_positions.expand_as(clean_values)] = 1.0

    flat_clean_mask = torch.zeros_like(flat_noisy_mask)

    flat_clean_mask.scatter_(
        dim=1,
        index=clean_indices,
        src=clean_values,
    )

    clean_mask = flat_clean_mask.reshape(B, 1, H, W)

    # --------------------------------------------------
    # 4. Apply dilation
    # --------------------------------------------------

    noisy_mask = _dilate_binary_mask(
        noisy_mask,
        kernel_size=dilation_kernel,
    )

    clean_mask = _dilate_binary_mask(
        clean_mask,
        kernel_size=dilation_kernel,
    )

    # --------------------------------------------------
    # 5. Enforce nested-mask constraint
    # --------------------------------------------------

    # Dilation can cause clean pixels to extend outside noisy mask.
    # Project clean mask back into noisy mask.
    clean_mask = clean_mask * noisy_mask

    # Binary guarantee.
    noisy_mask = (noisy_mask > 0).to(saliency.dtype)
    clean_mask = (clean_mask > 0).to(saliency.dtype)

    # --------------------------------------------------
    # 6. Collect statistics
    # --------------------------------------------------

    noisy_coverage = noisy_mask.mean(dim=(1, 2, 3))
    clean_coverage = clean_mask.mean(dim=(1, 2, 3))

    info = {
        "noisy_coverage": noisy_coverage.detach(),
        "clean_coverage": clean_coverage.detach(),
        "noisy_topk": float(noisy_topk),
        "clean_ratio": float(clean_ratio),
        "dilation_kernel": int(dilation_kernel),
        "nested_constraint": bool(
            torch.all(clean_mask <= noisy_mask).item()
        ),
        "noisy_pixel_count": noisy_mask.sum(
            dim=(1, 2, 3)
        ).detach(),
        "clean_pixel_count": clean_mask.sum(
            dim=(1, 2, 3)
        ).detach(),
    }

    if return_info:
        return noisy_mask, clean_mask, info

    return noisy_mask, clean_mask
