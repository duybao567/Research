
"""Mask refinement utilities for M5 Adaptive Mask in ACE.

Features:
    - Morphological closing
    - Connected-component filtering
    - Entropy regularization
    - Total Variation regularization
    - Gaussian smoothing

The original refine_mask() API is preserved for compatibility.
"""

import math

import torch
import torch.nn.functional as F

from .config import AdaptiveMaskConfig


# =========================================================
# 1. CONNECTED COMPONENT FILTERING
# =========================================================

def _remove_small_components(
    mask_2d: torch.Tensor,
    min_area: int,
) -> torch.Tensor:
    """Remove small 8-connected components. Requires scipy."""

    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError(
            "Connected-component filtering requires scipy. "
            "Install it with `pip install scipy`."
        ) from exc

    array = (
        mask_2d.detach()
        .cpu()
        .numpy()
        .astype(bool)
    )

    labels, count = ndimage.label(
        array,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    if count == 0:
        return torch.zeros_like(mask_2d)

    sizes = np.bincount(labels.ravel())

    keep = sizes >= min_area
    keep[0] = False

    cleaned = keep[labels]

    return torch.as_tensor(
        cleaned,
        device=mask_2d.device,
        dtype=mask_2d.dtype,
    )


# =========================================================
# 2. MORPHOLOGICAL CLOSING
# =========================================================

def _morphological_closing(
    mask: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """Fill small holes and gaps in a binary mask."""

    if kernel_size <= 1:
        return mask

    if kernel_size % 2 == 0:
        raise ValueError(
            "closing kernel size must be odd."
        )

    pad = kernel_size // 2

    # Dilation.
    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )

    # Erosion via min pooling.
    padded = F.pad(
        dilated,
        (pad, pad, pad, pad),
        mode="constant",
        value=1.0,
    )

    eroded = -F.max_pool2d(
        -padded,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
    )

    return (eroded > 0.5).to(mask.dtype)


# =========================================================
# 3. TOTAL VARIATION LOSS
# =========================================================

def total_variation_loss(
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute spatial Total Variation loss.

    Args:
        mask: Soft mask [B, 1, H, W].

    Returns:
        Scalar TV loss.
    """

    vertical = torch.abs(
        mask[:, :, 1:, :] - mask[:, :, :-1, :]
    )

    horizontal = torch.abs(
        mask[:, :, :, 1:] - mask[:, :, :, :-1]
    )

    return (
        vertical.mean() + horizontal.mean()
    )


# =========================================================
# 4. BINARY ENTROPY REGULARIZATION
# =========================================================

def binary_entropy_loss(
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean binary entropy of a soft mask.

    Minimizing this term encourages mask values toward 0 or 1.
    """

    mask = mask.clamp(
        min=eps,
        max=1.0 - eps,
    )

    entropy = -(
        mask * torch.log(mask)
        + (1.0 - mask) * torch.log(1.0 - mask)
    )

    return entropy.mean()


# =========================================================
# 5. GAUSSIAN SMOOTHING
# =========================================================

def gaussian_smoothing(
    mask: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply Gaussian smoothing to a mask.

    Supports [B, 1, H, W] masks.
    Output remains in [0, 1].
    """

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            "mask must have shape [B, 1, H, W]."
        )

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            "Gaussian kernel size must be a positive odd number."
        )

    if sigma <= 0:
        raise ValueError("sigma must be > 0.")

    if kernel_size == 1:
        return mask.clamp(0.0, 1.0)

    device = mask.device
    dtype = mask.dtype

    coordinates = torch.arange(
        kernel_size,
        device=device,
        dtype=dtype,
    ) - kernel_size // 2

    gaussian_1d = torch.exp(
        -(coordinates ** 2) / (2.0 * sigma ** 2)
    )

    gaussian_1d = gaussian_1d / gaussian_1d.sum()

    gaussian_2d = (
        gaussian_1d[:, None]
        * gaussian_1d[None, :]
    )

    kernel = gaussian_2d.view(
        1, 1, kernel_size, kernel_size
    )

    padding = kernel_size // 2

    # Replicate padding avoids artificial zero values at image borders.
    padded = F.pad(
        mask,
        (padding, padding, padding, padding),
        mode="replicate",
    )

    smoothed = F.conv2d(
        padded,
        kernel,
    )

    return smoothed.clamp(0.0, 1.0)


# =========================================================
# 6. ORIGINAL BINARY MASK REFINEMENT
# =========================================================

def refine_mask(
    mask: torch.Tensor,
    closing_kernel_size: int = 5,
    min_component_area: int = 16,
    remove_small_components: bool = True,
) -> torch.Tensor:
    """Close small holes and optionally remove small components.

    Original API is preserved for compatibility.

    Input/output:
        Binary float mask [B, 1, H, W].
    """

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            "mask must have shape [B, 1, H, W]."
        )

    if closing_kernel_size < 1:
        raise ValueError(
            "closing_kernel_size must be >= 1."
        )

    if closing_kernel_size % 2 == 0:
        raise ValueError(
            "closing_kernel_size must be odd."
        )

    if min_component_area < 1:
        raise ValueError(
            "min_component_area must be >= 1."
        )

    if not torch.isfinite(mask).all():
        raise ValueError("mask contains NaN or Inf.")

    binary = (mask > 0.5).to(mask.dtype)

    binary = _morphological_closing(
        binary,
        closing_kernel_size,
    )

    if remove_small_components:
        cleaned = [
            _remove_small_components(
                binary[i, 0],
                min_component_area,
            ).unsqueeze(0).unsqueeze(0)
            for i in range(binary.shape[0])
        ]

        binary = torch.cat(
            cleaned,
            dim=0,
        )

    return binary


# =========================================================
# 7. M5 MASK OPTIMIZATION
# =========================================================

def refine_adaptive_mask(
    initial_mask: torch.Tensor,
    saliency: torch.Tensor,
    config: AdaptiveMaskConfig = None,
    iterations: int = 100,
    learning_rate: float = 0.05,
    coverage_weight: float = 10.0,
    apply_closing: bool = False,
    remove_small_components: bool = False,
    return_info: bool = False,
):
    """Optimize a soft mask using CAM alignment, entropy and TV.

    Objective:

        L = -L_CAM
            + lambda_ent * L_entropy
            + lambda_tv * L_TV
            + lambda_cov * L_coverage

    where:

        L_CAM:
            Encourages the mask to overlap with important saliency.

        L_entropy:
            Encourages mask values toward 0 or 1.

        L_TV:
            Encourages spatial smoothness.

        L_coverage:
            Keeps mask area close to the initial coverage.

    Args:
        initial_mask:
            Initial mask [B, 1, H, W].

        saliency:
            Non-negative CAM/saliency map [B, 1, H, W].

        config:
            AdaptiveMaskConfig.

        iterations:
            Number of optimization steps.

        learning_rate:
            Adam learning rate.

        coverage_weight:
            Weight for coverage preservation.

        apply_closing:
            Whether to apply binary morphological closing
            before optimization.

        remove_small_components:
            Whether to remove small components before optimization.

        return_info:
            Whether to return optimization diagnostics.

    Returns:
        Optimized soft mask [B, 1, H, W].
    """

    if config is None:
        config = AdaptiveMaskConfig()

    config.validate()

    if initial_mask.ndim != 4 or initial_mask.shape[1] != 1:
        raise ValueError(
            "initial_mask must have shape [B, 1, H, W]."
        )

    if saliency.shape != initial_mask.shape:
        raise ValueError(
            "saliency and initial_mask must have identical shapes."
        )

    if not torch.isfinite(initial_mask).all():
        raise ValueError("initial_mask contains NaN or Inf.")

    if not torch.isfinite(saliency).all():
        raise ValueError("saliency contains NaN or Inf.")

    if iterations < 1:
        raise ValueError("iterations must be >= 1.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0.")

    if coverage_weight <= 0:
        raise ValueError("coverage_weight must be > 0.")

    # -----------------------------------------------------
    # 7.1 Initial binary cleanup
    # -----------------------------------------------------

    initial = initial_mask.detach().clamp(0.0, 1.0)

    if apply_closing or remove_small_components:
        initial = refine_mask(
            initial,
            closing_kernel_size=(
                config.closing_kernel_size
                if apply_closing else 1
            ),
            min_component_area=config.min_component_area,
            remove_small_components=remove_small_components,
        )

    # Keep a fixed coverage target for each image.
    target_coverage = initial.flatten(1).mean(dim=1).detach()

    # -----------------------------------------------------
    # 7.2 Normalize saliency per image
    # -----------------------------------------------------

    saliency = saliency.detach().clamp_min(0.0)

    flat_saliency = saliency.flatten(1)

    saliency_min = flat_saliency.min(
        dim=1,
        keepdim=True,
    ).values

    saliency_max = flat_saliency.max(
        dim=1,
        keepdim=True,
    ).values

    normalized_saliency = (
        (flat_saliency - saliency_min)
        / (saliency_max - saliency_min).clamp_min(config.eps)
    )

    normalized_saliency = normalized_saliency.view_as(saliency)

    # -----------------------------------------------------
    # 7.3 Optimize mask
    # -----------------------------------------------------

    optimized = torch.nn.Parameter(
        initial.clone()
    )

    optimizer = torch.optim.Adam(
        [optimized],
        lr=learning_rate,
    )

    for _ in range(iterations):

        optimizer.zero_grad()

        soft_mask = optimized.clamp(0.0, 1.0)

        # CAM alignment: maximize saliency inside the mask.
        cam_alignment = (
            soft_mask * normalized_saliency
        ).mean()

        cam_loss = -cam_alignment

        # Entropy regularization.
        if config.entropy_enabled:
            entropy_loss = binary_entropy_loss(
                soft_mask,
                config.eps,
            )
        else:
            entropy_loss = soft_mask.new_zeros(())

        # Total Variation regularization.
        if config.tv_enabled:
            tv_loss = total_variation_loss(
                soft_mask
            )
        else:
            tv_loss = soft_mask.new_zeros(())

        # Coverage preservation.
        current_coverage = soft_mask.flatten(1).mean(dim=1)

        coverage_loss = (
            (current_coverage - target_coverage) ** 2
        ).mean()

        # Combined objective.
        loss = (
            cam_loss
            + config.entropy_regularization_strength * entropy_loss
            + config.tv_regularization_strength * tv_loss
            + coverage_weight * coverage_loss
        )

        loss.backward()
        optimizer.step()

        # Project mask values into [0, 1].
        with torch.no_grad():
            optimized.clamp_(0.0, 1.0)

    # -----------------------------------------------------
    # 7.4 Final soft mask
    # -----------------------------------------------------

    result = optimized.detach().clamp(0.0, 1.0)

    # Gaussian smoothing is applied after optimization.
    if config.gaussian_smoothing_enabled:
        result = gaussian_smoothing(
            result,
            kernel_size=config.gaussian_kernel_size,
            sigma=config.gaussian_sigma,
        )

    # -----------------------------------------------------
    # 7.5 Diagnostics
    # -----------------------------------------------------

    if return_info:

        final_coverage = result.flatten(1).mean(dim=1)

        info = {
            "target_coverage": target_coverage.detach(),
            "final_coverage": final_coverage.detach(),
            "entropy_loss": entropy_loss.detach(),
            "tv_loss": tv_loss.detach(),
            "cam_alignment": cam_alignment.detach(),
            "coverage_loss": coverage_loss.detach(),
        }

        return result, info

    return result
