"""CAM-guided adaptive mask generation for ACE.

This module converts a non-negative saliency map into a binary
adaptive mask.

Features:
- Per-image entropy-aware coverage
- Exact Top-K mask selection
- Fixed morphological dilation (MaskDiME-inspired)
- Adaptive-radius dilation (original M5)
- Coverage enforced on the FINAL mask (after dilation)
- Diagnostic information

Coverage semantics (config.coverage_mode, default "final"):

    "final":
        [min_coverage, max_coverage] bounds the coverage of the final
        mask, i.e. AFTER dilation. The top-k size is chosen per image by
        bisection so that the dilated mask does not exceed the target
        coverage. The final coverage can be slightly below the target
        (a single extra top-k pixel adds up to (2r+1)^2 dilated pixels).

    "raw":
        Legacy behaviour. The coverage bounds only the top-k selection;
        dilation then enlarges the mask, so the final coverage can be
        several times larger than the target. This is closer to how
        MaskDiME uses top-k followed by a 5x5 dilation.

Entropy and TV mask optimization are handled separately by
the mask refinement module.

Note:
Fixed 5x5 dilation is inspired by the dilation kernel described
in MaskDiME. This module does not implement the complete
dynamic dual-mask sampling procedure of MaskDiME.
"""

import math

import torch
import torch.nn.functional as F

from .config import AdaptiveMaskConfig


# =========================================================
# 1. NORMALIZED ENTROPY
# =========================================================

def _normalized_entropy(
    saliency: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute normalized spatial entropy for each image.

    Args:
        saliency: Non-negative saliency tensor [B, 1, H, W].
        eps: Numerical stability constant.

    Returns:
        Entropy tensor [B], approximately in [0, 1].
    """

    probabilities = saliency.flatten(1)

    probabilities = probabilities / probabilities.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(eps)

    entropy = -(
        probabilities
        * probabilities.clamp_min(eps).log()
    ).sum(dim=1)

    num_pixels = probabilities.shape[1]
    max_entropy = math.log(max(num_pixels, 2))

    return (entropy / max_entropy).clamp(0.0, 1.0)


# =========================================================
# 2. EXACT TOP-K MASK GENERATION
# =========================================================

def _select_topk_mask(
    saliency: torch.Tensor,
    coverage: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exactly the top-k saliency pixels.

    Args:
        saliency: Single-image saliency tensor [1, 1, H, W].
        coverage: Desired coverage fraction.

    Returns:
        mask: Binary mask [1, 1, H, W].
        threshold: Lowest selected saliency value.
    """

    values = saliency.flatten()

    num_pixels = values.numel()

    k = int(round(coverage * num_pixels))
    k = max(1, min(k, num_pixels))

    selected_values, selected_indices = torch.topk(
        values,
        k=k,
        largest=True,
        sorted=False,
    )

    flat_mask = torch.zeros_like(values)

    flat_mask.scatter_(
        dim=0,
        index=selected_indices,
        value=1.0,
    )

    mask = flat_mask.view_as(saliency)

    threshold = selected_values.min()

    return mask, threshold


# =========================================================
# 3. FIXED MORPHOLOGICAL DILATION
# =========================================================

def _dilate_fixed(
    mask: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Apply fixed-size binary morphological dilation.

    Uses max pooling with a square structuring element.

    Args:
        mask: Binary mask [B, 1, H, W].
        kernel_size: Odd kernel size, e.g. 5.

    Returns:
        Binary dilated mask with the same shape.
    """

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            "kernel_size must be a positive odd number."
        )

    mask_float = (mask > 0).float()

    radius = kernel_size // 2

    dilated = F.max_pool2d(
        mask_float,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )

    return (dilated > 0).to(mask.dtype)


# =========================================================
# 4. ADAPTIVE-RADIUS DILATION (ORIGINAL M5)
# =========================================================

def _dilate(
    mask: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Apply adaptive-radius dilation using max pooling.

    Args:
        mask: Binary mask [B, 1, H, W].
        radius: Dilation radius in pixels.

    Returns:
        Binary dilated mask with the same shape.
    """

    if radius < 0:
        raise ValueError("radius must be >= 0.")

    if radius == 0:
        return (mask > 0).to(mask.dtype)

    kernel_size = 2 * radius + 1

    dilated = F.max_pool2d(
        (mask > 0).float(),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )

    return (dilated > 0).to(mask.dtype)


def _compute_dilation_radius(
    raw_coverage: float,
    config: AdaptiveMaskConfig,
) -> int:
    """Compute adaptive dilation radius from raw mask coverage."""

    min_radius = config.min_dilation_radius
    max_radius = config.max_dilation_radius

    if max_radius == min_radius:
        return min_radius

    ratio = (
        raw_coverage
        / max(config.dilation_area_reference, config.eps)
    )

    ratio = max(0.0, min(ratio, 1.0))

    radius = round(
        min_radius
        + (max_radius - min_radius) * ratio
    )

    return max(
        min_radius,
        min(radius, max_radius),
    )


def _apply_dilation(
    mask: torch.Tensor,
    raw_coverage: float,
    config: AdaptiveMaskConfig,
) -> tuple[torch.Tensor, int]:
    """Dilate a raw top-k mask according to the configured method.

    Returns:
        (dilated_mask, radius)
    """

    if config.dilation_method == "fixed":

        dilated = _dilate_fixed(
            mask,
            kernel_size=config.dilation_kernel_size,
        )

        radius = config.dilation_kernel_size // 2

    elif config.dilation_method == "adaptive":

        radius = _compute_dilation_radius(
            raw_coverage,
            config,
        )

        dilated = _dilate(
            mask,
            radius,
        )

    else:
        raise ValueError(
            f"Unsupported dilation method: "
            f"{config.dilation_method}"
        )

    return dilated, radius


# =========================================================
# 5. TOP-K SELECTION THAT RESPECTS FINAL COVERAGE
# =========================================================

def _fit_topk_to_final_coverage(
    saliency: torch.Tensor,
    target_coverage: float,
    config: AdaptiveMaskConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Choose the largest top-k whose dilated mask fits the target.

    The top-k masks are nested (k grows -> mask grows), and both fixed
    and adaptive dilation are monotone in the mask and in k, so the
    dilated pixel count is non-decreasing in k and bisection is valid.

    Args:
        saliency: Single-image saliency tensor [1, 1, H, W].
        target_coverage: Maximum allowed final coverage fraction.
        config: AdaptiveMaskConfig instance.

    Returns:
        raw_mask: Top-k mask before dilation [1, 1, H, W].
        dilated_mask: Final mask after dilation [1, 1, H, W].
        threshold: Lowest selected saliency value.
        radius: Dilation radius used for the chosen k.
    """

    values = saliency.flatten()
    num_pixels = values.numel()

    order = torch.argsort(
        values,
        descending=True,
        stable=True,
    )

    rank = torch.empty_like(order)
    rank[order] = torch.arange(
        num_pixels,
        device=values.device,
    )

    max_final_pixels = int(
        math.floor(target_coverage * num_pixels + 1e-4)
    )
    max_final_pixels = max(1, min(max_final_pixels, num_pixels))

    def build(k: int):

        raw = (rank < k).to(saliency.dtype).view_as(saliency)

        dilated, radius = _apply_dilation(
            raw,
            k / num_pixels,
            config,
        )

        return raw, dilated, radius, int(dilated.sum().item())

    # Dilation never shrinks a mask, so the raw top-k can never need
    # more pixels than the final target.
    low = 1
    high = max_final_pixels

    if build(low)[3] > max_final_pixels:
        k = 1
    else:
        while low < high:

            mid = (low + high + 1) // 2

            if build(mid)[3] <= max_final_pixels:
                low = mid
            else:
                high = mid - 1

        k = low

    raw_mask, dilated_mask, radius, _ = build(k)

    threshold = values[order[k - 1]]

    return raw_mask, dilated_mask, threshold, radius


# =========================================================
# 6. MAIN MASK GENERATION
# =========================================================

def generate_adaptive_mask(
    saliency: torch.Tensor,
    config: AdaptiveMaskConfig = None,
    return_info: bool = False,
):
    """Generate per-image adaptive binary masks from saliency maps.

    Args:
        saliency:
            Non-negative saliency tensor [B, 1, H, W].

        config:
            AdaptiveMaskConfig instance. The optional attribute
            ``coverage_mode`` ("final" or "raw", default "final")
            selects whether the coverage bounds apply to the final
            dilated mask or only to the raw top-k selection.

        return_info:
            If True, return diagnostic information.

    Returns:
        If return_info=False:
            Binary mask [B, 1, H, W].

        If return_info=True:
            (mask, info), where info contains:
                - normalized_entropy
                - target_coverage
                - raw_coverage
                - dilation_radius
                - dilation_method
                - dilation_kernel_size
                - coverage_mode
                - final_coverage
                - selection_threshold

    Mask convention:
        1 = selected saliency region
        0 = unselected region

    This module does not determine whether ACE interprets
    mask value 1 as an inpainting or preserved region.
    Verify the sampler's mask convention before integration.
    """

    # -----------------------------------------------------
    # 6.1 CONFIGURATION
    # -----------------------------------------------------

    if config is None:
        config = AdaptiveMaskConfig()

    config.validate()

    coverage_mode = getattr(config, "coverage_mode", "final")

    if coverage_mode not in ("final", "raw"):
        raise ValueError(
            "coverage_mode must be 'final' or 'raw', "
            f"got {coverage_mode!r}."
        )

    # -----------------------------------------------------
    # 6.2 INPUT VALIDATION
    # -----------------------------------------------------

    if not isinstance(saliency, torch.Tensor):
        raise TypeError("saliency must be a torch.Tensor.")

    if saliency.ndim != 4 or saliency.shape[1] != 1:
        raise ValueError(
            "saliency must have shape [B, 1, H, W]."
        )

    if not saliency.is_floating_point():
        raise TypeError(
            "saliency must be a floating-point tensor."
        )

    if saliency.shape[0] < 1:
        raise ValueError(
            "saliency batch must not be empty."
        )

    if saliency.shape[2] < 1 or saliency.shape[3] < 1:
        raise ValueError(
            "saliency spatial dimensions must be positive."
        )

    if not torch.isfinite(saliency).all():
        raise ValueError(
            "saliency contains NaN or Inf values."
        )

    # Saliency must be non-negative.
    saliency = saliency.detach().clamp_min(0.0)

    batch_size, _, height, width = saliency.shape

    num_pixels = height * width

    # -----------------------------------------------------
    # 6.3 CHECK SALIENCY QUALITY
    # -----------------------------------------------------

    flat_saliency = saliency.flatten(1)

    saliency_max = flat_saliency.max(dim=1).values
    saliency_min = flat_saliency.min(dim=1).values

    invalid_maps = (
        (saliency_max <= config.eps)
        | ((saliency_max - saliency_min) <= config.eps)
    )

    if invalid_maps.any():
        invalid_indices = (
            torch.nonzero(invalid_maps)
            .flatten()
            .tolist()
        )

        raise ValueError(
            "Invalid or spatially constant saliency map "
            f"for batch indices {invalid_indices}. "
            "Check classifier gradients or Grad-CAM target layer."
        )

    # -----------------------------------------------------
    # 6.4 COMPUTE NORMALIZED ENTROPY
    # -----------------------------------------------------

    entropy = _normalized_entropy(
        saliency,
        config.eps,
    )

    # -----------------------------------------------------
    # 6.5 ENTROPY-AWARE ADAPTIVE COVERAGE
    # -----------------------------------------------------

    if config.adaptive_coverage_enabled:

        entropy_factor = (
            config.entropy_coverage_strength * entropy
        ).clamp(0.0, 1.0)

        coverage = (
            config.min_coverage
            + (
                config.max_coverage
                - config.min_coverage
            ) * entropy_factor
        )

        coverage = coverage.clamp(
            config.min_coverage,
            config.max_coverage,
        )

    else:

        # Fixed coverage when adaptive coverage is disabled.
        coverage = torch.full_like(
            entropy,
            config.min_coverage,
        )

    # -----------------------------------------------------
    # 6.6 TOP-K SELECTION AND DILATION
    # -----------------------------------------------------

    raw_masks = []
    refined_masks = []
    thresholds = []
    raw_coverages = []
    dilation_radii = []

    for i in range(batch_size):

        image_saliency = saliency[i:i + 1]

        image_coverage = float(coverage[i].item())

        if coverage_mode == "final":

            raw_mask, dilated_mask, threshold, radius = (
                _fit_topk_to_final_coverage(
                    image_saliency,
                    image_coverage,
                    config,
                )
            )

        else:

            raw_mask, threshold = _select_topk_mask(
                image_saliency,
                image_coverage,
            )

            raw_for_radius = (
                raw_mask.sum() / num_pixels
            ).item()

            dilated_mask, radius = _apply_dilation(
                raw_mask,
                raw_for_radius,
                config,
            )

        raw_coverages.append(
            (raw_mask.sum() / num_pixels).item()
        )

        raw_masks.append(raw_mask)
        refined_masks.append(dilated_mask)
        thresholds.append(threshold)
        dilation_radii.append(radius)

    result = torch.cat(
        refined_masks,
        dim=0,
    ).clamp(0.0, 1.0)

    # -----------------------------------------------------
    # 6.7 FINAL COVERAGE
    # -----------------------------------------------------

    final_coverage = result.flatten(1).mean(dim=1)

    # -----------------------------------------------------
    # 6.8 DIAGNOSTIC INFORMATION
    # -----------------------------------------------------

    if return_info:

        info = {
            "normalized_entropy": entropy.detach(),

            "target_coverage": coverage.detach(),

            "raw_coverage": torch.tensor(
                raw_coverages,
                device=saliency.device,
                dtype=saliency.dtype,
            ),

            "dilation_radius": dilation_radii,

            "dilation_method": config.dilation_method,

            "dilation_kernel_size": (
                config.dilation_kernel_size
                if config.dilation_method == "fixed"
                else None
            ),

            "coverage_mode": coverage_mode,

            "final_coverage": final_coverage.detach(),

            "selection_threshold": torch.stack(
                thresholds
            ).detach(),
        }

        return result, info

    return result