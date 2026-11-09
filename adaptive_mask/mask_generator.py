"""Per-image adaptive saliency thresholding and dilation."""
import math
import torch
import torch.nn.functional as F

from .config import AdaptiveMaskConfig


def _normalized_entropy(saliency: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalized spatial entropy per image, in [0, 1]."""
    p = saliency.flatten(1).clamp_min(0)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(p * p.clamp_min(eps).log()).sum(dim=1)
    max_entropy = math.log(max(p.shape[1], 2))
    return (entropy / max_entropy).clamp(0, 1)


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def generate_adaptive_mask(
    saliency: torch.Tensor,
    config: AdaptiveMaskConfig = None,
    return_info: bool = False,
):
    """Generate binary masks [B,1,H,W] from saliency [B,1,H,W].

    Coverage is selected independently for each image. Higher normalized
    saliency entropy increases the target coverage within configured bounds.
    Dilation radius then adapts to the selected mask's area fraction.

    Returns a float mask containing 0/1. This function does not assume whether
    ACE interprets 1 as "inpaint" or "preserve"; verify that before integration.
    """
    if config is None:
        config = AdaptiveMaskConfig()
    config.validate()

    if saliency.ndim != 4 or saliency.shape[1] != 1:
        raise ValueError("saliency must have shape [B,1,H,W]")
    if not saliency.is_floating_point():
        raise TypeError("saliency must be floating point")

    s = saliency.detach().clamp_min(0)
    batch, _, height, width = s.shape
    entropy = _normalized_entropy(s, config.eps)

    coverage = (
        config.min_coverage
        + config.entropy_coverage_strength * entropy
    ).clamp(config.min_coverage, config.max_coverage)

    masks = []
    radii = []
    for i in range(batch):
        values = s[i, 0].flatten()
        # Select the top-k saliency pixels; avoids threshold ties causing
        # unexpectedly large masks on flat/quantized saliency maps.
        k = max(1, min(values.numel(), int(round(float(coverage[i]) * values.numel()))))
        threshold = torch.topk(values, k=k, largest=True, sorted=True).values[-1]
        raw = (s[i:i+1] >= threshold).to(s.dtype)

        area_fraction = raw.mean().item()
        ratio = area_fraction / max(config.dilation_area_reference, config.eps)
        radius = int(round(
            config.min_dilation_radius
            + (config.max_dilation_radius - config.min_dilation_radius)
            * min(max(ratio, 0.0), 1.0)
        ))
        radius = max(config.min_dilation_radius, min(radius, config.max_dilation_radius))
        masks.append(_dilate(raw, radius))
        radii.append(radius)

    result = torch.cat(masks, dim=0).clamp(0, 1)
    if return_info:
        info = {
            "normalized_entropy": entropy.detach(),
            "target_coverage": coverage.detach(),
            "dilation_radius": radii,
            "final_coverage": result.flatten(1).mean(dim=1).detach(),
        }
        return result, info
    return result
