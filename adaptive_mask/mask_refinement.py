"""Optional binary-mask cleanup using closing and connected components."""
import torch
import torch.nn.functional as F


def _remove_small_components(mask_2d: torch.Tensor, min_area: int) -> torch.Tensor:
    """Remove small 8-connected components. Requires scipy."""
    try:
        import numpy as np
        from scipy import ndimage
    except ImportError as exc:
        raise ImportError(
            "Connected-component filtering requires scipy. Install it with `pip install scipy`."
        ) from exc

    array = mask_2d.detach().cpu().numpy().astype(bool)
    labels, count = ndimage.label(array, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return torch.zeros_like(mask_2d)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_area
    keep[0] = False
    cleaned = keep[labels]
    return torch.as_tensor(cleaned, device=mask_2d.device, dtype=mask_2d.dtype)


def refine_mask(
    mask: torch.Tensor,
    closing_kernel_size: int = 5,
    min_component_area: int = 16,
    remove_small_components: bool = True,
) -> torch.Tensor:
    """Close small holes/gaps and optionally remove small components.

    Input/output shape: [B,1,H,W]. Output is binary float mask.
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B,1,H,W]")
    if closing_kernel_size < 1 or min_component_area < 1:
        raise ValueError("kernel size and component area must be >= 1")

    binary = (mask > 0.5).to(mask.dtype)
    if closing_kernel_size > 1:
        k = closing_kernel_size
        pad = k // 2
        dilated = F.max_pool2d(binary, kernel_size=k, stride=1, padding=pad)
        # Erosion via min-pooling = -max_pool(-x). Pad with ones so
        # image-border pixels are not incorrectly erased by implicit -inf padding.
        padded = F.pad(dilated, (pad, pad, pad, pad), mode="constant", value=1.0)
        binary = -F.max_pool2d(-padded, kernel_size=k, stride=1, padding=0)
        binary = (binary > 0.5).to(mask.dtype)

    if remove_small_components:
        cleaned = [
            _remove_small_components(binary[i, 0], min_component_area).unsqueeze(0).unsqueeze(0)
            for i in range(binary.shape[0])
        ]
        binary = torch.cat(cleaned, dim=0)

    return binary
