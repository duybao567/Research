
"""Visualization helpers for the M5 Adaptive Mask pipeline."""

import torch
import matplotlib.pyplot as plt
import numpy as np


def _to_display_image(image: torch.Tensor, image_range: str = "zero_one"):
    """Convert [C,H,W] or [B,C,H,W] tensor to a displayable NumPy image."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")

    image = image.detach().float().cpu()

    if image.ndim == 4:
        if image.shape[0] < 1:
            raise ValueError("image batch must not be empty")
        image = image[0]

    if image.ndim != 3:
        raise ValueError("image must have shape [C,H,W] or [B,C,H,W]")

    if image.shape[0] not in (1, 3):
        raise ValueError("Only 1-channel or 3-channel images are supported")

    if image_range == "minus_one_one":
        image = (image + 1.0) / 2.0
    elif image_range != "zero_one":
        raise ValueError(
            "image_range must be 'zero_one' or 'minus_one_one'"
        )

    image = image.clamp(0, 1)

    if image.shape[0] == 1:
        return image[0].numpy(), "gray"

    return image.permute(1, 2, 0).numpy(), None


def _to_display_map(tensor: torch.Tensor, name: str):
    """Convert [H,W], [1,H,W], or [B,1,H,W] to a 2D NumPy map."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")

    tensor = tensor.detach().float().cpu()

    if tensor.ndim == 4:
        if tensor.shape[0] < 1 or tensor.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B,1,H,W]")
        tensor = tensor[0, 0]
    elif tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1,H,W]")
        tensor = tensor[0]
    elif tensor.ndim != 2:
        raise ValueError(
            f"{name} must have shape [H,W], [1,H,W], or [B,1,H,W]"
        )

    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf values")

    return tensor.clamp(0, 1).numpy()


def _make_overlay(image, mask, image_cmap=None, alpha=0.45):
    """Overlay the mask in red on top of the input image."""
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()

    if mask.shape != image_rgb.shape[:2]:
        raise ValueError("Mask and image must have the same spatial dimensions")

    overlay = image_rgb.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 1.0

    active = mask[..., None]
    overlay = overlay * (1.0 - alpha * active) + red * (alpha * active)

    return np.clip(overlay, 0, 1)


def _coverage(mask):
    """Return soft coverage and thresholded coverage."""
    soft = float(mask.mean())
    binary = float((mask >= 0.5).mean())
    return soft, binary


def visualize_mask_pipeline(
    image: torch.Tensor,
    saliency: torch.Tensor,
    initial_mask: torch.Tensor,
    refined_mask: torch.Tensor = None,
    save_path: str = None,
    image_range: str = "zero_one",
    overlay_alpha: float = 0.45,
    show_statistics: bool = True,
):
    """
    Visualize the M5 Adaptive Mask pipeline.

    Args:
        image:
            Input image [C,H,W] or [B,C,H,W].
        saliency:
            Saliency / Grad-CAM map [H,W], [1,H,W], or [B,1,H,W].
        initial_mask:
            Initial adaptive mask, same spatial size as image.
        refined_mask:
            Optional mask after Entropy + TV optimization and smoothing.
        save_path:
            Optional path to save the figure.
        image_range:
            "zero_one" for images in [0,1].
            "minus_one_one" for images in [-1,1].
        overlay_alpha:
            Opacity of the red mask overlay.
        show_statistics:
            Whether to show mask coverage in panel titles.

    Returns:
        Matplotlib figure.
    """
    if not 0.0 <= overlay_alpha <= 1.0:
        raise ValueError("overlay_alpha must be in [0,1]")

    img, image_cmap = _to_display_image(image, image_range)
    sal = _to_display_map(saliency, "saliency")
    initial = _to_display_map(initial_mask, "initial_mask")

    if sal.shape != img.shape[:2]:
        raise ValueError("Saliency and image must have the same spatial dimensions")

    if initial.shape != img.shape[:2]:
        raise ValueError("Initial mask and image must have the same spatial dimensions")

    refined = None
    if refined_mask is not None:
        refined = _to_display_map(refined_mask, "refined_mask")
        if refined.shape != img.shape[:2]:
            raise ValueError("Refined mask and image must have the same spatial dimensions")

    panels = [
        ("Input image", img, image_cmap),
        ("Grad-CAM / Saliency", sal, "inferno"),
    ]

    initial_soft, initial_binary = _coverage(initial)
    initial_title = "Initial mask"
    if show_statistics:
        initial_title += (
            f"\nSoft: {initial_soft:.1%} | Binary: {initial_binary:.1%}"
        )
    panels.append((initial_title, initial, "gray"))

    if refined is not None:
        refined_soft, refined_binary = _coverage(refined)
        refined_title = "Refined mask"
        if show_statistics:
            refined_title += (
                f"\nSoft: {refined_soft:.1%} | Binary: {refined_binary:.1%}"
            )
        panels.append((refined_title, refined, "gray"))

    overlay_mask = refined if refined is not None else initial
    overlay_title = (
        "Refined mask overlay" if refined is not None
        else "Initial mask overlay"
    )
    overlay = _make_overlay(img, overlay_mask, alpha=overlay_alpha)
    panels.append((overlay_title, overlay, None))

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4 * len(panels), 4.5),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (title, data, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)

        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
