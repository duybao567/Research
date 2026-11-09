"""Visualization helpers for images, saliency maps, and masks."""
import torch
import matplotlib.pyplot as plt


def _to_display_image(image: torch.Tensor):
    image = image.detach().float().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError("image must have shape [C,H,W] or [B,C,H,W]")
    image = image.clamp(0, 1)
    if image.shape[0] == 1:
        return image[0].numpy(), "gray"
    if image.shape[0] == 3:
        return image.permute(1, 2, 0).numpy(), None
    raise ValueError("Only 1-channel or 3-channel images are supported")


def visualize_mask_pipeline(
    image: torch.Tensor,
    saliency: torch.Tensor,
    initial_mask: torch.Tensor,
    refined_mask: torch.Tensor = None,
    save_path: str = None,
):
    """Plot input, saliency, generated mask, and optional refined mask."""
    img, cmap = _to_display_image(image)
    sal = saliency.detach().float().cpu()
    mask = initial_mask.detach().float().cpu()
    if sal.ndim == 4:
        sal = sal[0, 0]
    elif sal.ndim == 3:
        sal = sal[0]
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]

    panels = [
        ("Input", img, cmap),
        ("Saliency", sal.numpy(), "inferno"),
        ("Initial mask", mask.numpy(), "gray"),
    ]
    if refined_mask is not None:
        refined = refined_mask.detach().float().cpu()
        if refined.ndim == 4:
            refined = refined[0, 0]
        elif refined.ndim == 3:
            refined = refined[0]
        panels.append(("Refined mask", refined.numpy(), "gray"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, data, color_map) in zip(axes, panels):
        ax.imshow(data, cmap=color_map, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
