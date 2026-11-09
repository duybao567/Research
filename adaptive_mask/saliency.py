"""Gradient-based saliency estimation.

Expected input:
    images: float tensor [B, C, H, W], normally in [0, 1].
    classifier_fn: callable mapping images to logits [B, num_classes].
    target_index: integer class index whose evidence is being explained.

If the classifier needs normalization, pass a classifier_fn wrapper that applies
the exact preprocessing used during classifier training.
"""
import torch


def compute_gradient_saliency(
    images: torch.Tensor,
    classifier_fn,
    target_index: int,
    absolute: bool = True,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return saliency maps with shape [B, 1, H, W], scaled to [0, 1].

    Uses gradients of the target-class logit with respect to the input. It does
    not populate model parameter .grad fields.
    """
    if images.ndim != 4:
        raise ValueError(f"images must have shape [B,C,H,W], got {tuple(images.shape)}")
    if not images.is_floating_point():
        raise TypeError("images must be a floating-point tensor")

    x = images.detach().clone().requires_grad_(True)
    logits = classifier_fn(x)

    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("classifier_fn must return logits with shape [B,num_classes]")
    if logits.shape[0] != x.shape[0]:
        raise ValueError("Classifier output batch size does not match input batch")
    if not (0 <= target_index < logits.shape[1]):
        raise IndexError(
            f"target_index={target_index} is outside [0, {logits.shape[1] - 1}]"
        )

    score = logits[:, target_index].sum()
    gradient = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]

    if absolute:
        saliency = gradient.abs().mean(dim=1, keepdim=True)
    else:
        saliency = gradient.mean(dim=1, keepdim=True).relu()

    if normalize:
        flat = saliency.flatten(1)
        lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
        hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
        saliency = (saliency - lo) / (hi - lo).clamp_min(eps)

    return saliency.detach()
