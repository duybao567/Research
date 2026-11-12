
"""Dynamic classifier-gradient saliency for adaptive counterfactual masks.

The saliency map is computed from the current diffusion state x_t.
It can be recomputed at every reverse-diffusion timestep.
"""

from typing import Optional

import torch


def compute_dynamic_gradient(
    x_t: torch.Tensor,
    classifier,
    target_label: int,
    loss_fn=None,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute classifier-gradient saliency from the current diffusion state.

    Args:
        x_t:
            Current diffusion state, shape [B, C, H, W].
        classifier:
            Differentiable classifier mapping images to logits [B, K].
            It must include the same preprocessing used during ACE evaluation.
        target_label:
            Target class index.
        loss_fn:
            Optional callable loss_fn(logits, target).
            If None, uses cross-entropy loss.
        normalize:
            Normalize each image's saliency map to [0, 1].
        eps:
            Numerical stability constant.

    Returns:
        saliency:
            Gradient-based saliency map, shape [B, 1, H, W].
    """

    if not isinstance(x_t, torch.Tensor):
        raise TypeError("x_t must be a torch.Tensor.")

    if x_t.ndim != 4:
        raise ValueError(
            f"x_t must have shape [B, C, H, W], got {tuple(x_t.shape)}"
        )

    if not x_t.is_floating_point():
        raise TypeError("x_t must be a floating-point tensor.")

    if not callable(classifier):
        raise TypeError("classifier must be callable.")

    if not isinstance(target_label, int):
        raise TypeError("target_label must be an integer.")

    # Keep the original diffusion state unchanged.
    x_input = x_t.detach().requires_grad_(True)

    # Forward pass through the classifier.
    logits = classifier(x_input)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if not isinstance(logits, torch.Tensor):
        raise TypeError("classifier must return a logits tensor.")

    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits [B, K], got {tuple(logits.shape)}"
        )

    if logits.shape[0] != x_t.shape[0]:
        raise ValueError("Classifier output batch size does not match x_t.")

    if not 0 <= target_label < logits.shape[1]:
        raise ValueError(
            f"target_label={target_label} is invalid for "
            f"{logits.shape[1]} classes."
        )

    # Target-class loss.
    targets = torch.full(
        (logits.shape[0],),
        target_label,
        device=logits.device,
        dtype=torch.long,
    )

    if loss_fn is None:
        loss = torch.nn.functional.cross_entropy(
            logits,
            targets,
            reduction="sum",
        )
    else:
        loss = loss_fn(logits, targets)

    # Compute gradient with respect to the current diffusion state.
    gradient = torch.autograd.grad(
        outputs=loss,
        inputs=x_input,
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]

    # Aggregate channel-wise gradient magnitudes.
    saliency = gradient.abs().mean(
        dim=1,
        keepdim=True,
    )

    # Remove invalid values.
    saliency = torch.nan_to_num(
        saliency,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Normalize each sample independently.
    if normalize:
        B = saliency.shape[0]

        flat = saliency.reshape(B, -1)
        max_values = flat.max(dim=1)[0].view(B, 1, 1, 1)

        saliency = saliency / (max_values + eps)

    return saliency.detach()
