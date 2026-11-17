"""Dynamic classifier-gradient saliency for adaptive counterfactual masks.

The saliency map is computed from the current state of the counterfactual
(e.g. the current PGD estimate or the current diffusion state x_t), so it
can be recomputed at every step, as in MaskDiME.

M5-v2 changes:
    - Works with single-logit binary classifiers (outputs [B] or [B, 1]),
      such as the STEEX DenseNet used for CelebA-HQ Smile. The previous
      version required [B, K] logits and cross-entropy, which raises an
      error for this classifier.
    - Optional Gaussian smoothing of the map before it is used for top-k.
    - `gradient_to_saliency` turns an EXISTING gradient (for example the
      one PGD already computes) into a saliency map, so no extra backward
      pass is needed.
    - `compute_dynamic_dual_mask` combines saliency with generate_dual_mask
      (MaskDiME-style nested noisy/clean masks).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from .dual_mask import generate_dual_mask
from .mask_generator import _smooth_saliency


def gradient_to_saliency(
    gradient: torch.Tensor,
    smoothing_sigma: float = 0.0,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Channel-averaged |gradient| -> saliency [B, 1, H, W].

    Use this to reuse a gradient that is already available (e.g. the PGD
    gradient w.r.t. the perturbation) instead of running another backward.
    """

    if not isinstance(gradient, torch.Tensor) or gradient.ndim != 4:
        raise ValueError(
            "gradient must be a tensor of shape [B, C, H, W]."
        )

    saliency = gradient.detach().abs().mean(dim=1, keepdim=True)

    saliency = torch.nan_to_num(
        saliency,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if normalize:
        peak = saliency.flatten(1).max(dim=1).values
        saliency = saliency / (peak.view(-1, 1, 1, 1) + eps)

    if smoothing_sigma and smoothing_sigma > 0:
        saliency = _smooth_saliency(saliency, smoothing_sigma)

    return saliency.detach()


def _counterfactual_loss(
    logits: torch.Tensor,
    target_label: Optional[int],
    target_index: int,
) -> torch.Tensor:
    """Loss whose gradient points toward the counterfactual class.

    Single-logit classifier ([B] or [B, 1]):
        target_label None -> flip the current prediction
        target_label 0/1  -> push toward that class
        uses binary cross-entropy on the logit.

    Multi-class classifier ([B, K]):
        target_label must be given; uses cross-entropy.
    """

    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):

        z = logits.reshape(-1)

        if logits.ndim == 2 and logits.shape[1] == 1:
            z = logits[:, 0]

        if target_label is None:
            target = (z.detach() < 0).to(z.dtype)   # opposite of prediction
        else:
            if target_label not in (0, 1):
                raise ValueError(
                    "For a single-logit classifier target_label must be "
                    "0, 1 or None."
                )
            target = torch.full_like(z, float(target_label))

        return F.binary_cross_entropy_with_logits(
            z, target, reduction="sum"
        )

    if logits.ndim == 2:

        if target_label is None:
            raise ValueError(
                "target_label is required for a multi-class classifier."
            )

        if not 0 <= target_label < logits.shape[1]:
            raise ValueError(
                f"target_label={target_label} is invalid for "
                f"{logits.shape[1]} classes."
            )

        targets = torch.full(
            (logits.shape[0],),
            target_label,
            device=logits.device,
            dtype=torch.long,
        )

        return F.cross_entropy(logits, targets, reduction="sum")

    raise ValueError(
        f"Unsupported logits shape {tuple(logits.shape)}."
    )


def compute_dynamic_gradient(
    x_t: torch.Tensor,
    classifier,
    target_label: Optional[int] = None,
    loss_fn=None,
    normalize: bool = True,
    eps: float = 1e-8,
    target_index: int = 0,
    smoothing_sigma: float = 0.0,
) -> torch.Tensor:
    """
    Compute classifier-gradient saliency from the current state.

    Args:
        x_t:
            Current image estimate / diffusion state, shape [B, C, H, W].
        classifier:
            Differentiable classifier. Must apply the same preprocessing
            as ACE (e.g. the Normalizer wrapper). Logits [B], [B, 1] or
            [B, K].
        target_label:
            Single-logit: None (flip prediction), 0 or 1.
            Multi-class: target class index.
        loss_fn:
            Optional callable loss_fn(logits, target_label) overriding the
            default counterfactual loss.
        normalize:
            Normalize each image's saliency to [0, 1].
        smoothing_sigma:
            Optional Gaussian smoothing (pixels) of the saliency map.

    Returns:
        saliency: [B, 1, H, W].
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

    x_input = x_t.detach().clone().requires_grad_(True)

    with torch.enable_grad():

        logits = classifier(x_input)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        if not isinstance(logits, torch.Tensor):
            raise TypeError("classifier must return a logits tensor.")

        if logits.shape[0] != x_t.shape[0]:
            raise ValueError(
                "Classifier output batch size does not match x_t."
            )

        if loss_fn is None:
            loss = _counterfactual_loss(logits, target_label, target_index)
        else:
            loss = loss_fn(logits, target_label)

        gradient = torch.autograd.grad(
            outputs=loss,
            inputs=x_input,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]

    return gradient_to_saliency(
        gradient,
        smoothing_sigma=smoothing_sigma,
        normalize=normalize,
        eps=eps,
    )


def compute_dynamic_dual_mask(
    x_t: torch.Tensor,
    classifier,
    target_label: Optional[int] = None,
    noisy_topk: float = 0.05,
    clean_ratio: float = 0.25,
    dilation_kernel: int = 5,
    smoothing_sigma: float = 0.0,
    gradient: Optional[torch.Tensor] = None,
    return_info: bool = False,
):
    """
    MaskDiME-style nested masks from the CURRENT state.

    Defaults follow MaskDiME for CelebA-HQ Smile:
        noisy_topk = 0.05 (k), clean_ratio = 0.25 (rho), dilation 5.

    If `gradient` is given (e.g. the PGD gradient), it is reused and no
    extra backward pass is run.

    Returns:
        noisy_mask, clean_mask (and info if return_info=True),
        each [B, 1, H, W] with 1 = editable.
    """

    if gradient is not None:
        saliency = gradient_to_saliency(
            gradient,
            smoothing_sigma=smoothing_sigma,
        )
    else:
        saliency = compute_dynamic_gradient(
            x_t,
            classifier,
            target_label=target_label,
            smoothing_sigma=smoothing_sigma,
        )

    return generate_dual_mask(
        saliency,
        noisy_topk=noisy_topk,
        clean_ratio=clean_ratio,
        dilation_kernel=dilation_kernel,
        return_info=return_info,
    )