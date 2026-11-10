
"""Gradient-based and Grad-CAM saliency estimation for ACE.

Supported methods:
    - Input Gradient
    - Grad-CAM

Expected input:
    images: float tensor [B, C, H, W].
    classifier_fn: callable mapping images to classifier logits.
    target_layer: convolutional module used by Grad-CAM.

The classifier_fn should apply exactly the same preprocessing
as the original classifier.
"""

import torch
import torch.nn.functional as F


# =========================================================
# 1. COMMON UTILITIES
# =========================================================

def _validate_images(images):
    """Validate input image tensor."""

    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor.")

    if images.ndim != 4:
        raise ValueError(
            f"Expected images [B,C,H,W], got {tuple(images.shape)}."
        )

    if not images.is_floating_point():
        raise TypeError("images must be floating point.")

    if not torch.isfinite(images).all():
        raise ValueError("images contain NaN or Inf.")


def _select_target_scores(logits, target_index, batch_size):
    """Select target scores from classifier outputs.

    Supports:
        [B]       : scalar output per image.
        [B, 1]    : single-output classifier.
        [B, C]    : multi-output classifier.
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError("classifier_fn must return a torch.Tensor.")

    if logits.ndim == 1:
        if logits.shape[0] != batch_size:
            raise ValueError("Classifier output batch size mismatch.")

        if target_index != 0:
            raise IndexError(
                "A scalar classifier output only supports target_index=0."
            )

        return logits

    if logits.ndim == 2:
        if logits.shape[0] != batch_size:
            raise ValueError("Classifier output batch size mismatch.")

        if not (0 <= target_index < logits.shape[1]):
            raise IndexError(
                f"target_index={target_index} is outside "
                f"[0, {logits.shape[1] - 1}]."
            )

        return logits[:, target_index]

    raise ValueError(
        "classifier_fn must return [B], [B,1], or [B,C] logits."
    )


def _normalize_saliency(
    saliency,
    eps=1e-8,
):
    """Normalize each saliency map independently to [0,1]."""

    flat = saliency.flatten(1)

    minimum = flat.min(
        dim=1,
        keepdim=True,
    ).values

    maximum = flat.max(
        dim=1,
        keepdim=True,
    ).values

    normalized = (
        (flat - minimum)
        / (maximum - minimum).clamp_min(eps)
    )

    return normalized.view_as(saliency).clamp(0.0, 1.0)


# =========================================================
# 2. INPUT GRADIENT SALIENCY
# =========================================================

def compute_gradient_saliency(
    images: torch.Tensor,
    classifier_fn,
    target_index: int = 0,
    absolute: bool = True,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute input-gradient saliency.

    Returns:
        Saliency maps [B,1,H,W] in [0,1] if normalize=True.
    """

    _validate_images(images)

    x = images.detach().clone().requires_grad_(True)

    with torch.enable_grad():

        logits = classifier_fn(x)

        scores = _select_target_scores(
            logits,
            target_index,
            x.shape[0],
        )

        gradient = torch.autograd.grad(
            scores.sum(),
            x,
            retain_graph=False,
            create_graph=False,
        )[0]

    if absolute:
        saliency = gradient.abs().mean(
            dim=1,
            keepdim=True,
        )
    else:
        saliency = gradient.mean(
            dim=1,
            keepdim=True,
        ).relu()

    if normalize:
        saliency = _normalize_saliency(
            saliency,
            eps,
        )

    return saliency.detach()


# =========================================================
# 3. GRAD-CAM SALIENCY
# =========================================================

def compute_gradcam_saliency(
    images: torch.Tensor,
    classifier_fn,
    target_layer: torch.nn.Module,
    target_index: int = 0,
    normalize: bool = True,
    eps: float = 1e-8,
    fallback_to_input_gradient: bool = True,
):
    """
    Compute Grad-CAM saliency maps with degeneracy detection.

    Grad-CAM is the primary method.
    Input Gradient is used only for samples whose Grad-CAM
    maps are invalid or spatially constant.

    Returns:
        Saliency maps [B,1,H,W].
    """

    _validate_images(images)

    if not isinstance(target_layer, torch.nn.Module):
        raise TypeError(
            "target_layer must be a torch.nn.Module."
        )

    activations = []

    def forward_hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "Grad-CAM target layer must return a Tensor."
            )

        if output.ndim != 4:
            raise ValueError(
                "Expected target layer output [B,C,H,W], "
                f"got {tuple(output.shape)}."
            )

        activations.append(output)

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:
        x = images.detach()

        with torch.enable_grad():

            logits = classifier_fn(x)

            if len(activations) != 1:
                raise RuntimeError(
                    "Expected target layer to execute once, "
                    f"observed {len(activations)} executions."
                )

            feature_maps = activations[0]

            scores = _select_target_scores(
                logits,
                target_index,
                x.shape[0],
            )

            gradients = torch.autograd.grad(
                scores.sum(),
                feature_maps,
                retain_graph=False,
                create_graph=False,
            )[0]

        # ==========================================
        # 1. COMPUTE STANDARD GRAD-CAM
        # ==========================================

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = (
            weights * feature_maps
        ).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # ==========================================
        # 2. DETECT INVALID CAM
        # ==========================================

        flat = cam.flatten(1)

        cam_min = flat.min(dim=1).values
        cam_max = flat.max(dim=1).values
        cam_std = flat.std(dim=1, unbiased=False)

        valid = (
            torch.isfinite(flat).all(dim=1)
            & ((cam_max - cam_min) > eps)
            & (cam_std > eps)
        )

        invalid_indices = (
            (~valid).nonzero(as_tuple=True)[0]
        )

        # ==========================================
        # 3. NORMALIZE VALID CAM
        # ==========================================

        if normalize:
            saliency = _normalize_saliency(
                cam,
                eps,
            )
        else:
            saliency = cam.clone()

        # ==========================================
        # 4. INPUT GRADIENT FALLBACK
        # ==========================================

        if len(invalid_indices) > 0:

            print(
                "[Grad-CAM WARNING] Degenerate CAM "
                f"for batch indices: "
                f"{invalid_indices.tolist()}"
            )

            print(
                "[Grad-CAM DEBUG] CAM min:",
                cam_min.tolist()
            )

            print(
                "[Grad-CAM DEBUG] CAM max:",
                cam_max.tolist()
            )

            print(
                "[Grad-CAM DEBUG] CAM std:",
                cam_std.tolist()
            )

            if not fallback_to_input_gradient:
                raise RuntimeError(
                    "Grad-CAM produced invalid saliency maps "
                    f"for indices {invalid_indices.tolist()}."
                )

            fallback_images = x.index_select(
                0,
                invalid_indices,
            )

            fallback_saliency = compute_gradient_saliency(
                images=fallback_images,
                classifier_fn=classifier_fn,
                target_index=target_index,
                absolute=True,
                normalize=True,
                eps=eps,
            )

            fallback_std = fallback_saliency.flatten(1).std(
                dim=1,
                unbiased=False,
            )

            fallback_valid = (
                torch.isfinite(fallback_saliency).flatten(1).all(dim=1)
                & (fallback_std > eps)
            )

            if not fallback_valid.all():
                failed = invalid_indices[
                    ~fallback_valid
                ].tolist()

                raise RuntimeError(
                    "Both Grad-CAM and Input Gradient "
                    f"failed for batch indices {failed}."
                )

            # Replace only invalid samples.
            saliency[invalid_indices] = fallback_saliency

            print(
                "[M5] Input Gradient fallback used for "
                f"{len(invalid_indices)} sample(s)."
            )

        return saliency.detach()

    finally:
        handle.remove()

# =========================================================
# 4. CONFIG-BASED SALIENCY DISPATCHER
# =========================================================

def compute_saliency(
    images: torch.Tensor,
    classifier_fn,
    config,
    target_layer=None,
):
    """Compute saliency using the method specified in config.

    Supported config.saliency_method:
        - "gradient"
        - "gradcam"
    """

    config.validate()

    method = config.saliency_method

    if method == "gradient":

        return compute_gradient_saliency(
            images=images,
            classifier_fn=classifier_fn,
            target_index=config.target_index,
            eps=config.eps,
        )

    if method == "gradcam":

        if target_layer is None:
            raise ValueError(
                "Grad-CAM requires target_layer. "
                "Provide the actual convolutional layer "
                "from the classifier."
            )

        return compute_gradcam_saliency(
            images=images,
            classifier_fn=classifier_fn,
            target_layer=target_layer,
            target_index=config.target_index,
            eps=config.eps,
        )

    raise ValueError(
        f"Unsupported saliency method: {method}"
    )
