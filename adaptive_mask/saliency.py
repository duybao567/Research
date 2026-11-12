"""Gradient-based and Grad-CAM saliency estimation for ACE.

Supported methods:
    - Input Gradient (optionally SmoothGrad-averaged)
    - Grad-CAM

Expected input:
    images: float tensor [B, C, H, W].
    classifier_fn: callable mapping images to classifier logits.
    target_layer: convolutional module used by Grad-CAM.

The classifier_fn should apply exactly the same preprocessing
as the original classifier, and the classifier must be in eval().

Key design points:
    1. Direction-aware scores for single-output classifiers.
       A single logit z means "attribute present" when z > 0 and
       "attribute absent" when z < 0. Explaining +z for a negative
       prediction gives a CAM whose evidence is all negative, which
       ReLU wipes out, leaving only border noise. By default we
       explain the PREDICTED side: score = sign(z) * z.
       Note: for the input-gradient method with absolute=True the
       direction has no effect, because |grad(-z)| == |grad(z)|.
    2. CAM validity is judged on the raw (pre-ReLU) CAM, relative to
       its own magnitude, not with an absolute eps. A CAM whose positive
       part is tiny compared with its negative part is degenerate and
       triggers the fallback instead of being stretched to [0, 1].
    3. Max-normalization (no min subtraction) so that background is not
       lifted.
    4. Optional SmoothGrad for the input-gradient method.
       Raw input gradients of a CNN classifier are noisy and scattered.
       A top-k selection on such a map yields isolated pixels, which a
       later dilation inflates into a much larger mask. Averaging the
       gradient over several noisy copies of the image gives a smoother
       map, so the selected pixels cluster more. Disabled by default
       (smoothgrad_samples = 1 reproduces the plain input gradient).
       The noise comes from a private generator, so enabling SmoothGrad
       does not change the global random stream used by the diffusion
       sampler.
"""

import torch
import torch.nn.functional as F


VALID_DIRECTIONS = ("predicted", "positive", "negative")


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


def _is_single_output(logits):
    """True for [B] or [B,1] classifier outputs."""

    return logits.ndim == 1 or (
        logits.ndim == 2 and logits.shape[1] == 1
    )


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


def _directional_scores(logits, target_index, batch_size, direction):
    """Select target scores and orient them for explanation.

    direction:
        "predicted": explain the side the model predicted.
            Single-output: score = sign(logit) * logit (so a negative
            prediction is explained by evidence for "attribute absent").
        "positive": always explain +logit (evidence FOR the attribute).
        "negative": always explain -logit (evidence AGAINST it).

    For multi-output classifiers the selected class logit is used
    as-is (the sign flip only makes sense for a single logit).

    Returns:
        scores [B], sign [B] (sign is +1 for multi-output).
    """

    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {VALID_DIRECTIONS}, "
            f"got {direction!r}."
        )

    scores = _select_target_scores(logits, target_index, batch_size)

    ones = torch.ones_like(scores.detach())

    if not _is_single_output(logits) or direction == "positive":
        sign = ones
    elif direction == "negative":
        sign = -ones
    else:  # "predicted"
        sign = torch.where(scores.detach() >= 0, ones, -ones)

    return scores * sign, sign


def _normalize_saliency(
    saliency,
    eps=1e-8,
):
    """Normalize each non-negative saliency map by its own maximum.

    Max-normalization (no min subtraction) keeps the background at 0
    instead of lifting it, which is what we want after ReLU.
    """

    flat = saliency.flatten(1).clamp_min(0.0)

    maximum = flat.max(
        dim=1,
        keepdim=True,
    ).values

    normalized = flat / maximum.clamp_min(eps)

    return normalized.view_as(saliency).clamp(0.0, 1.0)


# =========================================================
# 2. INPUT GRADIENT SALIENCY (OPTIONAL SMOOTHGRAD)
# =========================================================

def compute_gradient_saliency(
    images: torch.Tensor,
    classifier_fn,
    target_index: int = 0,
    absolute: bool = True,
    normalize: bool = True,
    eps: float = 1e-8,
    direction: str = "predicted",
    smoothgrad_samples: int = 1,
    smoothgrad_noise: float = 0.1,
    smoothgrad_seed: int = 0,
) -> torch.Tensor:
    """Compute input-gradient saliency.

    Args:
        smoothgrad_samples:
            Number of noisy copies to average. 1 means the plain
            gradient of the clean image (previous behaviour).
        smoothgrad_noise:
            Noise standard deviation as a fraction of each image's
            value range (max - min). Ignored when smoothgrad_samples=1.
        smoothgrad_seed:
            Seed of the private noise generator.

    Returns:
        Saliency maps [B,1,H,W] in [0,1] if normalize=True.
    """

    _validate_images(images)

    if smoothgrad_samples < 1:
        raise ValueError("smoothgrad_samples must be >= 1.")

    if smoothgrad_noise < 0:
        raise ValueError("smoothgrad_noise must be >= 0.")

    base = images.detach()

    use_smoothing = (
        smoothgrad_samples > 1 and smoothgrad_noise > 0
    )

    generator = None
    noise_std = None
    in_unit_range = False

    if use_smoothing:

        generator = torch.Generator(device=base.device)
        generator.manual_seed(int(smoothgrad_seed))

        spread = (
            base.amax(dim=(1, 2, 3), keepdim=True)
            - base.amin(dim=(1, 2, 3), keepdim=True)
        ).clamp_min(eps)

        noise_std = smoothgrad_noise * spread

        in_unit_range = bool(
            base.min() >= 0.0 and base.max() <= 1.0
        )

    total = None

    for _ in range(smoothgrad_samples if use_smoothing else 1):

        if use_smoothing:

            noise = torch.randn(
                base.shape,
                generator=generator,
                device=base.device,
                dtype=base.dtype,
            )

            x = base + noise * noise_std

            if in_unit_range:
                x = x.clamp(0.0, 1.0)

        else:
            x = base.clone()

        x = x.requires_grad_(True)

        with torch.enable_grad():

            logits = classifier_fn(x)

            scores, _ = _directional_scores(
                logits,
                target_index,
                x.shape[0],
                direction,
            )

            gradient = torch.autograd.grad(
                scores.sum(),
                x,
                retain_graph=False,
                create_graph=False,
            )[0]

        if absolute:
            sample_saliency = gradient.abs().mean(
                dim=1,
                keepdim=True,
            )
        else:
            sample_saliency = gradient.mean(
                dim=1,
                keepdim=True,
            ).relu()

        total = (
            sample_saliency
            if total is None
            else total + sample_saliency
        )

    saliency = total / (
        smoothgrad_samples if use_smoothing else 1
    )

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
    direction: str = "predicted",
    min_cam_strength: float = 0.05,
    return_info: bool = False,
):
    """Compute Grad-CAM saliency maps with degeneracy detection.

    Args:
        direction:
            "predicted" (default), "positive" or "negative".
            See _directional_scores.
        min_cam_strength:
            A CAM is considered degenerate when
                max(relu(raw_cam)) / max(|raw_cam|) < min_cam_strength
            i.e. when almost all of the layer's evidence points the
            wrong way. This is a RELATIVE test, so it works regardless
            of activation scale (unlike an absolute eps).
        return_info:
            If True, also return per-sample diagnostics.

    Returns:
        Saliency maps [B,1,H,W]   (or (saliency, info) if return_info).
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

        # The same tensor is stored and returned to the network. The
        # clone keeps a following in-place op from overwriting the
        # layer's own output. When such an op follows the layer
        # (torchvision DenseNet: norm5 -> ReLU(inplace=True)), the
        # stored activation is therefore expected to be the post-ReLU
        # feature map, the usual Grad-CAM feature map for DenseNet.
        output = output.clone()
        activations.append(output)
        return output

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:
        x = images.detach().clone().requires_grad_(True)

        with torch.enable_grad():

            logits = classifier_fn(x)

            # The hook is only needed for this single forward pass.
            # Removing it now keeps the fallback below from running
            # the hook again.
            handle.remove()

            if len(activations) != 1:
                raise RuntimeError(
                    "Expected target layer to execute once, "
                    f"observed {len(activations)} executions."
                )

            feature_maps = activations[0]

            scores, sign = _directional_scores(
                logits,
                target_index,
                x.shape[0],
                direction,
            )

            gradients = torch.autograd.grad(
                scores.sum(),
                feature_maps,
                retain_graph=False,
                create_graph=False,
            )[0]

        feature_maps = feature_maps.detach()
        gradients = gradients.detach()

        # ==========================================
        # 1. COMPUTE STANDARD GRAD-CAM (raw, pre-ReLU)
        # ==========================================

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        raw_cam = (
            weights * feature_maps
        ).sum(dim=1, keepdim=True)

        # ==========================================
        # 2. VALIDITY ON RAW CAM (relative test)
        # ==========================================

        raw_flat = raw_cam.flatten(1)

        pos_max = raw_flat.clamp_min(0.0).max(dim=1).values
        abs_max = raw_flat.abs().max(dim=1).values

        strength = pos_max / abs_max.clamp_min(eps)

        valid = (
            torch.isfinite(raw_flat).all(dim=1)
            & (abs_max > eps)
            & (strength >= min_cam_strength)
        )

        invalid_indices = (
            (~valid).nonzero(as_tuple=True)[0]
        )

        # ==========================================
        # 3. RELU, UPSAMPLE, NORMALIZE
        # ==========================================

        cam = F.relu(raw_cam)

        cam = F.interpolate(
            cam,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

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
                "[Grad-CAM WARNING] Degenerate CAM for batch "
                f"indices: {invalid_indices.tolist()}"
            )
            print(
                "[Grad-CAM DEBUG] CAM strength (pos_max/abs_max):",
                [round(v, 4) for v in strength.tolist()],
            )
            print(
                "[Grad-CAM DEBUG] score sign:",
                sign.tolist(),
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
                direction=direction,
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
            saliency[invalid_indices] = fallback_saliency.to(
                saliency.dtype
            )

            print(
                "[M5] Input Gradient fallback used for "
                f"{len(invalid_indices)} sample(s)."
            )

        saliency = saliency.detach()

        if return_info:
            info = {
                "score_sign": sign.detach(),
                "cam_strength": strength.detach(),
                "cam_positive_max": pos_max.detach(),
                "cam_absolute_max": abs_max.detach(),
                "fallback_indices": invalid_indices.tolist(),
            }
            return saliency, info

        return saliency

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

    Optional config attributes (defaults used if absent):
        - saliency_direction: "predicted" | "positive" | "negative"
        - min_cam_strength: float, default 0.05
        - smoothgrad_samples: int, default 1 (SmoothGrad off)
        - smoothgrad_noise: float, default 0.1 (fraction of value range)
        - smoothgrad_seed: int, default 0
    """

    config.validate()

    method = config.saliency_method

    direction = getattr(config, "saliency_direction", "predicted")

    if method == "gradient":

        return compute_gradient_saliency(
            images=images,
            classifier_fn=classifier_fn,
            target_index=config.target_index,
            eps=config.eps,
            direction=direction,
            smoothgrad_samples=getattr(config, "smoothgrad_samples", 1),
            smoothgrad_noise=getattr(config, "smoothgrad_noise", 0.1),
            smoothgrad_seed=getattr(config, "smoothgrad_seed", 0),
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
            direction=direction,
            min_cam_strength=getattr(config, "min_cam_strength", 0.05),
        )

    raise ValueError(
        f"Unsupported saliency method: {method}"
    )