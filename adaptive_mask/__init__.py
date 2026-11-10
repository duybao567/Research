
"""Adaptive mask utilities for ACE research.

This package provides mask generation utilities independently
from ACE's diffusion sampling implementation.

M5 extensions:
    - Grad-CAM saliency
    - Entropy and TV regularization
    - Gaussian mask smoothing

Integration into ACE sampling must only be performed after
verifying ACE's mask convention and inpainting behavior.
"""

from .config import AdaptiveMaskConfig

# Existing gradient-based saliency
from .saliency import (
    compute_gradient_saliency,
    compute_gradcam_saliency,
    compute_saliency,
)
# Adaptive mask generation
from .mask_generator import generate_adaptive_mask

# Mask refinement
from .mask_refinement import refine_mask


__all__ = [
    "AdaptiveMaskConfig",
    "compute_gradient_saliency",
    "generate_adaptive_mask",
    "refine_mask",
]
 