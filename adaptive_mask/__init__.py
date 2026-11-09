"""Adaptive mask utilities for ACE research.

This package is intentionally independent of ACE's sampling code.
Integration must be done only after checking ACE's mask convention.
"""
from .config import AdaptiveMaskConfig
from .saliency import compute_gradient_saliency
from .mask_generator import generate_adaptive_mask
from .mask_refinement import refine_mask

__all__ = [
    "AdaptiveMaskConfig",
    "compute_gradient_saliency",
    "generate_adaptive_mask",
    "refine_mask",
]
