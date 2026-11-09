"""Configuration for adaptive mask generation."""
from dataclasses import dataclass


@dataclass
class AdaptiveMaskConfig:
    # Fraction of pixels initially selected by the saliency quantile.
    min_coverage: float = 0.05
    max_coverage: float = 0.35

    # Entropy controls the selected coverage: diffuse saliency -> larger area.
    entropy_coverage_strength: float = 0.20

    # Adaptive dilation radius, in pixels.
    min_dilation_radius: int = 1
    max_dilation_radius: int = 9
    dilation_area_reference: float = 0.15

    # Refinement.
    closing_kernel_size: int = 5
    min_component_area: int = 16

    # Numerical stability.
    eps: float = 1e-8

    def validate(self):
        if not (0.0 < self.min_coverage <= self.max_coverage < 1.0):
            raise ValueError("Require 0 < min_coverage <= max_coverage < 1.")
        if self.min_dilation_radius < 0:
            raise ValueError("min_dilation_radius must be >= 0.")
        if self.max_dilation_radius < self.min_dilation_radius:
            raise ValueError("max_dilation_radius must be >= min_dilation_radius.")
        if self.closing_kernel_size < 1:
            raise ValueError("closing_kernel_size must be >= 1.")
        if self.min_component_area < 1:
            raise ValueError("min_component_area must be >= 1.")
