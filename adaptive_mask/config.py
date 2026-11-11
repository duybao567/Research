
"""Configuration for M5 Adaptive Mask generation in ACE."""

from dataclasses import dataclass


@dataclass
class AdaptiveMaskConfig:
    # =====================================================
    # 1. SALIENCY CONFIGURATION
    # =====================================================

    # "gradient" = Input Gradient
    # "gradcam"  = Grad-CAM
    saliency_method: str = "gradcam"

    # Classifier output index.
    # CelebA-HQ Smile classifier: target_index = 0
    target_index: int = 0

    # Saliency direction:
    # "predicted" = explain the classifier's current prediction
    # "positive"  = positive classifier evidence
    # "negative"  = negative classifier evidence
    saliency_direction: str = "predicted"

    # =====================================================
    # 2. ADAPTIVE COVERAGE
    # =====================================================

    min_coverage: float = 0.05
    max_coverage: float = 0.12

    # Controls how entropy affects adaptive coverage.
    # This is NOT entropy regularization strength.
    entropy_coverage_strength: float = 1.0

    # =====================================================
    # 3. MASK GENERATION / DILATION
    # =====================================================

    min_dilation_radius: int = 1
    max_dilation_radius: int = 9
    dilation_area_reference: float = 0.15

    # =====================================================
    # 4. ENTROPY REGULARIZATION
    # =====================================================

    entropy_enabled: bool = True
    entropy_regularization_strength: float = 0.05

    # =====================================================
    # 5. TOTAL VARIATION REGULARIZATION
    # =====================================================

    tv_enabled: bool = True
    tv_regularization_strength: float = 0.02

    # =====================================================
    # 6. GAUSSIAN MASK SMOOTHING
    # =====================================================

    gaussian_smoothing_enabled: bool = True
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0

    # =====================================================
    # 7. MASK REFINEMENT
    # =====================================================

    closing_kernel_size: int = 5
    min_component_area: int = 16

    # =====================================================
    # 8. NUMERICAL STABILITY
    # =====================================================

    eps: float = 1e-8

    def validate(self):
        """Validate all adaptive mask configuration values."""

        # Saliency
        if self.saliency_method not in ("gradient", "gradcam"):
            raise ValueError(
                "saliency_method must be 'gradient' or 'gradcam'."
            )

        if self.target_index < 0:
            raise ValueError("target_index must be >= 0.")

        if self.saliency_direction not in (
            "predicted", "positive", "negative"
        ):
            raise ValueError(
                "saliency_direction must be "
                "'predicted', 'positive', or 'negative'."
            )

        # Coverage
        if not (
            0.0 < self.min_coverage
            <= self.max_coverage
            < 1.0
        ):
            raise ValueError(
                "Require 0 < min_coverage <= max_coverage < 1."
            )

        if not (0.0 <= self.entropy_coverage_strength <= 1.0):
            raise ValueError(
                "entropy_coverage_strength must be in [0, 1]."
            )

        # Dilation
        if self.min_dilation_radius < 0:
            raise ValueError(
                "min_dilation_radius must be >= 0."
            )

        if self.max_dilation_radius < self.min_dilation_radius:
            raise ValueError(
                "max_dilation_radius must be >= min_dilation_radius."
            )

        if not (0.0 < self.dilation_area_reference < 1.0):
            raise ValueError(
                "dilation_area_reference must be in (0, 1)."
            )

        # Entropy regularization
        if self.entropy_regularization_strength < 0:
            raise ValueError(
                "entropy_regularization_strength must be >= 0."
            )

        # TV regularization
        if self.tv_regularization_strength < 0:
            raise ValueError(
                "tv_regularization_strength must be >= 0."
            )

        # Gaussian smoothing
        if self.gaussian_kernel_size < 1:
            raise ValueError(
                "gaussian_kernel_size must be >= 1."
            )

        if self.gaussian_kernel_size % 2 == 0:
            raise ValueError(
                "gaussian_kernel_size must be an odd number."
            )

        if self.gaussian_sigma <= 0:
            raise ValueError(
                "gaussian_sigma must be > 0."
            )

        # Mask refinement
        if self.closing_kernel_size < 1:
            raise ValueError(
                "closing_kernel_size must be >= 1."
            )

        if self.min_component_area < 1:
            raise ValueError(
                "min_component_area must be >= 1."
            )

        # Numerical stability
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")

        return self
