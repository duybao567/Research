"""Configuration for M5 Adaptive Gradient Mask generation in ACE.

Supports:
- Input Gradient / Grad-CAM saliency (optional SmoothGrad averaging)
- Adaptive coverage (entropy-aware)
- Coverage accounting: "final" (cap AFTER dilation) or "raw" (cap on top-k only)
- Multi-region Top-K
- Fixed 5x5 morphological dilation (MaskDiME-inspired)
- Entropy and TV regularization
- Adaptive Gaussian smoothing
- Optional mask refinement

Field order note: fields added in this revision (section 9) are appended
after the original ones, so existing positional construction still works.
"""

from dataclasses import dataclass


@dataclass
class AdaptiveMaskConfig:

    # =====================================================
    # 1. SALIENCY CONFIGURATION
    # =====================================================

    saliency_method: str = "gradient"

    target_index: int = 0

    saliency_direction: str = "predicted"

    # =====================================================
    # 2. ADAPTIVE COVERAGE
    # =====================================================

    adaptive_coverage_enabled: bool = True

    min_coverage: float = 0.05
    max_coverage: float = 0.12

    saliency_mass_ratio: float = 0.80

    entropy_coverage_strength: float = 1.0

    # =====================================================
    # 3. MULTI-REGION TOP-K
    # =====================================================

    multi_region_enabled: bool = False

    multi_region_grid_size: int = 4

    multi_region_min_regions: int = 2
    multi_region_max_regions: int = 8

    multi_region_topk_ratio: float = 0.10

    # =====================================================
    # 4. MASK GENERATION / DILATION
    # =====================================================

    # "fixed"    = Fixed morphological dilation
    # "adaptive" = Adaptive radius dilation
    dilation_method: str = "fixed"

    # MaskDiME-inspired fixed kernel size
    dilation_kernel_size: int = 5

    # Retained for optional adaptive dilation experiments
    adaptive_dilation_enabled: bool = False

    min_dilation_radius: int = 1
    max_dilation_radius: int = 9

    dilation_area_reference: float = 0.15

    # =====================================================
    # 5. ENTROPY REGULARIZATION
    # =====================================================

    entropy_enabled: bool = True
    entropy_regularization_strength: float = 0.05

    # =====================================================
    # 6. TOTAL VARIATION REGULARIZATION
    # =====================================================

    tv_enabled: bool = True
    tv_regularization_strength: float = 0.02

    # =====================================================
    # 7. GAUSSIAN MASK SMOOTHING
    # =====================================================

    gaussian_smoothing_enabled: bool = True

    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0

    adaptive_gaussian_enabled: bool = False

    gaussian_min_sigma: float = 0.5
    gaussian_max_sigma: float = 1.2

    # =====================================================
    # 8. MASK REFINEMENT
    # =====================================================

    mask_refinement_enabled: bool = False

    closing_kernel_size: int = 5
    min_component_area: int = 16

    # =====================================================
    # 9. NUMERICAL STABILITY
    # =====================================================

    eps: float = 1e-8

    # =====================================================
    # 10. COVERAGE MODE  (added)
    # =====================================================

    # "final": min/max_coverage bound the mask AFTER dilation (the mask
    #          ACE actually uses). Top-k is shrunk by bisection until the
    #          dilated mask fits.
    # "raw"  : bounds apply to the top-k mask only, dilation then inflates
    #          it (old behaviour; ~3x larger in practice, like MaskDiME's
    #          top-k + 5x5 dilation).
    coverage_mode: str = "final"

    # =====================================================
    # 11. SALIENCY NOISE REDUCTION / CAM  (added)
    # =====================================================

    # SmoothGrad: average the gradient saliency over this many noisy
    # copies of the input. 1 = off (plain gradient).
    smoothgrad_samples: int = 1

    # Noise std as a fraction of each image's (max - min).
    smoothgrad_noise: float = 0.1

    # Seed of the private generator used for the SmoothGrad noise.
    smoothgrad_seed: int = 0

    # Grad-CAM only: minimum CAM strength (relative) to accept a map.
    min_cam_strength: float = 0.05

    # =====================================================
    # 12. VALIDATION
    # =====================================================

    def validate(self) -> "AdaptiveMaskConfig":

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

        if not (0.0 < self.saliency_mass_ratio <= 1.0):
            raise ValueError(
                "saliency_mass_ratio must be in (0, 1]."
            )

        if not (0.0 <= self.entropy_coverage_strength <= 1.0):
            raise ValueError(
                "entropy_coverage_strength must be in [0, 1]."
            )

        if self.coverage_mode not in ("final", "raw"):
            raise ValueError(
                "coverage_mode must be 'final' or 'raw'."
            )

        # Multi-region
        if self.multi_region_grid_size < 1:
            raise ValueError(
                "multi_region_grid_size must be >= 1."
            )

        if self.multi_region_min_regions < 1:
            raise ValueError(
                "multi_region_min_regions must be >= 1."
            )

        if self.multi_region_max_regions < self.multi_region_min_regions:
            raise ValueError(
                "multi_region_max_regions must be >= "
                "multi_region_min_regions."
            )

        if self.multi_region_max_regions > self.multi_region_grid_size ** 2:
            raise ValueError(
                "multi_region_max_regions cannot exceed "
                "grid_size squared."
            )

        if not (0.0 < self.multi_region_topk_ratio <= 1.0):
            raise ValueError(
                "multi_region_topk_ratio must be in (0, 1]."
            )

        # Dilation
        if self.dilation_method not in ("fixed", "adaptive"):
            raise ValueError(
                "dilation_method must be 'fixed' or 'adaptive'."
            )

        if self.dilation_kernel_size < 1:
            raise ValueError(
                "dilation_kernel_size must be >= 1."
            )

        if self.dilation_kernel_size % 2 == 0:
            raise ValueError(
                "dilation_kernel_size must be odd."
            )

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

        # Entropy
        if self.entropy_regularization_strength < 0:
            raise ValueError(
                "entropy_regularization_strength must be >= 0."
            )

        # TV
        if self.tv_regularization_strength < 0:
            raise ValueError(
                "tv_regularization_strength must be >= 0."
            )

        # Gaussian
        if self.gaussian_kernel_size < 1:
            raise ValueError(
                "gaussian_kernel_size must be >= 1."
            )

        if self.gaussian_kernel_size % 2 == 0:
            raise ValueError(
                "gaussian_kernel_size must be odd."
            )

        if self.gaussian_sigma <= 0:
            raise ValueError(
                "gaussian_sigma must be > 0."
            )

        if self.gaussian_min_sigma <= 0:
            raise ValueError(
                "gaussian_min_sigma must be > 0."
            )

        if self.gaussian_max_sigma < self.gaussian_min_sigma:
            raise ValueError(
                "gaussian_max_sigma must be >= gaussian_min_sigma."
            )

        # Refinement
        if self.closing_kernel_size < 1:
            raise ValueError(
                "closing_kernel_size must be >= 1."
            )

        if self.closing_kernel_size % 2 == 0:
            raise ValueError(
                "closing_kernel_size must be odd."
            )

        if self.min_component_area < 1:
            raise ValueError(
                "min_component_area must be >= 1."
            )

        # SmoothGrad / CAM
        if self.smoothgrad_samples < 1:
            raise ValueError(
                "smoothgrad_samples must be >= 1."
            )

        if self.smoothgrad_noise < 0:
            raise ValueError(
                "smoothgrad_noise must be >= 0."
            )

        if not (0.0 <= self.min_cam_strength <= 1.0):
            raise ValueError(
                "min_cam_strength must be in [0, 1]."
            )

        # Numerical stability
        if self.eps <= 0:
            raise ValueError("eps must be > 0.")

        return self