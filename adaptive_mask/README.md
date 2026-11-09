# Adaptive Mask module (initial research implementation)

This package is a standalone starting point for experimenting with saliency-driven
mask generation. It is **not yet integrated with ACE**.

## Requirements

- PyTorch
- matplotlib
- scipy (only for connected-component cleanup)

```bash
pip install torch matplotlib scipy
```

## Minimal usage

```python
import torch
from adaptive_mask import (
    AdaptiveMaskConfig,
    compute_gradient_saliency,
    generate_adaptive_mask,
    refine_mask,
)
from adaptive_mask.visualization import visualize_mask_pipeline

# images: [B, 3, H, W], float, in [0, 1]
# model: your trained attribute classifier
# Apply the exact classifier normalization inside this wrapper if required.
def classifier_fn(x):
    return model(x)

saliency = compute_gradient_saliency(
    images, classifier_fn, target_index=1
)

config = AdaptiveMaskConfig()
mask, info = generate_adaptive_mask(
    saliency, config=config, return_info=True
)
mask_refined = refine_mask(
    mask,
    closing_kernel_size=config.closing_kernel_size,
    min_component_area=config.min_component_area,
)

visualize_mask_pipeline(images, saliency, mask, mask_refined)
```

## Important integration notes

1. `target_index` must be the classifier's actual output index, not the dataset
   attribute ID. For the user's STEEX CelebAMask-HQ classifier, verify the
   `label_query` mapping and preprocessing in `get_classifier`.
2. The classifier wrapper must apply the same input normalization used in
   training. Do not assume the classifier accepts raw `[0,1]` images.
3. This implementation returns `1` for selected salient pixels. ACE may use
   the opposite mask convention. Inspect ACE's inpainting/sampling code before
   connecting this output to `sampling_inpaint`.
4. Hyperparameters are initial hypotheses, not validated optimal values.
   Tune them on a validation subset and compare against the original fixed
   threshold/dilation baseline.
5. Gradient saliency and connected-component cleanup are research baselines.
   Evaluate localization, counterfactual success, perceptual quality, runtime,
   and stability across seeds before claiming improvement.
