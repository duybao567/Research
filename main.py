import os
import yaml
import copy
import random
import argparse
import numpy as np
import os.path as osp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from time import time

import torch
from torch.utils import data

# Diffusion Model imports
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from guided_diffusion.sample_utils import (
    ImageSaver,
    SlowSingleLabel,
    ChunkedDataset,
)
from guided_diffusion.image_datasets import (
    get_dataset,
    BINARYDATASET,
)

# Core imports
from core.utils import print_dict, generate_mask
from core.metrics import accuracy, get_prediction
from core.attacks_and_models import JointClassifierDDPM, get_attack
from models import get_classifier

# M5 Adaptive Mask modules
from adaptive_mask.config import AdaptiveMaskConfig
from adaptive_mask.saliency import compute_saliency
from adaptive_mask.mask_generator import generate_adaptive_mask
from adaptive_mask.mask_refinement import refine_adaptive_mask
from adaptive_mask.visualization import visualize_mask_pipeline


# =======================================================
# Grad-CAM target-layer discovery
# =======================================================

def find_gradcam_target_layer(classifier, layer_name="norm5"):
    """
    Find a DenseNet121 convolutional layer inside
    ACE's DecisionDensenetModel.

    Supported:
        norm5
        transition3
        denseblock4_last
    """

    decision_model = None

    for module in classifier.modules():
        if module.__class__.__name__ == "DecisionDensenetModel":
            decision_model = module
            break

    if decision_model is None:
        raise RuntimeError(
            "Could not find DecisionDensenetModel inside classifier."
        )

    try:
        features = (
            decision_model
            .feat_extract
            .feat_extract
            .features
        )
    except AttributeError as exc:
        raise RuntimeError(
            "Could not access DenseNet121 feature extractor. "
            "Check the classifier architecture."
        ) from exc

    layer_choices = {
        "norm5": features.norm5,
        "transition3": features.transition3.conv,
        "denseblock4_last": (
            features.denseblock4.denselayer16.conv2
        ),
    }

    if layer_name not in layer_choices:
        raise ValueError(
            f"Unsupported Grad-CAM layer: {layer_name}. "
            f"Available: {list(layer_choices.keys())}"
        )

    target_layer = layer_choices[layer_name]

    if not isinstance(target_layer, torch.nn.Module):
        raise TypeError(
            "Grad-CAM target layer must be a torch.nn.Module."
        )

    return target_layer


# =======================================================
# Arguments
# =======================================================

def create_args():

    defaults = dict(
        clip_denoised=True,
        batch_size=1,
        gpu="0",
        save_images=True,
        num_samples=10,
        cudnn_deterministic=False,

        # Paths
        model_path="",
        classifier_path="",
        output_path="results",
        exp_name="exp",

        # Attack
        seed=4,
        attack_method="PGD",
        attack_iterations=50,
        attack_epsilon=255,
        attack_step=1.0,
        attack_joint=True,
        attack_joint_checkpoint=False,
        attack_checkpoint_backward_steps=1,
        attack_joint_shortcut=False,

        # Distance
        dist_l1=0.0,
        dist_l2=0.0,
        dist_schedule="none",

        # ACE sampling
        sampling_time_fraction=0.1,
        sampling_stochastic=True,
        sampling_inpaint=0.15,
        sampling_dilation=15,

        # M5 Adaptive Mask
        adaptive_mask=True,
        adaptive_mask_target_index=0,

        adaptive_mask_saliency_method="gradcam",
        adaptive_mask_saliency_direction="predicted",
        adaptive_mask_gradcam_layer="norm5",

        adaptive_mask_min_coverage=0.05,
        adaptive_mask_max_coverage=0.12,
        adaptive_mask_entropy_strength=1.0,

        adaptive_mask_refinement=True,
        adaptive_mask_refinement_iterations=100,
        adaptive_mask_refinement_lr=0.05,

        adaptive_mask_entropy_weight=0.05,
        adaptive_mask_tv_weight=0.02,
        adaptive_mask_coverage_weight=10.0,

        adaptive_mask_gaussian=True,
        adaptive_mask_gaussian_kernel=5,
        adaptive_mask_gaussian_sigma=1.0,

        # "final": coverage bounds apply AFTER dilation (the mask ACE uses)
        # "raw"  : bounds apply to top-k only; dilation then inflates it
        adaptive_mask_coverage_mode="final",

        # "fixed" (k x k dilation, MaskDiME-like) or "adaptive"
        adaptive_mask_dilation_method="fixed",
        adaptive_mask_dilation_kernel=5,

        # SmoothGrad (1 = plain gradient). Gradient saliency only.
        adaptive_mask_smoothgrad_samples=1,
        adaptive_mask_smoothgrad_noise=0.1,
        adaptive_mask_smoothgrad_seed=0,

        adaptive_mask_save_visualizations=True,
        adaptive_mask_debug_only=False,

        # Labels
        label_query=-1,
        label_target=-1,

        # Dataset
        image_size=256,
        data_dir="",
        dataset="ImageNet",
        chunks=1,
        chunk=0,
        merge_chunks=False,
    )

    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    from guided_diffusion.script_util import add_dict_to_argparser
    add_dict_to_argparser(parser, defaults)

    return parser.parse_args()


# =======================================================
# Sampling filter
# =======================================================

@torch.no_grad()
def filter_fn(
    diffusion,
    attack,
    model,
    shape,
    steps,
    x,
    classifier,
    device,
    stochastic,
    target,
    inpaint,
    dilation,
    adaptive_mask_enabled=False,
    adaptive_mask_config=None,
    adaptive_mask_target_layer=None,
    adaptive_mask_refinement=True,
    adaptive_mask_refinement_iterations=100,
    adaptive_mask_refinement_lr=0.05,
    adaptive_mask_coverage_weight=10.0,
    adaptive_mask_save_dir=None,
    visualization_prefix="sample",
    adaptive_mask_debug_only=False,
):

    indices = list(range(steps))[::-1]

    # ---------------------------------------------------
    # 1. Generate pre-explanation
    # ---------------------------------------------------

    with torch.enable_grad():
        pe = attack.perturb(x, target)

    # ---------------------------------------------------
    # 2. Generate M5 Adaptive Mask
    # ---------------------------------------------------

    if adaptive_mask_enabled:

        if adaptive_mask_config is None:
            raise ValueError(
                "adaptive_mask_config is required."
            )

        if (
            adaptive_mask_config.saliency_method == "gradcam"
            and adaptive_mask_target_layer is None
        ):
            raise ValueError(
                "Grad-CAM target layer is missing."
            )

        with torch.enable_grad():

            saliency = compute_saliency(
                images=x.detach(),
                classifier_fn=classifier,
                config=adaptive_mask_config,
                target_layer=adaptive_mask_target_layer,
            )

            initial_edit_mask, mask_info = generate_adaptive_mask(
                saliency,
                config=adaptive_mask_config,
                return_info=True,
            )

            if adaptive_mask_refinement:

                refined_edit_mask, refinement_info = (
                    refine_adaptive_mask(
                        initial_mask=initial_edit_mask,
                        saliency=saliency,
                        config=adaptive_mask_config,
                        iterations=adaptive_mask_refinement_iterations,
                        learning_rate=adaptive_mask_refinement_lr,
                        coverage_weight=adaptive_mask_coverage_weight,
                        apply_closing=False,
                        remove_small_components=False,
                        return_info=True,
                    )
                )

            else:
                refined_edit_mask = initial_edit_mask
                refinement_info = {}

        # M5 convention:
        # 1 = editable
        # ACE convention:
        # 1 = preserve original

        adaptive_edit_mask = (
            refined_edit_mask.detach().clamp(0, 1)
        )

        boolmask = 1.0 - adaptive_edit_mask

        if inpaint == 0:
            boolmask = torch.zeros_like(boolmask)

        mask = adaptive_edit_mask

        # ------------------------------------------------
        # Save visualizations
        # ------------------------------------------------

        if adaptive_mask_save_dir is not None:

            os.makedirs(
                adaptive_mask_save_dir,
                exist_ok=True,
            )

            for batch_index in range(x.shape[0]):

                image_id = (
                    f"{visualization_prefix}_{batch_index:03d}"
                )

                figure_path = osp.join(
                    adaptive_mask_save_dir,
                    image_id + "_m5_pipeline.png",
                )

                fig = visualize_mask_pipeline(
                    image=x[
                        batch_index:batch_index + 1
                    ].detach(),

                    saliency=saliency[
                        batch_index:batch_index + 1
                    ].detach(),

                    initial_mask=initial_edit_mask[
                        batch_index:batch_index + 1
                    ].detach(),

                    refined_mask=adaptive_edit_mask[
                        batch_index:batch_index + 1
                    ].detach(),

                    save_path=figure_path,
                )

                plt.close(fig)

        print(
            "[M5] target coverage:",
            mask_info.get("target_coverage"),
            "| generated coverage:",
            mask_info.get("final_coverage"),
            "| refined coverage:",
            adaptive_edit_mask.mean(
                dim=(1, 2, 3)
            ).detach().cpu().tolist(),
        )

        # Extra diagnostics: raw (pre-dilation) coverage, dilation radius,
        # and how much of the PGD perturbation survives inside the mask.
        # pe is computed on the whole image, so perturbation outside the
        # mask is replaced by the original pixels and is lost.
        with torch.no_grad():
            pe_delta = (pe.detach() - x.detach()).abs().mean(
                dim=1, keepdim=True
            )
            inside_energy = (
                (pe_delta * adaptive_edit_mask).sum(dim=(1, 2, 3))
                / pe_delta.sum(dim=(1, 2, 3)).clamp_min(1e-12)
            )

        raw_cov = mask_info.get("raw_coverage")

        print(
            "[M5] raw coverage:",
            raw_cov.cpu().tolist()
            if torch.is_tensor(raw_cov) else raw_cov,
            "| dilation radius:",
            mask_info.get("dilation_radius"),
            "| PGD energy inside mask:",
            [round(v, 3) for v in inside_energy.cpu().tolist()],
        )

        if adaptive_mask_debug_only:
            return None, pe, None, mask

    else:

        # Original ACE mask
        mask, dil_mask = generate_mask(
            x, pe, dilation
        )

        boolmask = (dil_mask < inpaint).float()

    # ---------------------------------------------------
    # 3. ACE Diffusion Sampling
    # ---------------------------------------------------

    ce = (pe.detach() - 0.5) / 0.5
    orig = (x.detach() - 0.5) / 0.5

    noise_fn = (
        torch.randn_like
        if stochastic
        else torch.zeros_like
    )

    noise_x = None

    for idx, t in enumerate(indices):

        t = torch.tensor(
            [t] * ce.size(0),
            device=ce.device,
        )

        if idx == 0:

            ce = diffusion.q_sample(
                ce,
                t,
                noise=noise_fn(ce),
            )

            noise_x = ce.clone().detach()

        if inpaint != 0:

            ce = (
                ce * (1 - boolmask)
                + boolmask * diffusion.q_sample(
                    orig,
                    t,
                    noise=noise_fn(ce),
                )
            )

        out = diffusion.p_mean_variance(
            model,
            ce,
            t,
            clip_denoised=True,
        )

        ce = out["mean"]

        if stochastic and idx != (steps - 1):

            noise = torch.randn_like(ce)

            ce += (
                torch.exp(0.5 * out["log_variance"])
                * noise
            )

    ce = (
        ce * (1 - boolmask)
        + boolmask * orig
    )

    ce = (ce * 0.5) + 0.5
    ce = ce.clamp(0, 1)

    noise_x = (
        (noise_x * 0.5) + 0.5
    ).clamp(0, 1)

    return ce, pe, noise_x, mask


# =======================================================
# Main
# =======================================================

def main():

    args = create_args()

    # ---------------------------------------------------
    # Merge chunks
    # ---------------------------------------------------

    if args.merge_chunks:
        raise NotImplementedError(
            "Chunk merging is disabled in this version. "
            "Please run with --merge_chunks=False."
        )

    # ---------------------------------------------------
    # Validate configuration
    # ---------------------------------------------------

    if args.timestep_respacing == "":
        raise ValueError(
            "timestep_respacing must not be empty."
        )

    respaced_steps = int(
        args.sampling_time_fraction
        * int(args.timestep_respacing)
    )

    if respaced_steps <= 0:
        raise ValueError(
            "sampling_time_fraction produces zero sampling steps."
        )

    if args.adaptive_mask:

        if not (
            0 < args.adaptive_mask_min_coverage
            <= args.adaptive_mask_max_coverage
            <= 1
        ):
            raise ValueError(
                "Coverage must satisfy "
                "0 < min <= max <= 1."
            )

    # ---------------------------------------------------
    # Output paths
    # ---------------------------------------------------

    output_root = osp.abspath(
        osp.expanduser(args.output_path)
    )

    if (
        osp.basename(
            osp.normpath(output_root)
        ).lower() == "results"
    ):
        output_root = osp.dirname(
            osp.normpath(output_root)
        )

    args.output_path = output_root

    results_root = osp.join(
        args.output_path,
        "Results",
    )

    experiment_root = osp.join(
        results_root,
        args.exp_name,
    )

    original_correct_root = osp.join(
        args.output_path,
        "Original",
        "Correct",
    )

    original_incorrect_root = osp.join(
        args.output_path,
        "Original",
        "Incorrect",
    )

    print(args)

    print("[ACE output structure]")
    print("Output root:", args.output_path)
    print("Results:", results_root)
    print("Experiment:", experiment_root)

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(experiment_root, exist_ok=True)
    os.makedirs(original_correct_root, exist_ok=True)
    os.makedirs(original_incorrect_root, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ---------------------------------------------------
    # Seeds
    # ---------------------------------------------------

    torch.backends.cudnn.deterministic = (
        args.cudnn_deterministic
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---------------------------------------------------
    # Dataset
    # ---------------------------------------------------

    dataset = get_dataset(args)

    target = -1

    if args.label_target != -1:

        target = (
            1 - args.label_target
            if args.dataset in BINARYDATASET
            else args.label_query
        )

    dataset = SlowSingleLabel(
        target,
        dataset,
        args.num_samples,
    )

    dataset = ChunkedDataset(
        dataset=dataset,
        chunk=args.chunk,
        num_chunks=args.chunks,
    )

    print(
        "Images on the dataset:",
        len(dataset),
    )

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---------------------------------------------------
    # Load diffusion model
    # ---------------------------------------------------

    print("Loading diffusion model...")

    model, respaced_diffusion = create_model_and_diffusion(
        **args_to_dict(
            args,
            model_and_diffusion_defaults().keys(),
        )
    )

    model.load_state_dict(
        dist_util.load_state_dict(
            args.model_path,
            map_location="cpu",
        )
    )

    model.to(dist_util.dev())

    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()

    # ---------------------------------------------------
    # Load classifier
    # ---------------------------------------------------

    print("Loading classifier...")

    classifier = get_classifier(args)

    classifier.to(dist_util.dev()).eval()

    # ---------------------------------------------------
    # Initialize M5
    # ---------------------------------------------------

    adaptive_mask_config = None
    adaptive_mask_target_layer = None
    adaptive_mask_vis_dir = None

    if args.adaptive_mask:

        adaptive_mask_config = AdaptiveMaskConfig(
            saliency_method=args.adaptive_mask_saliency_method,
            target_index=args.adaptive_mask_target_index,
            saliency_direction=args.adaptive_mask_saliency_direction,

            min_coverage=args.adaptive_mask_min_coverage,
            max_coverage=args.adaptive_mask_max_coverage,
            entropy_coverage_strength=args.adaptive_mask_entropy_strength,

            tv_regularization_strength=args.adaptive_mask_tv_weight,
            entropy_regularization_strength=args.adaptive_mask_entropy_weight,

            gaussian_smoothing_enabled=args.adaptive_mask_gaussian,
            gaussian_kernel_size=args.adaptive_mask_gaussian_kernel,
            gaussian_sigma=args.adaptive_mask_gaussian_sigma,

            coverage_mode=args.adaptive_mask_coverage_mode,

            dilation_method=args.adaptive_mask_dilation_method,
            dilation_kernel_size=args.adaptive_mask_dilation_kernel,

            smoothgrad_samples=args.adaptive_mask_smoothgrad_samples,
            smoothgrad_noise=args.adaptive_mask_smoothgrad_noise,
            smoothgrad_seed=args.adaptive_mask_smoothgrad_seed,
        ).validate()

        if adaptive_mask_config.saliency_method == "gradcam":

            adaptive_mask_target_layer = (
                find_gradcam_target_layer(
                    classifier,
                    layer_name=args.adaptive_mask_gradcam_layer,
                )
            )

        if args.adaptive_mask_save_visualizations:

            adaptive_mask_vis_dir = osp.join(
                experiment_root,
                "m5_visualizations",
            )

            os.makedirs(
                adaptive_mask_vis_dir,
                exist_ok=True,
            )

        print(
            "[M5 Adaptive Mask] ENABLED",
            "| method:", args.adaptive_mask_saliency_method,
            "| direction:", args.adaptive_mask_saliency_direction,
            "| target index:", args.adaptive_mask_target_index,
            "| Grad-CAM layer:", args.adaptive_mask_gradcam_layer,
            "| coverage:",
            (
                args.adaptive_mask_min_coverage,
                args.adaptive_mask_max_coverage,
            ),
            "| coverage mode:", args.adaptive_mask_coverage_mode,
            "| dilation:",
            (
                args.adaptive_mask_dilation_method,
                args.adaptive_mask_dilation_kernel,
            ),
            "| SmoothGrad:",
            (
                args.adaptive_mask_smoothgrad_samples,
                args.adaptive_mask_smoothgrad_noise,
                args.adaptive_mask_smoothgrad_seed,
            ),
        )

    else:

        print(
            "[M5 Adaptive Mask] DISABLED "
            "(Original ACE mask)"
        )

    # ---------------------------------------------------
    # Joint classifier
    # ---------------------------------------------------

    if args.attack_joint and not (
        args.attack_joint_checkpoint
        or args.attack_joint_shortcut
    ):

        joint_classifier = JointClassifierDDPM(
            classifier=classifier,
            ddpm=model,
            diffusion=respaced_diffusion,
            steps=respaced_steps,
            stochastic=args.sampling_stochastic,
        )

        joint_classifier.eval()

    else:

        joint_classifier = None

    # ---------------------------------------------------
    # Distance function
    # ---------------------------------------------------

    def get_dist_fn():

        any_loss = False
        l1_loss = None
        l2_loss = None

        if args.dist_l2 != 0.0:

            l2_loss = lambda x, x_adv: (
                args.dist_l2
                * torch.linalg.norm(
                    (x - x_adv).view(x.size(0), -1),
                    dim=1,
                ).sum()
            )

            any_loss = True

        if args.dist_l1 != 0.0:

            l1_loss = lambda x, x_adv: (
                args.dist_l1 * (x - x_adv).abs().sum()
            )

            any_loss = True

        if not any_loss:
            return None

        def dist_fn(x, x_adv):

            loss = 0

            if l1_loss is not None:
                loss += l1_loss(x, x_adv)

            if l2_loss is not None:
                loss += l2_loss(x, x_adv)

            return loss

        return dist_fn

    dist_fn = get_dist_fn()

    # ---------------------------------------------------
    # Attack
    # ---------------------------------------------------

    main_args = {
        "predict": (
            joint_classifier
            if args.attack_joint
            and not (
                args.attack_joint_checkpoint
                or args.attack_joint_shortcut
            )
            else classifier
        ),

        "loss_fn": None,
        "dist_fn": dist_fn,
        "eps": args.attack_epsilon / 255,
        "nb_iter": args.attack_iterations,
        "dist_schedule": args.dist_schedule,
        "binary": args.dataset in BINARYDATASET,
        "step": args.attack_step / 255,
    }

    attack = get_attack(
        args.attack_method,
        args.attack_joint and args.attack_joint_checkpoint,
        args.attack_joint and args.attack_joint_shortcut,
    )

    if args.attack_joint and (
        args.attack_joint_checkpoint
        or args.attack_joint_shortcut
    ):

        attack = attack(
            diffusion=respaced_diffusion,
            ddpm=model,
            steps=respaced_steps,
            stochastic=args.sampling_stochastic,
            backward_steps=args.attack_checkpoint_backward_steps,
            **main_args,
        )

    else:

        attack = attack(**main_args)

    # ---------------------------------------------------
    # Output setup
    # ---------------------------------------------------

    start_time = time()

    save_imgs = {
        "pre-explanation": (
            ImageSaver(
                args.output_path,
                osp.join(args.exp_name, "pre-explanation"),
            )
            if args.save_images
            else None
        ),

        "explanation": (
            ImageSaver(
                args.output_path,
                osp.join(args.exp_name, "explanation"),
            )
            if args.save_images
            else None
        ),
    }

    stats_template = {
        "cf": 0,
        "cf5": 0,
        "untargeted": 0,
        "untargeted5": 0,
        "l1": 0,
        "l inf": 0,
    }

    stats = {
        "n": 0,
        "clean acc": 0,
        "clean acc5": 0,
        "explanation": copy.deepcopy(stats_template),
        "pre-explanation": copy.deepcopy(stats_template),
    }

    # ---------------------------------------------------
    # Generation loop
    # ---------------------------------------------------

    print("Starting Image Generation")

    for idx, (indexes, img, lab) in enumerate(loader):

        print(
            f"[Chunks ({args.chunk}+1) / {args.chunks}] "
            f"{idx} / {len(loader)} "
            f"| Time: {int(time() - start_time)}s"
        )

        img = img.to(dist_util.dev())

        lab = lab.to(
            dist_util.dev(),
            dtype=(
                torch.float
                if args.dataset in BINARYDATASET
                else torch.long
            ),
        )

        # Initial classification
        c_log, c_pred = get_prediction(
            classifier,
            img,
            args.dataset in BINARYDATASET,
        )

        # Construct target
        target = None

        if args.label_target != -1:

            target = torch.ones_like(lab) * args.label_target
            target[lab != c_pred] = lab[lab != c_pred]

        elif args.dataset in BINARYDATASET:

            target = 1 - c_pred
            target[lab != c_pred] = lab[lab != c_pred]

        acc1, acc5 = accuracy(
            c_log,
            lab,
            binary=args.dataset in BINARYDATASET,
        )

        stats["clean acc"] += acc1.sum().item()
        stats["clean acc5"] += acc5.sum().item()
        stats["n"] += lab.size(0)

        # ------------------------------------------------
        # Generate explanation
        # ------------------------------------------------

        ce, pe, noise, explanation_mask = filter_fn(
            diffusion=respaced_diffusion,
            attack=attack,
            model=model,
            shape=img.shape,
            steps=respaced_steps,
            x=img,
            classifier=classifier,
            device=dist_util.dev(),
            stochastic=args.sampling_stochastic,
            target=target,
            inpaint=args.sampling_inpaint,
            dilation=args.sampling_dilation,

            adaptive_mask_enabled=args.adaptive_mask,
            adaptive_mask_config=adaptive_mask_config,
            adaptive_mask_target_layer=adaptive_mask_target_layer,

            adaptive_mask_refinement=args.adaptive_mask_refinement,
            adaptive_mask_refinement_iterations=args.adaptive_mask_refinement_iterations,
            adaptive_mask_refinement_lr=args.adaptive_mask_refinement_lr,
            adaptive_mask_coverage_weight=args.adaptive_mask_coverage_weight,

            adaptive_mask_save_dir=adaptive_mask_vis_dir,
            visualization_prefix=f"batch_{idx:05d}",
            adaptive_mask_debug_only=args.adaptive_mask_debug_only,
        )

        if args.adaptive_mask_debug_only:

            print(
                "[M5 Debug] Mask generation completed. "
                "Skipping diffusion and evaluation."
            )

            continue

        # ------------------------------------------------
        # Convert outputs
        # ------------------------------------------------

        noise = (
            (noise * 255)
            .to(torch.uint8)
            .detach()
            .cpu()
        )

        if args.adaptive_mask:

            # M5 mask: 1 = editable.
            ce_mask = explanation_mask.detach().clone()

        else:

            ce_mask = generate_mask(
                img,
                ce,
                1,
            )[0]

        pe_mask = (
            (explanation_mask * 255)
            .to(torch.uint8)
            .detach()
            .cpu()
        )

        ce_mask = (
            (ce_mask * 255)
            .to(torch.uint8)
            .detach()
            .cpu()
        )

        # ------------------------------------------------
        # Evaluate counterfactuals
        # ------------------------------------------------

        with torch.no_grad():

            for data_type, data_img, data_mask in zip(
                ["pre-explanation", "explanation"],
                [pe, ce],
                [pe_mask, ce_mask],
            ):

                data_log, data_pred = get_prediction(
                    classifier,
                    data_img,
                    binary=args.dataset in BINARYDATASET,
                )

                cf, cf5 = accuracy(
                    data_log,
                    target,
                    binary=args.dataset in BINARYDATASET,
                )

                un, un5 = accuracy(
                    data_log,
                    c_pred,
                    binary=args.dataset in BINARYDATASET,
                )

                stats[data_type]["cf"] += cf.sum().item()
                stats[data_type]["cf5"] += cf5.sum().item()

                stats[data_type]["untargeted"] += (
                    un.size(0) - un.sum().item()
                )

                stats[data_type]["untargeted5"] += (
                    un5.size(0) - un5.sum().item()
                )

                l1 = (
                    (img - data_img)
                    .abs()
                    .view(img.size(0), -1)
                    .mean(dim=1)
                    .detach()
                    .cpu()
                )

                linf = (
                    (img - data_img)
                    .abs()
                    .view(img.size(0), -1)
                    .max(dim=1)[0]
                    .detach()
                    .cpu()
                )

                stats[data_type]["l1"] += l1.sum().item()
                stats[data_type]["l inf"] += linf.sum().item()

                img255 = (
                    (img * 255)
                    .to(torch.uint8)
                    .detach()
                    .cpu()
                )

                data_img_uint8 = (
                    (data_img * 255)
                    .to(torch.uint8)
                    .detach()
                    .cpu()
                )

                if args.save_images:

                    save_imgs[data_type](
                        imgs=img255.permute(
                            0, 2, 3, 1
                        ).numpy(),

                        cfs=data_img_uint8.permute(
                            0, 2, 3, 1
                        ).numpy(),

                        noises=noise.permute(
                            0, 2, 3, 1
                        ).numpy(),

                        target=(
                            target
                            if target is not None
                            else lab
                        ),

                        label=lab,
                        pred=c_pred,
                        pred_cf=data_pred,
                        l_inf=linf,
                        l_1=l1,

                        indexes=indexes.numpy(),

                        masks=data_mask.permute(
                            0, 2, 3, 1
                        ).squeeze(-1).numpy(),
                    )

        # ------------------------------------------------
        # Progress
        # ------------------------------------------------

        if (idx + 1) % 50 == 0:

            print("=" * 50)
            print("Current Stats at iteration", idx + 1)
            print_dict(stats)
            print("=" * 50)

        if (idx + 1) == len(loader):

            print(
                f"[Chunks ({args.chunk}+1) / {args.chunks}] "
                f"{idx + 1} / {len(loader)} "
                f"| Time: {int(time() - start_time)}s"
            )

            print("Done")
            break

    # ---------------------------------------------------
    # Final statistics
    # ---------------------------------------------------

    if stats["n"] == 0:

        if args.adaptive_mask_debug_only:
            print(
                "[M5 Debug] No evaluation performed. "
                "Mask visualizations, if generated, are saved."
            )
            return

        raise RuntimeError(
            "No images were processed. "
            "Check dataset and num_samples."
        )

    for data_type in [
        "pre-explanation",
        "explanation",
    ]:

        for key in stats[data_type]:
            stats[data_type][key] /= stats["n"]

    stats["clean acc"] /= stats["n"]
    stats["clean acc5"] /= stats["n"]

    if args.chunks == 1:

        print("=" * 50)
        print("Results:")
        print_dict(stats)
        print("=" * 50)

    prefix = (
        ""
        if args.chunks == 1
        else f"c-{args.chunk}_{args.chunks}-"
    )

    summary_path = osp.join(
        args.output_path,
        "Results",
        args.exp_name,
        prefix + "summary.yaml",
    )

    os.makedirs(
        osp.dirname(summary_path),
        exist_ok=True,
    )

    with open(summary_path, "w") as f:

        yaml.safe_dump(
            stats,
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    print(
        "[ACE] Summary saved to:",
        summary_path,
    )


if __name__ == "__main__":
    main()