import os
import yaml
import copy
import math
import random
import argparse
import itertools
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from PIL import Image
from time import time
from os import path as osp
from multiprocessing import Pool

import torch
from torch.utils import data
from torch.nn import functional as F
from torchvision import transforms
from torchvision import datasets

# Diffusion Model imports
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
    create_classifier,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.sample_utils import (
    get_DiME_iterative_sampling,
    clean_class_cond_fn,
    dist_cond_fn,
    ImageSaver,
    SlowSingleLabel,
    load_from_DDP_model,
    ChunkedDataset,
)
from guided_diffusion.gaussian_diffusion import _extract_into_tensor
from guided_diffusion.image_datasets import get_dataset, BINARYDATASET, MULTICLASSDATASETS

# Core imports
from core.utils import print_dict, merge_all_chunks, generate_mask
from core.metrics import accuracy, get_prediction
from core.attacks_and_models import JointClassifierDDPM, get_attack
from models import get_classifier

import matplotlib
matplotlib.use('Agg')


# =======================================================
# Adaptive Mask implementation (kept in this file)
# =======================================================

def compute_gradient_saliency(
    images,
    classifier,
    target_index=0,
    eps=1e-8,
):
    """
    Compute input-gradient saliency for ACE classifier.

    Supports:
        - Multi-output classifier: [B, num_classes]
        - Binary single-output classifier: [B] or [B, 1]

    Returns:
        Normalized saliency map [B, 1, H, W] in [0, 1].
    """
    if images.ndim != 4:
        raise ValueError(
            f"Expected images [B,C,H,W], got {tuple(images.shape)}"
        )

    with torch.enable_grad():
        x = images.detach().requires_grad_(True)
        output = classifier(x)

        if isinstance(output, (tuple, list)):
            output = output[0]

        if isinstance(output, dict):
            raise TypeError(
                "Classifier returned a dict; expected a differentiable tensor."
            )

        if not torch.is_tensor(output):
            raise TypeError(
                f"Classifier must return a tensor, got {type(output)}"
            )

        # Convert single-output binary classifier to [B, 1].
        if output.ndim == 1:
            if output.shape[0] != images.shape[0]:
                raise ValueError(
                    f"Output batch size {output.shape[0]} "
                    f"does not match input batch size {images.shape[0]}."
                )
            output = output.unsqueeze(1)

        elif output.ndim == 0 and images.shape[0] == 1:
            output = output.reshape(1, 1)

        if output.ndim != 2:
            raise ValueError(
                f"Expected classifier output [B, num_outputs], "
                f"got {tuple(output.shape)}"
            )

        if output.shape[0] != images.shape[0]:
            raise ValueError(
                f"Output batch size {output.shape[0]} "
                f"does not match input batch size {images.shape[0]}."
            )

        num_outputs = output.shape[1]

        if not (0 <= target_index < num_outputs):
            raise ValueError(
                f"adaptive_mask_target_index={target_index} is invalid. "
                f"Classifier has {num_outputs} output(s); "
                f"valid index: 0 to {num_outputs - 1}."
            )

        score = output[:, target_index].sum()

        if not score.requires_grad:
            raise RuntimeError(
                "Classifier output has no gradient. "
                "Check whether the classifier uses detach(), "
                "no_grad(), or non-differentiable predictions."
            )

        gradients = torch.autograd.grad(
            score,
            x,
            retain_graph=False,
            create_graph=False,
        )[0]

    # Channel-averaged absolute input gradients.
    saliency = gradients.abs().mean(dim=1, keepdim=True)

    # Normalize each image independently.
    flat = saliency.flatten(start_dim=1)

    min_values = flat.min(dim=1)[0].view(-1, 1, 1, 1)
    max_values = flat.max(dim=1)[0].view(-1, 1, 1, 1)

    saliency = (
        saliency - min_values
    ) / (
        max_values - min_values + eps
    )

    return saliency.detach()


def generate_adaptive_mask(
    saliency,
    min_coverage=0.05,
    max_coverage=0.35,
    entropy_coverage_strength=0.20,
):
    """
    Select high-saliency pixels using an image-adaptive coverage ratio.

    Returns a binary mask [B,1,H,W]. A value of 1 marks salient/editable
    pixels. The ACE preservation mask is its inverse.
    """
    if saliency.ndim != 4 or saliency.shape[1] != 1:
        raise ValueError(
            f"Expected saliency [B,1,H,W], got {tuple(saliency.shape)}"
        )
    if not (0.0 < min_coverage <= max_coverage <= 1.0):
        raise ValueError("Require 0 < min_coverage <= max_coverage <= 1.")

    batch_size = saliency.shape[0]
    flat = saliency.flatten(start_dim=1).clamp(0, 1)
    eps = 1e-8

    # Normalized entropy: diffuse saliency -> larger selected area;
    # concentrated saliency -> smaller selected area.
    probabilities = flat + eps
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    entropy = -(probabilities * probabilities.log()).sum(dim=1)
    max_entropy = math.log(max(flat.shape[1], 2))
    normalized_entropy = (entropy / max_entropy).clamp(0, 1)

    coverage = min_coverage + (
        entropy_coverage_strength * normalized_entropy
    )
    coverage = coverage.clamp(min_coverage, max_coverage)

    masks = []
    num_pixels = flat.shape[1]
    for i in range(batch_size):
        k = max(1, min(num_pixels, int(round(float(coverage[i]) * num_pixels))))
        indices = torch.topk(flat[i], k=k, largest=True, sorted=False).indices
        mask_flat = torch.zeros_like(flat[i])
        mask_flat[indices] = 1.0
        masks.append(mask_flat.view(1, *saliency.shape[-2:]))

    return torch.stack(masks, dim=0).to(
        device=saliency.device, dtype=saliency.dtype
    )


# =======================================================
# Arguments
# =======================================================

def create_args():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        gpu='0',
        save_images=True,
        num_samples=500000000000,
        cudnn_deterministic=False,

        # path args
        model_path='',
        classifier_path='',
        output_path='results',
        exp_name='exp',

        # attack args
        seed=4,
        attack_method='PGD',
        attack_iterations=50,
        attack_epsilon=255,
        attack_step=1.0,
        attack_joint=True,
        attack_joint_checkpoint=False,
        attack_checkpoint_backward_steps=1,
        attack_joint_shortcut=False,

        # distance args
        dist_l1=0.0,
        dist_l2=0.0,
        dist_schedule='none',

        # filtering args
        sampling_time_fraction=0.1,
        sampling_stochastic=True,

        # post processing (ACE baseline)
        sampling_inpaint=0.15,
        sampling_dilation=15,

        # Adaptive Mask
        adaptive_mask=False,
        adaptive_mask_target_index=1,
        adaptive_mask_min_coverage=0.05,
        adaptive_mask_max_coverage=0.35,
        adaptive_mask_entropy_strength=0.20,

        # query and target label
        label_query=-1,
        label_target=-1,

        # dataset
        image_size=256,
        data_dir='',
        dataset='ImageNet',
        chunks=1,
        chunk=0,
        merge_chunks=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
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
    adaptive_mask_target_index=1,
    adaptive_mask_min_coverage=0.05,
    adaptive_mask_max_coverage=0.35,
    adaptive_mask_entropy_strength=0.20,
):
    indices = list(range(steps))[::-1]

    # Generate pre-explanation (PE).
    with torch.enable_grad():
        pe = attack.perturb(x, target)

    # Generate preservation mask.
    if adaptive_mask_enabled:
        # High-saliency pixels are treated as editable regions.
        saliency = compute_gradient_saliency(
            x.detach(),
            classifier,
            target_index=adaptive_mask_target_index,
        )
        adaptive_edit_mask = generate_adaptive_mask(
            saliency,
            min_coverage=adaptive_mask_min_coverage,
            max_coverage=adaptive_mask_max_coverage,
            entropy_coverage_strength=adaptive_mask_entropy_strength,
        )

        # In ACE, boolmask=1 is the region restored from the original image.
        # Thus preserve the complement of the selected editable region.
        boolmask = 1.0 - adaptive_edit_mask
        if inpaint == 0:
            boolmask = torch.zeros_like(boolmask)

        # Save the selected editable region as the PE mask visualization.
        mask = adaptive_edit_mask
    else:
        # Original ACE implementation, kept unchanged.
        mask, dil_mask = generate_mask(x, pe, dilation)
        boolmask = (dil_mask < inpaint).float()

    ce = (pe.detach() - 0.5) / 0.5
    orig = (x.detach() - 0.5) / 0.5
    noise_fn = torch.randn_like if stochastic else torch.zeros_like

    noise_x = None

    for idx, t in enumerate(indices):
        t = torch.tensor([t] * ce.size(0), device=ce.device)

        if idx == 0:
            ce = diffusion.q_sample(ce, t, noise=noise_fn(ce))
            noise_x = ce.clone().detach()

        if inpaint != 0:
            ce = (
                ce * (1 - boolmask)
                + boolmask * diffusion.q_sample(
                    orig, t, noise=noise_fn(ce)
                )
            )

        out = diffusion.p_mean_variance(
            model, ce, t,
            clip_denoised=True
        )
        ce = out['mean']

        if stochastic and (idx != (steps - 1)):
            noise = torch.randn_like(ce)
            ce += torch.exp(0.5 * out["log_variance"]) * noise

    ce = ce * (1 - boolmask) + boolmask * orig
    ce = (ce * 0.5) + 0.5
    ce = ce.clamp(0, 1)
    noise_x = ((noise_x * 0.5) + 0.5).clamp(0, 1)

    return ce, pe, noise_x, mask


# =======================================================
# Main
# =======================================================

def main():
    args = create_args()

    if args.merge_chunks:
        merge_all_chunks(args.chunks, args.output_path, args.exp_name)
        return

    respaced_steps = int(
        args.sampling_time_fraction * int(args.timestep_respacing)
    )

    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(osp.join(args.output_path, 'Results'), exist_ok=True)
    os.makedirs(
        osp.join(args.output_path, 'Results', args.exp_name),
        exist_ok=True,
    )

    # Set seeds.
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset.
    dataset = get_dataset(args)

    target = -1
    if args.label_target != -1:
        target = (
            1 - args.label_target
            if args.dataset in BINARYDATASET
            else args.label_query
        )

    dataset = SlowSingleLabel(target, dataset, args.num_samples)
    dataset = ChunkedDataset(
        dataset=dataset, chunk=args.chunk, num_chunks=args.chunks
    )

    print('Images on the dataset:', len(dataset))

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load diffusion model.
    print('Loading Model and diffusion model')
    model, respaced_diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location='cpu')
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Load classifier.
    print('Loading Classifier')
    classifier = get_classifier(args)
    classifier.to(dist_util.dev()).eval()

    if args.adaptive_mask:
        print(
            '[Adaptive Mask] ENABLED | '
            f'target_index={args.adaptive_mask_target_index}, '
            f'coverage=[{args.adaptive_mask_min_coverage}, '
            f'{args.adaptive_mask_max_coverage}]'
        )
        if args.adaptive_mask_min_coverage > args.adaptive_mask_max_coverage:
            raise ValueError(
                'adaptive_mask_min_coverage must be <= '
                'adaptive_mask_max_coverage'
            )
    else:
        print('[Adaptive Mask] DISABLED (original ACE mask)')

    if args.attack_joint and not (
        args.attack_joint_checkpoint or args.attack_joint_shortcut
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

    # Load attack.
    def get_dist_fn():
        any_loss = False
        l1_loss = None
        l2_loss = None

        if args.dist_l2 != 0.0:
            l2_loss = lambda x, x_adv: (
                args.dist_l2
                * torch.linalg.norm(
                    (x - x_adv).view(x.size(0), -1), dim=1
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
            if l2_loss is not None:
                loss += l2_loss(x, x_adv)
            if l1_loss is not None:
                loss += l1_loss(x, x_adv)
            return loss

        return dist_fn

    dist_fn = get_dist_fn()

    main_args = {
        'predict': (
            joint_classifier
            if args.attack_joint
            and not (args.attack_joint_checkpoint or args.attack_joint_shortcut)
            else classifier
        ),
        'loss_fn': None,
        'dist_fn': dist_fn,
        'eps': args.attack_epsilon / 255,
        'nb_iter': args.attack_iterations,
        'dist_schedule': args.dist_schedule,
        'binary': args.dataset in BINARYDATASET,
        'step': args.attack_step / 255,
    }

    attack = get_attack(
        args.attack_method,
        args.attack_joint and args.attack_joint_checkpoint,
        args.attack_joint and args.attack_joint_shortcut,
    )

    if args.attack_joint and (
        args.attack_joint_checkpoint or args.attack_joint_shortcut
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

    # Output setup.
    start_time = time()
    save_imgs = {
        'pre-explanation': (
            ImageSaver(
                args.output_path,
                osp.join(args.exp_name, 'pre-explanation'),
            ) if args.save_images else None
        ),
        'explanation': (
            ImageSaver(
                args.output_path,
                osp.join(args.exp_name, 'explanation'),
            ) if args.save_images else None
        ),
    }

    stats_template = {
        'cf': 0,
        'cf5': 0,
        'untargeted': 0,
        'untargeted5': 0,
        'l1': 0,
        'l inf': 0,
    }
    stats = {
        'n': 0,
        'clean acc': 0,
        'clean acc5': 0,
        'explanation': copy.deepcopy(stats_template),
        'pre-explanation': copy.deepcopy(stats_template),
    }

    print('Starting Image Generation')
    for idx, (indexes, img, lab) in enumerate(loader):
        print(
            f'[Chunks ({args.chunk}+1) / {args.chunks}] '
            f'{idx} / {len(loader)} | Time: {int(time() - start_time)}s'
        )

        img = img.to(dist_util.dev())
        lab = lab.to(
            dist_util.dev(),
            dtype=torch.float if args.dataset in BINARYDATASET else torch.long,
        )

        # Initial classification.
        c_log, c_pred = get_prediction(
            classifier, img, args.dataset in BINARYDATASET
        )

        # Construct target.
        target = None
        if args.label_target != -1:
            target = torch.ones_like(lab) * args.label_target
            target[lab != c_pred] = lab[lab != c_pred]
        elif args.dataset in BINARYDATASET:
            target = 1 - c_pred
            target[lab != c_pred] = lab[lab != c_pred]

        acc1, acc5 = accuracy(
            c_log, lab, binary=args.dataset in BINARYDATASET
        )
        stats['clean acc'] += acc1.sum().item()
        stats['clean acc5'] += acc5.sum().item()
        stats['n'] += lab.size(0)

        # Generate explanation.
        ce, pe, noise, pe_mask = filter_fn(
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
            adaptive_mask_target_index=args.adaptive_mask_target_index,
            adaptive_mask_min_coverage=args.adaptive_mask_min_coverage,
            adaptive_mask_max_coverage=args.adaptive_mask_max_coverage,
            adaptive_mask_entropy_strength=args.adaptive_mask_entropy_strength,
        )

        noise = (noise * 255).to(dtype=torch.uint8).detach().cpu()
        pe_mask = (pe_mask * 255).to(dtype=torch.uint8).detach().cpu()
        ce_mask = (
            generate_mask(img, ce, 1)[0] * 255
        ).to(dtype=torch.uint8).detach().cpu()

        # Evaluate counterfactuals.
        with torch.no_grad():
            for data_type, data_img, data_mask in zip(
                ['pre-explanation', 'explanation'],
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

                stats[data_type]['cf'] += cf.sum().item()
                stats[data_type]['cf5'] += cf5.sum().item()
                stats[data_type]['untargeted'] += (
                    un.size(0) - un.sum().item()
                )
                stats[data_type]['untargeted5'] += (
                    un5.size(0) - un5.sum().item()
                )

                l1 = (
                    (img - data_img).abs()
                    .view(img.size(0), -1)
                    .mean(dim=1)
                    .detach()
                    .cpu()
                )
                linf = (
                    (img - data_img).abs()
                    .view(img.size(0), -1)
                    .max(dim=1)[0]
                    .detach()
                    .cpu()
                )
                stats[data_type]['l1'] += l1.sum().item()
                stats[data_type]['l inf'] += linf.sum().item()

                img255 = (
                    (img * 255).to(dtype=torch.uint8).detach().cpu()
                )
                data_img_uint8 = (
                    (data_img * 255).to(dtype=torch.uint8).detach().cpu()
                )

                if args.save_images:
                    save_imgs[data_type](
                        imgs=img255.permute(0, 2, 3, 1).numpy(),
                        cfs=data_img_uint8.permute(0, 2, 3, 1).numpy(),
                        noises=noise.permute(0, 2, 3, 1).numpy(),
                        target=target if target is not None else lab,
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

        if (idx + 1) % 50 == 0:
            print('=' * 50)
            print('\nCurrent Stats at iteration', idx + 1, ':')
            print_dict(stats)
            print('=' * 50)

        if (idx + 1) == len(loader):
            print(
                f'[Chunks ({args.chunk}+1) / {args.chunks}] '
                f'{idx + 1} / {len(loader)} | '
                f'Time: {int(time() - start_time)}s'
            )
            print('\nDone')
            break

    if stats['n'] == 0:
        raise RuntimeError('No images were processed. Check dataset and num_samples.')

    for data_type in ['pre-explanation', 'explanation']:
        for key in stats[data_type]:
            stats[data_type][key] /= stats['n']

    stats['clean acc'] /= stats['n']
    stats['clean acc5'] /= stats['n']

    if args.chunks == 1:
        print('=' * 50, '\nResults:\n\n')
        print_dict(stats)
        print('=' * 50)

    prefix = '' if args.chunks == 1 else f'c-{args.chunk}_{args.chunks}-'
    summary_path = osp.join(
        args.output_path, 'Results', args.exp_name, prefix + 'summary.yaml'
    )
    with open(summary_path, 'w') as f:
        f.write(str(stats))


if __name__ == '__main__':
    main()
