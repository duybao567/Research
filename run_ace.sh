#!/bin/bash
#SBATCH --job-name=m5-v2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:30:00
#SBATCH --array=0-5
#SBATCH --output=/media/lhbac38/logs/%A_%a_m5v2.txt

# M5-v2 sweep on the same 200 images as ace_baseline_200 (seed 4).
# 3 mask settings x 2 chunks = 6 array tasks:
#   task 0,1 -> m5v2_mass     : coverage_rule=mass (50% of saliency mass), 5-20%, smoothed saliency
#   task 2,3 -> m5v2_fixed10  : fixed 10% coverage, smoothed saliency
#   task 4,5 -> m5v2_fixed20  : fixed 20% coverage, smoothed saliency
# Requires the updated adaptive_mask/ (config.py, mask_generator.py, dynamic_gradient.py).

source /media/lhbac38/venv/bin/activate
export HF_HOME=/media/lhbac38/.cache/huggingface
export XDG_CACHE_HOME=/media/lhbac38/.cache
export OMP_NUM_THREADS=8
cd /media/lhbac38/ACE

SETTING=$(( SLURM_ARRAY_TASK_ID / 2 ))
CHUNK=$(( SLURM_ARRAY_TASK_ID % 2 ))

export M5_SALIENCY_SIGMA=4
case $SETTING in
  0) EXP_NAME="m5v2_mass";    export M5_COVERAGE_RULE=mass;  export M5_MASS_RATIO=0.5; MIN_COV=0.05; MAX_COV=0.20 ;;
  1) EXP_NAME="m5v2_fixed10"; export M5_COVERAGE_RULE=fixed; MIN_COV=0.05; MAX_COV=0.10 ;;
  2) EXP_NAME="m5v2_fixed20"; export M5_COVERAGE_RULE=fixed; MIN_COV=0.05; MAX_COV=0.20 ;;
esac
echo "Setting $SETTING ($EXP_NAME), chunk $CHUNK | rule=$M5_COVERAGE_RULE mass=$M5_MASS_RATIO sigma=$M5_SALIENCY_SIGMA cov=[$MIN_COV,$MAX_COV]"

MODELPATH=/media/lhbac38/ACE/models/celebahq-ddpm.pt
CLASSIFIERPATH=/media/lhbac38/ACE/models/classifier.pth
DATAPATH=/media02/lhbac38/Datasets
OUTPUT_PATH=/media/lhbac38/ACE_Results

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 500 \
--learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 \
--num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1"

python -u main.py $MODEL_FLAGS $SAMPLE_FLAGS \
    --gpu 0 \
    --seed 4 \
    --num_samples 200 \
    --chunks 2 \
    --chunk $CHUNK \
    --model_path "$MODELPATH" \
    --classifier_path "$CLASSIFIERPATH" \
    --output_path "$OUTPUT_PATH" \
    --exp_name "$EXP_NAME" \
    --attack_method PGD \
    --attack_iterations 50 \
    --attack_joint True \
    --attack_joint_checkpoint False \
    --attack_checkpoint_backward_steps 1 \
    --dist_l1 0.001 \
    --dist_l2 0.0 \
    --timestep_respacing 25 \
    --sampling_time_fraction 0.2 \
    --sampling_stochastic True \
    --sampling_inpaint 0.15 \
    --label_query 31 \
    --label_target -1 \
    --image_size 256 \
    --data_dir "$DATAPATH" \
    --dataset CelebAHQ \
    --adaptive_mask True \
    --adaptive_mask_target_index 0 \
    --adaptive_mask_saliency_method gradient \
    --adaptive_mask_gradcam_layer none \
    --adaptive_mask_refinement False \
    --adaptive_mask_min_coverage $MIN_COV \
    --adaptive_mask_max_coverage $MAX_COV \
    --adaptive_mask_entropy_strength 1.0 \
    --adaptive_mask_tv_weight 0.02 \
    --adaptive_mask_gaussian True \
    --adaptive_mask_gaussian_kernel 5 \
    --adaptive_mask_gaussian_sigma 3