#!/bin/bash
# Test-Time RL (TTRL) Training Script for SimpleVLA-RL
# This script demonstrates how to use the TTRL modifications for multi-trajectory sampling with efficiency-based rewards

set -x

export NCCL_DEBUG=WARN 
export WANDB_API_KEY='YOUR WANDB KEY'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export ROBOT_PLATFORM=LIBERO

PROJECT_NAME='SimpleVLA-RL-TTRL'
EXPERIMENT_NAME='TTRL-Debug-4samples'  # Change this for your experiments

# Model and data paths
SFT_MODEL_PATH="YOUR SFT_MODEL_PATH"
CKPT_PATH="YOUR CKPT_PATH"
DATASET_NAME="libero_10"
VLA_NAME="openvla-oft"

# Hardware configuration
NUM_GPUS=8
NUM_NODES=1
ALIGN_PATH="YOUR PATH TO SimpleVLA-RL/align.json"

# Test-Time RL Configuration
N_SAMPLES=4  # Number of trajectories per task (4-8 for debugging, 8 for production)
ENABLE_TTRL=True
BASE_TEMPERATURE=0.6
TEMPERATURE_RANGE=0.3

echo "Starting Test-Time RL training with $N_SAMPLES samples per task..."
echo "Base temperature: $BASE_TEMPERATURE"
echo "Temperature range: $TEMPERATURE_RANGE"

bash examples/overwrite_vla_ckpt_utils.sh $SFT_MODEL_PATH 

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=$N_SAMPLES \
    data.filter_accuracy=False \
    data.oversample_factor=1 \
    data.train_batch_size=64 \
    data.val_batch_size=496 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    actor_rollout_ref.model.max_steps=500 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.target_kl_coef=0.04 \
    actor_rollout_ref.actor.init_kl_coef=0.02 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    critic.optim.lr=1e-5 \
    critic.optim.warmup_style=constant \
    critic.ppo_mini_batch_size=64 \
    critic.ppo_micro_batch_size=$NUM_GPUS \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.use_remove_padding=True \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.adv_estimator=grpo \
    algorithm.adv_params.reward_model_gamma=1.0 \
    algorithm.adv_params.verifier_gamma=1.0 \
    trainer.logger='console' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.default_local_dir=$CKPT_PATH \
    trainer.total_epochs=10 \
    trainer.critic_warmup=0 \
    trainer.log_interval=10 \
    verifier.reward_coef=1.0 \
    '+test_time_rl=True' \
    '+base_temperature='$BASE_TEMPERATURE \
    '+temperature_range='$TEMPERATURE_RANGE \
    "+enable_test_time_rl=True" \
    "$@"

echo "Training completed!"
