#!/bin/bash
# Test-Time RL (TTRL) Training Script for SimpleVLA-RL
# This script demonstrates how to use the TTRL modifications for multi-trajectory sampling with efficiency-based rewards

set -x

export MUJOCO_GL=osmesa
export NCCL_DEBUG=WARN 
export WANDB_API_KEY='e24cdb09c08933b35e62651afdeac3d18a99ff30'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export ROBOT_PLATFORM=LIBERO

PROJECT_NAME='SimpleVLA-RL-TTRL'
EXPERIMENT_NAME='TTRL-Debug-4samples'  # Change this for your experiments

# Model and data paths
SFT_MODEL_PATH="/inspire/hdd/project/realtimedecisionmaking/yishenghong-CZXS25230064/sychen/vla_models/openvla-oft-trajall-libero_10"
CKPT_PATH="/inspire/hdd/project/realtimedecisionmaking/yishenghong-CZXS25230064/sychen/simplettrl-vla_1023/checkpoint"
DATASET_NAME="libero_10"
VLA_NAME="openvla-oft"

# Hardware configuration
NUM_GPUS=8
NUM_NODES=1
ALIGN_PATH="/inspire/hdd/project/realtimedecisionmaking/yishenghong-CZXS25230064/sychen/simplettrl-vla_1023/align.json"


# Test-Time RL Configuration
N_SAMPLES=4  # Number of trajectories per task (4-8 for debugging, 8 for production)
ENABLE_TTRL=True
BASE_TEMPERATURE=1.6
TEMPERATURE_RANGE=0.0
TTRL_REWARD_MODE="env_sparse"

echo "Starting Test-Time RL training with $N_SAMPLES samples per task..."
echo "Base temperature: $BASE_TEMPERATURE"
echo "Temperature range: $TEMPERATURE_RANGE"
echo "Reward mode: $TTRL_REWARD_MODE"

bash examples/overwrite_vla_ckpt_utils.sh $SFT_MODEL_PATH 

HYDRA_FULL_ERROR=1 python -u -m verl.trainer.main_ppo \
    data.task_suite_name=$DATASET_NAME \
    data.num_trials_per_task=50 \
    data.n_samples=$N_SAMPLES \
    data.filter_accuracy=True \
    data.accuracy_lower_bound=0.1 \
    data.accuracy_upper_bound=0.9 \
    data.oversample_factor=1 \
    data.train_batch_size=32 \
    data.val_batch_size=256 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.model.vla=$VLA_NAME \
    actor_rollout_ref.model.action_token_len=7 \
    actor_rollout_ref.model.action_chunks_len=8 \
    +actor_rollout_ref.model.max_steps=500 \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$NUM_GPUS \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.num_images_in_input=1 \
    actor_rollout_ref.actor.traj_mini_batch_size=16 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0. \
    +actor_rollout_ref.actor.use_kl_loss=True \
    +actor_rollout_ref.actor.target_kl_coef=0.04 \
    +actor_rollout_ref.actor.init_kl_coef=0.02 \
    actor_rollout_ref.rollout.num_images_in_input=1 \
    actor_rollout_ref.rollout.use_proprio=False \
    actor_rollout_ref.rollout.val_micro_batch_size=8 \
    actor_rollout_ref.rollout.temperature=$BASE_TEMPERATURE \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.experiment_name=$EXPERIMENT_NAME \
    actor_rollout_ref.rollout.micro_batch_size=1 \
    actor_rollout_ref.rollout.unnorm_key=$DATASET_NAME \
    actor_rollout_ref.rollout.model_family=openvla \
    actor_rollout_ref.rollout.task_suite_name=$DATASET_NAME \
    actor_rollout_ref.rollout.num_steps_wait=10 \
    actor_rollout_ref.rollout.pretrained_checkpoint=$SFT_MODEL_PATH \
    actor_rollout_ref.rollout.center_crop=True \
    actor_rollout_ref.rollout.max_prompt_length=512 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.optim.warmup_style=constant \
    critic.ppo_mini_batch_size=32 \
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
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=$NUM_NODES \
    trainer.save_freq=25 \
    trainer.test_freq=4 \
    trainer.total_epochs=100 \
    trainer.critic_warmup=0 \
    +trainer.log_interval=10 \
    trainer.runtime_env=$ALIGN_PATH \
    trainer.wandb_mode=offline \
    trainer.val_before_train=False \
    verifier.reward_coef=1.0 \
    '+test_time_rl=True' \
    '+base_temperature='$BASE_TEMPERATURE \
    '+temperature_range='$TEMPERATURE_RANGE \
    '+test_time_rl_reward='$TTRL_REWARD_MODE \
    "+enable_test_time_rl=True" \
    "$@"

echo "Training completed!"
