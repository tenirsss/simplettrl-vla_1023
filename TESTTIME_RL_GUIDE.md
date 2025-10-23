# Test-Time RL (TTRL) Integration Guide for SimpleVLA-RL

## Overview

This guide explains how to use the Test-Time Reinforcement Learning (TTRL) modifications integrated into SimpleVLA-RL.

## Key Changes

### 1. **Multi-Trajectory Sampling** (`rob_rollout.py`)

The rollout module now supports generating multiple trajectories per prompt with different temperatures to encourage diversity:

- **n_samples**: Number of trajectories to generate per task (default: 1)
- **enable_test_time_rl**: Enable test-time RL mode (boolean flag)
- **base_temperature**: Starting temperature for trajectory diversity (default: 0.6)
- **temperature_range**: Range of temperature variation (default: 0.3)

When `n_samples > 1` and `enable_test_time_rl=True`, temperatures are varied across samples:
- Sample 0: 0.6
- Sample 1: 0.75
- Sample 2: 0.9
- ...

### 2. **Efficiency-Based Trajectory Rewards** (`main_ppo.py`)

The `RobRewardManager` now supports two reward modes:

#### Standard Mode (test_time_rl=False)
```
reward = 1.0 if task_complete else 0.0
```

#### Test-Time RL Mode (test_time_rl=True)
```
Efficiency reward = 1.0 / (1.0 + steps / max_steps)

Example:
- Complete in 100 steps (max=500): reward = 1.0 / (1.0 + 100/500) = 0.833
- Complete in 250 steps (max=500): reward = 1.0 / (1.0 + 250/500) = 0.667  
- Complete in 500 steps (max=500): reward = 1.0 / (1.0 + 500/500) = 0.5
- Failed: reward = 0.0
```

This encourages:
- Task completion (if not completed, reward=0)
- Efficiency (completing in fewer steps gets higher reward)
- Natural trade-off exploration (different trajectories can explore different speed/accuracy tradeoffs)

## Usage

### Configuration via Command Line

Add these parameters when running training:

```bash
python -m verl.trainer.main_ppo \
    ... existing parameters ... \
    data.n_samples=4 \
    +test_time_rl=True \
    +base_temperature=0.6 \
    +temperature_range=0.3
```

### Configuration via YAML

Create or modify your config file to include:

```yaml
# Test-Time RL settings
test_time_rl: True
base_temperature: 0.6
temperature_range: 0.3

# Data sampling
data:
  n_samples: 4  # Number of trajectories per task (4-8 recommended for debugging)

# Maximum steps for trajectory length normalization
actor_rollout_ref:
  model:
    max_steps: 500  # Adjust based on your task max_steps
```

### Minimal Example

For debugging with 4 trajectories:

```bash
python -m verl.trainer.main_ppo \
    --config-name=ppo_trainer.yaml \
    data.train_files='["path/to/train.parquet"]' \
    data.val_files='["path/to/val.parquet"]' \
    data.n_samples=4 \
    +test_time_rl=True
```

## Implementation Details

### Multi-Sample Trajectory Generation Flow

```
For each task prompt:
  For each sample i in [0, n_samples):
    1. Determine temperature = base_temp + (temp_range / (n_samples-1)) * i
    2. Execute environment rollout with this temperature
    3. Collect trajectory and final step count
    4. Return all n_samples trajectories with their finish_step counts

PPO Training:
  For each trajectory:
    Calculate reward = 1.0 / (1.0 + finish_step / max_steps)
    Use this trajectory-level reward for advantage estimation
```

### Data Flow

```
Sampling Phase (rob_rollout.py):
  - Generate N trajectories with varied temperatures
  - Record: complete, finish_step, actions, responses
  - No downsampling (all N trajectories used)

Reward Calculation (main_ppo.py):
  - Input: complete, finish_step for each trajectory
  - Output: efficiency reward (or 0 if failed)
  - Each trajectory gets its own reward based on efficiency

PPO Update:
  - Use trajectory-level reward for advantage/return calculation
  - All N trajectories participate in policy gradient update
```

## Expected Behavior

With Test-Time RL enabled:

1. **Trajectory Diversity**: Different trajectories explore different solution paths due to temperature variation
2. **Efficiency Learning**: Model learns to balance speed and accuracy through reward signal
3. **No Downsampling**: All generated trajectories are used for training (increases compute, no filtering)
4. **Trajectory-Level Rewards**: Each trajectory gets reward based on its efficiency, not just success/failure

## Debugging Tips

1. **Verify Temperature Variation**:
   - Check logs for temperature values being used
   - Ensure temperatures are actually varying across samples

2. **Check Finish Step Distribution**:
   - Print finish_step values to verify trajectories complete at different speeds
   - If all trajectories finish at same step, temperature variation might not be effective

3. **Monitor Reward Distribution**:
   - Look at train/verifier metric in logs
   - Should see diversity in trajectory rewards (not all 0 or 1)

4. **Start Small**:
   - Begin with n_samples=4 for debugging
   - Monitor memory usage and training speed
   - Scale up to 8 if needed

## Performance Considerations

- **Memory**: n_samples × normal_memory required per trajectory
- **Compute**: n_samples × environment execution time per task
- **Training Speed**: Slight overhead in reward calculation, but should be minimal

For 8 GPUs with n_samples=4-8:
- Expect 4-8× environment execution time per batch
- 8 × H200 GPUs can handle this for typical VLA tasks

## Future Extensions

Potential improvements (not yet implemented):

1. **Clustering** (optional, post-TTRL): Group similar trajectories and weighted voting
2. **Multi-Objective Rewards**: Combine efficiency with other metrics (energy, smoothness, etc.)
3. **Adaptive Temperature**: Adjust temperature based on trajectory diversity statistics
4. **Selective Training**: Only use most diverse/efficient trajectories (downsampling)

## References

Related to:
- TTRL Paper: https://arxiv.org/abs/2504.16084
- OpenVLA: Vision-Language Models for Embodied AI
- SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning
