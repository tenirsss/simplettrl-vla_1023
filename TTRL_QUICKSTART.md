---
noteId: "0d6884d0afcf11f0a9d95f67904b6761"
tags: []

---

# Quick Start - Test-Time RL for SimpleVLA-RL

## 30-Second Setup

### 1. Update your training command:

```bash
python -m verl.trainer.main_ppo \
    ... your existing parameters ... \
    data.n_samples=4 \
    +test_time_rl=True \
    +base_temperature=0.6 \
    +temperature_range=0.3
```

That's it! You've enabled test-time RL.

### 2. What this does:

- ✅ Generates 4 trajectories per task (instead of 1)
- ✅ Varies temperature: [0.6, 0.7, 0.8, 0.9] for diversity
- ✅ Rewards efficiency: `reward = 1.0 / (1.0 + steps/max_steps)` if completed
- ✅ All 4 trajectories participate in PPO training

---

## Minimal Debug Script

Create `run_ttrl_debug.sh`:

```bash
#!/bin/bash

export WANDB_API_KEY='your_key'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u -m verl.trainer.main_ppo \
    data.task_suite_name=libero_10 \
    data.n_samples=4 \
    data.train_batch_size=32 \
    data.val_batch_size=128 \
    actor_rollout_ref.model.path="YOUR_MODEL_PATH" \
    actor_rollout_ref.model.vla=openvla-oft \
    trainer.total_epochs=2 \
    trainer.test_freq=5 \
    trainer.n_gpus_per_node=8 \
    '+test_time_rl=True' \
    '+base_temperature=0.6' \
    '+temperature_range=0.3'
```

Run it:
```bash
bash run_ttrl_debug.sh
```

---

## What to Watch For

### In Training Logs:

1. **Temperature Variation:**
   - Should see different temperature values used
   - Indicates diverse trajectory sampling

2. **Reward Distribution:**
   - `train/verifier` should be ~0.3-0.7 (not just 0 or 1)
   - Indicates efficiency-based rewards working

3. **Success Metrics:**
   - Watch for improvement vs. baseline
   - May see different learning curves

### Key Metrics to Log:

```python
# Check these in wandb/console logs:
train/verifier           # Average trajectory reward (should vary)
train/reward_all         # Combined reward signal
timing/gen              # Generation time (should be ~n_samples × baseline)
```

---

## Common Issues & Fixes

### Issue 1: "Config key 'test_time_rl' not found"
**Fix:** Use `+test_time_rl=True` (note the `+` prefix for new keys)

### Issue 2: "Only 1 sample generated instead of 4"
**Fix:** Make sure `data.n_samples=4` is set correctly

### Issue 3: "Memory error with n_samples=8"
**Fix:** Reduce `data.train_batch_size` or `data.val_batch_size`

### Issue 4: "Temperatures not varying"
**Fix:** Check `base_temperature` and `temperature_range` are in config
Ensure `enable_test_time_rl=True` is being passed

---

## Comparison: Before vs After

### Before (Standard RL)
```
Task → Generate 1 trajectory → Reward = 0 or 1 → PPO update
```

### After (Test-Time RL)
```
Task → Generate 4 trajectories with different temperatures
     → Each gets efficiency reward (based on completion speed)
     → All 4 participate in PPO update
```

---

## Scaling Up

### Debugging (4 samples):
```bash
data.n_samples=4
+base_temperature=0.6
+temperature_range=0.3
```

### Production (8 samples):
```bash
data.n_samples=8
+base_temperature=0.5
+temperature_range=0.4
```

---

## Configuration Presets

### Ultra-Conservative (similar to baseline)
```
data.n_samples=2
base_temperature=0.7
temperature_range=0.1
```

### Balanced (recommended for debugging)
```
data.n_samples=4
base_temperature=0.6
temperature_range=0.3
```

### Aggressive (maximum diversity)
```
data.n_samples=8
base_temperature=0.5
temperature_range=0.4
```

---

## Next Steps

1. **Try the debug script** with your model
2. **Monitor the metrics** in logs/wandb
3. **Compare with baseline** (test_time_rl=False)
4. **Adjust temperature range** if needed
5. **Scale to n_samples=8** if debugging goes well

---

## Full Example Command

```bash
#!/bin/bash

python -m verl.trainer.main_ppo \
    data.task_suite_name=libero_10 \
    data.n_samples=4 \
    data.train_batch_size=64 \
    actor_rollout_ref.model.path=/path/to/sft_model \
    actor_rollout_ref.model.vla=openvla-oft \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=10 \
    trainer.test_freq=10 \
    '+test_time_rl=True' \
    '+base_temperature=0.6' \
    '+temperature_range=0.3'
```

---

## Documentation

- 📖 **Full Guide:** See `TESTTIME_RL_GUIDE.md`
- 🔧 **Code Changes:** See `TTRL_CODE_CHANGES.md`
- 📝 **Example Script:** `examples/run_openvla_oft_rl_libero_ttrl.sh`

---

## Support

- **Issues with trajectory generation?** Check `rob_rollout.py` modifications
- **Reward not working?** Check `RobRewardManager` in `main_ppo.py`
- **Config not recognized?** Use `+` prefix for new parameters

Happy training! 🚀
