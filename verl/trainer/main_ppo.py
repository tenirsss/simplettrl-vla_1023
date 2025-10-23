# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json
import os
import statistics
from functools import partial

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, countdown, multiply, logic
from verl.trainer.ppo.ray_trainer import RayTrainer
import warnings
warnings.filterwarnings("ignore", message="Batch mode enable graph is only supported with num_graph_seeds==1")

class RobRewardManager():
    """The reward manager with Test-Time RL efficiency-based rewards.
    """
    def __init__(self, num_examine, config) -> None:
        self.num_examine = num_examine
        self.config = config
        # Test-Time RL mode: use efficiency-based rewards instead of 0/1
        self.test_time_rl_mode = config.get("test_time_rl", False)

    def verify(self, data):
        """Compute efficiency-based rewards for test-time RL."""
        completes = data.batch['complete'].tolist()
        finish_steps = data.batch['finish_step'].tolist()
        batch_size = data.batch['responses'].size(0)
        assert len(completes) == batch_size
        assert len(finish_steps) == batch_size
        
        score = []
        if self.test_time_rl_mode:
            # Efficiency-based reward: reward = success / steps
            # Higher reward for completing in fewer steps
            for complete, steps in zip(completes, finish_steps):
                if complete:
                    # Normalize: reward = 1.0 / (steps / max_steps)
                    # This encourages efficiency: completing in half steps gets 2x reward
                    max_steps = self.config.actor_rollout_ref.model.max_steps if hasattr(self.config.actor_rollout_ref.model, 'max_steps') else 500
                    efficiency_reward = 1.0 / (1.0 + steps / max_steps)  # Range: [0, 1]
                    score.append(efficiency_reward)
                else:
                    # Failed trajectory gets 0 reward
                    score.append(0.0)
        else:
            # Standard 0/1 reward
            score = [float(item) for item in completes]
        
        format = [1.0 for _ in range(len(completes))]

        data.batch['acc'] = torch.tensor(score, dtype=torch.float32, device=data.batch['responses'].device)
        data.batch['format_correctness'] = torch.tensor(format, dtype=torch.float32, device=data.batch['responses'].device)
        
        reward_metrics = {}
        format_metrics = {}
        reward_format_metrics = {}
            
        reward_metrics['all'] = data.batch['acc'].mean().item()
        format_metrics['all'] = data.batch['format_correctness'].mean().item()
        reward_format_metrics['all'] = data.batch['acc'].mean().item()

        return score, reward_metrics, format_metrics, reward_format_metrics

    def __call__(self, data: DataProto):
        
        # aggregate all available reward tensors

        reward_tensor_dict={}
        reward_metrics={}
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32) # batch * 64 * 56
        verifier_reward=torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_tensor = reward_tensor.reshape((reward_tensor.shape[0],-1))
        verifier_reward = verifier_reward.reshape((verifier_reward.shape[0],-1))
        
        valid_response_length = data.batch['finish_step'] * self.config.actor_rollout_ref.model.action_token_len 
       
        if 'acc' in data.batch:
            # the separated rewards have been logged; now we add format correctness back for reward shaping
            verifier_score = data.batch['acc'].cpu().numpy().tolist()
        else:
            verifier_score, verifier_metrics, format_metrics, reward_format_metrics = self.verify(data)
            reward_metrics.update(verifier_metrics)
        for i in range(verifier_reward.shape[0]):
            verifier_reward[i,valid_response_length[i]-1] += verifier_score[i]
            
        reward_tensor_dict['gt_scores'] = verifier_reward

        if self.config.verifier.reward_coef!=0:
            
            reward_metrics['verifier'] = reward_tensor_dict['gt_scores'].sum(dim=1).mean().item()
            reward_tensor += self.config.verifier.reward_coef * reward_tensor_dict['gt_scores']

        reward_tensor_dict['all'] = reward_tensor
        reward_metrics['reward_all'] = reward_tensor.sum(dim=-1).mean(dim=0).item()

        return reward_tensor_dict, reward_metrics

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        if os.path.isfile(str(config.trainer.runtime_env)):
            with open(str(config.trainer.runtime_env), 'r') as f:
                runtime_env = json.load(f)
            ray.init(runtime_env=runtime_env)
        else:
            ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker, RobActorRolloutRefWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker, RobActorRolloutRefWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(RobActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(RobActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable and config.reward_model.rm_coef!=0.:
        if config.reward_model.rm_type == 'normal':
            if config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        elif config.reward_model.rm_type == 'prime':
            from verl.workers.fsdp_workers import PRIMERewardModelWorker
            role_worker_mapping[Role.RewardModel] = ray.remote(PRIMERewardModelWorker)
        else:
            raise NotImplementedError
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RobRewardManager( num_examine=0, config=config) # note: verifier is called both inside reward_fn and outside.

    # Note that we always use function-based RM for validation
    val_reward_fn = RobRewardManager( num_examine=1,config=config)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
