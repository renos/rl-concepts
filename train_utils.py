# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn
import torch.optim
from concept_module import ConceptNetwork
from models.concept_net import make_ppo_modules_crafter, make_ppo_modules_pixels
from models.crafter_impala import (
    make_ppo_modules_crafter_impala,
    make_ppo_modules_crafter_impala_achievements,
)
from models.crafter_pyramid import make_ppo_modules_crafter_pyramid
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.data.tensor_specs import DiscreteBox
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvCreator,
    ExplorationType,
    GrayScale,
    GymWrapper,
    NoopResetEnv,
    ParallelEnv,
    Resize,
    RewardClipping,
    RewardSum,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)
from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

from torchsummary import summary

import numpy as np
from atari import make_parallel_env as make_parallel_env_atari
from crafter import make_parallel_env as make_parallel_env_crafter
from crafter import make_parallel_env_impala as make_parallel_env_crafter_impala
from crafter import make_parallel_env_concept as make_parallel_env_crafter_concept

# ====================================================================
# Environment utils
# --------------------------------------------------------------------


# ====================================================================
# Model utils
# --------------------------------------------------------------------

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math


class CombinedLRScheduler(_LRScheduler):
    def __init__(
        self, optimizer, milestones, lr_vals, gamma=1.0, last_epoch=-1, verbose=False
    ):
        self.milestones = milestones
        self.lr_vals = lr_vals
        self.current_milestone_index = 0  # Start with the first milestone
        self.gamma = gamma
        super(CombinedLRScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch > self.milestones[-1]:
            # If beyond the last milestone, return the last lr_val
            return [self.lr_vals[-1] for _ in self.optimizer.param_groups]

        if self.last_epoch in self.milestones:
            self.current_milestone_index = self.milestones.index(self.last_epoch)
            return [
                self.lr_vals[self.current_milestone_index]
                for _ in self.optimizer.param_groups
            ]

        if self.last_epoch < self.milestones[0]:
            # Before the first milestone, use the initial LR
            return [self.lr_vals[0] for _ in self.optimizer.param_groups]

        # Calculate cosine annealing between milestones
        base_lr = self.lr_vals[self.current_milestone_index]
        next_lr = self.lr_vals[self.current_milestone_index + 1]
        start_milestone = self.milestones[self.current_milestone_index]
        end_milestone = self.milestones[self.current_milestone_index + 1]
        progress = (self.last_epoch - start_milestone) / (
            end_milestone - start_milestone
        )

        lr = next_lr + 0.5 * (base_lr - next_lr) * (1 + math.cos(math.pi * progress))

        return [lr for _ in self.optimizer.param_groups]


# class CombinedLRScheduler(_LRScheduler):
#     def __init__(
#         self, optimizer, milestones, lr_vals, gamma=1.0, last_epoch=-1, verbose=False
#     ):
#         self.milestones = milestones
#         self.lr_vals = lr_vals
#         self.current_milestone_index = 0  # Start with the first milestone
#         self.gamma = gamma
#         super(CombinedLRScheduler, self).__init__(optimizer, last_epoch, verbose)

#     def get_lr(self):
#         if self.last_epoch > self.milestones[-1]:
#             # If beyond the last milestone, return the last lr_val
#             return [self.lr_vals[-1] for _ in self.optimizer.param_groups]

#         if self.last_epoch in self.milestones:
#             self.current_milestone_index = self.milestones.index(self.last_epoch)
#             return [
#                 self.lr_vals[self.current_milestone_index]
#                 for _ in self.optimizer.param_groups
#             ]

#         if self.last_epoch < self.milestones[0]:
#             # Before the first milestone, use the initial LR
#             return [self.lr_vals[0] for _ in self.optimizer.param_groups]

#         # Calculate exponential decay between milestones
#         base_lr = self.lr_vals[self.current_milestone_index]
#         next_lr = self.lr_vals[self.current_milestone_index + 1]
#         start_milestone = self.milestones[self.current_milestone_index]
#         end_milestone = self.milestones[self.current_milestone_index + 1]
#         progress = (self.last_epoch - start_milestone) / (
#             end_milestone - start_milestone
#         )
#         decay_rate = (next_lr / base_lr) ** (1 / (end_milestone - start_milestone))
#         lr = base_lr * (decay_rate ** (self.last_epoch - start_milestone))

#         return [lr for _ in self.optimizer.param_groups]


# # Example usage
# optimizer = torch.optim.Adam(...)  # Your model's optimizer
# milestones = [10, 20, 30]  # Epoch milestones
# lr_vals = [1e-3, 5e-4, 1e-4, 5e-5]  # Learning rates for each phase

# scheduler = CombinedLRScheduler(optimizer, milestones, lr_vals)


def make_ppo_models(env_name, model_type, category, model_params, task="default"):
    if category == "atari":
        proof_environment = make_parallel_env_atari(env_name, task, 1, device="cpu")
    elif category == "crafter":
        if model_type == "impala" or model_type == "impala_achievements":
            proof_environment = make_parallel_env_crafter_impala(
                env_name, task, 1, device="cpu"
            )
        elif model_type == "pyramid":
            proof_environment = make_parallel_env_crafter_concept(
                env_name, task, 1, device="cpu"
            )
        else:
            proof_environment = make_parallel_env_crafter(
                env_name, task, 1, device="cpu"
            )
    else:
        assert (
            0
        ), f"Category {category} not supported. Supported categories: atari, crafter"

    if category == "atari":
        common_module, policy_module, value_module = make_ppo_modules_pixels(
            proof_environment,
        )
    elif category == "crafter":
        if model_type == "impala":
            common_module, policy_module, value_module = (
                make_ppo_modules_crafter_impala(proof_environment)
            )
        elif model_type == "impala_achievements":
            common_module, policy_module, value_module = (
                make_ppo_modules_crafter_impala_achievements(
                    proof_environment, model_params
                )
            )
        elif model_type == "pyramid":
            common_module, policy_module, value_module = (
                make_ppo_modules_crafter_pyramid(proof_environment)
            )
        else:
            common_module, policy_module, value_module = make_ppo_modules_crafter(
                proof_environment
            )
    else:
        assert (
            0
        ), f"Category {category} not supported. Supported categories: atari, crafter"

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    )

    with torch.no_grad():
        td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
        td = actor_critic(td)
        del td

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    del proof_environment

    return actor, critic


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
    del td_test
    return torch.cat(test_rewards, 0).mean()


def save_experiment(folder_name, total_frames, state_dict) -> None:
    """Checkpoint trainer"""
    checkpoint_folder = folder_name / "checkpoints"
    checkpoint_folder.mkdir(parents=False, exist_ok=True)
    checkpoint_file = checkpoint_folder / f"checkpoint_{total_frames}.pt"
    torch.save(state_dict, checkpoint_file)


def load_experiment(folder_name, model, optimizer) -> int:
    pass


import matplotlib.pyplot as plt


def plot_lr_curve(milestones, lr_vals, epochs, batch_size, gamma=1.0, log_scale=False):
    lr_rates = []

    model = torch.nn.Linear(1, 1)  # Simple model
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_vals[0])
    scheduler = CombinedLRScheduler(optimizer, milestones, lr_vals, gamma=gamma)
    # Temporarily save the current state of the scheduler to restore later
    last_epoch = scheduler.last_epoch
    scheduler.last_epoch = -1  # Reset to initial state

    for epoch in range(epochs):
        scheduler.step()
        lr_rates.append(scheduler.get_last_lr()[0])

    # Restore the scheduler to its original state
    scheduler.last_epoch = last_epoch

    # Plotting the learning rate curve
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.array(range(epochs)) * batch_size,
        lr_rates,
        marker="o",
        linestyle="-",
        color="b",
    )
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    if log_scale:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()
