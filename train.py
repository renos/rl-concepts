# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script reproduces the Proximal Policy Optimization (PPO) Algorithm
results from Schulman et al. 2017 for the on Atari Environments.
"""
from concept_module import ClipPPOLossWithConceptLoss
from crafter import update_info_dict
import hydra
from torchrl._utils import logger as torchrl_logger
import os

from train_utils import CombinedLRScheduler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4"


@hydra.main(config_path="./configs/", config_name="config_atari", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    import time

    import torch.optim
    import tqdm

    from tensordict import TensorDict
    from torchrl.collectors import SyncDataCollector
    from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.envs import ExplorationType, set_exploration_type
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value.advantages import GAE
    from torchrl.record.loggers import generate_exp_name, get_logger
    from train_utils import eval_model, make_ppo_models, save_experiment
    from atari import make_parallel_env as make_parallel_env_atari
    from crafter import make_parallel_env as make_parallel_env_crafter
    from crafter import make_parallel_env_impala as make_parallel_env_crafter_impala
    from crafter import make_parallel_env_concept as make_parallel_env_crafter_concept
    import importlib
    from pathlib import Path

    device = "cpu" if not torch.cuda.device_count() else "cuda"

    _has_hydra = importlib.util.find_spec("hydra") is not None
    if _has_hydra:
        from hydra.core.hydra_config import HydraConfig
    if _has_hydra and HydraConfig.initialized():
        folder_name = Path(HydraConfig.get().runtime.output_dir)
    else:
        assert 0, f"Hydra not found or not initialized."

    # Correct for frame_skip
    category = cfg.env.category
    if category == "atari":
        frame_skip = 4
    else:
        frame_skip = 1
    total_frames = cfg.collector.total_frames // frame_skip
    frames_per_batch = cfg.collector.frames_per_batch // frame_skip
    mini_batch_size = cfg.loss.mini_batch_size // frame_skip
    test_interval = cfg.logger.test_interval // frame_skip

    # Create models (check utils_atari.py)

    if category == "atari":
        make_parallel_env = make_parallel_env_atari
    elif category == "crafter":
        if cfg.loss.model == "impala" or cfg.loss.model == "impala_achievements":
            make_parallel_env = make_parallel_env_crafter_impala
        elif cfg.loss.model == "pyramid":
            make_parallel_env = make_parallel_env_crafter_concept
        else:
            make_parallel_env = make_parallel_env_crafter
    else:
        raise ValueError(f"Category {category} not recognized")

    actor, critic = make_ppo_models(
        cfg.env.env_name,
        cfg.loss.model,
        category,
        cfg.loss.model_params,
        task=cfg.env.task,
    )
    actor, critic = actor.to(device), critic.to(device)

    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_parallel_env(
            cfg.env.env_name, cfg.env.task, cfg.env.num_envs, "cpu"
        ),
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        policy_device=device,
        device="cpu",
        storing_device="cpu",
        max_frames_per_traj=-1,
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch),
        sampler=sampler,
        batch_size=mini_batch_size,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )

    loss_module = ClipPPOLossWithConceptLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
        concept_coef=cfg.loss.concept_coef,
    )

    # use end-of-life as done key
    if cfg.env.category == "atari":
        adv_module.set_keys(done="end-of-life", terminated="end-of-life")
        loss_module.set_keys(done="end-of-life", terminated="end-of-life")
    if cfg.env.category == "crafter":
        adv_module.set_keys(value="denormalized_state_value")
    # adv_module.set_keys(done="end-of-life", terminated="end-of-life")
    # loss_module.set_keys(done="end-of-life", terminated="end-of-life")

    # Create optimizer
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.eps,
    )

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name(
            "PPO", f"{cfg.logger.exp_name}_{cfg.env.env_name}_{cfg.env.task}"
        )
        logger = get_logger(
            cfg.logger.backend,
            logger_name="ppo",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Create test environment
    test_env = make_parallel_env(
        cfg.env.env_name, cfg.env.task, 1, device, is_test=True
    )
    test_env.eval()

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (
        (total_frames // frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
    )

    sampling_start = time.time()

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes
    cfg_optim_max_grad_norm = cfg.optim.max_grad_norm
    cfg.loss.clip_epsilon = cfg_loss_clip_epsilon
    losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    collection_steps = total_frames // frames_per_batch

    if cfg_optim_anneal_lr:
        annealing_milestones = cfg.optim.anneal_milestones
        annealing_end_batch = cfg.optim.annealing_end_batch
        milestones = [
            int(annealing_end_batch * n) for n in annealing_milestones
        ]  # Epoch milestones
        lr_vals = cfg.optim.anneal_lrs  # Learning rates for each phase

        scheduler = CombinedLRScheduler(optim, milestones, lr_vals)

    for i, data in enumerate(collector):

        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * frame_skip
        pbar.update(data.numel())

        # Get training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "terminated"]]
        # original_rewards = data["next", "original_reward_sum"][
        #     data["next", "terminated"].squeeze(-1)
        # ]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "terminated"]]
            log_info.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                    # "train/original_reward": original_rewards.mean().item(),
                }
            )
            end_achievements = data["next", "achievements"][data["next", "terminated"]]

            if category == "crafter":
                log_info.update(update_info_dict(end_achievements))

        # Compute GAE
        with torch.no_grad():
            data = adv_module(data.to(device, non_blocking=True))
        data_reshape = data.reshape(-1)

        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):

            # Update the data buffer
            data_buffer.extend(data_reshape)

            for k, batch in enumerate(data_buffer):

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                alpha = 1 - (num_network_updates / total_network_updates)
                # if cfg_optim_anneal_lr:
                #     for group in optim.param_groups:
                #         group["lr"] = cfg_optim_lr * alpha
                if cfg_loss_anneal_clip_eps:
                    loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                num_network_updates += 1
                # Get a data batch
                batch = batch.to(device, non_blocking=True)

                # Forward pass PPO loss
                loss = loss_module(batch)
                if "loss_visual_concept" in loss.keys():
                    losses[j, k] = loss.select(
                        "loss_critic",
                        "loss_entropy",
                        "loss_objective",
                        "loss_visual_concept",
                        "average_acc",
                    ).detach()
                    loss_sum = (
                        loss["loss_critic"].mean()
                        + loss["loss_objective"].mean()
                        + loss["loss_entropy"].mean()
                        + loss["loss_visual_concept"].mean()
                    )
                else:
                    losses[j, k] = loss.select(
                        "loss_critic", "loss_entropy", "loss_objective"
                    ).detach()
                    loss_sum = (
                        loss["loss_critic"].mean()
                        + loss["loss_objective"].mean()
                        + loss["loss_entropy"].mean()
                    )
                # Backward pass
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=cfg_optim_max_grad_norm
                )

                # Update the networks
                optim.step()
                optim.zero_grad()
        scheduler.step()

        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        if not cfg_optim_anneal_lr:
            current_lr = cfg_optim_lr
            current_clip_epsilon = cfg_loss_clip_epsilon
        else:
            current_lr = scheduler.get_last_lr()[0]
            current_clip_epsilon = loss_module.clip_epsilon.item()
        log_info.update(
            {
                "train/lr": current_lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/clip_epsilon": current_clip_epsilon,
            }
        )

        # Get test rewards
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if ((i - 1) * frames_in_batch * frame_skip) // test_interval < (
                i * frames_in_batch * frame_skip
            ) // test_interval:
                actor.eval()
                # eval_start = time.time()
                # test_rewards = eval_model(
                #     actor, test_env, num_episodes=cfg_logger_num_test_episodes
                # )
                # eval_time = time.time() - eval_start
                # log_info.update(
                #     {
                #         "eval/reward": test_rewards.mean(),
                #         "eval/time": eval_time,
                #     }
                # )
                # save experiment dict
                save_experiment(
                    folder_name,
                    collected_frames,
                    actor.state_dict(),
                )
                actor.train()

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()
        collector.reset()
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")
    return episode_rewards.mean().item()


if __name__ == "__main__":
    main()
