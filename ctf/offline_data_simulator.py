from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector

from ctf_rl.wrappers.make_pettingzoo import make_env

import numpy as np
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.sample_batch import SampleBatch


def _new_episode(env_id, policy_map, policy_mapping_fn):
    episode = Episode(
        policy_map,
        policy_mapping_fn,
        # SimpleListCollector will find or create a
        # simple_list_collector._PolicyCollector as batch_builder
        # for this episode later. Here we simply provide a None factory.
        lambda: None,  # batch_builder_factory
        extra_batch_callback=None,
        env_id=env_id,
    )
    return episode


def simulate_episode(env_id, env_config, policy_map, policy_mapping_fn, clip_rewards):
    batch_builder = SimpleListCollector(
        policy_map=policy_map,
        clip_rewards=clip_rewards,
        callbacks=None,
        multiple_episodes_in_batch=True,
        rollout_fragment_length=200,
        count_steps_by="env_steps",
    )
    for j in range(100):
        episode_object = _new_episode(env_id, policy_map, policy_mapping_fn)

        env = make_env(env_config)
        observation, info = env.reset()

        for agent_id in observation.keys():
            batch_builder.add_init_obs(
                episode=episode_object,
                agent_id=agent_id,
                env_id=env_id,
                policy_id=policy_mapping_fn(agent_id, episode=None),
                init_obs=observation[agent_id],
                init_infos=info[agent_id],
                t=-1,
            )

        terminated = truncated = False
        t = 0
        prep = get_preprocessor(env.observation_space)(env.observation_space)

        for t in range(1000):
            fake_actions = {
                k: env.par_env.action_space(k).sample() for k in env.alive_agents
            }
            new_obs, reward, terminated, truncated, info = env.step(fake_actions)

            actions = {i: info[i]["chosen_actions"][0] for i in info.keys()}

            for i, agent_id in enumerate(info.keys()):
                # assert (
                #     0
                # ), f"{policy_mapping_fn(agent_id, episode=None)=}, {eps_id=} {agent_id=}"
                batch_builder.add_action_reward_next_obs(
                    episode_id=episode_object.episode_id,
                    agent_id=agent_id,
                    env_id=env_id,
                    policy_id=policy_mapping_fn(agent_id, episode=None),
                    agent_done=(agent_id in terminated and terminated[agent_id])
                    or (agent_id in truncated and truncated[agent_id]),
                    values={
                        SampleBatch.NEXT_OBS: new_obs[
                            agent_id
                        ],  # put the true action probability here
                        SampleBatch.ACTIONS: actions[agent_id],
                        SampleBatch.REWARDS: reward[agent_id],
                        SampleBatch.TERMINATEDS: agent_id in terminated
                        and terminated[agent_id],
                        SampleBatch.TRUNCATEDS: agent_id in truncated
                        and truncated[agent_id],
                        SampleBatch.INFOS: info[agent_id],
                    },
                )
            if terminated["__all__"]:
                print(batch_builder.total_env_steps())
                break
    return batch_builder.try_build_truncated_episode_multi_agent_batch()
