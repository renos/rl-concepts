from ray.tune.registry import register_env


# from ray.rllib.env import PettingZooEnv
import copy

import supersuit as ss

from rlrep.utils import possibly_gridsearch
import gymnasium as gym

from pettingzoo.utils import aec_to_parallel
import pettingzoo
import numpy as np
from ctf_rl.env.ctf.simple_ctf import parallel_env
from ctf_rl.env.rllib_wrapper import PettingZooEnv
from rlrep.envs.ctf.ctf import AddConceptintervention
from rlrep.concepts import concept_config as multi_concept_gen
from argparse import Namespace
import json
from ctf_rl.wrappers.make_pettingzoo import make_env as env_creater
from ray.rllib.policy.policy import PolicySpec


def scenario(experiment_class, CustomPolicy):
    args = experiment_class.arguments
    policy_params = Namespace(**experiment_class.policy_parameters)

    if policy_params.scenario == "ctf":
        if len(experiment_class.model_parameters) > 1:

            def policy_mapping_fn(agent_id, episode, **kwargs):
                if "team_0_" in agent_id:
                    return "good"
                elif "team_1_" in agent_id:
                    return "adversary"
                else:
                    assert False, f"Unknown agent {agent_id}"

        else:

            def policy_mapping_fn(agent_id, episode, **kwargs):
                return "default_policy"

        env_config = experiment_class.env_config
        env = env_creater(env_config)

        obs_space_dict = env.observation_space
        action_space_dict = env.action_space

        experiment_class.agent_ids = env.par_env.env.aec_env.agents

        register_env("custom_env", env_creater)
        policies = {}
        for model_name, val in experiment_class.model_parameters.items():
            if not isinstance(action_space_dict, gym.spaces.Discrete):
                action_space_dict = gym.spaces.Discrete(action_space_dict.n)
            if not isinstance(obs_space_dict, gym.spaces.Box):
                obs_space_dict = gym.spaces.Box(
                    low=obs_space_dict.low,
                    high=obs_space_dict.high,
                    shape=obs_space_dict.shape,
                    dtype=obs_space_dict.dtype,
                )
            model_dict = {
                "obs_space_dict": obs_space_dict,
                "act_space_dict": action_space_dict,
                "model": val,
            }

            policies[model_name] = (
                CustomPolicy,
                obs_space_dict,
                action_space_dict,
                model_dict,
            )

        assert (
            policy_mapping_fn is not None
        ), f"Need to define a policy mapping function for {args.sub_scenario}"

        if len(experiment_class.model_parameters) > 1:
            experiment_class.policy_mapping_fn = policy_mapping_fn
            experiment_class.policies = policies
        else:
            policies = {}
            policies["default_policy"] = experiment_class.model_parameters
            experiment_class.policies = policies
            experiment_class.observation_space = env.observation_space
            experiment_class.action_space = env.action_space

        experiment_class.compute_concepts = "ctf"

        return experiment_class
