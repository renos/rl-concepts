import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper
import copy


class AddConceptintervention(BaseParallelWrapper):
    """Observation wrapper that flattens the observation.
    Example:
        >>> import gymnasium
        >>> env = gym.make('CarRacing-v1')
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservation(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs = env.reset()
        >>> obs.shape
        (27648,)
    """

    def __init__(
        self,
        env: gym.Env,
        length: int = 0,
        intervention=False,
        agent_configs=None,
        return_frame=False,
        observation_mode="raw",
    ):
        """Flattens the observations of an environment.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.env = env
        self.agent_configs = agent_configs
        self.concept_length = length
        self.intervention = intervention
        self.return_frame = return_frame
        self.observation_mode = observation_mode

    def observation_space(self, agent):
        original_shape = np.prod(self.env.observation_space(agent).shape)
        return spaces.Box(
            shape=(original_shape + self.concept_length,),
            low=-np.inf,
            high=np.inf,
        )

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs = self.env.reset(**kwargs)
        frame = self.render()
        infos = {agent_name: {} for agent_name in self.agents}
        for agent_name in infos.keys():
            infos[agent_name]["frame"] = copy.deepcopy(frame)
        return self.observation(obs, {}), infos

    @property
    def agents(self):
        return self.env.aec_env.env.agents

    def step(self, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        observations, rewards, terminateds, truncateds, infos = self.env.step(action)
        if self.return_frame:
            frame = self.render()
            for agent_name in infos.keys():
                infos[agent_name]["frame"] = copy.deepcopy(frame)
        return (
            self.observation(observations, infos),
            rewards,
            terminateds,
            truncateds,
            infos,
        )

    def observation(self, observations, infos):
        """Flattens an observation.
        Args:
            observation: The observation to flatten
        Returns:
            The flattened observation
        """
        if not self.intervention:
            concept_intervention = np.zeros(self.concept_length)
            obs = {
                key: np.concatenate((value.flatten(), concept_intervention))
                for key, value in observations.items()
            }
            # flatten obs, then return a concatenated version of the flattened obs and the concept intervention
            return obs
        concept_targets = {key: np.zeros((1, 0)) for key in self.agents}

        for concept in self.agent_configs.configs:
            concept_targets = self._compute_concepts(
                concept, observation, [info], concept_targets
            )
        obs = observation.flatten()
        concept_intervention = np.zeros(self.concept_length)
        # flatten obs, then return a concatenated version of the flattened obs and the concept intervention
        return np.concatenate((obs, concept_intervention))

    def _compute_concepts(self, concept, observation, info, acc):
        vec = np.zeros((acc.shape[0], concept.length))
        for i, info in enumerate(info):
            if concept.concept_type == "regression":
                if concept.name == "dots_eaten":
                    vec[i, 0] = (
                        info["labels"]["dots_eaten_count"] / concept.scaling_factor
                    )
                elif concept.name.endswith("_pos"):
                    newName = concept.name[:-3]
                    vec[i, :] = [
                        info["labels"][f"{newName}x"] / float(concept.scaling_factor),
                        info["labels"][f"{newName}y"] / float(concept.scaling_factor),
                    ]
                elif concept.name == "block_bit_map":
                    vec[i, :] = np.array(
                        [
                            info["labels"][f"{concept.name}_{j}"]
                            / concept.scaling_factor
                            for j in range(30)
                        ]
                    )
                elif concept.name == "tile_color":
                    vec[:, :] = np.array(
                        [info["labels"][f"{concept.name}_{j}"] == 26 for j in range(21)]
                    )
                else:
                    vec[i, 0] = info["labels"][concept.name] / concept.scaling_factor
            else:
                toAdd = np.zeros(concept.length)
                if concept.name == "player_direction":
                    toAdd[info["labels"][concept.name] // 64] = 1
                elif concept.name == "blue_tank_facing_direction":
                    if info["labels"][concept.name] == 17:
                        toAdd[0] = 1
                    elif info["labels"][concept.name] == 21:
                        toAdd[1] = 1
                    elif info["labels"][concept.name] == 29:
                        toAdd[2] = 1
                elif concept.name == "crosshairs_color":
                    if info["labels"][concept.name] == 0:
                        toAdd[0] = 1
                    elif info["labels"][concept.name] == 46:
                        toAdd[1] = 1
                elif (
                    concept.name == "left_tread_position"
                    or concept.name == "right_tread_position"
                ):
                    toAdd[info["labels"][concept.name] % 8] = 1
                else:
                    toAdd[info["labels"][concept.name]] = 1
                vec[i, :] = toAdd
        return np.concatenate((acc, vec), axis=-1)
