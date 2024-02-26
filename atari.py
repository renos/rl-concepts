import gymnasium as gym
from ocatari.core import OCAtari
import numpy as np
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


class PositionHistoryEnv(gym.ObservationWrapper):
    """OCAtari Environment that behaves like a gymnasium environment and passes env_check. Based on RAM, the observation space is object-centric.
    More specifically it is a list position history informations of objects detected. No hud information.
    The observation space is a vector where every object position history has a fixed place.
    If an object is not detected its information entries are set to 0.

    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.ocatari_env = env
        self.reference_list = self._init_ref_vector()
        self.vector_size = 7
        self.current_vector = np.zeros(
            (1, self.vector_size * len(self.reference_list)), dtype=np.float32
        )

    @property
    def observation_space(self):
        vl = len(self.reference_list) * self.vector_size
        return gym.spaces.Box(
            low=-(2**63), high=2**63 - 2, shape=(1, vl), dtype=np.float32
        )

    @property
    def action_space(self):
        return self.ocatari_env.action_space

    def step(self, *args, **kwargs):
        obs, reward, truncated, terminated, info = self.ocatari_env.step(
            *args, **kwargs
        )
        self._obj2vec()
        return self.current_vector, reward, truncated, terminated, info

    def reset(self, *args, **kwargs):
        obs, info = self.ocatari_env.reset(*args, **kwargs)
        self._obj2vec()
        return self.current_vector, info

    def render(self, *args, **kwargs):
        return self.ocatari_env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.ocatari_env.close(*args, **kwargs)

    def _init_ref_vector(self):
        reference_list = []
        obj_counter = {}
        for o in self.ocatari_env.max_objects:
            if o.category not in obj_counter.keys():
                obj_counter[o.category] = 0
            obj_counter[o.category] += 1
        for k in list(obj_counter.keys()):
            reference_list.extend([k for i in range(obj_counter[k])])
        return reference_list

    def _obj2vec(self):
        temp_ref_list = self.reference_list.copy()
        for o in self.ocatari_env.objects:  # populate out_vector with object instance
            idx = temp_ref_list.index(
                o.category
            )  # at position of first category occurance
            start = idx * self.vector_size
            flat = list(o.xywh)
            flat.extend(list(o.rgb))
            self.current_vector[0, start : start + self.vector_size] = (
                flat  # write the slice
            )
            temp_ref_list[idx] = ""  # remove reference from reference list
        for i, d in enumerate(temp_ref_list):
            if d != "":  # fill not populated category instances wiht 0.0's
                self.current_vector[
                    0, i * self.vector_size : i * self.vector_size + self.vector_size
                ] = [0.0 for _ in range(self.vector_size)]


def make_base_env(env_name="BreakoutNoFrameskip-v4", frame_skip=4, is_test=False):
    env = OCAtari(
        env_name, mode="revised", hud=True, obs_mode=None, render_mode="rgb_array"
    )
    env = GymWrapper(
        PositionHistoryEnv(
            env
        ),  # OCAtari(env_name, mode="raw", hud=True, render_mode="rgb_array"),
        frame_skip=frame_skip,
        from_pixels=True,
        pixels_only=False,
        device="cpu",
    )
    env = TransformedEnv(env)
    env.append_transform(NoopResetEnv(noops=30, random=True))
    if not is_test:
        env.append_transform(EndOfLifeTransform())
    return env


def make_parallel_env(env_name, num_envs, device, is_test=False):
    env = ParallelEnv(
        num_envs,
        EnvCreator(lambda: make_base_env(env_name)),
        serial_for_single=True,
        device=device,
    )
    env = TransformedEnv(env)
    env.append_transform(ToTensorImage())
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3, in_keys=["pixels"]))
    env.append_transform(CatFrames(N=4, dim=-2, in_keys=["observation"]))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500))
    if not is_test:
        env.append_transform(RewardClipping(-1, 1))
    env.append_transform(DoubleToFloat())
    env.append_transform(VecNorm(in_keys=["pixels"]))
    return env
