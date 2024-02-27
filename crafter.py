from crafter_utils import update_and_merge_closest_blocks
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
    ObservationNorm,
    UnsqueezeTransform,
)
from torchrl.envs import default_info_dict_reader
import torch
from contrib import crafter
from crafter_description import id_to_item


class ConceptEnv(gym.ObservationWrapper):
    """OCAtari Environment that behaves like a gymnasium environment and passes env_check. Based on RAM, the observation space is object-centric.
    More specifically it is a list position history informations of objects detected. No hud information.
    The observation space is a vector where every object position history has a fixed place.
    If an object is not detected its information entries are set to 0.

    """

    def __init__(self, env, task, use_pixels=True, is_test=False) -> None:
        super().__init__(env)
        self.env = env
        self.use_pixels = use_pixels

        self.achievements_len = 22  # + 16
        self.status_concepts = (22 + 16, 1)

        self.block_types_to_track = {
            "stone": 5,
            "wood": 1,
            "water": 1,
            "tree": 5,
            "plant": 1,
            "coal": 1,
            "iron": 1,
            "diamond": 1,
            "table": 1,
            "furnace": 1,
            "cow": 1,
            "skeleton": 1,
            "zombie": 1,
            "lava": 1,
        }
        total_blocks = sum(self.block_types_to_track.values())
        self.history_concepts = (total_blocks + 1, 2)
        self.test = is_test
        self.task = task
        self.k = 5

    @property
    def observation_space(self):
        # surroundings
        surroundings = 9 * 7
        inventory_len = 16
        facing_len = 2
        visual_concepts_len = surroundings + inventory_len + facing_len
        obs_space = {
            "pixels": gym.spaces.Box(
                low=0,
                high=255,
                shape=self.env.observation_space.shape,
                dtype=np.float32,
            ),
            "visual_concepts": gym.spaces.Box(
                low=0, high=19, shape=(19, 9, 9), dtype=np.float32
            ),
            "status_concepts": gym.spaces.Box(
                low=0, high=1, shape=self.status_concepts, dtype=np.float32
            ),
            "history_concepts": gym.spaces.Box(
                low=-10, high=10, shape=self.history_concepts, dtype=np.float32
            ),
            "achievements": gym.spaces.Box(
                low=0, high=100, shape=(1, self.achievements_len), dtype=np.float32
            ),
        }
        if not self.use_pixels:
            del obs_space["pixels"]
        if self.test:
            obs_space["render_pixels"] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(224, 224, 3),
                dtype=np.float32,
            )
        return obs_space

    @property
    def action_space(self):
        return gym.spaces.discrete.Discrete(self.env.action_space.n)

    def pad_and_slice_semantic_map(self, info):
        view_height, view_width = info["view"]  # Desired view size (9x7)
        map_height, map_width = info["semantic"].shape  # Original map size
        player_y, player_x = info["player_pos"]  # Player's position (y, x)

        # Calculate padding sizes to ensure the view can be centered around the player
        pad_top = pad_bottom = view_height // 2
        pad_left = pad_right = view_width // 2

        # Create a padded map with default values (e.g., -1) around the original semantic map
        padded_map = np.full(
            (map_height + pad_top + pad_bottom, map_width + pad_left + pad_right),
            fill_value=0,  # Use a distinct value to indicate out-of-bounds areas
            dtype=info["semantic"].dtype,
        )

        # Insert the original semantic map into the center of the padded map
        padded_map[pad_top : pad_top + map_height, pad_left : pad_left + map_width] = (
            info["semantic"]
        )

        # Calculate the start coordinates for slicing the padded map, ensuring we don't go out of bounds
        start_y = info["player_pos"][0] - info["view"][0] // 2 + pad_top
        end_y = info["player_pos"][0] + info["view"][0] // 2 + 1 + pad_top

        start_x = info["player_pos"][1] - info["view"][1] // 2 + 1 + pad_left
        end_x = info["player_pos"][1] + info["view"][1] // 2 + pad_left

        # Slice the padded map to get the view centered around the player's position
        semantic_view = padded_map[start_y:end_y, start_x:end_x]

        return semantic_view

    def into_to_semantic(self, info):
        semantic = info["semantic"][
            info["player_pos"][0]
            - info["view"][0] // 2 : info["player_pos"][0]
            + info["view"][0] // 2
            + 1,
            info["player_pos"][1]
            - info["view"][1] // 2
            + 1 : info["player_pos"][1]
            + info["view"][1] // 2,
        ]
        if semantic.shape != (9, 7):
            semantic = self.pad_and_slice_semantic_map(info)
        return semantic.reshape(9, 7)

    def info_to_inventory(self, info):
        inventory = info["inventory"]
        inventory_list = [0] * 16
        for i, item in enumerate(inventory):
            inventory_list[i] = inventory[item]
        return np.array(inventory_list).reshape(1, 16)

    def info_to_status_concepts(self, info):
        achievements = info["achievements"]
        achievements_list = [0] * (self.status_concepts[0] - 16)
        pos = 0
        for achievement, achievement_value in achievements.items():
            # if achievement == "collect_stone" or achievement == "collect_wood":
            #     continue
            achievements_list[pos] = int(achievement_value > 0)
            pos += 1
        # inventory = info["inventory"]
        # achievements_list[-4] = int(inventory["health"] < 5)
        # achievements_list[-3] = int(inventory["food"] < 5)
        # achievements_list[-2] = int(inventory["drink"] < 5)
        # achievements_list[-1] = int(inventory["energy"] < 5)
        inventory = np.array(self.info_to_inventory(info)).reshape(-1)

        # print(inventory)

        return np.concatenate((achievements_list, inventory)).reshape(
            1, self.status_concepts[0]
        )

    # def info_to_history_concepts(self, info):
    #     history_concepts = np.zeros(self.history_concepts)
    #     cur_pos = 0
    #     for block_type in [
    #         "water",
    #         "stone",
    #         "tree",
    #         "plant",
    #         "coal",
    #         "iron",
    #         "diamond",
    #         "table",
    #         "furnace",
    #         "cow",
    #         "skelton",
    #         "zombie",
    #         "lava",
    #     ]:
    #         if block_type in self.old_closest_blocks:
    #             history_concepts[cur_pos][:2] = self.old_closest_blocks[block_type][
    #                 "relative_position"
    #             ]
    #             # history_concepts[cur_pos][2] = 1.0
    #         else:
    #             history_concepts[cur_pos] = np.array([99, 99])
    #         cur_pos += 1
    #     history_concepts[-1][:2] = np.array(info["player_facing"])
    #     # history_concepts[-1][2] = 1.0
    #     return history_concepts
    def info_to_history_concepts(self, info):
        cur_pos = 0
        history_concepts = np.zeros(self.history_concepts)

        for block_type, count in self.block_types_to_track.items():
            if block_type in self.old_closest_blocks:
                closest_blocks = self.old_closest_blocks[block_type]
                num_blocks_to_add = min(len(closest_blocks), count)
                for i in range(num_blocks_to_add):
                    history_concepts[cur_pos][:2] = closest_blocks[i][
                        "relative_position"
                    ]
                    cur_pos += 1
                # Fill the remaining slots if fewer blocks found than needed
                for _ in range(count - num_blocks_to_add):
                    history_concepts[cur_pos] = np.array([99, 99])
                    cur_pos += 1
            else:
                # If no blocks of this type, fill all slots for this block type with [99, 99]
                for _ in range(count):
                    history_concepts[cur_pos] = np.array([99, 99])
                    cur_pos += 1
        history_concepts[-1][:2] = np.array(info["player_facing"])
        # history_concepts[-1][2] = 1.0

        return history_concepts

    def into_to_achievements(self, info):
        achievements = info["achievements"]
        achievements_list = [0] * (self.achievements_len)
        for i, achievement in enumerate(achievements):
            achievements_list[i] = achievements[achievement]

        achievements_list = np.array(achievements_list)

        return achievements_list.reshape(1, self.achievements_len)

    def info_to_misc(self, info):
        misc_list = [0 for _ in range(2)]
        if info["player_facing"] == (-1, 0):
            misc_list[0] = 0
        elif info["player_facing"] == (1, 0):
            misc_list[0] = 1
        elif info["player_facing"] == (0, -1):
            misc_list[0] = 2
        elif info["player_facing"] == (0, 1):
            misc_list[0] = 3
        return np.array(misc_list).reshape(1, 2)

    def step(self, *args, **kwargs):
        obs, reward, done, info = self.env.step(*args, **kwargs)
        self.semantic = self.into_to_semantic(info)
        self.old_closest_blocks = update_and_merge_closest_blocks(
            old_blocks=self.old_closest_blocks,
            old_pos=self.old_pos,
            new_pos=info["player_pos"],
            new_semantic_map=self.semantic,
            k=self.k,
        )
        self.old_pos = info["player_pos"]

        new_obs = self._generate_concepts(obs, info)
        info["original_reward"] = reward
        new_reward = self.task_reward(info, reward)

        return (new_obs, new_reward, done, done, info)

    def task_reward(self, info, reward):
        survival_reward = info["survival_reward"]
        if self.task == "default":
            return reward
        elif self.task == "survive":
            return survival_reward
        elif self.task == "all":
            return reward + survival_reward
        else:
            raise ValueError(f"Task {self.task} not recognized")

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(return_info=True, *args, **kwargs)
        self.semantic = self.into_to_semantic(info)
        self.old_closest_blocks = update_and_merge_closest_blocks(
            old_blocks=None,
            old_pos=None,
            new_pos=info["player_pos"],
            new_semantic_map=self.semantic,
            k=self.k,
        )
        self.old_pos = info["player_pos"]

        new_obs = self._generate_concepts(obs, info)
        info["original_reward"] = 0

        return new_obs, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.env.close(*args, **kwargs)

    def _visual_concepts(self, info):
        not_onehot = np.concatenate(
            [
                self.semantic.reshape((1, 9 * 7)),
                self.info_to_inventory(info),
                self.info_to_misc(info),
            ],
            axis=-1,
        ).reshape(1, 9, 9)
        onehot = np.zeros((19, 9, 9))
        x, y = np.meshgrid(np.arange(9), np.arange(9), indexing="ij")
        onehot[not_onehot.squeeze(), x, y] = 1
        return onehot

    def _generate_concepts(self, obs, info):
        if self.use_pixels:
            obs_dict = {"pixels": obs}
        else:
            obs_dict = {}
        if self.test:
            obs_dict["render_pixels"] = self.env.render(size=(224, 224))
        obs_dict["visual_concepts"] = self._visual_concepts(info)
        obs_dict["status_concepts"] = self.info_to_status_concepts(info) / 10.0
        obs_dict["history_concepts"] = self.info_to_history_concepts(info) / 100.0
        obs_dict["achievements"] = self.into_to_achievements(info)
        # print(obs_dict["status_concepts"])
        # print(obs_dict["history_concepts"])
        return obs_dict


def make_base_env(task, is_test=False, size=(224, 224), use_pixels=True):
    env = crafter.Env(size=size, use_pixels=use_pixels)
    reader = default_info_dict_reader(["original_reward"])
    env = GymWrapper(
        ConceptEnv(env, task, use_pixels=use_pixels, is_test=is_test),
        frame_skip=1,
        from_pixels=use_pixels,
        pixels_only=False,
        device="cpu",
    ).set_info_dict_reader(info_dict_reader=reader)
    env = TransformedEnv(env)
    # env.append_transform(NoopResetEnv(noops=30, random=True))
    return env


def make_parallel_env(env_name, task, num_envs, device, is_test=False):
    env = ParallelEnv(
        num_envs,
        EnvCreator(lambda: make_base_env(task, is_test=is_test)),
        serial_for_single=True,
        device=device,
    )
    env = TransformedEnv(env)
    env.append_transform(ToTensorImage())
    env.append_transform(GrayScale())
    env.append_transform(Resize(128, 128))
    # env.append_transform(CatFrames(N=4, dim=-3, in_keys=["pixels"]))
    env.append_transform(UnsqueezeTransform(unsqueeze_dim=-3, in_keys=["pixels"]))
    # env.append_transform(CatFrames(N=4, dim=-2, in_keys=["visual_concepts"]))
    env.append_transform(RewardSum())
    env.append_transform(
        RewardSum(in_keys=["original_reward"], out_keys=["original_reward_sum"])
    )
    env.append_transform(StepCounter(max_steps=4500))
    if not is_test:
        env.append_transform(RewardClipping(-1, 1))
    env.append_transform(DoubleToFloat())
    # env.append_transform(VecNorm(in_keys=["pixels"]))
    env.append_transform(ObservationNorm(in_keys=["pixels"], loc=-0.5, scale=1 / 255.0))
    return env


def make_parallel_env_impala(env_name, task, num_envs, device, is_test=False):
    env = ParallelEnv(
        num_envs,
        EnvCreator(lambda: make_base_env(task, is_test=is_test, size=(64, 64))),
        serial_for_single=True,
        device=device,
    )
    env = TransformedEnv(env)
    env.append_transform(ToTensorImage())
    # env.append_transform(GrayScale())
    # env.append_transform(Resize(128, 128))
    # env.append_transform(CatFrames(N=4, dim=-3, in_keys=["pixels"]))
    # env.append_transform(UnsqueezeTransform(unsqueeze_dim=-3, in_keys=["pixels"]))
    # env.append_transform(CatFrames(N=4, dim=-2, in_keys=["visual_concepts"]))
    env.append_transform(RewardSum())
    # env.append_transform(
    #     RewardSum(in_keys=["original_reward"], out_keys=["original_reward_sum"])
    # )
    env.append_transform(StepCounter(max_steps=4500))
    # if not is_test:
    #     env.append_transform(RewardClipping(-1, 1))
    env.append_transform(DoubleToFloat())
    # env.append_transform(VecNorm(in_keys=["pixels"]))
    env.append_transform(ObservationNorm(in_keys=["pixels"], loc=0, scale=1 / 255.0))
    return env


def make_parallel_env_concept(env_name, task, num_envs, device, is_test=False):
    env = ParallelEnv(
        num_envs,
        EnvCreator(lambda: make_base_env(task, size=(64, 64), use_pixels=False)),
        serial_for_single=True,
        device=device,
    )
    env = TransformedEnv(env)
    # env.append_transform(ToTensorImage())
    # env.append_transform(GrayScale())
    # env.append_transform(Resize(128, 128))
    # env.append_transform(CatFrames(N=4, dim=-3, in_keys=["pixels"]))
    # env.append_transform(UnsqueezeTransform(unsqueeze_dim=-3, in_keys=["pixels"]))
    # env.append_transform(CatFrames(N=4, dim=-2, in_keys=["visual_concepts"]))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500))
    if not is_test:
        env.append_transform(RewardClipping(-1, 1))
    env.append_transform(DoubleToFloat())
    # env.append_transform(VecNorm(in_keys=["pixels"]))
    # env.append_transform(ObservationNorm(in_keys=["pixels"], loc=-0.5, scale=1 / 255.0))
    return env


def calculate_success_rate(achievements_complete):
    achievements_bool = achievements_complete >= 1
    # Calculate mean along the tensor, converting True/False to 1s and 0s implicitly
    success_rate = 100.0 * achievements_bool.float().mean(dim=0)
    return success_rate


def mean_achievement_per_episode(achievements_complete):
    return achievements_complete.float().mean(dim=0)


def update_info_dict(achievements_complete):
    out_dict = {}
    success_rate = calculate_success_rate(achievements_complete)
    mean_achieved = mean_achievement_per_episode(achievements_complete)
    achievements = [
        "collect_coal",
        "collect_diamond",
        "collect_drink",
        "collect_iron",
        "collect_sapling",
        "collect_stone",
        "collect_wood",
        "defeat_skeleton",
        "defeat_zombie",
        "eat_cow",
        "eat_plant",
        "make_iron_pickaxe",
        "make_iron_sword",
        "make_stone_pickaxe",
        "make_stone_sword",
        "make_wood_pickaxe",
        "make_wood_sword",
        "place_furnace",
        "place_plant",
        "place_stone",
        "place_table",
        "wake_up",
    ]

    for i, achievement in enumerate(achievements):
        out_dict[f"train/{achievement} score"] = success_rate[i].item()
        out_dict[f"train/{achievement}_mean"] = mean_achieved[i].item()

    out_dict[f"train/score"] = (
        torch.exp(torch.nanmean(np.log(1 + success_rate), -1)) - 1
    ).item()
    return out_dict
