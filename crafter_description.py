import numpy as np
from contrib import crafter

env = crafter.Env(size=(224, 224))
action_space = env.action_space

vitals = [
    "health",
    "food",
    "drink",
    "energy",
]

rot = np.array([[0, -1], [1, 0]])
directions = ["front", "right", "back", "left"]

id_to_item = [0] * 19
import itertools
import difflib

for name, ind in itertools.chain(
    env._world._mat_ids.items(), env._sem_view._obj_ids.items()
):
    name = (
        str(name)[str(name).find("objects.") + len("objects.") : -2].lower()
        if "objects." in str(name)
        else str(name)
    )
    id_to_item[ind] = name
player_idx = id_to_item.index("player")


def describe_inventory(info):
    result = ""

    status_str = "Vitals:\n{}".format(
        "\n".join(["- {}: {}/9".format(v, info["inventory"][v]) for v in vitals])
    )
    result += status_str + "\n\n"

    inventory_str = "\n".join(
        [
            "- {}: {}".format(i, num)
            for i, num in info["inventory"].items()
            if i not in vitals and num != 0
        ]
    )
    inventory_str = (
        "Inventory:\n{}".format(inventory_str) if inventory_str else "Inventory: empty"
    )
    result += inventory_str  # + "\n\n"

    return result.strip()


REF = np.array([0, 1])


def rotation_matrix(v1, v2):
    dot = np.dot(v1, v2)
    cross = np.cross(v1, v2)
    rotation_matrix = np.array([[dot, -cross], [cross, dot]])
    return rotation_matrix


def describe_loc(ref, P, target_facing):
    # print(ref,P)
    desc = []
    if ref[1] > P[1]:
        desc.append("north")
    elif ref[1] < P[1]:
        desc.append("south")
    if ref[0] > P[0]:
        desc.append("west")
    elif ref[0] < P[0]:
        desc.append("east")

    result = "-".join(desc)

    if P[0] == target_facing[0] and P[1] == target_facing[1]:
        result += " (facing)"

    return result


def describe_env(info):
    assert info["semantic"][info["player_pos"][0], info["player_pos"][1]] == player_idx
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
    center = np.array([info["view"][0] // 2, info["view"][1] // 2 - 1])
    result = ""
    x = np.arange(semantic.shape[1])
    y = np.arange(semantic.shape[0])
    x1, y1 = np.meshgrid(x, y)
    loc = np.stack((y1, x1), axis=-1)
    dist = np.absolute(center - loc).sum(axis=-1)
    obj_info_list = []
    grass_idx = id_to_item.index("grass")

    facing = info["player_facing"]
    target_facing = (center[0] + facing[0], center[1] + facing[1])
    target = id_to_item[semantic[target_facing[0], target_facing[1]]]
    # obs = "You face {} to your front.".format(target, describe_loc(np.array([0,0]),facing))
    around = {}
    # around["below"]=id_to_item[semantic[center[0],center[1]]]
    for d in [
        [center[0] - 1, center[1]],
        [center[0] + 1, center[1]],
        [center[0], center[1] - 1],
        [center[0], center[1] + 1],
    ]:
        around[describe_loc(np.array([0, 0]), np.array(d) - center, facing)] = (
            id_to_item[semantic[d[0], d[1]]]
        )

    obs = "Observation (1-step):\n" + "\n".join(
        ["- {}: {}".format(o, d) for d, o in around.items()]
    )

    for idx in np.unique(semantic):
        if idx == player_idx or idx == grass_idx:
            continue

        distances = np.where(semantic == idx, dist, np.inf)
        smallest_indices = np.unravel_index(
            np.argsort(distances, axis=None), distances.shape
        )
        smallest_indices = [
            (smallest_indices[0][i], smallest_indices[1][i])
            for i in range(min(2, np.count_nonzero(semantic == idx)))
        ]

        for i in range(len(smallest_indices)):
            smallest = smallest_indices[i]
            obj_info_list.append(
                (
                    id_to_item[idx],
                    dist[smallest],
                    describe_loc(np.array([0, 0]), smallest - center, facing),
                )
            )

    if len(obj_info_list) > 0:
        status_str = "Observation (nearby):\n{}".format(
            "\n".join(
                [
                    "- {} {} steps to your {}".format(
                        name.replace("arrow", "flying-arrow"), dist, loc
                    )
                    for name, dist, loc in obj_info_list
                ]
            )
        )
    else:
        status_str = "Observation (nearby): nothing nearby"
    result += obs.strip() + "\n\n" + status_str

    return result.strip()


def describe_act(info):
    result = ""

    action_str = info["action"].replace("do_", "interact_")
    action_str = action_str.replace("move_up", "move 1 step north")
    action_str = action_str.replace("move_down", "move 1 step south")
    action_str = action_str.replace("move_left", "move 1 step west")
    action_str = action_str.replace("move_right", "move 1 step east")

    # act = "Player took action {}.".format(action_str)
    # result+= act #+ "\n\n"

    return action_str.strip()


def describe_status(info):

    if info["sleeping"]:
        return "Player is sleeping, and will not be able take actions until energy is full.\n\n"
    elif info["dead"]:
        return "Player died.\n\n"
    else:
        return ""


def describe_frame(info):
    result = ""

    # result+=describe_act(info)
    result += describe_status(info)
    # result+="\n\n"
    result += describe_env(info)
    result += "\n\n"
    result += describe_inventory(info)

    return describe_act(info).strip(), result.strip()


action_list = [
    "Noop",
    "Move West",
    "Move East",
    "Move North",
    "Move South",
    "Do",
    "Sleep",
    "Place Stone",
    "Place Table",
    "Place Furnace",
    "Place Plant",
    "Make Wood Pickaxe",
    "Make Stone Pickaxe",
    "Make Iron Pickaxe",
    "Make Wood Sword",
    "Make Stone Sword",
    "Make Iron Sword",
]
action_list = [a.lower() for a in action_list]


def match_act(string):
    matches = difflib.get_close_matches(string.lower(), action_list, n=1)
    if matches:
        print('Action matched "{}" to "{}"'.format(string, matches[0]))
        return action_list.index(matches[0])
    else:
        print('LLM failed with output "{}", taking action Do...'.format(string))
        return action_list.index("do")
