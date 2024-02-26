import rlrep.utils as utils
import numpy as np


def process_named_function(agent_id, concept, config, rollout, other_agent_batches):
    information = rollout["infos"]
    opponent_name = concept.opponents[0]

    vec = np.zeros((len(information), concept.length))
    if concept.concept_type == "regression":
        if concept.name == "dots_eaten":
            vec = (
                np.array([info["labels"]["dots_eaten_count"] for info in information])
                / concept.scaling_factor
            )
        else:
            vec = (
                np.array([info[concept.name][opponent_name] for info in information])
                / concept.scaling_factor
            )
        
    else:
        vec = np.array([info[concept.name][opponent_name] for info in information])
    vec = vec.reshape((len(information), concept.length))

    return vec
    

def relative_orientation(concept_config, config, rollout):
    """
    Calculates the relative orientation between the agent and all enemies
    """
    agent_id = int(concept_config.agent_id)
    opponent_id = int(concept_config.opponents[0])
    # Get the agent's position
    if rollout["infos"][-1] and type(rollout["infos"][0]) is dict:
        num_agents = config["model"]["custom_model_config"]["num_agents"]
        num_opp_agents = config["model"]["custom_model_config"]["num_opp_agents"]
        n_steps = rollout["obs"].shape[0]
        agent_team = "guard" if rollout["agent_index"][0] < 5 else "attacker"
        relative_orientation_concept = np.zeros((n_steps, 1))

        for i in range(n_steps):
            agent_pos = rollout["infos"][i]["ground_truth"][agent_id]
            opponent_obs = rollout["infos"][i]["ground_truth"][opponent_id]
            # Calculate the relative orientation wtih respect to each the enemy
            if opponent_obs[0] == 0:
                # If the enemy is not alive, we set the concept to indicate so
                relative_orientation_concept[i][0] = -1
            else:
                # If the enemy is alive, we calculate the relative orientation and use that to calculate the concept
                angle_between = utils.angle_between(
                    agent_pos[1:3], opponent_obs[1:3], agent_pos[3]
                )
                relative_orientation_concept[i][0] = angle_between
    else:
        n_steps = rollout["obs"].shape[0]
        relative_orientation_concept = np.zeros((n_steps, 1))
    # rollout[f"relative_orientation_concept_{opponent_id}"] = relative_orientation_concept
    rollout["concept_targets"] = (
        np.concatenate(
            [rollout["concept_targets"], relative_orientation_concept], axis=-1
        )
        if rollout["concept_targets"].size
        else relative_orientation_concept
    )
    # rollout['concept_lengths'].append(5*7)
    return rollout


def distance_between(concept_config, config, rollout):
    """
    Calculates the distance between the agent and all enemies
    """
    agent_id = int(concept_config.agent_id)
    opponent_id = int(concept_config.opponents[0])
    # Get the agent's position
    if rollout["infos"][-1] and type(rollout["infos"][0]) is dict:
        num_agents = config["model"]["custom_model_config"]["num_agents"]
        num_opp_agents = config["model"]["custom_model_config"]["num_opp_agents"]
        n_steps = rollout["obs"].shape[0]
        agent_team = "guard" if rollout["agent_index"][0] < 5 else "attacker"
        distance_between_concept = np.zeros((n_steps, 1))

        for i in range(n_steps):
            agent_pos = rollout["infos"][i]["ground_truth"][agent_id]
            opponent_obs = rollout["infos"][i]["ground_truth"][opponent_id]

            if opponent_obs[0] == 0:
                # If the enemy is not alive, we set the concept to indicate so
                distance_between_concept[i][0] = -1
            else:
                # If the enemy is alive, we calculate the relative orientation and use that to calculate the concept
                distance_between_ = utils.distance_between(
                    agent_pos[1:3], opponent_obs[1:3]
                )
                distance_between_concept[i][0] = distance_between_
    else:
        n_steps = rollout["obs"].shape[0]
        distance_between_concept = np.zeros((n_steps, 1))
    distance_between_concept = distance_between_concept.reshape(n_steps, 1)
    # rollout["distance_between_concept"] = distance_between_concept
    rollout["concept_targets"] = (
        np.concatenate([rollout["concept_targets"], distance_between_concept], axis=-1)
        if rollout["concept_targets"].size
        else distance_between_concept
    )
    # rollout['concept_lengths'].append(5*3)
    return rollout


def distance_from_base(concept_config, config, rollout):
    """
    Calculates the distance between the agent and the base
    """
    agent_id = int(concept_config.agent_id)
    n_steps = rollout["obs"].shape[0]
    # Get the agent's position
    if rollout["infos"][-1] and type(rollout["infos"][0]) is dict:
        distance_from_base_concept = np.zeros((n_steps, 1))

        for i in range(n_steps):
            agent_pos = rollout["infos"][i]["ground_truth"][agent_id]
            distance_from_base_concept[i][0] = utils.distance_between(
                agent_pos[1:3], [0, 0.8]
            )
    else:
        distance_between_concept = np.zeros((n_steps, 1))
    rollout["concept_targets"] = (
        np.concatenate(
            [rollout["concept_targets"], distance_from_base_concept], axis=-1
        )
        if rollout["concept_targets"].size
        else distance_from_base_concept
    )
    # rollout['concept_lengths'].append(5*3)
    return rollout


def can_shoot_ordinal(concept_config, config, rollout):
    """
    Calculates the relative orientation between the agent and all enemies
    """
    agent_id = int(concept_config.agent_id)
    opponent_id = int(concept_config.opponents[0])
    # Get the agent's position
    if rollout["infos"][-1] and type(rollout["infos"][0]) is dict:
        num_agents = config["model"]["custom_model_config"]["num_agents"]
        num_opp_agents = config["model"]["custom_model_config"]["num_opp_agents"]
        n_steps = rollout["obs"].shape[0]
        agent_team = "guard" if rollout["agent_index"][0] < 5 else "attacker"
        can_shoot_concept = np.zeros((n_steps, 2))

        for i in range(n_steps):
            agent_pos = rollout["infos"][i]["ground_truth"][agent_id]
            opponent_obs = rollout["infos"][i]["ground_truth"][opponent_id]

            # Calculate the relative orientation wtih respect to each the enemy
            if opponent_obs[0] == 0:
                # If the enemy is not alive, we set the concept to indicate so
                can_shoot_concept[i][0] = 1
            else:
                # If the enemy is alive, we calculate the relative orientation and use that to calculate the concept
                # angle_between = utils.angle_between(
                #     agent_pos[1:3], opponent_obs[1:3], agent_pos[3]
                # )
                # distance_between = utils.distance_between(
                #     agent_pos[1:3], opponent_obs[1:3]
                # )
                # if angle_between < 0.16 and distance_between < 0.8:
                #     can_shoot_concept[i][1] = 1
                # else:
                #     can_shoot_concept[i][0] = 1
                answer = utils.can_shoot(
                    agent_pos[1:3], agent_pos[3], opponent_obs[1:3]
                )
                if answer:
                    can_shoot_concept[i][1] = 1
                else:
                    can_shoot_concept[i][0] = 1
    else:
        n_steps = rollout["obs"].shape[0]
        can_shoot_concept = np.zeros((n_steps, 2))

    # rollout[f"can_shoot_concept_{opponent_id}"] = can_shoot_concept
    rollout["concept_targets"] = (
        np.concatenate([rollout["concept_targets"], can_shoot_concept], axis=-1)
        if rollout["concept_targets"].size
        else can_shoot_concept
    )
    return rollout


def agent_targeting_ordinal(concept_config, config, rollout):
    """
    Calculates the relative orientation between the agent and all enemies
    """
    num_agents = config["model"]["custom_model_config"]["num_agents"]
    num_opp_agents = config["model"]["custom_model_config"]["num_opp_agents"]

    n_steps = rollout["obs"].shape[0]
    agent_team = "guard" if rollout["agent_index"][0] < 5 else "attacker"
    agent_targeting_concept = np.zeros((n_steps, concept_config.length))
    agent_id = int(concept_config.agent_id)
    # Get the agent's position
    if rollout["infos"][-1] and type(rollout["infos"][0]) is dict:
        for i in range(n_steps):
            info_list = rollout["infos"][i]["info_list"][agent_id]
            attacker_pairing = info_list["attacker_pairing"]
            agent_targeting_concept[i][attacker_pairing] = 1

    # rollout[f"agent_targeting_concept"] = agent_targeting_concept
    rollout["concept_targets"] = (
        np.concatenate([rollout["concept_targets"], agent_targeting_concept], axis=-1)
        if rollout["concept_targets"].size
        else agent_targeting_concept
    )
    return rollout


def attacker_stratagy(concept_config, config, rollout):
    """
    Calculates the relative orientation between the agent and all enemies
    """
    num_agents = config["model"]["custom_model_config"]["num_agents"]
    num_opp_agents = config["model"]["custom_model_config"]["num_opp_agents"]

    n_steps = rollout["obs"].shape[0]
    agent_team = "guard" if rollout["agent_index"][0] < 5 else "attacker"
    agent_stratagy_concept = np.zeros((n_steps, concept_config.length))
    agent_id = int(concept_config.agent_id)
    # Get the agent's position
    if rollout["infos"][-1] and type(rollout["infos"][0]) is dict:
        for i in range(n_steps):
            attacker_stratagy = rollout["infos"][i]["stratagy"]
            agent_stratagy_concept[i][attacker_stratagy] = 1

    # rollout[f"agent_targeting_concept"] = agent_targeting_concept
    rollout["concept_targets"] = (
        np.concatenate([rollout["concept_targets"], agent_stratagy_concept], axis=-1)
        if rollout["concept_targets"].size
        else agent_stratagy_concept
    )
    return rollout


def tom_extraction(agent_id, total_length, rollout, other_agent_batches):
    n_steps = rollout["obs"].shape[0]
    if "concept_targets" in rollout:
        _, agent_rollout = other_agent_batches[int(agent_id)]
        tom_concept = agent_rollout["concept_targets"][:, :total_length]
    else:
        tom_concept = np.zeros((n_steps, total_length))


    rollout["concept_targets"] = (
        np.concatenate([rollout["concept_targets"], tom_concept], axis=-1)
        if rollout["concept_targets"].size
        else tom_concept
    )
    return rollout


def relative_orientation_ordinal(concept_config, config, rollout):
    """
    Calculates the relative orientation between the agent and all enemies
    """
    agent_id = int(concept_config.agent_id)
    opponent_id = int(concept_config.opponents[0])
    # Get the agent's position
    if rollout["infos"][-1] and type(rollout["infos"][0]) is dict:
        num_agents = config["model"]["custom_model_config"]["num_agents"]
        num_opp_agents = config["model"]["custom_model_config"]["num_opp_agents"]
        n_steps = rollout["obs"].shape[0]
        agent_team = "guard" if rollout["agent_index"][0] < 5 else "attacker"
        relative_orientation_concept = np.zeros((n_steps, 7))

        for i in range(n_steps):
            agent_pos = rollout["infos"][i]["ground_truth"][agent_id]
            opponent_obs = rollout["infos"][i]["ground_truth"][opponent_id]
            # Calculate the relative orientation wtih respect to each the enemy
            if opponent_obs[0] == 0:
                # If the enemy is not alive, we set the concept to indicate so
                relative_orientation_concept[i][0] = 1
            else:
                # If the enemy is alive, we calculate the relative orientation and use that to calculate the concept
                angle_between = utils.angle_between(
                    agent_pos[1:3], opponent_obs[1:3], agent_pos[3]
                )
                if angle_between > np.pi / 2 or angle_between < -np.pi / 2:
                    relative_orientation_concept[i][1] = 1
                elif angle_between > -np.pi / 10 and angle_between < np.pi / 10:
                    relative_orientation_concept[i][2] = 1
                elif angle_between < -np.pi / 10 and angle_between > -np.pi / 4:
                    relative_orientation_concept[i][3] = 1
                elif angle_between < -np.pi / 4 and angle_between > -np.pi / 2:
                    relative_orientation_concept[i][4] = 1
                elif angle_between > np.pi / 10 and angle_between < np.pi / 4:
                    relative_orientation_concept[i][5] = 1
                elif angle_between > np.pi / 4 and angle_between < np.pi / 2:
                    relative_orientation_concept[i][6] = 1

    else:
        n_steps = rollout["obs"].shape[0]
        relative_orientation_concept = np.zeros((n_steps, 7))
    # rollout[f"relative_orientation_concept_{opponent_id}"] = relative_orientation_concept
    rollout["concept_targets"] = (
        np.concatenate(
            [rollout["concept_targets"], relative_orientation_concept], axis=-1
        )
        if rollout["concept_targets"].size
        else relative_orientation_concept
    )
    # rollout['concept_lengths'].append(5*7)
    return rollout


def distance_between_ordinal(concept_config, config, rollout):
    """
    Calculates the distance between the agent and all enemies
    """
    agent_id = int(concept_config.agent_id)
    opponent_id = int(concept_config.opponents[0])
    # Get the agent's position
    if rollout["infos"][-1] and type(rollout["infos"][0]) is dict:
        num_agents = config["model"]["custom_model_config"]["num_agents"]
        num_opp_agents = config["model"]["custom_model_config"]["num_opp_agents"]
        n_steps = rollout["obs"].shape[0]
        agent_team = "guard" if rollout["agent_index"][0] < 5 else "attacker"
        distance_between_concept = np.zeros((n_steps, 10))

        for i in range(n_steps):
            agent_pos = rollout["infos"][i]["ground_truth"][agent_id]
            opponent_obs = rollout["infos"][i]["ground_truth"][opponent_id]

            if opponent_obs[0] == 0:
                # If the enemy is not alive, we set the concept to indicate so
                distance_between_concept[i][0] = 1
            else:
                # If the enemy is alive, we calculate the relative orientation and use that to calculate the concept
                a_x, a_y = agent_pos[1:3]
                opp_x, opp_y = opponent_obs[1:3]
                xdiff = a_x - opp_x
                ydiff = a_y - opp_y

                if xdiff < -0.75 and ydiff < -0.75:
                    distance_between_concept[i][1] = 1
                elif xdiff < -0.75 and ydiff > 0.75:
                    distance_between_concept[i][2] = 1
                elif xdiff > 0.75 and ydiff < -0.75:
                    distance_between_concept[i][3] = 1
                elif xdiff > 0.75 and ydiff > 0.75:
                    distance_between_concept[i][4] = 1
                elif xdiff < -0.75 and ydiff < 0.75 and ydiff > -0.75:
                    distance_between_concept[i][5] = 1
                elif xdiff > 0.75 and ydiff < 0.75 and ydiff > -0.75:
                    distance_between_concept[i][6] = 1
                elif xdiff < 0.75 and xdiff > -0.75 and ydiff < 0.75 and ydiff > -0.75:
                    distance_between_concept[i][7] = 1
                elif xdiff < 0.75 and xdiff > -0.75 and ydiff > 0.75:
                    distance_between_concept[i][8] = 1
                elif xdiff < 0.75 and xdiff > -0.75 and ydiff < -0.75:
                    distance_between_concept[i][9] = 1

                # distance_between_ = utils.distance_between(
                #     agent_pos[1:3], opponent_obs[1:3]
                # )
                # distance_between_concept[i][0] = distance_between_
    else:
        n_steps = rollout["obs"].shape[0]
        distance_between_concept = np.zeros((n_steps, 1))
    distance_between_concept = distance_between_concept.reshape(n_steps, 1)
    # rollout["distance_between_concept"] = distance_between_concept
    rollout["concept_targets"] = (
        np.concatenate([rollout["concept_targets"], distance_between_concept], axis=-1)
        if rollout["concept_targets"].size
        else distance_between_concept
    )
    # rollout['concept_lengths'].append(5*3)
    return rollout


def distance_between_mape(concept_config, config, rollout):
    """
    Calculates the distance between the agent and all enemies
    """
    agent_id = int(concept_config.agent_id)
    opponent_id = int(concept_config.opponents[0])
    # Get the agent's position
    if rollout["infos"][-1] and type(rollout["infos"][0]) is dict:
        num_agents = config["model"]["custom_model_config"]["num_agents"]
        num_opp_agents = config["model"]["custom_model_config"]["num_opp_agents"]
        n_steps = rollout["obs"].shape[0]
        agent_team = "guard" if rollout["agent_index"][0] < 5 else "attacker"
        distance_between_concept = np.zeros((n_steps, 1))

        for i in range(n_steps):
            agent_pos = rollout["infos"][i]["ground_truth"][agent_id]
            opponent_obs = rollout["infos"][i]["ground_truth"][opponent_id]

            if opponent_obs[0] == 0:
                # If the enemy is not alive, we set the concept to indicate so
                distance_between_concept[i][0] = -1
            else:
                # If the enemy is alive, we calculate the relative orientation and use that to calculate the concept
                distance_between_ = utils.distance_between(
                    agent_pos[1:3], opponent_obs[1:3]
                )
                distance_between_concept[i][0] = distance_between_
    else:
        n_steps = rollout["obs"].shape[0]
        distance_between_concept = np.zeros((n_steps, 1))
    distance_between_concept = distance_between_concept.reshape(n_steps, 1)
    # rollout["distance_between_concept"] = distance_between_concept
    rollout["concept_targets"] = (
        np.concatenate([rollout["concept_targets"], distance_between_concept], axis=-1)
        if rollout["concept_targets"].size
        else distance_between_concept
    )
    # rollout['concept_lengths'].append(5*3)
    return rollout


concept_function_dict = {
    "distance_between_mape": distance_between_mape,
    "distance_between_ordinal": distance_between_ordinal,
    "relative_orientation_ordinal": relative_orientation_ordinal,
    "attacker_stratagy": attacker_stratagy,
    "agent_targeting_ordinal" : agent_targeting_ordinal,
    "distance_from_base" : distance_from_base,
}