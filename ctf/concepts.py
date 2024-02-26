import numpy as np
import scipy.signal
from typing import Dict, List, Optional
from ray.rllib.policy.sample_batch import SampleBatch
from .concept_functions import concept_function_dict
from .concept_functions import process_named_function


def compute_concepts(config: Dict, rollout: SampleBatch, other_agent_batches):
    # Computes concepts from a rollout.
    # Args:  rollout (SampleBatch): The rollout to compute concepts from.
    #        include_concepts (bool): whether to compute concepts.
    #        concept_function_list (list): A list of concepts functions to iterativly compute.
    # Returns:
    #        SampleBatch: The postprocessed, modified SampleBatch.
    if not config["include_concepts"] or config["concept_configs"] == 0:
        return rollout
    agent_name = config["model"]["custom_model_config"]["policy_name"]
    concept_configs = config["concept_configs"][agent_name]

    if not ((rollout["infos"][-1] != 0) and type(rollout["infos"][0]) is dict):
        # if config["per_frame_concepts"]:
        #     frame_size = 4
        # else:
        #     frame_size = 1
        rollout[f"concept_targets_{agent_name}"] = np.zeros(
            (rollout["obs"].shape[0], concept_configs.total_length)
        )
    elif len(concept_configs.configs) > 0:
        rollout[f"concept_targets_{agent_name}"] = np.array([])

        agent_id = list(set(config["agent_ids"]) - set(other_agent_batches.keys()))[0]
        concept_configs = config["concept_configs"][agent_id]

        if "alive" not in rollout["infos"][0].keys():
            solve_one = list(rollout["infos"][0].values())[0]
            rollout["infos"][0] = solve_one[agent_id]

        for concept_config in concept_configs.configs:
            concept_name = concept_config.name
            if concept_name in rollout["infos"][0]:
                concept_to_append = process_named_function(
                    agent_id, concept_config, config, rollout, other_agent_batches
                )
            else:
                concept_to_append = concept_function_dict[concept_name](
                    agent_id, concept_config, config, rollout, other_agent_batches
                )

            rollout[f"concept_targets_{agent_name}"] = (
                np.concatenate(
                    [rollout[f"concept_targets_{agent_name}"], concept_to_append],
                    axis=-1,
                )
                if rollout[f"concept_targets_{agent_name}"].size
                else concept_to_append
            )

    # rollout[f"concept_targets_{agent_name}"] = np.expand_dims(
    #     rollout[f"concept_targets_{agent_name}"], 1
    # )
    return rollout
