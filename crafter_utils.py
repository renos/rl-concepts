import numpy as np
from crafter_description import id_to_item


def find_closest_blocks(semantic_map):
    agent_position = (3, 4)  # Center of a 9x7 map
    closest_positions = {
        item: {"distance": float("inf"), "relative_position": None}
        for item in id_to_item
    }

    for y in range(semantic_map.shape[0]):
        for x in range(semantic_map.shape[1]):
            block_id = semantic_map[y, x]
            block_name = id_to_item[block_id]
            distance = abs(agent_position[0] - x) + abs(agent_position[1] - y)

            if distance < closest_positions[block_name]["distance"]:
                closest_positions[block_name]["distance"] = distance
                # Store the relative position instead of the absolute position
                closest_positions[block_name]["relative_position"] = (
                    x - agent_position[0],
                    y - agent_position[1],
                )

    # Remove entries for blocks not found and 'None' type
    closest_positions = {
        k: v
        for k, v in closest_positions.items()
        if v["relative_position"] is not None and k != "None"
    }

    return closest_positions


def update_relative_positions(closest_blocks, dx, dy):
    """
    Update the relative positions of blocks based on player movement.

    Parameters:
    - closest_blocks: The dictionary of blocks with their relative positions and distances.
    - dx, dy: The change in x and y positions of the player.

    Returns:
    - A new dictionary with updated relative positions.
    """
    updated_blocks = {}
    for block, info in closest_blocks.items():
        # Update the relative position based on player movement
        new_rel_pos = (
            info["relative_position"][0] - dx,
            info["relative_position"][1] - dy,
        )
        # Recalculate the Manhattan distance based on the new relative position
        new_distance = abs(new_rel_pos[0]) + abs(new_rel_pos[1])
        updated_blocks[block] = {
            "distance": new_distance,
            "relative_position": new_rel_pos,
        }
    return updated_blocks


def merge_closest_blocks(old_blocks, new_blocks):
    """
    Merge the old and new closest block information, keeping the closest.

    Parameters:
    - old_blocks: The dictionary of old blocks with their relative positions and distances.
    - new_blocks: The dictionary of new blocks with their relative positions and distances.

    Returns:
    - A merged dictionary with the closest block information.
    """
    # Copy old blocks to start with
    merged_blocks = old_blocks.copy()
    for block, new_info in new_blocks.items():
        if (
            block not in merged_blocks
            or new_info["distance"] < merged_blocks[block]["distance"]
        ):
            merged_blocks[block] = new_info  # Update if closer in the new map
    return merged_blocks


def filter_blocks_by_distance(blocks):
    """
    Filter out blocks that are within a certain relative distance.

    Parameters:
    - blocks: The dictionary of blocks with their relative positions and distances.
    - max_rel_distance: The maximum relative distance threshold for keeping a block.

    Returns:
    - A filtered dictionary with blocks outside the specified relative distance.
    """
    filtered_blocks = {
        block: info
        for block, info in blocks.items()
        if abs(info["relative_position"][0]) > 3
        or abs(info["relative_position"][1]) > 4
    }
    return filtered_blocks


def update_and_merge_closest_blocks(old_blocks, old_pos, new_pos, new_semantic_map):
    """
    Update and merge closest blocks based on player movement and a new semantic map.
    If old blocks are None, return the closest blocks from the new semantic map directly.

    Parameters:
    - old_blocks: Dictionary of old blocks with their relative positions and distances, or None.
    - old_pos: Tuple (x, y) representing the old position of the player.
    - new_pos: Tuple (x, y) representing the new position of the player.
    - new_semantic_map: Numpy array representing the new semantic map of the environment.

    Returns:
    - A merged dictionary with the updated and closest block information, or the closest blocks from the new semantic map if old_blocks is None.
    """
    # If there are no old blocks, directly find and return the closest blocks from the new semantic map
    if old_blocks is None:
        return find_closest_blocks(new_semantic_map)

    # Calculate the difference in position
    dx, dy = new_pos[0] - old_pos[0], new_pos[1] - old_pos[1]

    # Update the relative positions of the old blocks
    updated_old_blocks = update_relative_positions(old_blocks, dx, dy)

    # Filter out blocks that are within render distance (the player could have altered them, so this representation is no longer useful)
    updated_old_blocks = filter_blocks_by_distance(updated_old_blocks)

    # Calculate the closest blocks from the new semantic map
    new_closest_blocks = find_closest_blocks(new_semantic_map)

    # Merge the updated old blocks with the new blocks
    merged_blocks = merge_closest_blocks(updated_old_blocks, new_closest_blocks)

    return merged_blocks
