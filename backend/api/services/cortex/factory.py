import gymnasium as gym

def create_cortex_configs_from_observation_space(observation_space):
    """
    Inspects a Gym observation space and returns a suitable cortex configuration
    dictionary for the ChimeraAgent.

    Args:
        observation_space: A gymnasium.spaces.Space object.

    Returns:
        A tuple containing (cortex_configs, cortex_id), where cortex_configs
        is a dictionary to be passed to the agent, and cortex_id is the
        primary ID for the created cortex.
    """
    if isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1:
        # 1D Vector observation space (e.g., CartPole)
        input_dim = observation_space.shape[0]
        cortex_id = "vector_cortex"
        cortex_configs = {
            cortex_id: {
                "type": "DenseCortex",
                "params": {"input_dim": input_dim}
            }
        }
        return cortex_configs, cortex_id

    elif isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 3:
        # 3D Image observation space (e.g., Atari games)
        input_shape = observation_space.shape
        cortex_id = "vision_cortex"
        cortex_configs = {
            cortex_id: {
                "type": "VisionCortex",
                "params": {"input_shape": input_shape}
            }
        }
        return cortex_configs, cortex_id

    else:
        raise NotImplementedError(
            f"Observation space type {type(observation_space)} with shape {observation_space.shape} is not supported yet."
        )
