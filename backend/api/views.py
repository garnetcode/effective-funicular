import os
import uuid
import json
import threading
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
from .services.chimera_agent import ChimeraAgent

# --- Helper Functions ---

def get_agent_service(agent_id):
    """Loads or initializes a ChimeraAgent instance."""
    if not all(c.isalnum() or c in '-_' for c in agent_id):
        return None
    try:
        # Pass the hyperparams from the request if creating a new agent,
        # otherwise, they are loaded from the saved state.
        return ChimeraAgent(agent_id=agent_id, load_from_storage=True)
    except FileNotFoundError:
        return None

def get_agent_config(agent_id):
    """Helper to get just the config part of an agent without loading the whole thing."""
    if not all(c.isalnum() or c in '-_' for c in agent_id):
        return None
    storage_path = os.path.join('backend', 'storage', f'{agent_id}.npz')
    if not os.path.exists(storage_path):
        return None
    with np.load(storage_path, allow_pickle=True) as data:
        cortex_configs_json = str(data['cortex_configs_json'])
        hyperparams_json = str(data['hyperparams_json'])
        config = {
            'cortex_configs': json.loads(cortex_configs_json),
            'hyperparams': json.loads(hyperparams_json),
            'dimensions': int(data['dimensions'])
        }
    return config


# --- API Views ---

class SessionCreate(APIView):
    """Creates a new game session."""
    def post(self, request, format=None):
        environment_id = request.data.get('environment_id')
        agent_tag = request.data.get('agent_tag')

        if not environment_id or not agent_tag:
            return Response({'error': 'environment_id and agent_tag are required'}, status=status.HTTP_400_BAD_REQUEST)

        session_id = f"session-{uuid.uuid4()}"

        # In a real application, you would store session details in a database or cache.
        # For the purpose of this training script, we just need to return a valid session ID
        # so the connector can proceed to the WebSocket connection.
        # The WebSocket consumer will handle the actual game state.

        return Response({
            'success': True,
            'session_id': session_id,
            'environment_id': environment_id,
            'agent_tag': agent_tag
        }, status=status.HTTP_201_CREATED)


class EnvironmentList(APIView):
    """Lists available Gymnasium environments."""
    def get(self, request, format=None):
        # A curated list of classic environments that don't require special dependencies
        environments = [
            {'id': 'CartPole-v1', 'name': 'CartPole'},
            {'id': 'Acrobot-v1', 'name': 'Acrobot'},
            {'id': 'MountainCar-v0', 'name': 'Mountain Car'},
            {'id': 'Pendulum-v1', 'name': 'Pendulum'},
        ]
        return Response(environments)


class CortexSpecificationList(APIView):
    """Lists available cortex types and their input specifications."""
    def get(self, request, format=None):
        # This could be automated further with class introspection, but a manual
        # definition is clearer and safer for now.
        specs = [
            {
                "type": "DenseCortex",
                "description": "Processes a fixed-size vector input. Requires an 'input_dim' parameter during agent creation.",
                "input_spec": {
                    "type": "vector",
                    "dtype": "float",
                    "shape": ["input_dim"]
                },
                "params": [
                    {"name": "input_dim", "type": "integer", "description": "The dimensionality of the input vector."}
                ]
            },
            {
                "type": "TextCortex",
                "description": "Processes a string of any length into a fixed-size embedding.",
                "input_spec": {
                    "type": "string"
                },
                "params": []
            },
            {
                "type": "VisionCortex",
                "description": "Processes an image into a fixed-size embedding. Requires a file path as input.",
                "input_spec": {
                    "type": "image_path",
                    "format": ["png", "jpg", "jpeg"]
                },
                "params": []
            }
        ]
        return Response(specs)


class AgentList(APIView):
    """List all agents or create a new one."""
    def get(self, request, format=None):
        storage_dir = os.path.join('backend', 'storage')
        agents = []
        if os.path.exists(storage_dir):
            for filename in os.listdir(storage_dir):
                if filename.endswith('.npz'):
                    agent_id = filename[:-4]
                    agents.append({'id': agent_id, **get_agent_config(agent_id)})
        return Response(agents)

    def post(self, request, format=None):
        """Creates a new Chimera Agent."""
        agent_id = f"agent-{uuid.uuid4()}"
        dimensions = request.data.get('dimensions', 64)
        cortex_configs = request.data.get('cortex_configs', {})
        hyperparams = request.data.get('hyperparams', {})

        ChimeraAgent(
            agent_id=agent_id,
            dimensions=dimensions,
            cortex_configs=cortex_configs,
            load_from_storage=False,
            **hyperparams
        )
        return Response({'id': agent_id, 'cortex_configs': cortex_configs, 'hyperparams': hyperparams}, status=status.HTTP_201_CREATED)


class AgentDetail(APIView):
    """Retrieve or delete an agent."""
    def get(self, request, agent_id, format=None):
        config = get_agent_config(agent_id)
        if config is None:
            return Response(status=status.HTTP_404_NOT_FOUND)
        return Response(config)

    def delete(self, request, agent_id, format=None):
        """Deletes an agent's state file."""
        if not all(c.isalnum() or c in '-_' for c in agent_id):
            return Response({'error': 'Invalid agent ID format.'}, status=status.HTTP_400_BAD_REQUEST)

        storage_path = os.path.join('backend', 'storage', f'{agent_id}.npz')

        if os.path.exists(storage_path):
            try:
                os.remove(storage_path)
                # Also remove the faiss index if it exists, for cleanup
                faiss_path = os.path.join('backend', 'storage', f'{agent_id}.faiss')
                if os.path.exists(faiss_path):
                    os.remove(faiss_path)
                return Response(status=status.HTTP_204_NO_CONTENT)
            except OSError as e:
                return Response({'error': f'Error deleting file: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(status=status.HTTP_404_NOT_FOUND)


class LearnAssociative(APIView):
    """
    Endpoint for LEARN Mode: learns a new pattern in the Hopfield core.
    """
    def post(self, request, agent_id, format=None):
        service = get_agent_service(agent_id)
        if service is None: return Response(status=status.HTTP_404_NOT_FOUND)

        cortex_id = request.data.get('cortex_id')
        raw_input = request.data.get('raw_input')
        if not all([cortex_id, raw_input is not None]):
            return Response({'error': 'cortex_id and raw_input are required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            embedding = service.perceive(cortex_id, raw_input)
            result = service.learn_associative(embedding)
            return Response(result)
        except Exception as e:
            return Response({'error': f'An unexpected error occurred: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class OrganizeMemory(APIView):
    """
    Endpoint for ORGANIZE Mode: organizes memory for a given pattern.
    """
    def post(self, request, agent_id, format=None):
        service = get_agent_service(agent_id)
        if service is None: return Response(status=status.HTTP_404_NOT_FOUND)

        pattern_id = request.data.get('pattern_id')
        if pattern_id is None:
            return Response({'error': 'pattern_id is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            result = service.organize_memory(int(pattern_id))
            return Response(result)
        except (ValueError, KeyError) as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': f'An unexpected error occurred: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ConsolidateMemories(APIView):
    """
    Endpoint for triggering offline memory consolidation.
    """
    def post(self, request, agent_id, format=None):
        service = get_agent_service(agent_id)
        if service is None: return Response(status=status.HTTP_404_NOT_FOUND)

        n_replays = request.data.get('n_replays', 1)

        try:
            result = service.consolidate_memories(int(n_replays))
            return Response(result)
        except Exception as e:
            return Response({'error': f'An unexpected error occurred: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AgentStructure(APIView):
    """Endpoint to get the agent's GNG/STAG hierarchical graph structure."""
    def get(self, request, agent_id, format=None):
        service = get_agent_service(agent_id)
        if service is None: return Response(status=status.HTTP_404_NOT_FOUND)
        return Response(service.get_graph_structure())


class ProbeActivity(APIView):
    """
    Endpoint for probing the agent's internal activity for a given input.
    Returns the activation path through the hierarchy.
    """
    def post(self, request, agent_id, format=None):
        service = get_agent_service(agent_id)
        if service is None: return Response(status=status.HTTP_404_NOT_FOUND)

        cortex_id = request.data.get('cortex_id')
        raw_input = request.data.get('raw_input')
        if not all([cortex_id, raw_input is not None]):
            return Response({'error': 'cortex_id and raw_input are required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            result = service.probe_activity(cortex_id, raw_input)
            return Response(result)
        except Exception as e:
            return Response({'error': f'An unexpected error occurred: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SelectAction(APIView):
    """Endpoint for an agent to select an action based on a perceived input."""
    def post(self, request, agent_id, format=None):
        service = get_agent_service(agent_id)
        if service is None: return Response(status=status.HTTP_404_NOT_FOUND)

        cortex_id = request.data.get('cortex_id')
        raw_input = request.data.get('raw_input')
        if not all([cortex_id, raw_input is not None]):
            return Response({'error': 'cortex_id and raw_input are required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            state_embedding = service.perceive(cortex_id, raw_input)
            action, log_prob, _ = service.select_action(state_embedding)
            return Response({'action': int(action), 'log_probability': float(log_prob)})
        except Exception as e:
            return Response({'error': f'An unexpected error occurred: {e}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# --- Training ---

def get_env_config(env):
    """Inspects a gymnasium environment to determine agent configuration."""
    import gymnasium as gym
    obs_space = env.observation_space
    if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 1:
        raise NotImplementedError("Only 1D Box observation spaces are supported.")

    act_space = env.action_space
    if not isinstance(act_space, gym.spaces.Discrete):
        raise NotImplementedError("Only Discrete action spaces are supported.")

    input_dim = obs_space.shape[0]
    cortex_configs = {"vector_input": {"type": "DenseCortex", "params": {"input_dim": input_dim}}}
    n_actions = act_space.n
    return cortex_configs, "vector_input", n_actions

def run_training_loop(agent, env, cortex_id, episodes=500):
    """The main training loop, adapted from train.py to run in a thread."""
    intrinsic_reward_coeff = agent.hyperparams.get('intrinsic_reward_coefficient', 0.1)
    print(f"Starting background training for agent '{agent.agent_id}' in '{env.spec.id}'...")
    total_rewards = []
    for episode in range(episodes):
        try:
            state, info = env.reset()
            terminated, truncated, episode_reward = False, False, 0
            while not (terminated or truncated):
                state_embedding = agent.perceive(cortex_id, state)

                # Check for novelty and create a new pattern if needed
                pattern_id = None
                if agent.is_novel(state_embedding):
                    result = agent.learn_associative(state_embedding)
                    pattern_id = result.get('pattern_id')

                action, log_prob, internal_state = agent.select_action(state_embedding)
                next_state, extrinsic_reward, terminated, truncated, _ = env.step(action)

                # Calculate intrinsic reward based on the novelty of the *next* state
                next_state_embedding = agent.perceive(cortex_id, next_state)
                novelty_error = agent.get_state_novelty_error(next_state_embedding)
                intrinsic_reward = intrinsic_reward_coeff * novelty_error

                total_reward = extrinsic_reward + intrinsic_reward

                agent.record_experience(internal_state, action, log_prob, total_reward, pattern_id=pattern_id)
                state = next_state
                episode_reward += extrinsic_reward

            agent.train()
            total_rewards.append(episode_reward)

            if episode % 10 == 0:
                avg_reward = np.mean(total_rewards[-100:])

                # Send metrics over WebSocket
                channel_layer = get_channel_layer()
                async_to_sync(channel_layer.group_send)(
                    f"training_{agent.agent_id}",
                    {
                        "type": "training.message",
                        "message": {
                            "episode": episode,
                            "total_reward": episode_reward,
                            "avg_reward": avg_reward,
                        },
                    },
                )
        except Exception as e:
            print(f"Error during training loop for agent {agent.agent_id}: {e}")
            break

    env.close()
    print(f"Finished background training for agent '{agent.agent_id}'.")


class StartTraining(APIView):
    """Starts a training session for an agent in a given environment."""
    def post(self, request, agent_id, format=None):
        import gymnasium as gym
        env_id = request.data.get('env_id')
        if not env_id:
            return Response({'error': 'env_id is required'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            env = gym.make(env_id)
            cortex_configs, cortex_id, n_actions = get_env_config(env)
        except Exception as e:
            return Response({'error': f'Failed to create environment: {e}'}, status=status.HTTP_400_BAD_REQUEST)

        # Get the agent, or create one if it doesn't exist
        try:
            agent = ChimeraAgent(agent_id=agent_id, load_from_storage=True)
            # Ensure the loaded agent is compatible with the environment
            if cortex_id not in agent.cortexes:
                agent.update_cortex_config(cortex_configs)
            if agent.n_actions != n_actions:
                agent.update_action_space(n_actions)
        except FileNotFoundError:
            agent = ChimeraAgent(
                agent_id=agent_id,
                dimensions=64, # Default dimension
                n_actions=n_actions,
                cortex_configs=cortex_configs,
                load_from_storage=False
            )

        # Run training in a background thread
        training_thread = threading.Thread(
            target=run_training_loop,
            args=(agent, env, cortex_id)
        )
        training_thread.daemon = True
        training_thread.start()

        return Response({'status': f'Training started for agent {agent_id} in {env_id}'}, status=status.HTTP_202_ACCEPTED)
