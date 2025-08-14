import os
import uuid
import json
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
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


class AgentStructure(APIView):
    """Endpoint to get the agent's GNG/STAG hierarchical graph structure."""
    def get(self, request, agent_id, format=None):
        service = get_agent_service(agent_id)
        if service is None: return Response(status=status.HTTP_404_NOT_FOUND)
        return Response(service.get_graph_structure())


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
