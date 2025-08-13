import os
import uuid
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services.cognitive_architecture_service import CognitiveArchitectureService, text_to_embedding

# --- Helper Functions ---

def get_network_service(network_id):
    """Loads or initializes a CognitiveArchitectureService instance."""
    # Basic security check for network_id to prevent directory traversal
    if not all(c.isalnum() or c in '-_' for c in network_id):
        return None
    try:
        return CognitiveArchitectureService(network_id=network_id, load_from_storage=True)
    except FileNotFoundError:
        return None

def get_network_config(network_id):
    """Helper to get just the config part of a network without loading the whole thing."""
    if not all(c.isalnum() or c in '-_' for c in network_id):
        return None
    storage_path = os.path.join('backend', 'storage', f'{network_id}.json')
    if not os.path.exists(storage_path):
        return None
    with open(storage_path, 'r') as f:
        data = json.load(f)
    return data.get('hopfield_state', {})


# --- API Views ---

class NetworkList(APIView):
    """
    List all networks or create a new one.
    """
    def get(self, request, format=None):
        """Lists all available networks by scanning the storage directory."""
        storage_dir = os.path.join('backend', 'storage')
        networks = []
        for filename in os.listdir(storage_dir):
            if filename.endswith('.json'):
                network_id = filename[:-5]
                networks.append({'id': network_id, 'config': get_network_config(network_id)})
        return Response(networks)

    def post(self, request, format=None):
        """Creates a new cognitive architecture."""
        network_id = f"network-{uuid.uuid4()}"
        dimensions = request.data.get('dimensions', 64)
        hyperparams = request.data.get('config', {})

        CognitiveArchitectureService(
            network_id=network_id,
            dimensions=dimensions,
            load_from_storage=False, # Force creation
            **hyperparams
        )
        return Response({'id': network_id}, status=status.HTTP_201_CREATED)


class NetworkDetail(APIView):
    """
    Retrieve or update a network's configuration.
    """
    def get(self, request, network_id, format=None):
        """Retrieves the configuration for a specific network."""
        config = get_network_config(network_id)
        if config is None:
            return Response(status=status.HTTP_404_NOT_FOUND)
        return Response(config)

    def patch(self, request, network_id, format=None):
        """Updates the hyperparameters for a network."""
        service = get_network_service(network_id)
        if service is None:
            return Response(status=status.HTTP_404_NOT_FOUND)

        new_config = request.data
        # Update Hopfield params
        service.hopfield.learning_rate = new_config.get('learning_rate', service.hopfield.learning_rate)
        service.hopfield.weight_decay = new_config.get('weight_decay', service.hopfield.weight_decay)
        # In a real app, you'd update GNG/STAG params too

        service.save_state()
        return Response(service.hopfield.get_state())


class LearnText(APIView):
    """Endpoint to learn a new pattern from text."""
    def post(self, request, network_id, format=None):
        service = get_network_service(network_id)
        if service is None:
            return Response(status=status.HTTP_404_NOT_FOUND)

        text = request.data.get('text')
        if not text:
            return Response({'error': 'Text is required'}, status=status.HTTP_400_BAD_REQUEST)

        result = service.learn_pattern(text)
        return Response(result)


class Organize(APIView):
    """Endpoint to trigger one step of the GNG/STAG organization process."""
    def post(self, request, network_id, format=None):
        service = get_network_service(network_id)
        if service is None:
            return Response(status=status.HTTP_404_NOT_FOUND)

        cue_text = request.data.get('cue_text', None)
        result = service.organize_step(cue_text)
        return Response(result)


class NetworkStructure(APIView):
    """Endpoint to get the GNG/STAG hierarchical graph structure."""
    def get(self, request, network_id, format=None):
        service = get_network_service(network_id)
        if service is None:
            return Response(status=status.HTTP_404_NOT_FOUND)

        structure = service.get_graph_structure()
        return Response(structure)
