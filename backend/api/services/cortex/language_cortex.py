# Implements the LanguageCortex for processing natural language input.
# This cortex uses a pre-trained language model to convert text into
# a numerical embedding that can be understood by the agent's WorldModel.

import torch
import torch.nn as nn
import numpy as np
from openai import OpenAI

class LanguageCortex(nn.Module):
    """
    A cortex that processes text input into a fixed-size embedding using an API,
    with a projection layer to match the required output dimension.
    """
    def __init__(self, model_path_or_id, output_dim, api_base=None, embedding_dim=None, device='cpu'):
        super(LanguageCortex, self).__init__()
        self.device = device
        self.model_id = model_path_or_id
        self.output_dim = output_dim

        if not api_base:
            raise ValueError("API base URL must be provided for LanguageCortex")
        if not embedding_dim:
            raise ValueError("Embedding dimension must be provided for LanguageCortex")

        # Initialize the OpenAI client to connect to the Ollama server
        self.client = OpenAI(
            base_url=api_base,
            api_key='ollama',  # Required for the client, but not used by Ollama
        )

        # A linear layer to project the API's embedding to the required output_dim.
        self.projection_layer = nn.Linear(embedding_dim, output_dim).to(self.device)

    def process(self, text_input: str) -> np.ndarray:
        """
        Processes a string of text and returns a fixed-size embedding vector.
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_id,
                input=text_input
            )
            embedding = response.data[0].embedding

            # Ensure the embedding is a numpy array and on the correct device
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)

            # Project the embedding to the desired output dimension
            with torch.no_grad():
                projected_embedding = self.projection_layer(embedding_tensor)

            # Return as a numpy array, detaching from the graph
            return projected_embedding.cpu().numpy().flatten()

        except Exception as e:
            print(f"Error getting embedding from Ollama: {e}")
            # Return a zero vector of the expected dimension on error
            return np.zeros(self.output_dim, dtype=np.float32)
