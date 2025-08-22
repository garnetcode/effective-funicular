# Implements the LanguageCortex for processing natural language input.
# This cortex uses a pre-trained language model to convert text into
# a numerical embedding that can be understood by the agent's WorldModel.

import torch
import torch.nn as nn
import numpy as np
from openai import OpenAI

class LanguageCortex(nn.Module):
    """
    A cortex that processes text input into a fixed-size embedding using an API.
    """
    def __init__(self, model_path_or_id, output_dim, api_base=None, device='cpu'):
        super(LanguageCortex, self).__init__()
        self.device = device
        self.model_id = model_path_or_id
        self.output_dim = output_dim

        if not api_base:
            raise ValueError("API base URL must be provided for LanguageCortex")

        # Initialize the OpenAI client to connect to the Ollama server
        self.client = OpenAI(
            base_url=api_base,
            api_key='ollama',  # Required for the client, but not used by Ollama
        )

    def process(self, text_input: str) -> np.ndarray:
        """
        Processes a string of text and returns a fixed-size embedding vector.
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_id,
                input=text_input
            )
            # The response contains a list of embeddings, we take the first one
            embedding = response.data[0].embedding

            # Ensure the embedding is a numpy array
            embedding_np = np.array(embedding, dtype=np.float32)

            # NOTE: The projection layer is removed. The caller must now handle
            # potential dimension mismatches between the embedding and the world model.
            # For now, we assume the dimensions are compatible.
            if embedding_np.shape[0] != self.output_dim:
                 print(f"Warning: Embedding dimension ({embedding_np.shape[0]}) does not match model's expected output_dim ({self.output_dim}). This may cause errors.")


            return embedding_np.flatten()

        except Exception as e:
            print(f"Error getting embedding from Ollama: {e}")
            # Return a zero vector of the expected dimension on error
            return np.zeros(self.output_dim, dtype=np.float32)
