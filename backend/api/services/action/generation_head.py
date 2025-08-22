# Implements the TextGenerationHead for producing natural language output.
# This module uses a causal language model API (like Ollama) to
# translate the agent's internal state into a text response.

import torch
import torch.nn as nn
from openai import OpenAI

class TextGenerationHead(nn.Module):
    """
    A module that generates text based on an agent's internal state via an API.
    """
    def __init__(self, model_path_or_id, input_dim, api_base=None, device='cpu'):
        super(TextGenerationHead, self).__init__()
        self.device = device
        self.model_id = model_path_or_id

        if not api_base:
            raise ValueError("API base URL must be provided for TextGenerationHead")

        # Initialize the OpenAI client to connect to the Ollama server
        self.client = OpenAI(
            base_url=api_base,
            api_key='ollama',  # Required for the client, but not used by Ollama
        )

        # A simple layer to create a text prompt from the agent's hidden state
        self.prompt_formatter = lambda h: f"Based on my current internal state, my response is: "

    def generate(self, agent_hidden_state: torch.Tensor, max_new_tokens=50):
        """
        Generates a text response from the agent's hidden state.
        """
        # 1. Create a prompt from the hidden state
        prompt = self.prompt_formatter(agent_hidden_state)

        # 2. Generate the response using the API
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_new_tokens
            )

            response_text = response.choices[0].message.content.strip()
            return response_text

        except Exception as e:
            print(f"Error getting completion from Ollama: {e}")
            return "I am currently unable to generate a response due to an API error."
