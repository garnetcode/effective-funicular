# Implements the TextGenerationHead for producing natural language output.
# This module uses a causal language model API (like Ollama) to
# translate the agent's internal state into a text response.

import json
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

    def _format_prompt(self, agent_state: dict) -> str:
        """
        Creates a detailed prompt for the language model, including the agent's
        state serialized as a JSON object.
        """
        state_json = json.dumps(agent_state, indent=2)

        prompt = (
            "You are the mind of a cognitive agent. Your internal state is provided below as a JSON object, "
            "containing your vital signs and a summary of your current neural state. "
            "Based on this information, generate a brief, first-person response that reflects your current condition or thought process.\n\n"
            "```json\n"
            f"{state_json}\n"
            "```\n\n"
            "Response:"
        )

        print(prompt)
        return prompt

    def generate(self, agent_state: dict, max_new_tokens=50):
        """
        Generates a text response from the agent's state dictionary.
        """
        # 1. Create a detailed prompt from the agent state
        prompt = self._format_prompt(agent_state)

        # 2. Generate the response using the API
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "user", "content": prompt} # The prompt now contains all context
                ],
                max_tokens=max_new_tokens
            )

            response_text = response.choices[0].message.content.strip()
            return response_text

        except Exception as e:
            print(f"Error getting completion from Ollama: {e}")
            return "I am currently unable to generate a response due to an API error."
