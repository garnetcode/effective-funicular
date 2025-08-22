# Implements the TextGenerationHead for producing natural language output.
# This module uses a pre-trained causal language model (like Gemma) to
# translate the agent's internal state into a text response.

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Gemma3ForCausalLM, BitsAndBytesConfig

class TextGenerationHead(nn.Module):
    """
    A module that generates text based on an agent's internal state.
    """
    def __init__(self, model_id, input_dim, device='cpu'):
        super(TextGenerationHead, self).__init__()
        self.device = device

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # We use the CausalLM model for text generation
        self.model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config
        ).to(self.device)

        # A simple layer to create a text prompt from the agent's hidden state
        # This is a placeholder; a more complex implementation could be used.
        self.prompt_formatter = lambda h: f"Based on my current internal state, my response is: "

    def generate(self, agent_hidden_state: torch.Tensor, max_new_tokens=50):
        """
        Generates a text response from the agent's hidden state.
        """
        self.model.eval()

        with torch.no_grad():
            # 1. Create a prompt from the hidden state
            # A simple approach is to use a fixed prefix. A more advanced
            # method would be to use the hidden state to select from multiple
            # prompt templates or even generate the prompt itself.
            prompt = self.prompt_formatter(agent_hidden_state)

            # 2. Tokenize the prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # 3. Generate the response
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            # 4. Decode and clean up the response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # The response often includes the prompt, so we remove it.
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()

        return response_text
