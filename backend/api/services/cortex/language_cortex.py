# Implements the LanguageCortex for processing natural language input.
# This cortex uses a pre-trained language model to convert text into
# a numerical embedding that can be understood by the agent's WorldModel.

import torch
import torch.nn as nn
from transformers import AutoTokenizer, Gemma3Model, BitsAndBytesConfig

class LanguageCortex(nn.Module):
    """
    A cortex that processes text input into a fixed-size embedding.
    """
    def __init__(self, model_path_or_id, output_dim, device='cpu'):
        super(LanguageCortex, self).__init__()
        self.device = device

        # For memory efficiency, we can use quantization
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        # We use the base model to get the last hidden state as an embedding
        self.model = Gemma3Model.from_pretrained(
            model_path_or_id,
            quantization_config=quantization_config
        ).to(self.device)

        # The output of the LLM might not match the desired output_dim for the WorldModel
        # We add a linear layer to project the LLM's hidden size to the required output_dim.
        llm_hidden_size = self.model.config.hidden_size
        self.projection_layer = nn.Linear(llm_hidden_size, output_dim).to(self.device)

    def process(self, text_input: str):
        """
        Processes a string of text and returns a fixed-size embedding vector.
        This is designed to run on a single text input, not a batch.
        """
        # Ensure the model is in evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Tokenize the input text
            inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)

            # Get the hidden states from the base model
            outputs = self.model(**inputs)

            # We use the last hidden state as the embedding.
            # For simplicity in mocking, we assume the model output is already pooled.
            last_hidden_state = outputs.last_hidden_state

            # Project the embedding to the desired output dimension
            embedding = self.projection_layer(last_hidden_state.squeeze(0))

        # Return as a numpy array, detaching from the graph
        return embedding.detach().cpu().numpy().flatten()
