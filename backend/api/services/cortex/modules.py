import abc
import numpy as np
import torch

class BaseCortex(abc.ABC):
    """Abstract base class for all sensory cortex modules."""

    @abc.abstractmethod
    def process(self, raw_input):
        """
        Processes raw input and returns a fixed-size embedding vector.

        Args:
            raw_input: The input data for the cortex.

        Returns:
            np.array: The resulting embedding vector.
        """
        pass

import torch

class DenseCortex(BaseCortex, torch.nn.Module):
    """A trainable cortex for processing vectorized input using PyTorch."""

    def __init__(self, input_dim, output_dim):
        super(DenseCortex, self).__init__()  # Calls __init__ for both parent classes
        self.input_dim = input_dim
        self.output_dim = output_dim
        # LayerNorm for input normalization
        self.norm = torch.nn.LayerNorm(input_dim)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = torch.nn.Tanh()

    def _pad_input(self, raw_input: np.array) -> np.array:
        """Pads a single numpy input vector to the expected input_dim."""
        input_len = raw_input.shape[0]
        if input_len > self.input_dim:
            raise ValueError(f"Input dimension {input_len} exceeds the maximum expected dimension of {self.input_dim}")

        if input_len < self.input_dim:
            padded_input = np.zeros(self.input_dim)
            padded_input[:input_len] = raw_input
            return padded_input
        return raw_input

    def process(self, raw_input: np.array) -> np.array:
        """
        Processes a single raw numpy vector for inference.
        Converts to tensor, processes, and converts back to numpy.
        """
        with torch.no_grad():
            padded_input = self._pad_input(raw_input)
            input_tensor = torch.from_numpy(padded_input).float()
            # Normalize the input tensor
            normalized_input = self.norm(input_tensor)
            output_tensor = self.activation(self.linear(normalized_input))
            return output_tensor.numpy()

    def forward(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of tensors through the layers.
        Used during training. This now includes padding.
        """
        batch_size, current_dim = batch_tensor.shape
        if current_dim < self.input_dim:
            # Create a new tensor with the target dimensions and fill it with zeros
            padded_batch = torch.zeros(batch_size, self.input_dim, device=batch_tensor.device)
            # Copy the original data into the padded tensor
            padded_batch[:, :current_dim] = batch_tensor
            batch_tensor = padded_batch

        # Normalize the input batch
        normalized_batch = self.norm(batch_tensor)
        return self.activation(self.linear(normalized_batch))

class TextCortex(BaseCortex):
    """A simple cortex for processing raw text into a reproducible vector."""

    def __init__(self, output_dim):
        self.output_dim = output_dim

    def process(self, raw_input: str):
        """
        Converts a string to a fixed-size bipolar vector using a hash seed.
        """
        # Constrain the seed to be a 32-bit unsigned integer for numpy
        seed = abs(hash(raw_input)) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.output_dim)
        # Convert to a bipolar (-1, 1) vector
        vec = np.sign(vec)
        # Ensure at least one element is non-zero
        if np.all(vec == 0):
            vec[0] = 1
        return vec

try:
    from PIL import Image
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms

    class VisionCortex(BaseCortex):
        """A cortex for processing image data using a pre-trained CNN."""

        def __init__(self, output_dim):
            self.output_dim = output_dim
            # Load a pre-trained model
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

            # Replace the classifier with a new layer to produce the desired embedding size
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier = torch.nn.Linear(num_ftrs, output_dim)

            self.model.eval() # Set to evaluation mode

            # Define the image transformations
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        def process(self, raw_input: str):
            """
            Processes an image file path.

            Args:
                raw_input (str): Path to the image file.

            Returns:
                np.array: The resulting embedding vector.
            """
            try:
                image = Image.open(raw_input).convert('RGB')
                image_tensor = self.transform(image).unsqueeze(0) # Add batch dimension

                with torch.no_grad():
                    embedding = self.model(image_tensor)

                # Convert to numpy array and ensure it's bipolar
                embedding_np = embedding.squeeze(0).numpy()
                embedding_np = np.sign(embedding_np)
                if np.all(embedding_np == 0): embedding_np[0] = 1 # Avoid zero vector

                return embedding_np
            except Exception as e:
                print(f"Failed to process image: {e}")
                return np.zeros(self.output_dim)

except ImportError:
    print("Warning: PyTorch or Pillow not installed. VisionCortex will not be available.")
    # Create a dummy class if dependencies are missing
    class VisionCortex(BaseCortex):
        def __init__(self, output_dim):
            raise ImportError("PyTorch and Pillow are required to use VisionCortex.")

        def process(self, raw_input):
            raise NotImplementedError("VisionCortex is not available due to missing dependencies.")
