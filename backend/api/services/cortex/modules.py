import abc
import numpy as np

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

class DenseCortex(BaseCortex):
    """A simple cortex for processing already-vectorized input."""

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize weights for a simple linear layer
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.biases = np.random.randn(output_dim) * 0.1

    def process(self, raw_input: np.array):
        if raw_input.shape[0] != self.input_dim:
            raise ValueError(f"Input dimension {raw_input.shape[0]} does not match expected {self.input_dim}")

        # Simple linear transformation + tanh activation
        output = np.tanh(np.dot(raw_input, self.weights) + self.biases)
        return output

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
