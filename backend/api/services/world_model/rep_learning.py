import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Computes the InfoNCE loss for self-supervised representation learning.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor (torch.Tensor): The anchor embeddings (e.g., h_t). Shape: (batch_size, embed_dim)
            positive (torch.Tensor): The positive embeddings (e.g., h_{t+1}). Shape: (batch_size, embed_dim)
            negatives (torch.Tensor): The negative embeddings. Shape: (batch_size, num_negatives, embed_dim)
        """
        # Normalize the embeddings
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)

        # Calculate similarity between anchor and positive
        l_pos = torch.einsum('bd,bd->b', [anchor, positive]).unsqueeze(-1)

        # Calculate similarity between anchor and negatives
        l_neg = torch.einsum('bd,bnd->bn', [anchor, negatives])

        # The logits are the similarities scaled by the temperature
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # The labels are all zeros, as the positive sample is always at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)

        # Calculate the cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss
