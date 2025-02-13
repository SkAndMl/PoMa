import torch
from torch import nn
from torch.nn import functional as F

class EnhancedCrossEntropyK(nn.Module):
    def __init__(
        self,
        k: int = 3,
        gamma: float = 0.1,
        alpha: float = 2.0,  # focal loss power factor
        beta: float = 0.1,   # minimum contribution from each position
        token_weights: dict = None  # optional token importance weights
    ) -> None:
        super().__init__()
        self.k = k
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.token_weights = token_weights

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        bs, n, vocab_size = logits.shape
        assert n + self.k == targets.size(-1)

        # Initialize total loss tensor
        total_loss = torch.zeros(size=(bs, n), dtype=torch.float32).to(logits.device)
        
        # Apply log softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        for i in range(self.k):
            targets_i = targets[:, i:i+n]
            
            # Gather predicted probabilities for target tokens
            probs = torch.gather(log_probs, dim=-1, index=targets_i.unsqueeze(-1)).squeeze(-1)
            
            # Apply focal loss modulation
            pt = torch.exp(probs)  # Convert log probs to probs
            focal_weight = (1 - pt) ** self.alpha
            
            # Apply position-based decay with minimum contribution
            position_weight = max(self.gamma**i, self.beta)
            
            # Apply token importance weighting if provided
            if self.token_weights is not None:
                token_weight = torch.tensor([self.token_weights.get(t.item(), 1.0) 
                                          for t in targets_i.view(-1)],
                                          device=logits.device).view(bs, -1)
                focal_weight = focal_weight * token_weight
            
            # Combine all weighting factors
            weighted_loss = -position_weight * focal_weight * probs
            total_loss += weighted_loss
        
        # Normalize by sequence length and batch size
        return total_loss.mean()

    def compute_token_weights(self, token_counts: dict) -> dict:
        """
        Compute token weights based on frequency.
        Rarer tokens get higher weights.
        """
        total_count = sum(token_counts.values())
        weights = {}
        for token, count in token_counts.items():
            freq = count / total_count
            # Inverse frequency weighting with smoothing
            weights[token] = 1.0 / (freq + 0.01) ** 0.5
        return weights