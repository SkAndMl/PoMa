import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict

class CrossEntropyK(nn.Module):
    def __init__(
        self,
        k: int = 3,
        gamma: float = 0.9,
        beta: float = 0.1,   
    ) -> None:
        super().__init__()
        self.k = k
        self.gamma = gamma
        self.beta = beta

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        bs, n, vocab_size = logits.shape
        assert n + self.k -1 == targets.size(-1)

        total_loss = torch.zeros(size=(bs, n), dtype=torch.float32).to(logits.device)
        log_probs = F.log_softmax(logits, dim=-1)
        
        losses = defaultdict(float)
        for i in range(self.k):
            targets_i = targets[:, i:i+n]
            probs = torch.gather(log_probs, dim=-1, index=targets_i.unsqueeze(-1)).squeeze(-1)
            
            position_weight = max(self.gamma**i, self.beta)
            weighted_loss = -position_weight * probs
            losses[i] = weighted_loss.mean().item()
            total_loss += weighted_loss
        
        return total_loss.mean(), losses