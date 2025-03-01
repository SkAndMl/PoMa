import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from typing import Dict

class CrossEntropyK(nn.Module):
    def __init__(
        self,
        k: int = 3,
    ) -> None:
        super().__init__()
        self.k = k

    def forward(
        self,
        logits: Dict[int, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        bs, n, vocab_size = logits[0].shape
        assert n + self.k - 1 == targets.size(-1)

        total_loss = torch.zeros(size=(bs, n), dtype=torch.float32).to(logits[0].device)        
        losses = defaultdict(float)
        for i in range(self.k):
            targets_i = targets[:, i:i+n]
            logits_i = logits[i]
            # print(logits_i.shape, targets_i.shape)
            loss_i = F.cross_entropy(logits_i.view(-1, vocab_size), targets_i.reshape(-1))
            
            losses[i] = loss_i.mean().item()
            total_loss += loss_i
        
        return total_loss.mean(), losses