import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
import math
import inspect

@dataclass
class ModelArgs:
    n_heads: int = 8
    n_dim: int = 1024
    n_layers: int = 8
    ctx_length: int = 512
    vocab_size: int = 1000
    dropout_p: float = 0.01
    ffn_multiplier: int = 4


class Attention(nn.Module):

    def __init__(self, args: ModelArgs) -> None:

        super().__init__()

        assert args.n_dim % args.n_heads==0

        self.n_heads = args.n_heads
        self.WKV = nn.Linear(args.n_dim, args.n_dim*3, bias=False)
        self.O = nn.Linear(args.n_dim, args.n_dim)
        self.D = nn.Dropout(p=args.dropout_p)
    
    def forward(
            self, 
            x:torch.Tensor,
            mask:torch.Tensor=None
        ) -> torch.Tensor:
        # bs, n, d
        bs, n, d = x.size()
        head_dim = d//self.n_heads
        w, k, v = self.WKV(x).split(d, dim=-1)

        w = w.view(bs, n, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(bs, n, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(bs, n, self.n_heads, head_dim).transpose(1, 2)

        attn = (w @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            attn += mask
        attn = F.softmax(attn, dim=-1)
        attn = self.D(attn)

        y = attn @ v
        y = y.transpose(1, 2).contiguous().view(bs, n, d)
        return self.O(y)


class FFN(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.n_dim, args.n_dim*args.ffn_multiplier),
            nn.ReLU(),
            nn.Linear(args.n_dim*args.ffn_multiplier, args.n_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attn = Attention(args)
        self.ffn = FFN(args)
        self.ln_1 = nn.LayerNorm(args.n_dim)
        self.ln_2 = nn.LayerNorm(args.n_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor=None
    ) -> torch.Tensor:
        
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.ffn(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(args.vocab_size, args.n_dim)
        self.pos_emb = nn.Embedding(args.ctx_length, args.n_dim)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.lm_head = nn.Linear(args.n_dim, args.vocab_size)

        mask = torch.triu(
            torch.ones(args.ctx_length, args.ctx_length) * float("-inf"),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        self.register_buffer('mask', tensor=mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        bs, n = x.size()
        
        x_tok = self.tok_emb(x)
        x_pos = self.pos_emb(torch.arange(n).to(x.device)).unsqueeze(0)

        x = x_tok + x_pos
        for block in self.blocks:
            x = block(x, self.mask[:, :, :n, :n])
        
        logits = self.lm_head(x)
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer