from tokenizer import Tokenizer
import torch
from model import Transformer, ModelArgs
from pathlib import Path
import json
import os
import logging
from typing import Dict
from datetime import datetime

@torch.inference_mode()
def generate(model, max_tokens: int, prompt: str, tokenizer: Tokenizer, device: str) -> str:
    """
    generates content
    """
    assert device in {"cpu", "cuda"}
    model = model.to(device)
    model.eval()
    tokens = torch.tensor([tokenizer.encode(prompt, bos=True, eos=False)]).to(device)
    for _ in range(max_tokens):
        logits: torch.Tensor = model(tokens, 0) # 1, seq_len, vocab_size
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=-1).to(device)
    return tokenizer.decode(tokens[0].tolist())


def load_model(ckpt_path: str, device: str, max_batch_size: int, max_seq_len: int):
    """
    loads the weights into the model
    """
    assert device in {"cpu", "cuda"}
    assert os.path.isdir(ckpt_path)
    cp = list(Path(ckpt_path).glob("*.pth"))
    assert len(cp)>0, f"{ckpt_path} has no .pth files!"
    wt = torch.load(cp[0], map_location=device)

    with open(Path(ckpt_path) / "params.json", "r") as f:
        params: dict = json.loads(f.read())
        params['max_batch_size'] = max_batch_size
        params['max_seq_len'] = max_seq_len
        # remove extra params
        params = {k:params[k] for k in ModelArgs.__match_args__}
    
    model_args = ModelArgs(**params)
    model = Transformer(model_args)
    model.load_state_dict(wt)
    return model.to(device)

def create_logger():
    current_date = datetime.now().strftime('%Y-%m-%d')
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'training_{current_date}.log', mode='w')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def calculate_per_token_accuracy(logits_dict: Dict[int, torch.Tensor],
                                 masked_batch: torch.Tensor,
                                 pad_id: int) -> Dict[int, float]:
    """as the name tells"""
    per_token_accuracy = {}
    for i, val in logits_dict.items():
        logits_i = val[:, :-i-1, :]
        tgt = masked_batch[:, i+1:]
        b, s = logits_i.shape[:-1]
        assert (b, s) == tuple(tgt.shape)

        logits_i, tgt = logits_i.reshape(-1, logits_i.size(-1)), tgt.reshape(-1)
        _, topk_indices = torch.topk(logits_i, k=5, dim=-1)
        top5_mask = torch.any(topk_indices == tgt.unsqueeze(-1), dim=-1)
        masked_equality = torch.where(tgt != pad_id, top5_mask, -1)
        num_correct = masked_equality[masked_equality!=-1].sum()
        total_num = masked_equality[masked_equality!=-1].shape[0]
        per_token_accuracy[i] = num_correct/total_num
    
    return per_token_accuracy