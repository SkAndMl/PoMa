from tokenizer import Tokenizer
import torch
from model import Transformer, ModelArgs
from pathlib import Path
import json
import os

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
    cp = torch.load(list(Path(ckpt_path).glob("*.pth"))[0], map_location=device)
    with open(Path(ckpt_path) / "params.json", "r") as f:
        params: dict = json.loads(f.read())
        params['max_batch_size'] = max_batch_size
        params['max_seq_len'] = max_seq_len
        # remove extra params
        params = {k:params[k] for k in ModelArgs.__match_args__}
    
    model_args = ModelArgs(**params)
    model = Transformer(model_args)
    model.load_state_dict(cp)
    return model.to(device)