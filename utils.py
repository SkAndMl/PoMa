from tokenizer import Tokenizer
import torch
import torch.nn.functional as F
from model import Transformer, ModelArgs
from pathlib import Path
import json
import os
import logging
from typing import Dict
from datetime import datetime
from collections import defaultdict
import re
import matplotlib.pyplot as plt

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

def create_logger(log_file_name):
    if not os.path.exists("logs"):
        os.mkdir("logs")
    current_date = datetime.now().strftime('%Y-%m-%d')
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'logs/training_{log_file_name}_{current_date}.log', mode='w')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def calculate_per_token_accuracy(logits_dict: Dict[int, torch.Tensor],
                                 masked_batch: torch.Tensor,
                                 pad_id: int,
                                 top_k: int) -> Dict[int, float]:
    """as the name tells"""
    per_token_accuracy = {}
    for i, val in logits_dict.items():
        logits_i = val[:, :-i-1, :]
        tgt = masked_batch[:, i+1:]
        b, s = logits_i.shape[:-1]
        assert (b, s) == tuple(tgt.shape)

        logits_i, tgt = logits_i.reshape(-1, logits_i.size(-1)), tgt.reshape(-1)
        _, topk_indices = torch.topk(logits_i, k=top_k, dim=-1)
        topk_mask = torch.any(topk_indices == tgt.unsqueeze(-1), dim=-1)
        masked_equality = torch.where(tgt != pad_id, topk_mask, -1)
        num_correct = masked_equality[masked_equality!=-1].sum()
        total_num = masked_equality[masked_equality!=-1].shape[0]
        per_token_accuracy[i] = num_correct/total_num
    
    return per_token_accuracy

def parse_log_filename(filename: str):
    pattern = r"training_(?P<model>.+?)_k=(?P<k>\d+)_top_k=(?P<top_k>\d+)_task=(?P<task>.+?)_\d{4}-\d{2}-\d{2}\.log"
    match = re.match(pattern, filename)
    if match:
        return {
            "model_name": match.group("model"),
            "k": int(match.group("k")),
            "top_k": int(match.group("top_k")),
            "task": match.group("task")
        }
    else:
        raise ValueError("Filename does not match expected pattern.")


def get_acc_scores_from_log_file(log_file: str) -> dict:
    val_acc_k_scores = parse_log_filename(log_file.split("/")[-1])
    val_acc_k_scores["k_pos_scores"] = defaultdict(list)
    acc_pattern = r'acc \d: (\d+\.\d+)'
    with open(log_file, "r") as f:
        for line in f.readlines():
            if "val" in line:
                scores = re.findall(acc_pattern, line.strip())
                assert len(scores) == val_acc_k_scores["k"]
                for i in range(len(scores)):
                    val_acc_k_scores["k_pos_scores"][i].append(float(scores[i]))
    return val_acc_k_scores

def plot_k_pos_scores(results_dict, save_fig: bool=False):
    k_pos_scores = results_dict['k_pos_scores']
    k = results_dict['k']
    model = results_dict['model_name']
    top_k_val = results_dict['top_k']
    task = results_dict['task'].capitalize()

    plt.figure(figsize=(8, 5.5))

    for i in range(k):
        plt.plot(
            k_pos_scores[i],
            marker='o',
            linewidth=2,
            label=f'Position {i+1}'
        )

    plt.title(
        f'{model} | Task: {task} | MTP (k={k}) | Top-{top_k_val} Accuracy',
        fontsize=14,
        fontweight='bold'
    )
    plt.xlabel('Training Step', fontsize=11)
    plt.ylabel(f'Top-{top_k_val} Accuracy', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_fig:
        if not os.path.exists("assets"):
            os.mkdir("assets")
        plt.savefig(
            f"assets/{model}_{task}_k={k}_top_k={top_k_val}.png",
            dpi=600,
            bbox_inches='tight'
        )
    plt.show()

@torch.inference_mode()
def run_inference(model, tokenizer, prompt: str, k: int = 4, top_k: int = 5, device: str = "cuda") -> Dict[int, list]:
    """
    Runs PoMA inference for a single prompt and returns top-k predictions at each of the k positions.
    """
    model.eval()
    input_ids = tokenizer.encode(prompt, bos=True, eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  # shape: (1, seq_len)
    start_pos = torch.tensor([len(input_ids)], dtype=torch.long).to(device)

    logits_dict = model(input_tensor, start_pos=0)  # {pos: logits} where logits are (1, seq_len - pos, vocab_size)

    topk_results = {}

    for i in range(k):
        logits_i = logits_dict[i][0, -1]  # shape: (vocab_size,) -> last position of predicted token for pos i
        probs_i = F.softmax(logits_i, dim=-1)
        topk = torch.topk(probs_i, top_k)
        top_tokens = [tokenizer.decode([idx.item()]) for idx in topk.indices]
        topk_results[i] = top_tokens

    return topk_results