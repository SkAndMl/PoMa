import torch
import time
import os
import random
import numpy as np
from tokenizer import Tokenizer
from code_data import *
from model import GPTk
from base_model import Transformer as base
from utils import run_inference, generate, load_model
import config as config

def set_seed(seed=42):
    """For deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Function for timed inference
def timed_inference(prompt: str, base_model, MTP_model, k: int, tokenizer: Tokenizer, repeat = 5):
    """
    Timed run to compare Base and MTP Models
    """
    # Warm-up (Optional, but improves first-time randomness)
    with torch.no_grad():
        _ = generate(model=base_model, max_tokens=k, prompt=prompt, tokenizer=tokenizer, device="cuda")
        _ = run_inference(model=MTP_model, tokenizer=tokenizer, prompt=prompt, k=k)
        torch.cuda.synchronize()

    # Inference on base
    torch.cuda.synchronize()
    start_base = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output_base = generate(model=base_model, max_tokens=k, prompt=prompt, tokenizer=tokenizer, device="cuda")
            torch.cuda.synchronize()
    end_base = time.time()

    # Inference on MTP
    torch.cuda.synchronize()
    start_mtp = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output_mtp = run_inference(model=MTP_model, tokenizer=tokenizer, prompt=prompt, k=k)
            torch.cuda.synchronize()
    end_mtp = time.time()

    print("-"*50 + " BASE MODEL " + "-"*50)
    print(f"Base Model Output: {output_base}\n")
    print(f"Base Model Time taken: {(end_base - start_base)/repeat:.6f} seconds")
    print("-"*50 + " MTP MODEL " + "-"*50)
    print(f"MTP Model Output: {output_mtp}\n")
    print(f"MTP Model Time taken: {(end_mtp - start_mtp)/repeat:.6f} seconds")

def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load Tokenizer
    tokenizer = Tokenizer(f"{config.llama_path}/tokenizer.model")

    # Load Base Transformer Model
    base_model = load_model(
        ckpt_path=os.path.join(config.llama_path),
        max_seq_len=config.max_seq_len,
        max_batch_size=config.max_batch_size,
        device=device
    )
    base_model.eval()
    print("Base Model Loaded")

    # Load GPTk MTP Model
    mtp_model = GPTk(
        ckpt_path=config.llama_path,
        device=device,
        max_seq_len=config.max_seq_len,
        max_batch_size=config.max_batch_size,
        freeze_lm_head=config.freeze_lm_head,
        k=config.k
    ).to(device)
    mtp_model.load_poma_weights(poma_ckpt_path="/home/users/ntu/ashwaths/workspace/PoMa/ckpts/Llama3.2-3B_k=3_top_k=5_task=alpaca.pt")
    mtp_model.eval()
    print("MTP Model Loaded with PoMa Weights")

    # Demo prompt
    prompt_alpaca = (
        "Generate code for the following instruction\n"
        "Instruction: Write a sql statement to sort a list of employees in descending order based on their salary\n"
    )
    prompt_conala = (
        "Generate single line code for the following instruction\n"
        "Instruction: sort a list of tuples `my_list` by second parameter in the tuple\n"
    )

    print(f"Prompt 1: {prompt_conala}\n")

    timed_inference(prompt=prompt_alpaca, base_model=base_model, MTP_model=mtp_model, k=config.k, tokenizer=tokenizer)

if __name__ == "__main__":
    main()
