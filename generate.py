from tokenizer import Tokenizer
from data import TranslationDataset, DataLoader
import torch
import time
from utils import *
from lstm import LSTMHead
import json
import os
from dataclasses import dataclass
from pathlib import Path

# Setting basics
SEED = 42
#device = "cuda" if torch.cuda.is_available() else "mps"
device = "mps"
# TODO: Optionally set deterministic seed for CUDA here.


# Config dataclass
@dataclass
class config:
    dim: int = 2048
    Llama_path: str = "/Users/ashwathsreeram/Desktop/Llama_model/Llama3.2-1B"
    max_seq_len: int = 2048  # maximum sequence length for context / Llama model
    layers: int = 2
    vocab_size: int = 128256
    max_batch_size: int = 32
    lr: float = 1e-3
    wd: float = 0.01
    epochs: int = 10
    k: int = 10

# Model Loading -- Llama
llm_model = load_model(
    ckpt_path=os.path.join(config.Llama_path),
    max_seq_len=config.max_seq_len,
    max_batch_size=config.max_batch_size,
    device=device
)
llm_model.eval()  # Freeze Llama model; no gradient computation here.
print("Llama Loaded")

# LSTM Model (prediction head) -- Load this model on the device as well.
# GIVE CHECKPOINT PATH
ckpt_path = "/Users/ashwathsreeram/Desktop/Llama_model/lstm-llama/model_epoch_7.pth"
assert os.path.exists(ckpt_path), "Invalid checkpoint path"
lstm = LSTMHead(
    input_dim=config.dim,
    hidden_dim=config.dim,
    num_layers=config.layers,
    vocab_size=config.vocab_size,
    dropout=0.1
).to(device)
lstm.load_state_dict(torch.load(ckpt_path, map_location = torch.device(device)))
lstm.eval()
print("LSTM loaded with weights")

# Tokenizer Loading -- Llama
tokenizer = Tokenizer(os.path.join(config.Llama_path, "tokenizer.model"))
print("Tokenzier loaded")

# Creating input -- the prelude
source_lang, target_lang = "de", "en"
few_shot_examples = [
        {
            "source": "Ich gehe morgen ins Kino.", 
            "target": "I am going to the cinema tomorrow."
        },
        {
            "source": "Er ist ein sehr guter Koch.", 
            "target": "He is a very good cook."
        },
        {
            "source": "Das Wetter ist heute schön.", 
            "target": "The weather is nice today."
        },
        {
            "source": "Wir haben gestern einen langen Spaziergang gemacht.", 
            "target": "We took a long walk yesterday."
        },
        {
            "source": "Kannst du mir bitte helfen?", 
            "target": "Can you please help me?"
        }
    ]

input_string =  f"Translate from {source_lang} to {target_lang}\n"
for sample in few_shot_examples:
    input_string += f"{source_lang}: {sample['source']}; {target_lang}: {sample['target']}\n"

# the actual line to translate
input_string += f"{source_lang}: Vielen Dank, Herr Segni, das will ich gerne tun.; {target_lang}"

print(generate_lstm(lstm = lstm, llama = llm_model, max_tokens= 3, prompt = input_string, tokenizer = tokenizer, device = device))