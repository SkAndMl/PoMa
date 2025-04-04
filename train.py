from utils import create_logger
from tokenizer import Tokenizer
from data import TranslationDataset, DataLoader
import config
import torch
from pathlib import Path
import time
from model import GPTk
from typing import Dict
from collections import defaultdict

logger = create_logger()
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"{device=}")
tokenizer = Tokenizer(Path(config.llama_path) / "tokenizer.model")
model = GPTk(ckpt_path=config.llama_path, 
            device=device,
            max_seq_len=config.max_batch_size, 
            max_batch_size=config.max_seq_len,
            k=config.k)
logger.info(f"loaded tokenizer and model")

train_ds = TranslationDataset(
    dataset_hf_id="de-en",
    source_lang="de",
    target_lang="en",
    split="train",
    tokenizer=tokenizer
)
val_ds = TranslationDataset(
    dataset_hf_id="de-en",
    source_lang="de",
    target_lang="en",
    split="validation",
    tokenizer=tokenizer 
)
train_dl = DataLoader(train_ds, config.max_batch_size, tokenizer)
val_ds = DataLoader(val_ds, config.max_batch_size, tokenizer)
# test set should also be loaded but i am too lazy to write that now

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.wd)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens["<|eot_id|>"])

@torch.inference_mode()
def evaluate() -> float:
    pass

def train():
    """
    this is just a template
    kindly modify the training loop based on the model api
    """
    for epoch in range(config.epochs):
        accumulated_loss = defaultdict(float)
        start_time = time.time()
        for batch, start_pos in train_dl:
            batch, start_pos = batch.to(device), start_pos.to(device)
            logits: Dict[int, torch.Tensor] = model(batch, 0) # {pos: tensor<bs, seq_len, embed_dim>}
            
            optimizer.zero_grad()
            for i, val in logits.items():
                if i==0: 
                    # lm_head is frozen, so no need to backprop for the ntp just yet
                    continue
                logits_i = val[:, :-i-1, :]
                tgt = batch[:, i+1:]
                b, s = logits_i.shape[:-1]
                assert (b, s) == tuple(tgt.shape)
                loss: torch.Tensor = loss_fn(logits_i.view(b*s, -1), tgt.view(-1,))
                loss.backward()
                accumulated_loss[i] += loss.item()
            
            optimizer.step()

        end_time = time.time()
        accumulated_loss = {i: accumulated_loss[i]/len(train_dl) for i in accumulated_loss}
        validation_loss = evaluate()
        logger.info(f"time took: {end_time-start_time:.2f}")
        logger.info(accumulated_loss)