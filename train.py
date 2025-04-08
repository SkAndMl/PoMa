from utils import create_logger
from tokenizer import Tokenizer
from data import TranslationDataset, DataLoader
import config
import torch
import time
from model import GPTk
from typing import Dict
from collections import defaultdict
import json
import gc

logger = create_logger()
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"{device=}")
if device=="cuda":
    torch.cuda.empty_cache()
    gc.collect()

tokenizer = Tokenizer(f"{config.llama_path}/tokenizer.model")
model = GPTk(ckpt_path=config.llama_path, 
            device=device,
            max_seq_len=config.max_seq_len, 
            max_batch_size=config.max_batch_size,
            k=config.k).to(device)
logger.info(f"loaded tokenizer and model")

train_ds = TranslationDataset(
    dataset_hf_id="de-en",
    source_lang="de",
    target_lang="en",
    split="train",
    tokenizer=tokenizer,
    max_seq_len=config.max_seq_len,
    num_instances=config.num_train_instances
)
val_ds = TranslationDataset(
    dataset_hf_id="de-en",
    source_lang="de",
    target_lang="en",
    split="validation",
    max_seq_len=config.max_seq_len,
    tokenizer=tokenizer,
    num_instances=config.num_eval_instances
)
train_dl = DataLoader(train_ds, config.max_batch_size, tokenizer)
val_dl = DataLoader(val_ds, config.max_batch_size, tokenizer)
# test set should also be loaded but i am too lazy to write that now

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.wd)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens["<|eot_id|>"])

@torch.inference_mode()
def evaluate() -> float:
    model.eval()
    accumulated_loss = defaultdict(float)
    for batch, start_pos in val_dl:
        batch, start_pos = batch.to(device), start_pos.to(device)
        logits: Dict[int, torch.Tensor] = model(batch, 0) # {pos: tensor<bs, seq_len, embed_dim>}
        for i, val in logits.items():
            if i==0: 
                continue
            logits_i = val[:, :-i-1, :]
            tgt = batch[:, i+1:]
            b, s = logits_i.shape[:-1]
            assert (b, s) == tuple(tgt.shape)
            loss: torch.Tensor = loss_fn(logits_i.reshape(-1, logits_i.size(-1)), tgt.reshape(-1))
            accumulated_loss[i] += loss.item()
    
    model.train()
    accumulated_loss = {i: accumulated_loss[i]/len(val_dl) for i in accumulated_loss}
    return accumulated_loss

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
                loss: torch.Tensor = loss_fn(logits_i.reshape(-1, logits_i.size(-1)), tgt.reshape(-1))
                loss.backward()
                accumulated_loss[i] += loss.item()
            
            optimizer.step()

        end_time = time.time()
        accumulated_loss = {i: accumulated_loss[i]/len(train_dl) for i in accumulated_loss}
        validation_accumlated_loss = evaluate()
        logger.info(f"time took: {end_time-start_time:.2f}")
        logger.info(f"train loss: {json.dumps(accumulated_loss)}")
        logger.info(f"validation loss: {json.dumps(validation_accumlated_loss)}")
    

if __name__ == "__main__":
    train()