from utils import create_logger, calculate_per_token_accuracy
from tokenizer import Tokenizer
from data import TranslationDataset, DataLoader
import config
import torch
from model import GPTk
from typing import Dict, Tuple
from collections import defaultdict
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
)
train_dl = DataLoader(train_ds, config.max_batch_size, tokenizer)
val_dl = DataLoader(val_ds, config.max_batch_size, tokenizer)
# test set should also be loaded but i am too lazy to write that now

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.wd)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens["<|eot_id|>"])

@torch.inference_mode()
def evaluate() -> Tuple[Dict[int, float]]:
    model.eval()
    accumulated_loss = defaultdict(float)
    accumulated_per_token_accuracy = defaultdict(float)
    for batch, start_pos in val_dl:
        batch, start_pos = batch.to(device), start_pos.to(device)
        logits_dict: Dict[int, torch.Tensor] = model(batch, 0) # {pos: tensor<bs, seq_len, embed_dim>}
        for i, val in logits_dict.items():
            logits_i = val[:, :-i-1, :]
            tgt = batch[:, i+1:]
            b, s = logits_i.shape[:-1]
            assert (b, s) == tuple(tgt.shape)
            loss: torch.Tensor = loss_fn(logits_i.reshape(-1, logits_i.size(-1)), tgt.reshape(-1))
            accumulated_loss[i] += loss.item()

        per_token_accuracy = calculate_per_token_accuracy(logits_dict, batch, tokenizer.special_tokens["<|eot_id|>"])
        for i in per_token_accuracy:
            accumulated_per_token_accuracy[i] += per_token_accuracy[i]
    
    model.train()
    accumulated_loss = {i: accumulated_loss[i]/len(val_dl) for i in accumulated_loss}
    accumulated_per_token_accuracy = {i: v/len(val_dl) for i,v in accumulated_per_token_accuracy.items()}
    return accumulated_loss, accumulated_per_token_accuracy

def train():
    """
    this is just a template
    kindly modify the training loop based on the model api
    """
    accumulated_loss = defaultdict(float)
    for step, (batch, start_pos) in enumerate(train_dl):
        batch, start_pos = batch.to(device), start_pos.to(device)
        logits: Dict[int, torch.Tensor] = model(batch, 0) # {pos: tensor<bs, seq_len, embed_dim>}
        
        optimizer.zero_grad()
        for i, val in logits.items():
            logits_i = val[:, :-i-1, :]
            tgt = batch[:, i+1:]
            b, s = logits_i.shape[:-1]
            assert (b, s) == tuple(tgt.shape)
            loss: torch.Tensor = loss_fn(logits_i.reshape(-1, logits_i.size(-1)), tgt.reshape(-1))
            loss.backward()
            accumulated_loss[i] += loss.item()
        
        optimizer.step()

        if (step+1)%config.print_loss_every==0:
            accumulated_loss = {i:accumulated_loss[i]/config.print_loss_every for i in accumulated_loss}
            log_str = f"train step: {step+1:5d} | "
            log_str += " | ".join(f"{k}: {v:.4f}" for k,v in accumulated_loss.items())
            logger.info(log_str)
            accumulated_loss = defaultdict(float)
        
        if (step+1)%config.eval_step==0:
            validation_accumulated_loss, validation_accumulated_per_token_accuracy = evaluate()
            log_str = f"val step: {step+1:5d} | "
            log_str += " | ".join(f"{k}: {v:.4f}" for k,v in validation_accumulated_loss.items())
            log_str += " " + " | ".join(f"acc {k}: {v*100:.2f}" for k,v in validation_accumulated_per_token_accuracy.items())
            logger.info(log_str)
    

if __name__ == "__main__":
    train()