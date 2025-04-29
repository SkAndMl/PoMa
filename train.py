from utils import create_logger, calculate_per_token_accuracy
from tokenizer import Tokenizer
from code_data import *
from torch.utils.data import DataLoader
import config
import torch
from model import GPTk
from typing import Dict, Tuple
from collections import defaultdict
import gc
import os

log_file_name = config.llama_path.split('/')[-1] + f"_k={config.k}_top_k={config.top_k}_task={config.task}"
logger = create_logger(log_file_name=log_file_name)
logger.info(f"Running model: {config.llama_path.split('/')[-1]}")
logger.info(f"Task: {config.task}")
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
            freeze_lm_head = config.freeze_lm_head,
            k=config.k).to(device)
logger.info(f"loaded tokenizer and model")

# Full dataset loading based on config
train_ds = CodeDataset(config=config, tokenizer = tokenizer, split = "train", max_seq_len = config.max_seq_len, num_samples = None)
val_ds = CodeDataset(config=config, tokenizer = tokenizer, split = "test", max_seq_len = config.max_seq_len, num_samples = None)

logger.info(f"Training instances: {len(train_ds)}; Validation instances: {len(val_ds)}")

collate_fn = Collator(pad_id= tokenizer.special_tokens["<|eot_id|>"], max_seq_len = config.max_seq_len)
train_dl = DataLoader(train_ds, batch_size = config.max_batch_size, shuffle = True, collate_fn = collate_fn,num_workers = 8)
val_dl = DataLoader(val_ds, batch_size = config.max_batch_size, shuffle = False, collate_fn = collate_fn, num_workers = 8)


optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.wd)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.special_tokens["<|eot_id|>"])

@torch.inference_mode()
def evaluate() -> Tuple[Dict[int, float]]:
    model.eval()
    accumulated_loss = defaultdict(float)
    accumulated_per_token_accuracy = defaultdict(float)
    # End position ignored for PoMa
    for batch, start_pos, _ in val_dl:
        batch, start_pos = batch.to(device), start_pos.to(device)
        masked_batch = torch.where(
            torch.arange(batch.shape[1], device=device).view(1, -1) < start_pos,
            tokenizer.special_tokens["<|eot_id|>"],
            batch
        ).to(device)
        logits_dict: Dict[int, torch.Tensor] = model(batch, 0) # {pos: tensor<bs, seq_len, embed_dim>}
        for i, val in logits_dict.items():
            logits_i = val[:, :-i-1, :]
            tgt = masked_batch[:, i+1:]
            b, s = logits_i.shape[:-1]
            assert (b, s) == tuple(tgt.shape)
            loss: torch.Tensor = loss_fn(logits_i.reshape(-1, logits_i.size(-1)), tgt.reshape(-1))
            accumulated_loss[i] += loss.item()

        per_token_accuracy = calculate_per_token_accuracy(logits_dict, masked_batch, tokenizer.special_tokens["<|eot_id|>"], top_k=config.top_k)
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
    if not os.path.exists("ckpts"):
        os.mkdir("ckpts")
    logger.info("Training started")
    logger.info(f"Total number of training steps: {len(train_dl)}")
    logger.info(f"Expect logs at every {config.eval_step} and {config.print_loss_every}")
    accumulated_loss = defaultdict(float)
    best_combined_accuracy = float("-inf")
    for _ in range(config.epochs):
        # End pos ignored for PoMa
        for step, (batch, start_pos, _) in enumerate(train_dl):
            batch, start_pos = batch.to(device), start_pos.to(device)
            masked_batch = torch.where(
                torch.arange(batch.shape[1], device=device).view(1, -1) < start_pos,
                tokenizer.special_tokens["<|eot_id|>"],
                batch
            ).to(device)
            logits: Dict[int, torch.Tensor] = model(batch, 0) # {pos: tensor<bs, seq_len, embed_dim>}
            
            optimizer.zero_grad()
            for i, val in logits.items():
                logits_i = val[:, :-i-1, :]
                tgt = masked_batch[:, i+1:]
                b, s = logits_i.shape[:-1]
                assert (b, s) == tuple(tgt.shape)
                loss: torch.Tensor = loss_fn(logits_i.reshape(-1, logits_i.size(-1)), tgt.reshape(-1))
                accumulated_loss[i] += loss.item()
                if i==0 and config.freeze_lm_head:
                    continue
                loss.backward()
                
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

                if config.save_weights:
                    combined_accuracy = sum([v*100 for v in validation_accumulated_per_token_accuracy.values()])
                    if combined_accuracy > best_combined_accuracy:
                        best_combined_accuracy = combined_accuracy
                        logger.info(f"Combined validation accuracy at step: {step+1} is the best so far. Saving the wts")
                        save_dict = {}
                        for i, layer in enumerate(model.pos_embeddings):
                            save_dict[f"pos_embeddings.{i}.embedding.weight"] = layer.embedding.weight.detach().cpu()
                            save_dict[f"pos_embeddings.{i}.norm.weight"] = layer.norm.weight.detach().cpu()

                        if not config.freeze_lm_head:
                            save_dict["lm_head.weight"] = model.base_model.output.weight.detach().cpu()

                        torch.save(save_dict, f"ckpts/{log_file_name}.pt")

if __name__ == "__main__":
    train()