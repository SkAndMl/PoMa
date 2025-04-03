from utils import load_model, create_logger
from tokenizer import Tokenizer
from data import TranslationDataset, DataLoader
import config
import torch
from pathlib import Path
import time

logger = create_logger()
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"{device=}")
tokenizer = Tokenizer(Path(config.llama_path) / "tokenizer.model")
model = load_model(ckpt_path=config.llama_path, device=device,
                   max_seq_len=config.max_batch_size, max_batch_size=config.max_seq_len)
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
loss_fn = torch.nn.CrossEntropyLoss()

@torch.inference_mode()
def evaluate() -> float:
    pass

def train():
    """
    this is just a template
    kindly modify the training loop based on the model api
    """
    for epoch in range(config.epochs):
        accumulated_loss = 0
        start_time = time.time()
        for batch, start_pos in train_dl:
            batch, start_pos = batch.to(device), start_pos.to(device)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            b, s = inp.shape
            logits: torch.Tensor = model(inp, 0) # bs, seq_len, embed_dim
            loss: torch.Tensor = loss_fn(logits.reshape(b*s, -1), tgt.reshape(-1,))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accumulated_loss += loss.item()
        end_time = time.time()
        accumulated_loss /= len(train_dl) 
        validation_loss = evaluate()
        logger.info(f"time took: {end_time-start_time:.2f}, training_loss: {accumulated_loss:.4f}, validation_loss: {validation_loss:.4f}")