from tokenizer import Tokenizer
from data import TranslationDataset, DataLoader
import torch
import time
from utils import load_model, create_logger
from lstm import LSTMHead
import json
import os
from dataclasses import dataclass
from pathlib import Path
import random

# Setting basics
SEED = 42
device = "cuda" if torch.cuda.is_available() else "mps"
# TODO: Optionally set deterministic seed for CUDA here.

# Logger creation
logger = create_logger()
logger.info(f"Seed Set; Device: {device}")

# Config dataclass
@dataclass
class config:
    dim: int = 2048
    Llama_path: str = "/home/msai/ashwaths001/ASHWATHS001/Llama_NLP/Llama3.2-1B"
    max_seq_len: int = 2048  # maximum sequence length for context / Llama model
    layers: int = 2
    vocab_size: int = 128256
    max_batch_size: int = 32
    lr: float = 1e-3
    wd: float = 0.05
    epochs: int = 10
    k: int = 3  # Number of steps per backpropagation segment

# Model Loading -- Llama
llm_model = load_model(
    ckpt_path=os.path.join(config.Llama_path),
    max_seq_len=config.max_seq_len,
    max_batch_size=config.max_batch_size,
    device=device
)
llm_model.eval()  # Freeze Llama model; no gradient computation here.
logger.info("Llama Model Locked and Loaded!")

# LSTM Model (prediction head) -- Load this model on the device as well.
model = LSTMHead(
    input_dim=config.dim,
    hidden_dim=config.dim,
    num_layers=config.layers,
    vocab_size=config.vocab_size,
    dropout=0.1
).to(device)
model.train()
logger.info("LSTM Model Locked and Loaded")

# Tokenizer Loading -- Llama
tokenizer = Tokenizer(os.path.join(config.Llama_path, "tokenizer.model"))
logger.info("Tokenizer Loaded")

# Reducing few shot for fitting into Vram
few_shot_examples = [
        {
            "source": "Ich gehe morgen ins Kino.", 
            "target": "I am going to the cinema tomorrow."
        },
        {
            "source": "Er ist ein sehr guter Koch.", 
            "target": "He is a very good cook."
        }
    ]

# Adding k to dataloader as Vram inadequate
train_ds = TranslationDataset(
    dataset_hf_id="de-en",
    source_lang="de",
    target_lang="en",
    split="train",
    tokenizer=tokenizer,
    few_shot_examples = few_shot_examples,
    max_seq_len = -1,
    num_instances = 5000
)

val_ds = TranslationDataset(
    dataset_hf_id="de-en",
    source_lang="de",
    target_lang="en",
    split="validation",
    tokenizer=tokenizer,
    few_shot_examples = few_shot_examples, 
    max_seq_len = -1
)
train_dl = DataLoader(train_ds, config.max_batch_size, tokenizer, k = config.k)
val_dl = DataLoader(val_ds, config.max_batch_size, tokenizer, k = config.k)
logger.info("Dataset Locked and Loaded")

# Optimizer and loss function
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.wd)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

# Variables to track best validation loss for checkpointing
best_val_loss = float("inf")
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Keeping track of train and val loss
training_loss = []
validation_loss = []

# ------------------
# Training Loop
# ------------------
for epoch in range(config.epochs):
    epoch_start = time.time()
    epoch_train_loss = []
    logger.info(f"Epoch {epoch+1} Training Start:")
    
    # Training Phase
    for i, (batch, start_pos) in enumerate(train_dl):
        optimizer.zero_grad()

        # Tracking loss for batch
        batch_loss = 0.0

        # Transfer batch to device.
        batch = batch.to(device)
        # Extract context length (start_pos is a tensor; no need to send an int to device)
        context_len = start_pos[0].item()
        logger.info(f"Iteration {i}; Sample Target Length {context_len}")
        # Extract context tokens (already on device)
        context_tensor = batch[:, :context_len]

        # Forward pass: Get context vector from Llama model (no gradients)
        with torch.no_grad():
            # Here we pass start_pos=0 because we assume the context was processed wholly.
            current_input = llm_model(context_tensor, 0)  # shape: (bsz, context_len, dim)
        
        # Get target tokens (the translation part) and ensure on device.
        target_tensor = batch[:, context_len:].to(device)
        # verification
        assert target_tensor.shape[1] == config.k, "target shape not k"

        # Clone current_input so that it can participate in gradient computations.
        current_input = current_input.clone()

        # Setting the loss tensor
        total_loss_tensor = torch.tensor(0.0, device=device)
        current_step = 0


        # Iterative multi-step prediction loop 
        while current_step < config.k:

            lstm_logits, _ = model(current_input)  # shape: (bsz, cur_seq_len, vocab)
            # Use logits from final time step for prediction:
            final_token_logit = lstm_logits[:, -1, :]  # shape: (bsz, vocab)

            # Compute loss for the current token prediction
            loss = loss_fn(final_token_logit, target_tensor[:, current_step])
            total_loss_tensor += loss
            
            # Get predicted token IDs (using argmax)
            final_token_ids = final_token_logit.argmax(dim=1) # shape: (bsz, 1)
            # Use the Llama embedding layer to obtain token embeddings.
            with torch.no_grad():
                # Using Llama's token embeddings; no need for autograd here.
                final_token_embeddings = llm_model.tok_embeddings(final_token_ids).unsqueeze(1) # shape: (bsz, 1, dim)
            
            # Append the predicted token embedding to the current input along sequence dimension
            current_input = torch.cat([current_input, final_token_embeddings], dim=1)

            current_step += 1

        # Backprop the loss for k steps
        mean_loss = total_loss_tensor / config.k
        mean_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Adding to batch loss
        batch_loss += mean_loss.item()

        # Averaging across all steps taken
        logger.info(f"Iteration {i}; Batch Loss: {batch_loss / current_step}")
        epoch_train_loss.append(batch_loss / current_step)
    
    avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
    logger.info(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
    training_loss.append(avg_train_loss)

    # ------------------
    # Validation Phase
    # ------------------
    model.eval()  # set LSTM head to evaluation mode
    val_loss = 0.0
    num_val_batches = 0

    # For token-level accuracy across the k steps, initialize accumulators
    # Here we accumulate counts for each step (0 to config.k - 1)
    token_correct = [0] * config.k
    token_total = [0] * config.k


    logger.info(f"Epoch {epoch+1} Validation Start:")
    print(f"EPOCH: {epoch+1} VALIDATION:")

    with torch.no_grad():
        for (val_batch, val_start_pos) in val_dl:
            val_batch = val_batch.to(device)

            

            # Obtain context length from the validation start positions.
            # Here, we use val_start_pos (not start_pos, which is for training)
            context_len = val_start_pos[0].item()

            # Create context tensor and forward pass through Llama
            context_tensor = val_batch[:, :context_len]
            current_input = llm_model(context_tensor, 0)  # shape: (batch_size, context_len, dim)

            # Create target tensor and ensure it's on device.
            target_tensor = val_batch[:, context_len:].to(device)
            # verification
            assert target_tensor.shape[1] == config.k, "target shape not k"


            # We will perform exactly config.k iterative steps.
            current_step = 0
            batch_loss_tensor = torch.tensor(0.0, device=device)
            batch_size = val_batch.shape[0]
            
            # Visualizing sample -- random
            visualization_tensor = context_tensor[0, :].unsqueeze(0)
            
            # Forward pass only for k tokens
            while current_step < config.k:
                lstm_logits, _ = model(current_input)  # shape: (batch_size, cur_seq_len, vocab)
                final_token_logit = lstm_logits[:, -1, :]  # shape: (batch_size, vocab)

                # Check and compute loss for the current step
                loss = loss_fn(final_token_logit, target_tensor[:, current_step])
                batch_loss_tensor += loss

                # Obtain predicted token IDs
                predicted_ids = final_token_logit.argmax(dim=1)  # shape: (batch_size,)
                # Adding to the visualization tensor
                visualization_tensor = torch.cat([visualization_tensor, torch.tensor([[predicted_ids[0]]], device = device)], dim = -1)
                
                # Update token-level accuracy for this step
                # Compare predicted_ids with the ground truth for current step
                ground_truth = target_tensor[:, current_step]  # shape: (batch_size,)
                # Count element-wise matches: convert boolean tensor to int
                token_correct[current_step] += (predicted_ids == ground_truth).sum().item()

                token_total[current_step] += batch_size


                # Obtain the embedding for the predicted tokens using Llama's embedding layer
                with torch.no_grad():
                    prediction_embeddings = llm_model.tok_embeddings(predicted_ids).unsqueeze(1)  # shape: (batch_size, 1, dim)

                
                current_input = torch.cat([current_input, prediction_embeddings], dim=1)
                current_step += 1

            # End of k-step loop for the batch
            batch_mean_loss = batch_loss_tensor / config.k
            val_loss += batch_mean_loss.item()
            num_val_batches += 1
            # printing random sample
            print(f"A Random Sample for Visualization:")
            print(tokenizer.decode(visualization_tensor[0].tolist()))
            print("\n")


    # Compute average validation loss over all batches
    avg_val_loss = val_loss / num_val_batches

    # Compute per-token (k-step) accuracy percentages:
    token_accuracies = [100 * (token_correct[i] / token_total[i]) for i in range(config.k)]


    # Log the results:
    logger.info(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
    for i, acc in enumerate(token_accuracies):
        logger.info(f"Epoch {epoch+1} Validation Token Accuracy at step {i+1}: {acc:.2f}%")
    validation_loss.append(avg_val_loss)
    model.train()  # switch back to training mode


    # ------------------
    # Checkpoint Saving
    # ------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"New best model saved to: {ckpt_path}")

    epoch_end = time.time()
    logger.info(f"Epoch {epoch+1} completed in {epoch_end - epoch_start:.2f} seconds")
    logger.info("-" * 100)

results_dict = {"Training Loss": training_loss, "Validation Loss": validation_loss}
with open("results.json", "w") as file:
    json.dump(results_dict, file)
logger.info(f"Results saved: results.json")

# End of training script.
