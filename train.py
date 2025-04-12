from tokenizer import Tokenizer
from data import TranslationDataset, DataLoader
import torch
import time
from utils import load_model, create_logger, generate_lstm
from lstm import LSTMHead
import json
import os
from dataclasses import dataclass
from pathlib import Path
from torchmetrics import Accuracy
import gc

torch.autograd.set_detect_anomaly(True)

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
    Llama_path: str = "/home/users/ntu/ashwaths/workspace/lstm-llama/Llama3.2-1B"
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
# Hidden dim reduced for reduced computation
model = LSTMHead(
    input_dim=config.dim,
    hidden_dim= config.dim,
    num_layers=config.layers,
    vocab_size=config.vocab_size,
    dropout=0.1
).to(device)
model.train()
logger.info("LSTM Model Locked and Loaded")

# Tokenizer Loading -- Llama
tokenizer = Tokenizer(os.path.join(config.Llama_path, "tokenizer.model"))
logger.info("Tokenizer Loaded")

# Loading the train and validation dataset 
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
# Running with no few shot since LSTM has limited context window
few_shot_examples = None

# Adding k to dataloader to reduce Vram spend
train_ds = TranslationDataset(
    dataset_hf_id="de-en",
    source_lang="de",
    target_lang="en",
    split="train",
    tokenizer=tokenizer,
    few_shot_examples = few_shot_examples,
    max_seq_len = -1,
    num_instances = 8000,
    k = config.k
)

val_ds = TranslationDataset(
    dataset_hf_id="de-en",
    source_lang="de",
    target_lang="en",
    split="validation",
    tokenizer=tokenizer,
    few_shot_examples = few_shot_examples, 
    max_seq_len = -1,
    k = config.k
)

train_dl = DataLoader(train_ds, config.max_batch_size, tokenizer, k = config.k)
val_dl = DataLoader(val_ds, config.max_batch_size, tokenizer, k = config.k)
logger.info("Dataset Locked and Loaded")

# Optimizer, loss function and Accuracy
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=config.wd)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
metric = Accuracy(top_k=5, task = "multiclass", num_classes = config.vocab_size).to(device)

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
    for i, batch in enumerate(train_dl):

        # llama input: Context tensor -- Padded at the start; hence, last k tokens target
        context_tensor = batch[:, :-config.k].to(device) #shape: (bsz, context_len)

        # Forward pass: Get context vector from Llama model (no gradients)
        with torch.no_grad():
            # Here we pass start_pos=0 because we assume the context was processed wholly.
            current_input = llm_model(context_tensor, 0)  # shape: (bsz, context_len, dim)

        #logger.info(f"[GPU] After llama output: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        llama_output = current_input.cpu()
        
        # Do not need context tensor or current input anymore
        del context_tensor
        del current_input

        # Preallocating tensor to reduce VRAM usage due to cat
        # Seq length is constant as it is padded at the start
        seq_length = llama_output.shape[1] + config.k
        context_tensor = torch.zeros((batch.shape[0], seq_length, config.dim), device = "cpu", pin_memory = True) #shape: (bsz, seq_len, dim)
        # Update leaving k tokens at the end for the step predictions
        context_tensor[:, :-config.k, :] = llama_output

        # Do no need current input anymore
        del llama_output

        # Get target tokens (the translation part) and ensure on device.
        target_pos = context_tensor.shape[1] - config.k
        target_tensor = batch[:,target_pos:].to(device)

        # verification
        assert target_tensor.shape[1] == config.k, "target shape not k"

        current_step = 0
        # Track of loss across k steps
        total_loss_tensor = torch.tensor(0.0).to(device)

        # Iterative multi-step prediction loop 
        while current_step < config.k:
            # Cloning tensor to GPU for forward pass; this way, only the input to the lstm is loaded to GPU
            lstm_input = context_tensor[:, :target_pos + current_step, :].clone().to(device, non_blocking = True)
            lstm_logits, _ = model(lstm_input)  # shape: (bsz, cur_seq_len, vocab)

            #logger.info(f"[GPU] After lstm forward step {current_step}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
            # Compute loss for the current token prediction -- use only the last word in lstm logits
            loss = loss_fn(lstm_logits[:, -1, :], target_tensor[:, current_step])
            total_loss_tensor += loss
            
            # Get predicted token IDs (using argmax)
            final_token_ids = lstm_logits[:, -1, :].argmax(dim=1) # shape: (bsz,)
            # Use the Llama embedding layer to obtain token embeddings.
            with torch.no_grad():
                # Using Llama's token embeddings; no need for autograd here.
                final_token_embeddings = llm_model.tok_embeddings(final_token_ids) # shape: (bsz, dim)
            
            # Append the predicted token embedding to the current input along sequence dimension
            context_tensor[:, target_pos + current_step, :] = final_token_embeddings.cpu()
            current_step += 1

        # Average loss across k steps
        mean_k_loss = total_loss_tensor / config.k
        batch_loss = mean_k_loss.item()
        #Backprop
        mean_k_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Averaging across all steps taken
        logger.info(f"Iteration {i}; Sample Context Length {seq_length - config.k}; Batch Loss: {batch_loss}")
        epoch_train_loss.append(batch_loss)
        # Garbage collection
        gc.collect()
        torch.cuda.empty_cache()
    
    avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
    logger.info(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
    training_loss.append(avg_train_loss)

    # ------------------
    # Validation Phase
    # ------------------
    model.eval()  # set LSTM head to evaluation mode
    val_loss = 0.0
    num_val_batches = 0
    # For token-level accuracy across the k steps
    accuracies = [0] * config.k

    logger.info(f"Epoch {epoch+1} Validation Start:")
    print(f"EPOCH: {epoch+1} VALIDATION:")

    with torch.no_grad():
        for i, val_batch in enumerate(val_dl):

            # llama input: Context tensor -- Padded at the start; hence, last k tokens target
            context_tensor = val_batch[:, :-config.k].to(device) #shape: (bsz, context_len)
            
            # Here we pass start_pos=0 because we assume the context was processed wholly.
            current_input = llm_model(context_tensor, 0)  # shape: (bsz, context_len, dim)
            llama_output = current_input.cpu()

            # Do not need context tensor and current input anymore
            del context_tensor
            del current_input

            # Preallocating tensor to reduce VRAM usage due to cat
            # Seq length is constant as it is padded at the start
            seq_length = llama_output.shape[1] + config.k
            context_tensor = torch.zeros((val_batch.shape[0], seq_length, config.dim), device = "cpu", pin_memory = True) #shape: (bsz, seq_len, dim)
            # Update leaving k tokens at the end for the step predictions
            context_tensor[:, :-config.k, :] = llama_output

            # Do no need llama_output anymore
            del llama_output

            # Get target tokens (the translation part) and ensure on device.
            target_pos = context_tensor.shape[1] - config.k
            target_tensor = val_batch[:,target_pos:].to(device)

            # verification
            assert target_tensor.shape[1] == config.k, "target shape not k"


            # We will perform exactly config.k iterative steps.
            current_step = 0
            batch_loss = 0.0
            batch_size = val_batch.shape[0]

            # Print for last batch first sample
            if i == len(val_dl) - 1:
                context = val_batch[0, :target_pos].tolist()
                print_tokens = []
            
            # Forward pass only for k tokens
            while current_step < config.k:

                # Cloning tensor to GPU for forward pass; this way, only the input to the lstm is loaded to GPU
                lstm_input = context_tensor[:, :target_pos + current_step, :].clone().to(device, non_blocking = True)
                lstm_logits, _ = model(lstm_input)  # shape: (batch_size, cur_seq_len, vocab)

                # Check and compute loss for the current step
                loss = loss_fn(lstm_logits[:, -1, :], target_tensor[:, current_step])
                batch_loss += loss.item()

                # Obtain predicted token IDs
                predicted_ids = lstm_logits[:, -1, :].argmax(dim=1)  # shape: (batch_size,)

                #OPTIONAL: Add predicted tokens for printing
                if i == len(val_dl) - 1:
                    print_tokens.append(predicted_ids[0].item())
                
                # Update token-level accuracy for this step
                # Compare predicted_ids with the ground truth for current step
                ground_truth = target_tensor[:, current_step]  # shape: (batch_size,)
                acc = metric(lstm_logits[:, -1, :], ground_truth)

                # Adding to accumulator
                accuracies[current_step] += acc.item()

                # Prediction embeddings -- id -> embeddings
                prediction_embeddings = llm_model.tok_embeddings(predicted_ids)  # shape: (batch_size, dim)

                
                # Append the predicted token embedding to the current input along sequence dimension
                context_tensor[:, target_pos + current_step, :] = prediction_embeddings.detach().cpu()
                current_step += 1

            # End of k-step loop for the batch
            batch_mean_loss = batch_loss / config.k
            val_loss += batch_mean_loss
            num_val_batches += 1
    
    # Compute average validation loss over all batches
    avg_val_loss = val_loss / num_val_batches
    # Compute average per token accuracy across batches
    for i in range(config.k):
        accuracies[i] = accuracies[i] / num_val_batches
    
    
    # Log the results:
    logger.info(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
    for i, acc in enumerate(accuracies):
        logger.info(f"Epoch {epoch+1} Validation Token Accuracy at step {i+1}: {acc*100:.2f}%")
    validation_loss.append(avg_val_loss)

    # print sample
    full_text = context + print_tokens
    decoded = tokenizer.decode(full_text)
    logger.info(f"SAMPLE: {decoded} \n")
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
