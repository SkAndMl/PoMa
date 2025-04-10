from tokenizer import Tokenizer
import torch
from model import Transformer, ModelArgs
from pathlib import Path
import json
import os
import logging
import random

@torch.inference_mode()
def generate(model, max_tokens: int, prompt: str, tokenizer: Tokenizer, device: str) -> str:
    """
    generates content
    """
    assert device in {"cpu", "cuda", "mps"}
    model = model.to(device)
    model.eval()
    tokens = torch.tensor([tokenizer.encode(prompt, bos=True, eos=False)]).to(device)
    for _ in range(max_tokens):
        logits: torch.Tensor = model(tokens, 0) # 1, seq_len, vocab_size
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        tokens = torch.cat([tokens, torch.tensor([[next_token]], device=device)], dim=-1).to(device)
    return tokenizer.decode(tokens[0].tolist())

def generate_lstm(lstm, llama,max_tokens: int, prompt: str, tokenizer: Tokenizer, device: str) -> str:
    """
    Module to generate and visualize sample
    Parameters:
        lstm: Trained lstm model
        llama: llama model of choice
        max_tokens: Tokens to generate for visualization
        prompt: Your context vector -- as seen in training
        tokenizer: Tokenizer to view the words generated
        device: The device for inference
    Return:
        string
    """
    # Assure valid device
    assert device in {"cpu", "cuda", "mps"}
    # Models to device and eval mode on
    lstm = lstm.to(device)
    lstm.eval()
    llama = llama.to(device)
    llama.eval()
    print("PROMPT: \n" + prompt)
    # Assuming that the prompt or context isn't yet tensored and tokenized
    tokens = torch.tensor([tokenizer.encode(prompt, bos = True, eos = False)]).to(device) # shape: (1, seq_len)
    # One step Llama context tensor; 0 given as there is no separation of text into context and target
    with torch.no_grad():
        context_tensor = llama(tokens, 0) # shape: (1, seq_len, dim)
    
    # n-step LSTM prediction
    for _ in range(max_tokens):
        # logits
        with torch.no_grad():
            logits, _ = lstm(context_tensor) # shape: (seq_len, vocab)

        # Next token 
        final_token_logit = logits[:, -1, :]  # shape: (1, vocab)
        # Get predicted token IDs (using argmax)
        final_token_id = final_token_logit.argmax(dim=1) # shape: (1, 1)
        # printing predicted token
        print(tokenizer.decode(final_token_id.tolist()))
        # Adding token to output
        tokens = torch.cat([tokens, torch.tensor([[final_token_id]], device = device)], dim = -1)

        # Use the Llama embedding layer to obtain token embeddings.
        with torch.no_grad():
            # Using Llama's token embeddings; no need for autograd here.
            final_token_embeddings = llama.tok_embeddings(final_token_id).unsqueeze(1) # shape: (bsz, 1, dim)
        
        # Append the predicted token embedding to the current input along sequence dimension
        context_tensor = torch.cat([context_tensor, final_token_embeddings], dim=1)
    
    return tokenizer.decode(tokens[0].tolist())



def load_model(ckpt_path: str, device: str, max_batch_size: int, max_seq_len: int):
    """
    loads the weights into the model
    """
    assert device in {"cpu", "cuda", "mps"}
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

def create_logger():
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('training.log', mode='w')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def set_seed(seed: int) -> None:
    """
    Sets the seed for generating random numbers in Python, NumPy, and PyTorch.
    This helps to ensure reproducibility across runs.

    Parameters:
        seed (int): The seed value to be used for reproducibility.

    Returns:
        None
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy's random module
    ##np.random.seed(seed)
    
    # PyTorch CPU and GPU seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Additional settings for PyTorch to ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False