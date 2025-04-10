import torch
import torch.nn as nn

class LSTMHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, vocab_size, dropout=0.1):
        """
        LSTM head to predict token probabilities from sequence representations.
        
        Args:
            input_dim (int): Dimensionality of the input features (e.g., Llama output dim).
            hidden_dim (int): Number of hidden units in the LSTM.
            num_layers (int): Number of stacked LSTM layers.
            vocab_size (int): Vocabulary size for output predictions.
            dropout (float): Dropout probability applied between LSTM layers (only if num_layers > 1).
        """
        super(LSTMHead, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Projection layer from LSTM hidden state to vocabulary logits.
        self.linear = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass for the LSTM head.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            hidden (tuple, optional): Tuple (h0, c0) of initial hidden states.
        
        Returns:
            logits (Tensor): Logits for each time step of shape (batch_size, seq_len, vocab_size).
            hidden (tuple): Updated hidden state tuple from the LSTM.
        """
        # Process the input sequence through the LSTM.
        out, hidden = self.lstm(x, hidden)
        # Project the LSTM outputs to obtain logits over the vocabulary.
        logits = self.linear(out)
        return logits, hidden

# Example usage:
if __name__ == "__main__":
    batch_size = 8
    seq_len = 50
    input_dim = 512       # e.g., dimension of frozen Llama outputs
    hidden_dim = 512
    num_layers = 2
    vocab_size = 32000    # adjust to your tokenizer's vocabulary size

    model = LSTMHead(input_dim, hidden_dim, num_layers, vocab_size, dropout=0.1)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    logits, _ = model(dummy_input)
    print("Logits shape:", logits.shape)  # expected: (8, 50, 32000)
