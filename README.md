# gpt-k: Next-token prediction (NTP) to Multi-token prediction (MTP)

The project undertaken by the authors aims to explore methods for speeding up LLM inference through prediction of multiple tokens in a single pass. Currently, 2 methods are being explored:


1.Matrix Based:

2. LSTM Based: K-steps forward pass is achieved through an LSTM model added on top of a Llama3 Large Language Model. Why: Traditional LLMs use Transformers as their backbone architecture, which need some form of positional embedding for linking a tokens in a sequence based on position. RNNs process the inputs sequentially and hence are Time-Aware, making them a natural fit for the MTP process. This Hypothesis is currently being tested. Branch: ashwath_edits


## Translation Task Experimental Setup:

Dataset : wmt14.
Since translation tasks have some amount of determinism in terms of the outputs expected, it eases the process of evaluating MTP methods. wmt14 has multiple language translations, lending itself well to rigourous testing of models. 

Loss: Cross-Entropy Loss
Cross Entropy Loss over the Vocabulary Size. 

Other Hyperparameters are to be optimized based on the needs of the experiment. 

### Scripts

1. data.py: Script for loading the dataset. Curated mainly for the Matrix Based model; LSTM version to be found in ashwath_edits branch.
2. train.py: Script for training. Curated mainly for the Matrix Based model; LSTM version to be found in ashwath_edits branch.
3. utils.py: Script for logger and Llama Model Loading
4. tokenizer.py: Llama Tokenizer Script
5. model.py: Llama Model Script

Most model loading functions and tokenizer functions adapted from: https://github.com/meta-llama/llama3/blob/main/llama/model.py


--------------------------------------------------TEMP--------------------------------------------------


LSTM Runs Notes:

Llama few shot: Uses few shot examples before the start of the training sample

Llama: Each sample is just the training sample

Model Parameters:

Large Language Model: Llama3: 1B

LSTM: 2 Layers, Hidden Dim 2048 + Feedforward LM Head with output size 128,256

K = 3. 

Training Parameters:
Epochs: 10; LR: 1e-3; WD: 0.05; BS: 32
Adam Optimizer and Cross Entropy Loss

Dataset: German-English, 8000 samples

--------------------------------------------------TEMP--------------------------------------------------
