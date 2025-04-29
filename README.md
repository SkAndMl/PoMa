# gpt-k: Next-token prediction (NTP) to Multi-token prediction (MTP)

## PoMa

The project undertaken by the authors aims to explore methods for speeding up LLM inference through prediction of multiple tokens in a single pass. Current method involves the use of Positional Matrices to perform parallel prediction of Tokens. We obtained 2-3X Speed up in token generation when compared to a vanilla NTP model. 

Llama Model code adapted from: https://github.com/meta-llama/llama3/blob/main/llama/model.py

### Script Running 

1. Ensure all Python packages are installed as mentioned in requirements.txt. 

2. Code requires the download of Llama3 Model weights; Download link for Llama Models:  https://www.llama.com/llama-downloads/

3. After downloading the llama model change the the LLAMA_PATH in config.py to the folder location.

4. Dataset samples can be visualized in visualization.ipynb.

5. Training Parameters are adjusted in config.py. Training can be run with train.py

6. eval.py is used to run timed evaluation and compare with the base model. Please change poma_ckpt_path accordingly to run eval on
trained models.

