# gpt-k: Next-token prediction (NTP) to Multi-token prediction (MTP)

The project undertaken by the authors aims to explore methods for speeding up LLM inference through prediction of multiple tokens in a single pass. 

Llama Model code adapted from: https://github.com/meta-llama/llama3/blob/main/llama/model.py

## Script Running

1. Ensure all Python packages are installed as mentioned in requirements.txt. 

2. Code requires the download of Llama3 Model weights; Download link for Llama Models:  https://www.llama.com/llama-downloads/

3. After downloading the llama model change the the LLAMA_PATH in config.py to the folder location.

4. Dataset samples can be visualized in visualization.ipynb.

5. Training Parameters are adjusted in config.py. Training can be run with train.py

6. eval.py is used to run timed evaluation and compare with the base model. Please change poma_ckpt_path accordingly to run eval on
trained models. 