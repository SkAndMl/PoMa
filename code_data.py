from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from tokenizer import Tokenizer
from typing import Tuple
from tqdm import tqdm
import re


class CodeDataset:
    """
    Unified Dataset object for Code Datasets; can be extended to other datasets as well.
    """
    def __init__(self, 
                config,
                 tokenizer: Tokenizer, 
                 split: str, 
                 max_seq_len: int, 
                 num_samples: int = None, 
                 seed: int = 2406, 
                 train_ratio: float = 0.9):
        self.task = config.task.lower()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Mapping task names to dataset names
        self.task_to_dataset = {
            "alpaca": config.CODE_ALPACA_DATASET,
            "conala": config.CONALA_DATASET,
            "evolinstruct": config.EVOLINSTRUCT_DATASET,
        }

        # Mapping task names to prompt preparation functions
        self.task_to_prompt = {
            "alpaca": self.prepare_codealpaca_prompt,
            "conala": self.prepare_conala_prompt,
            "evolinstruct": self.prepare_evolinstruct_prompt,
        }

        # Mapping task names to field mappings
        self.task_to_fields = {
            "alpaca": ("instruction", "input", "output"),
            "conala": ("rewritten_intent", None, "snippet"),
            "evolinstruct": ("instruction", None, "code"),
        }

        # Load the dataset
        dataset_path = self.task_to_dataset[self.task]
        print(f"Dataset path being loaded: {dataset_path}")
        # Alpaca and evolinstuct has no test set; needs to be created
        if config.task == "alpaca" or config.task == "evolinstruct":
            self.dataset = load_dataset(dataset_path, trust_remote_code=True)["train"]
            split_dataset = self.dataset.train_test_split(train_size=train_ratio, seed=seed)
            self.dataset = split_dataset[split]
        else:
            self.dataset = load_dataset(dataset_path, trust_remote_code=True)[split]

        # Specific to Evol Instruct
        if self.task == "evolinstruct":
            self.dataset = self._filter(self.dataset)
        self.dataset = self.dataset.shuffle(seed=seed)

        if num_samples is not None:
            self.dataset = self.dataset.select(range(num_samples))
        

    def __getitem__(self, idx):
        instance = self.dataset[idx]
        prepare_prompt_fn = self.task_to_prompt[self.task]
        try:
            input_str, output_str = prepare_prompt_fn(instance)
        except:
            print("NOT ABLE TO PROCESS SAMPLES")
        input_tokens = self.tokenizer.encode(input_str, bos=True, eos=False)
        start_pos = len(input_tokens)
        output_tokens = self.tokenizer.encode(output_str, bos=False, eos=True)
        end_pos = start_pos + len(output_tokens)
        return input_tokens, output_tokens, start_pos, end_pos

    def __len__(self):
        return len(self.dataset)

    # ----- Prompt Preparation Functions -----

    def prepare_codealpaca_prompt(self, instance):
        instr_field, input_field, output_field = self.task_to_fields[self.task]
        instruction = instance[instr_field]
        input_text = instance[input_field] if input_field and input_field in instance else ""
        output_text = instance[output_field]

        prompt = "Generate code for the following instruction"
        if input_text:
            prompt += " and the input given"
        prompt += f"\nInstruction: {instruction}\n"
        if input_text:
            prompt += f"Input: {input_text}\n"
        return prompt, output_text

    def prepare_conala_prompt(self, instance):
        instr_field, _, output_field = self.task_to_fields[self.task]
        instruction = instance[instr_field]
        output_text = instance[output_field]

        prompt = "Generate single line code for the following instruction"
        prompt += f"\nInstruction: {instruction}\n"
        return prompt, output_text

    def prepare_evolinstruct_prompt(self, instance):
        instr_field, _, output_field = self.task_to_fields[self.task]
        instruction = instance[instr_field]
        output_text = instance[output_field]

        prompt = "Generate python code for the following instruction"
        prompt += f"\nInstruction: {instruction}\n"
        return prompt, output_text

    # ----Filtering specific to EvolInstruct----
    def _filter(self, ds):
        # only samples which are python code blocks
        pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
        # Child function for pattern searching
        def extract_python_code(example):
            match = pattern.search(example['output'])
            return {'has_code': bool(match), 'code': match.group(1) if match else ""}
        # Filtering out the entire dataset
        ds = ds.map(extract_python_code)
        # Filtered dataset
        filtered_ds = ds.filter(lambda x: x['has_code'])
        return filtered_ds

class Collator:
    """
    Collate function for padding
    """
    def __init__(self, pad_id: int, max_seq_len: int = None):
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        """
        Collates a batch of samples into a padded batch tensor.
        
        batch: list of (input_tokens, output_tokens, start_pos) tuples
        """
        contexts, targets, start_positions, end_positions = zip(*batch)

        # Build final sequences by concatenating context + target
        sequences = [ctx + tgt for ctx, tgt in zip(contexts, targets)]

        # Find maximum sequence length in this batch
        max_len = max(len(seq) for seq in sequences)
        # If cropping based on seq length
        if self.max_seq_len:
            sequences = [seq[:self.max_seq_len] for seq in sequences]
            max_len = self.max_seq_len
            end_positions = [end if end < max_len else max_len for end in end_positions]

        # Batch size
        bsz = len(sequences)

        # Create batch tensor filled with pad_id
        batch_tensor = torch.full((bsz, max_len), self.pad_id, dtype=torch.long)

        # Fill batch tensor with sequences
        for i, seq in enumerate(sequences):
            batch_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        # Make start_positions and end positions a tensor (bsz, 1)
        start_positions = torch.tensor(start_positions, dtype=torch.long).reshape(bsz, 1)
        end_positions = torch.tensor(end_positions, dtype = torch.long).reshape(bsz, 1)
        return batch_tensor, start_positions, end_positions
