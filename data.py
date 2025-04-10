from torch.utils.data import Dataset
from datasets import load_dataset
import os
import torch
from tokenizer import Tokenizer
from typing import Tuple, List, Optional, Dict
#import config

class TranslationDataset(Dataset):
    """
    loads any translation dataset in the wmt14 family, provided the correct ID is given
    """
    def __init__(
        self, 
        dataset_hf_id: str, 
        source_lang: str, 
        target_lang: str, 
        split: str,
        tokenizer: Tokenizer,
        max_seq_len: int,
        num_instances: Optional[int] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None
    ) -> None:
        """
        constructs
        """
        assert split in ["train", "validation", "test"], f"split must be train, validation or test"
        try:
            ds = load_dataset("wmt14", dataset_hf_id, split=split)
        except Exception as e:
            raise ValueError(f"{dataset_hf_id} is invalid")
        
        self.few_shot_examples = few_shot_examples
        # Edit made to fit my code
        """if few_shot_examples is None and dataset_hf_id in config.FEW_SHOT_EXAMPLES:
                self.few_shot_examples = config.FEW_SHOT_EXAMPLES[dataset_hf_id]"""

        self.tokenizer = tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_seq_len = max_seq_len

        num_instances = min(ds.num_rows, num_instances) if num_instances is not None else ds.num_rows
        self.tokens = []
        for instance in ds['translation'][:num_instances]:
            tokens, start_position = self._prepare_instance(instance)
            self.tokens.append([tokens[:self.max_seq_len], start_position])
    
    def _prepare_instance(self, instance) -> Tuple[torch.Tensor, int]:
        """prepares the string, tokenizes, returns tokens and start position"""
        input_string = f"Translate from {self.source_lang} to {self.target_lang}\n"
        if self.few_shot_examples is not None:
            for fs_instance in self.few_shot_examples:
                input_string += f"{self.source_lang}: {fs_instance['source']}; {self.target_lang}: {fs_instance['target']}\n"
        input_string += f"{self.source_lang}: {instance[self.source_lang]}; {self.target_lang}: "
        input_tokens = self.tokenizer.encode(input_string, bos=True, eos=False)
        start_position = len(input_tokens)
        target_tokens = self.tokenizer.encode(instance[self.target_lang], bos=False, eos=True)
        return input_tokens+target_tokens, start_position


    def __getitem__(self, idx: int) -> list:
        """
        returns tuple of src and tgt tokens
        """
        if isinstance(idx, slice):
            slice_range = range(*idx.indices(len(self.tokens)))
            tokens = [self.tokens[idx][0] for idx in slice_range]
            start_positions = [self.tokens[idx][1] for idx in slice_range]
            return tokens, start_positions
            
        assert 0<=idx<len(self.tokens), f"{idx} must be >=0 and < {len(self.tokens)}"
        return self.tokens[idx]
    
    def __len__(self) -> int:
        return len(self.tokens)


class DataLoader:
    """
    lightweight dataloader
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        tokenizer: Tokenizer,
        k: int
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.idx = 0
        self.k = k
    
    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor]:
        if self.idx >= len(self.dataset):
            raise StopIteration
        batch, start_positions = self.prepare_batch()
        self.idx += self.batch_size
        return batch, start_positions

    def __len__(self) -> int:
        """returns the length"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    

    def prepare_batch(self) -> Tuple[torch.Tensor]:
        """
        Creating batches with padding for context and target
        """
        tokens, start_positions = self.dataset[self.idx:self.idx+self.batch_size]
        # padding it to max context -- the lstm way of doing it
        max_context_len = max(start_positions)
        # Max target length is k
        max_target_len = self.k
        # Max token length is context + target
        max_token_len = max_context_len + max_target_len
        # Creating batch tensor
        batch = self.tokenizer.special_tokens["<|eot_id|>"]*torch.ones(size=(len(tokens), max_token_len), dtype = torch.long) # shape = (bsz, max_token_len)
        # Iterating
        for i, token in enumerate(tokens):
            start_pos = start_positions[i]
            # Assigning context with padding to batch
            batch[i][:start_pos] =  torch.tensor(token[:start_pos], dtype = torch.long)
            # Assigning the target -- k tokens
            # In case there are fewer tokens than k -- can't believe this exists
            target_len = min(self.k, len(token[start_pos:]))
            batch[i][max_context_len: max_context_len + target_len] = torch.tensor(token[start_pos: start_pos + target_len], dtype = torch.long)

        start_position = torch.tensor([max_context_len], dtype = torch.long)
        return batch, start_position


if __name__ == "__main__":
    from config import llama_path
    from tokenizer import Tokenizer
    few_shot_examples = [
        {
            "source": "Ich gehe morgen ins Kino.", 
            "target": "I am going to the cinema tomorrow."
        },
        {
            "source": "Er ist ein sehr guter Koch.", 
            "target": "He is a very good cook."
        },
    ]
    tokenizer = Tokenizer(model_path=f"{llama_path}/tokenizer.model")
    ds = TranslationDataset(
        dataset_hf_id="de-en",
        source_lang="de",
        target_lang = "en",
        split = "validation",
        tokenizer=tokenizer,
        few_shot_examples=few_shot_examples
    )
    print(ds[0])