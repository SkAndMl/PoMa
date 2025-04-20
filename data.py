from torch.utils.data import Dataset
from datasets import load_dataset
import os
import torch
from tokenizer import Tokenizer
from typing import Tuple, List, Optional, Dict
import config
from tqdm import tqdm

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
        use_few_shot: Optional[bool] = True,
    ) -> None:
        """
        constructs
        """
        assert split in ["train", "validation", "test"], f"split must be train, validation or test"
        try:
            ds = load_dataset("wmt14", dataset_hf_id, split=split)
        except Exception as e:
            raise ValueError(f"{dataset_hf_id} is invalid")
        
        self.few_shot_examples = None
        if use_few_shot and dataset_hf_id in config.FEW_SHOT_EXAMPLES:
                self.few_shot_examples = config.FEW_SHOT_EXAMPLES[dataset_hf_id]

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

class PyDataset(Dataset):
    """
    loads the python dataset
    """
    def __init__(
        self, 
        hf_id: str, 
        split: str,
        tokenizer: Tokenizer,
        max_seq_len: int,
    ) -> None:
        """
        constructs
        """
        assert split in ["train", "test"], f"split must be train or test"
        try:
            ds: Dataset = load_dataset(hf_id)['train']
            ds = ds.shuffle(seed=2406)
            idx = int(ds.num_rows * 0.9)
            ds = ds[:idx] if split=="train" else ds[idx:]
        except Exception as e:
            raise ValueError(f"{hf_id} is invalid")
        
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.tokens = []
        for i in tqdm(range(len(ds['input']))):
            instance = {k:ds[k][i] for k in ds}
            tokens, start_position = self._prepare_instance(instance)
            self.tokens.append([tokens[:self.max_seq_len], start_position])
    
    def _prepare_instance(self, instance) -> Tuple[torch.Tensor, int]:
        """prepares the string, tokenizes, returns tokens and start position"""
        input_string = f"Create python code for the following instruction" + \
                       " and the input given\n" if len(instance['input'])>0 else "\n"
        input_string += f"Instruction: {instance['instruction']}\n"
        if len(instance['input']) > 0:
            input_string += f"Input: {instance['input']}\n"
        input_tokens = self.tokenizer.encode(input_string, bos=True, eos=False)
        start_position = len(input_tokens)
        target_tokens = self.tokenizer.encode(instance['output'], bos=False, eos=True)
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
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.idx = 0
    
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
        batches inputs into the same seq_len 
        """
        tokens, start_positions = self.dataset[self.idx:self.idx+self.batch_size]
        # padding it to max_len per seq
        # it will either be efficient or won't use gpu parallelization properly
        # but it will save space
        max_seq_len = max(len(token) for token in tokens)
        batch = self.tokenizer.special_tokens["<|eot_id|>"]*torch.ones(size=(len(tokens), max_seq_len), dtype=torch.long)
        for i, token in enumerate(tokens):
            batch[i][:len(token)] = torch.tensor(token, dtype=torch.long)
        start_positions = torch.tensor(start_positions, dtype=torch.long).reshape((batch.shape[0], 1))
        return batch, start_positions


if __name__ == "__main__":
    import config
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
        {
            "source": "Das Wetter ist heute schön.", 
            "target": "The weather is nice today."
        },
        {
            "source": "Wir haben gestern einen langen Spaziergang gemacht.", 
            "target": "We took a long walk yesterday."
        },
        {
            "source": "Kannst du mir bitte helfen?", 
            "target": "Can you please help me?"
        }
    ]
    tokenizer = Tokenizer(model_path=f"{config.llama_path}/tokenizer.model")
    ds = TranslationDataset(
        dataset_hf_id="de-en",
        source_lang="de",
        target_lang = "en",
        split = "train",
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        num_instances=config.num_train_instances,
        use_few_shot=False
    )
    print(ds[0])