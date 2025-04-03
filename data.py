from torch.utils.data import Dataset
from datasets import load_dataset
import os
import torch
from tokenizer import Tokenizer
from typing import Tuple

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
        tokenizer_path: str
    ) -> None:
        """
        constructs
        """
        assert split in ["train", "validation", "test"], f"split must be train, validation or test"
        assert os.path.exists(tokenizer_path), OSError(f"{tokenizer_path} does not exit")
        try:
            ds = load_dataset("wmt14", dataset_hf_id)
        except Exception as e:
            raise ValueError(f"{dataset_hf_id} is invalid")
        
        self.tokenizer = Tokenizer(model_path=tokenizer_path)
        data = ds[split]
        self.tokens = []
        for instance in data:
            instance_toks = self.tokenizer.encode("Translate: ", bos=True, eos=False)
            instance_toks.extend(
                self.tokenizer.encode(f"{source_lang.capitalize()}: {instance['translation'][source_lang]} {target_lang.capitalize()}: ", bos=False, eos=False)
            )
            end_pos = len(instance_toks)
            instance_toks.extend(
                self.tokenizer.encode(instance['translation'][target_lang], bos=False, eos=True)
            )
            self.tokens.append([instance_toks, end_pos])

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
    

    def prepare_batch(self) -> Tuple[torch.Tensor]:
        """
        batches inputs into the same seq_len 
        """
        tokens, start_positions = self.dataset[self.idx:self.idx+self.batch_size]
        # padding it to max_len per seq
        # it will either be efficient or won't use gpu parallelization properly
        # but it will save space
        max_seq_len = max(len(token) for token in tokens)
        batch = self.tokenizer.pad_id*torch.ones(size=(len(tokens), max_seq_len), dtype=torch.long)
        for i, token in enumerate(tokens):
            batch[i][:len(token)] = torch.tensor(token, dtype=torch.long)
        start_positions = torch.tensor(start_positions, dtype=torch.long).reshape((batch.shape[0], 1))
        return batch, start_positions


if __name__ == "__main__":
    from config import LLAMA_PATH
    ds = TranslationDataset(
        dataset_hf_id="de-en",
        source_lang="de",
        target_lang = "en",
        split = "validation",
        tokenizer_path = f"{LLAMA_PATH}/tokenizer.model"
    )
    print(ds[0])