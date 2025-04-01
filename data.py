from torch.utils.data import Dataset
from datasets import load_dataset
import os
import torch
from tokenizer import Tokenizer

class TranslationDataset(Dataset):
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
                self.tokenizer.encode(f"{source_lang.capitalize()}: {instance['translation'][source_lang]} {target_lang.capitalize()}", bos=False, eos=False)
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
        assert 0<=idx<len(self.tokens), f"{idx} must be >=0 and < {len(self.tokens)}"
        return self.tokens[idx]
    
    def __len__(self) -> int:
        return len(self.tokens)


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