import json
from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset
from datasets.splits import dataclasses

DATA_DIR = Path(__file__).parent.parent / "data"

def load_data(split):
  return load_dataset("Akhil-Theerthala/Kuvera-PersonalFinance-V2.1", split=f"train[{split}]")


class Dataset:
    def __init__(self, split: str):
        if split=='train':
          self.data = load_data(":1000").select_columns(['query','response'])
        else:
          self.data = load_data("1000:1100").select_columns(['query','response'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


if __name__ == "__main__":
    print(Dataset("train")[0])
