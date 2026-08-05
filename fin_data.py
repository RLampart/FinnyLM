import json
from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset
from datasets.splits import dataclasses

DATA_DIR = Path(__file__).parent.parent / "data"

def load_data():
  return load_dataset("Akhil-Theerthala/Kuvera-PersonalFinance-V2.1")['train']


class Dataset:
    def __init__(self, split: str):
        data = load_data().select_columns(['query','response'])
        if split=='train':
          data = data[:1000]
        else:
          data = data[1000:1100]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


if __name__ == "__main__":
    print(Dataset("train")[0])
