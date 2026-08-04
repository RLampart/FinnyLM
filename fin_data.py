import json
from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset

from base_llm import BaseLLM

DATA_DIR = Path(__file__).parent.parent / "data"

def load_data():
  return load_dataset("Akhil-Theerthala/Kuvera-PersonalFinance-V2.1")


class Dataset:
    def __init__(self, split: str):
        with (DATA_DIR / f"{split}.json").open() as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def is_answer_valid(answer: float, correct_answer: float, relative_tolerance: float = 0.05) -> bool:
    return abs(round(answer, 3) - round(correct_answer, 3)) < relative_tolerance * abs(round(correct_answer, 3))


if __name__ == "__main__":
    print(Dataset("train")[0])
