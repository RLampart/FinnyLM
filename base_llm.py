from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fin_data import load_data

checkpoint = "Qwen/Qwen2.5-0.5B-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, prompt: list) -> str:
        return (" ").join(prompt)+"Describe the position of the business based on these statements."

    def generate(self, prompt: str) -> str:
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.
        """
        from tqdm import tqdm  # Importing tqdm for progress bar
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]
        tokens = self.tokenizer(prompts, padding=True, return_tensors='pt', padding_side='left').to(device)
        sample =False
        if num_return_sequences==None:
          num_return_sequences=1
        if temperature > 0:
           sample = True
           output = self.model.generate(**tokens, max_new_tokens=512, temperature=temperature, do_sample=sample, num_return_sequences=num_return_sequences, eos_token_id=self.tokenizer.eos_token_id)
        else:
          output = self.model.generate(**tokens, max_new_tokens=512, num_return_sequences=num_return_sequences, eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.batch_decode(output[:, len(tokens["input_ids"][0]) :])

    def answer(self, *questions) -> list[float]:
      pass

def answer():
  model = BaseLLM()
  prompt = model.format_prompt(text[:16])
  #print(prompt)
  answer = model.generate(prompt)
  print("Response:", answer)

def test_model():
    dataset = load_data()
    queries = dataset['train']['query'][:5]
    model = BaseLLM()
    answer = model.batched_generate(queries)
    for idx, query in enumerate(queries):
      print("Input:\n", query)
      print("Output:\n", answer[idx])


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "answer":answer})
