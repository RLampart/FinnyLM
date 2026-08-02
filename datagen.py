from .cot import CoTModel
from .base_llm import BaseLLM
from .data import Dataset, benchmark
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct"

def parse_answer(prompt,answer):
  response = []
  response.append(prompt)
  try:
    value = float(answer.split("<answer>")[1].split("</answer>")[0])
  except (IndexError, ValueError):
    value = float("nan")
  response.append(value)
  response.append(answer.split("<|im_end|>")[0])
  return response

def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    model = CoTModel()
    model.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    data = Dataset("train")
    result = []
    failed = []
    length = len(data)
    rate = 0
    count = 0
    prompts = [model.format_prompt(d[0]) for d in data]
    generations = [model.batched_generate([p], num_return_sequences=5, temperature=temperature) for p in prompts]
    for idx, g in enumerate(generations):
        for sample in g:
          if abs(model.parse_answer(sample)-data[idx][1])<1e-2:
            result.append(parse_answer(data[idx][0],sample))
            rate+=1
            break   
          count+=1
          if count==5:
            failed.append(parse_answer(data[idx][0],sample))
        count = 0
    print(rate/length)
    with open(output_json, "w", encoding="utf-8") as rfile:
      json.dump(result,rfile)
    with open("data/fail.json", "w", encoding="utf-8") as ffile:
      json.dump(failed,ffile)
    #return result


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
