from torch.nnn.functional import cosine_similarity
from base_llm import BaseLLM
from fin_data import Dataset, benchmark
from transformers import Trainer
from transformers.training_args import TrainingArguments

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "finny"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, query: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{query} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    query_len = len(tokenizer(query)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * query_len + input_ids[query_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a query / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    
    return {"query":prompt, "answer":answer}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - query: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    from peft import get_peft_model, LoraConfig
    config = LoraConfig(r=8, lora_alpha=32, target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
    llm = BaseLLM()
    model = get_peft_model(llm.model, config)
    model.enable_input_require_grads()
    epochs = kwargs.get("epoch", 5)
    batch = kwargs.get("batch", 32)
    args = TrainingArguments(output_dir=output_dir, logging_dir=output_dir, report_to="tensorboard", gradient_checkpointing=True, learning_rate=1e-3, num_train_epochs=epochs, per_device_train_batch_size=batch)
    t_dataset = TokenizedDataset(llm.tokenizer, Dataset("train"), format_example)
   # v_dataset = TokenizedDataset(llm.tokenizer, Dataset("valid"), format_example)
    training = Trainer(model,args,train_dataset=t_dataset)
    training.train()
    training.save_model(output_dir)
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    base= llm.tokenizer(testset['response'])
    check = llm.tokenizer(llm.model.batched_generate(testset['query']))
    similarity = cosine_similarity(base, check, dim=1)
    print(f"{similarity}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
