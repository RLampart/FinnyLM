from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        message = [{"role":"system", "content": "You are a unit convertor. Convert the given value and unit into the target unit. Enclose the converted value in this format <answer>value</answer>. DO NOT INCLUDE THE UNIT between the <answer></answer> tags. Feel free to think about the problem before providing the answer."},
                    {"role":"user", "content": "Convert [num][unit1] into [unit2]."} ,{"role":"assistant", "content":"1 [unit1] = x [unit2]. [num][unit1] = [num]*x = y. <answer>y</answer>"},{"role":"user", "content": "Tell me how many minutes are in 3 hours."} ,
                    {"role":"assistant", "content":"1 hour = 60 minutes. 3 hours = 60*3 = 180. <answer>180</answer>"},
                     {"role":"user", "content":"What is the conversion of 4 mph to ft/s?"} ,
                     {"role":"assistant", "content":"1 mph is equivalent to 1.46667 ft/s. Therefore 4 mph = 4*1.46667 = 5.86667. <answer>5.86667</answer>"},{"role":"user", "content":"Convert 2 centuries into weeks."} ,
                     {"role":"assistant", "content":"1 century is equal to 100 years and 1 year is equal to 52.177457 weeks. That means 2 centuries = 2*100 years = 200*52.177457 = 10435.4914 weeks. <answer>10435.4914</answer>"},
                   {"role":"user", "content": question}]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
