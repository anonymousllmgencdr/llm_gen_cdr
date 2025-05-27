
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class GenQWen(object):
    def __init__(self, model_path, max_new_tokens=512, max_model_len=2048, temperature=0.7, top_p=0.8,
                 gpu_memory_utilization=0.9, repetition_penalty=1.05, n=1):

        self.sampling_params = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                                              repetition_penalty=repetition_penalty, max_tokens=max_new_tokens)
        self.model = LLM(model=model_path, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                         enable_lora=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt):
        texts = []
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        else:
            for val in prompt:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": val}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)
        response = []

        outputs = self.model.generate(texts, sampling_params=self.sampling_params)
        for output in outputs:
            now_res = []
            for idx in range(len(output.outputs)):
                now_res.append(output.outputs[idx].text)
            response.append(now_res)
        return response
