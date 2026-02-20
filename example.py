import os
from nanovllm import LLM, SamplingParams
# Tokenizer class from Hugging Face Transformers library. knows how to load any tokenizer.
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    # enforce_eager disable CUDA graph; tensor_parallel_size=1 only one GPU is used.
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # temperature controls randomness, higher more creative; max_tokens max
    # length of generated text.
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            # Here we don't tokenize the prompt, prompt still string, but chat
            # formatted it. The real tokenization happens later in llm.generate()
            tokenize=False,
            # Append generation prompt to the end of the user prompt, so that
            # the model can better understand where to start generating. For
            # Qwen, append `<|im_start|>assistant\n` to the end of the user prompt.
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
