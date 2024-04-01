import torch
from peft import PeftModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config.constants import CHECKPOINT_WITH_LORA_PATH


def generate_text(prompt, model_dir=CHECKPOINT_WITH_LORA_PATH):
    initial_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    model = PeftModel.from_pretrained(initial_model, model_dir)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=128, )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    return decoded_output

def main():
    prompt = input("Enter prompt: ")
#     prompt = "As the Nameless officially do not exist"
    generated_text = generate_text(prompt)
    print(f"Generate text: {generated_text}")

if __name__ == "__main__":
    main()