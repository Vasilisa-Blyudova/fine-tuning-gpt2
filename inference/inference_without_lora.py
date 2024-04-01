import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config.constants import CHECKPOINT_WITHOUT_LORA_PATH


def generate_text(prompt, model_name=CHECKPOINT_WITHOUT_LORA_PATH):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=128, )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    return decoded_output

def main():
    prompt = input("Enter prompt: ")
    generated_text = generate_text(prompt)
    print(f"Generate text: {generated_text}")

if __name__ == "__main__":
    main()
