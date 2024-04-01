# Fine-tuning of GPT2 and LoRA fine-tuning of GPT2

This repository contains a pipeline with a fine-tuning gpt2 model for the generation task, 
as well as a LoRA fine-tuning pipeline of GPT2 model.
The **dataset** can be found [here](https://huggingface.co/datasets/wikitext).
The **initial model** can be found [here](https://huggingface.co/openai-community/gpt2).
Fine-tuning models can be found
[here](https://drive.google.com/drive/folders/19m0J3kwyaRbj7D4qaNIN-ST_G7N6Fgv4?usp=sharing).

To test models, run the modules `inference_without_lora.py` and `inference_with_lora.py`. It is 
located in [inference](./inference).

### Results

| Model                     | Time     | Memory           |
|:--------------------------|:---------|:-----------------|
| Fine-tuning of GPT2       | 3690.71  | 1627230208 bytes |
| LoRA fine-tuning of GPT2  | 3478.39  | 1114918912 bytes |

### BLEU metric

* Fine-tuning of GPT2: 0.0166
* LoRA fine-tuning of GPT2: 0.0235

