import os
import random
import time

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (DataCollatorForLanguageModeling, GPT2LMHeadModel,
                          GPT2Tokenizer, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)

from config.constants import MODEL_WITH_LORA_PATH
from data_preprocessor import DataImporter, DataPreprocessor


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(42)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
        all_param += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def measure_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss

def main():
    importer = DataImporter('wikitext', 'wikitext-2-raw-v1')
    importer.obtain()
    train_data = importer.get_train_data()
    val_data = importer.get_val_data()

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    preprocessor = DataPreprocessor(train_data, val_data, test_data, tokenizer)
    preprocessor.transform()
    preprocessor.tokenize_texts()

    train_tokenized = preprocessor.get_train_tokenized()
    val_tokenized = preprocessor.get_val_tokenized()

    gpt2_model_with_lora = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to('cpu')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    gpt2_model_with_lora.resize_token_embeddings(len(tokenizer))

    for param in gpt2_model_with_lora.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    gpt2_model_with_lora.gradient_checkpointing_enable()
    gpt2_model_with_lora.enable_input_require_grads()

    config = LoraConfig(
        r = 64,
        lora_alpha = 256,
        inference_mode = False,
        lora_dropout = 0.05,
        bias = "none",
        task_type = TaskType.CAUSAL_LM
    )

    gpt2_model_with_lora = get_peft_model(gpt2_model_with_lora, config)
    print_trainable_parameters(gpt2_model_with_lora)

    training_args_with_lora = TrainingArguments(
        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        num_train_epochs=2,
        learning_rate = 0.001,
        evaluation_strategy="epoch",
        logging_strategy = "epoch",
        save_steps = 100,
        logging_steps = 100,
        eval_steps = 100,
        save_strategy="epoch",
        output_dir=MODEL_WITH_LORA_PATH,
    )

    trainer_with_lora = Trainer(
        model=gpt2_model_with_lora,
        tokenizer=tokenizer,
        args=training_args_with_lora,
        data_collator=data_collator,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized
    )

    initial_memory = measure_memory_usage()
    start_time = time.time()

    trainer_with_lora.train()
    trainer_with_lora.evaluate()

    end_time = time.time()
    final_memory = measure_memory_usage()

    print(f"Memory used by model (CPU): {final_memory - initial_memory} bytes")
    print(f'This is total time for train: {end_time - start_time}')


if __name__ == "__main__":
    main()