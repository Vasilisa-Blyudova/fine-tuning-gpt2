import time

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel
from transformers import (DataCollatorForLanguageModeling, GPT2LMHeadModel,
                          GPT2Tokenizer, Trainer, TrainerCallback,
                          TrainerControl, TrainerState, TrainingArguments)
import random
import os
import numpy as np
from config.constants import MODEL_WITHOUT_LORA_PATH
from data_preprocessor import DataImporter, DataPreprocessor

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(42)

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

    gpt2_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").to('cpu')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    gpt2_model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
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
        output_dir=MODEL_WITHOUT_LORA_PATH,
    )

    trainer = Trainer(
        model=gpt2_model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized
    )

    initial_memory = measure_memory_usage()
    start_time = time.time()

    trainer.train()
    trainer.evaluate()

    end_time = time.time()
    final_memory = measure_memory_usage()

    print(f"Memory used by model (CPU): {final_memory - initial_memory} bytes")
    print(f'This is total time for train: {end_time - start_time}')


if __name__ == "__main__":
    main()


