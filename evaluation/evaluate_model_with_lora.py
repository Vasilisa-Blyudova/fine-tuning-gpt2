import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from evaluate import load
from peft import PeftModel
from config.constants import CHECKPOINT_WITH_LORA_PATH
from data_preprocessor import DataImporter
from evaluate_model_without_lora import PromptBasedModel, Evaluator

class LoRAPromptBasedModel(PromptBasedModel):
    def __init__(self, model_dir, dataframe):
        super().__init__(model_dir, dataframe)
        initial_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        self.model = PeftModel.from_pretrained(initial_model, model_dir)
        self.model.eval()

def main():
    importer = DataImporter('wikitext', 'wikitext-2-raw-v1')
    importer.obtain()
    test_data = importer.get_test_data()
    # print(test_data)
    prompt_model = LoRAPromptBasedModel(CHECKPOINT_WITH_LORA_PATH, test_data)
    prompt_model.process_dataset("text")
    prompt_model.generate_predictions("prompt")
    test_data = prompt_model.get_dataframe()

    evaluator = Evaluator("text", "model_result")
    bleu_score = evaluator.calculate_bleu(test_data)
    print("BLEU score:", bleu_score)

if __name__ == "__main__":
    main()
