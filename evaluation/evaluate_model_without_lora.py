import pandas as pd
from evaluate import load
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config.constants import CHECKPOINT_WITHOUT_LORA_PATH
from data_preprocessor import DataImporter


class PromptBasedModel:
    def __init__(self, model_dir, dataframe):
        self.model = GPT2LMHeadModel.from_pretrained(model_dir)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        self._dataframe = dataframe

    def process_dataset(self, text_column):
        self._dataframe = self._dataframe[self._dataframe[text_column].apply(lambda x: len(x.split()) > 20)]
        self._dataframe['prompt'] = self._dataframe[text_column].apply(lambda x: ' '.join(x.split()[:3]))

    def generate_predictions(self, prompt_column):
        prompts = self._dataframe[prompt_column].tolist()
        generated_texts = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, padding=True)
            output = self.model.generate(input_ids, max_length=128, num_return_sequences=1)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
        self._dataframe['model_result'] = generated_texts

    def get_dataframe(self):
        return self._dataframe

class Evaluator:
    def __init__(self, reference_column, generated_column):
        self.reference_column = reference_column
        self.generated_column = generated_column

    def calculate_bleu(self, dataframe):
        references = dataframe[self.reference_column].tolist()
        predictions = dataframe[self.generated_column].tolist()
        bleu_metric = load('bleu')
        result_bleu = bleu_metric.compute(predictions=predictions, references=references)
        return result_bleu['bleu']

def main():
    importer = DataImporter('wikitext', 'wikitext-2-raw-v1')
    importer.obtain()
    test_data = importer.get_test_data()
    prompt_model = PromptBasedModel(CHECKPOINT_WITHOUT_LORA_PATH, test_data)
    prompt_model.process_dataset("text")
    prompt_model.generate_predictions("prompt")
    test_data = prompt_model.get_dataframe()

    evaluator = Evaluator("text", "model_result")
    bleu_score = evaluator.calculate_bleu(test_data)
    print("BLEU score:", bleu_score)

if __name__ == "__main__":
    main()
