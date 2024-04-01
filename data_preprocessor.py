from datasets import Dataset, load_dataset
from transformers import GPT2Tokenizer


class DataImporter:

    def __init__(self, hf_name, subset):

        self._hf_name = hf_name
        self._subset = subset
        self._train_raw_data = None
        self._val_raw_data = None
        self._test_raw_data = None

    def obtain(self):
        train_dataset = load_dataset(self._hf_name, self._subset, split='train[:3000]').to_pandas()
        val_dataset = load_dataset(self._hf_name, self._subset, split='validation[:1000]').to_pandas()
        test_dataset = load_dataset(self._hf_name, self._subset, split='test[:50]').to_pandas()

        self._train_raw_data = train_dataset
        self._val_raw_data = val_dataset
        self._test_raw_data = test_dataset

    def get_train_data(self):
        return self._train_raw_data

    def get_val_data(self):
        return self._val_raw_data

    def get_test_data(self):
        return self._test_raw_data


class DataPreprocessor:

    def __init__(self, train_data, val_data, test_data, tokenizer):
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data
        self._tokenizer = tokenizer
        self._train_tokenized = None
        self._val_tokenized = None

    def transform(self):
        self._train_data.drop_duplicates(inplace=True)
        self._val_data.drop_duplicates(inplace=True)
        self._test_data.drop_duplicates(inplace=True)

        self._train_data.dropna(inplace=True)
        self._val_data.dropna(inplace=True)
        self._test_data.dropna(inplace=True)

        self._train_data.reset_index(inplace=True, drop=True)
        self._val_data.reset_index(inplace=True, drop=True)
        self._test_data.reset_index(inplace=True, drop=True)

        self._train_data = Dataset.from_pandas(self._train_data)
        self._val_data = Dataset.from_pandas(self._val_data)
        self._test_data = Dataset.from_pandas(self._test_data)

    def tokenize_texts(self):
        self._train_tokenized = self._train_data.map(lambda x: self._tokenizer(x['text'], truncation=True,
        padding='max_length', max_length=128), batched=True)
        self._val_tokenized = self._val_data.map(lambda x: self._tokenizer(x['text'], truncation=True,
        padding='max_length', max_length=128), batched=True)
        self._train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask'])
        self._val_tokenized.set_format('torch', columns=['input_ids', 'attention_mask'])

    def get_train_tokenized(self):
        return self._train_tokenized

    def get_val_tokenized(self):
        return self._val_tokenized
