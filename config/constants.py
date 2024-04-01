from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_WITHOUT_LORA_PATH = PROJECT_ROOT / 'models' / 'gpt2_finetuned_without_lora'
MODEL_WITH_LORA_PATH = PROJECT_ROOT / 'models' / 'gpt2_finetuned_with_lora'

CHECKPOINT_WITHOUT_LORA_PATH = PROJECT_ROOT / 'models' / 'gpt2_finetuned_without_lora' / 'checkpoint-370'
CHECKPOINT_WITH_LORA_PATH = PROJECT_ROOT / 'models' / 'gpt2_finetuned_with_lora' / 'checkpoint-370'
