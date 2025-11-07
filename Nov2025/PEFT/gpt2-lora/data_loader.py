"""
Handles loading and preprocessing of the dataset.
"""
from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokenized_datasets(model_name, max_length=128):
    """
    Loads and tokenizes the rotten_tomatoes dataset.

    Args:
        model_name (str): The name of the pre-trained model to use for tokenization.
        max_length (int): The maximum sequence length for tokenization.

    Returns:
        datasets.DatasetDict: The tokenized datasets.
    """
    # 1. Load Dataset
    print("  → Loading rotten_tomatoes dataset from Hugging Face...")
    dataset = load_dataset("rotten_tomatoes")

    # 2. Load Tokenizer
    print("  → Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Preprocess Dataset
    print("  → Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    print("  ✓ Dataset loaded and tokenized successfully!")
    return tokenized_datasets, tokenizer
