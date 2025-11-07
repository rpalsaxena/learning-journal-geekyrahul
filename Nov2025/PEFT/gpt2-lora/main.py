"""
Main script to run the PEFT-LoRA fine-tuning and evaluation pipeline.
"""
from data_loader import get_tokenized_datasets
from model_handler import get_base_model, train_peft_model, evaluate_model, load_peft_model_for_inference
from config import peft_config
from transformers import Trainer, TrainingArguments
import numpy as np
import evaluate

def main():
    """
    Main function to execute the fine-tuning and evaluation pipeline.
    """
    model_name = "gpt2"
    output_dir = "./peft-lora-rotten-tomatoes"

    print("="*60)
    print("PEFT-LoRA Fine-Tuning Pipeline Started")
    print("="*60)
    
    # 1. Load and preprocess data
    print("\n[STEP 1/5] Loading and preprocessing dataset...")
    tokenized_datasets, tokenizer = get_tokenized_datasets(model_name)
    
    # Use smaller subsets for quicker execution
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))
    print(f"✓ Training samples: {len(small_train_dataset)}")
    print(f"✓ Evaluation samples: {len(small_eval_dataset)}")

    # 2. Load base model
    print("\n[STEP 2/5] Loading base GPT-2 model...")
    base_model = get_base_model(model_name, tokenizer)

    # Define evaluation metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # 3. Evaluate the base model before fine-tuning
    print("\n[STEP 3/5] Evaluating base model (before fine-tuning)...")
    # We need a dummy Trainer to run the evaluation
    base_model_trainer = Trainer(
        model=base_model,
        args=TrainingArguments(
            output_dir="./temp_eval",
            per_device_eval_batch_size=8,
        ),
        compute_metrics=compute_metrics,
    )
    evaluate_model(base_model_trainer, small_eval_dataset, "Original Base")

    # 4. Train the PEFT model
    print("\n[STEP 4/5] Training PEFT model with LoRA...")
    print("This may take several minutes. Please be patient...")
    peft_trainer = train_peft_model(
        model=base_model,
        peft_config=peft_config,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        output_dir=output_dir
    )
    
    # The Trainer automatically evaluates the model at the end of each epoch.
    # The final evaluation results are printed during the training process.

    # 5. Perform inference with the saved PEFT model
    print("\n[STEP 5/5] Loading fine-tuned model for inference...")
    inference_model = load_peft_model_for_inference(model_name, tokenizer, output_dir)
    
    # Create a new Trainer for the inference model
    inference_trainer = Trainer(
        model=inference_model,
        args=TrainingArguments(
            output_dir="./temp_inference_eval",
            per_device_eval_batch_size=8,
        ),
        compute_metrics=compute_metrics,
    )
    evaluate_model(inference_trainer, small_eval_dataset, "Fine-Tuned (Inference)")

    print("\n" + "="*60)
    print("PEFT-LoRA Fine-Tuning Pipeline Completed Successfully!")
    print("="*60)


if __name__ == "__main__":
    main()