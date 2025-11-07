"""
Handles model loading, training, and evaluation.
"""
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import get_peft_model

def get_base_model(model_name, tokenizer):
    """
    Loads the base sequence classification model.

    Args:
        model_name (str): The name of the pre-trained model.
        tokenizer: The tokenizer to associate with the model.

    Returns:
        transformers.PreTrainedModel: The loaded model.
    """
    print(f"  → Loading {model_name} model for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    print("  ✓ Base model loaded successfully!")
    return model

def train_peft_model(model, peft_config, train_dataset, eval_dataset, output_dir="./peft-lora-rotten-tomatoes"):
    """
    Trains the PEFT model.

    Args:
        model (transformers.PreTrainedModel): The base model.
        peft_config (LoraConfig): The PEFT configuration.
        train_dataset (datasets.Dataset): The training dataset.
        eval_dataset (datasets.Dataset): The evaluation dataset.
        output_dir (str): The directory to save the trained model.

    Returns:
        transformers.Trainer: The trained trainer object.
    """
    # Create PEFT model
    print("  → Creating PEFT model with LoRA adapters...")
    peft_model = get_peft_model(model, peft_config)
    print("\n  → Trainable parameters:")
    peft_model.print_trainable_parameters()
    print()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Define evaluation metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Create Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("\n  → Starting training process...")
    print("  → Progress will be logged every 10 steps\n")
    trainer.train()
    print(f"\n  ✓ Training complete! Model saved to {output_dir}")
    
    return trainer

def evaluate_model(trainer, eval_dataset, model_description):
    """
    Evaluates a given model and prints the accuracy.

    Args:
        trainer (transformers.Trainer): The trainer with the model to evaluate.
        eval_dataset (datasets.Dataset): The dataset to evaluate on.
        model_description (str): A description of the model being evaluated.
    """
    print(f"  → Running evaluation on {len(eval_dataset)} samples...")
    results = trainer.evaluate(eval_dataset)
    
    # Check if we have accuracy in results, if not calculate it manually
    if 'eval_accuracy' in results:
        accuracy = results['eval_accuracy']
    elif 'eval_loss' in results:
        # If we only have loss, we need to compute predictions and accuracy
        predictions = trainer.predict(eval_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=-1)
        accuracy = (predicted_labels == predictions.label_ids).mean()
    else:
        accuracy = 0.0
    
    print(f"\n  ╔{'═'*58}╗")
    print(f"  ║ {model_description} Model Performance {' '*(35-len(model_description))}║")
    print(f"  ╠{'═'*58}╣")
    print(f"  ║ Accuracy: {accuracy:.4f} (or {accuracy*100:.2f}%) {' '*27}║")
    print(f"  ╚{'═'*58}╝")

def load_peft_model_for_inference(model_name, tokenizer, peft_model_path):
    """
    Loads a PEFT model for inference.

    Args:
        model_name (str): The name of the base pre-trained model.
        tokenizer: The tokenizer associated with the model.
        peft_model_path (str): The path to the saved PEFT adapter.

    Returns:
        peft.PeftModel: The loaded PEFT model.
    """
    from peft import PeftModel

    # Load the base model
    print("  → Loading base model...")
    model = get_base_model(model_name, tokenizer)
    
    # Load the PEFT model
    print(f"  → Loading LoRA adapter from {peft_model_path}...")
    peft_model = PeftModel.from_pretrained(model, peft_model_path)
    peft_model.eval()  # Set the model to evaluation mode
    
    print(f"  ✓ PEFT model loaded successfully for inference!")
    return peft_model
