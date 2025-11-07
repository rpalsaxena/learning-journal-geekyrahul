"""
Configuration for the PEFT-LoRA model.
"""
from peft import LoraConfig, TaskType

# LoRA configuration for the sequence classification task
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn"],  # Target attention layers in GPT-2
    bias="none"
)
