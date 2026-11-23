"""Step 2: train a LoRA adapter on data/train.jsonl.

The base model stays frozen. Only the small adapter is trained and saved.
Uses the GPU automatically if there is one.
"""

import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from common import ADAPTER_DIR, BASE_MODEL, chat

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# Add the adapter. r = adapter size. Bigger r = more to learn with, slower to train.
model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ),
)
model.print_trainable_parameters()


def to_features(example):
    """Text -> token ids. Labels are -100 (ignored) for the prompt, so the
    model is only graded on the answer it should write."""
    prompt = tokenizer.apply_chat_template(chat(example["input"]), tokenize=False, add_generation_prompt=True)
    answer = example["output"] + "<|im_end|>"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return {
        "input_ids": torch.tensor(prompt_ids + answer_ids),
        "attention_mask": torch.ones(len(prompt_ids) + len(answer_ids), dtype=torch.long),
        "labels": torch.tensor([-100] * len(prompt_ids) + answer_ids),
    }


with open("data/train.jsonl") as f:
    train_data = [to_features(json.loads(line)) for line in f]

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    args=TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=2,
        per_device_train_batch_size=1,  # 1 at a time = no padding needed
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=42,
    ),
)
trainer.train()

model.save_pretrained(ADAPTER_DIR)
print(f"adapter saved to {ADAPTER_DIR}/")
