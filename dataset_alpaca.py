"""
Alpaca dataset preparation module.
Handles tokenization and DataLoader creation for training and validation.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from config import CFG_M

# Initialize configuration
cfg_m = CFG_M()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model and tokenizer setup
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Add pad_token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading Alpaca dataset...")
ds = load_dataset("tatsu-lab/alpaca")["train"].train_test_split(test_size=0.1, seed=42)
train_ds = ds["train"]
val_ds = ds["test"]
print(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")


def tokenize_example(example, max_length=1024):
    """
    Tokenize a single Alpaca example with instruction-following format.
    
    Args:
        example: Dictionary with 'instruction', 'input', and 'output' keys
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'labels'
    """
    instr = example["instruction"]
    inp = example.get("input", "").strip()
    out = example["output"]
    
    # Create full prompt with instruction-following format
    if inp:
        full_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instr}

### Input:
{inp}

### Response:
{out}{tokenizer.eos_token}"""
        prompt_part = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instr}

### Input:
{inp}

### Response:"""
    else:
        full_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instr}

### Response:
{out}{tokenizer.eos_token}"""
        prompt_part = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instr}

### Response:"""
    
    # Tokenize full prompt and prompt-only part
    enc_full = tokenizer(full_prompt, truncation=True, max_length=max_length, add_special_tokens=False)
    enc_prompt = tokenizer(prompt_part, truncation=True, max_length=max_length, add_special_tokens=False)
    
    # Create labels: mask prompt tokens with -100 (only supervise response)
    input_ids = enc_full["input_ids"]
    labels = input_ids.copy()
    labels[:len(enc_prompt["input_ids"])] = [-100] * len(enc_prompt["input_ids"])
    
    return {
        "input_ids": input_ids,
        "attention_mask": enc_full["attention_mask"],
        "labels": labels,
    }


# Tokenize datasets
print("Tokenizing datasets...")
tokenized_train = train_ds.map(tokenize_example, batched=False, remove_columns=train_ds.column_names)
tokenized_val = val_ds.map(tokenize_example, batched=False, remove_columns=val_ds.column_names)
print("Tokenization complete!")


class DataCollatorForCausalLMWithLabels(DataCollatorWithPadding):
    """
    Custom data collator that handles padding for both inputs and labels.
    Labels are padded with -100 (ignored in loss computation).
    """
    def __call__(self, features):
        # Extract and remove labels
        labels = [f.pop("labels") for f in features]
        
        # Pad inputs using parent class
        batch = super().__call__(features)
        
        # Pad labels to match input length
        max_length = batch["input_ids"].size(1)
        padded_labels = [lab + [-100] * (max_length - len(lab)) for lab in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch


# Create data collator
data_collator = DataCollatorForCausalLMWithLabels(tokenizer=tokenizer, return_tensors="pt")

# Create DataLoaders
train_dataloader = DataLoader(
    tokenized_train, 
    batch_size=4, 
    shuffle=True, 
    collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_val, 
    batch_size=4, 
    shuffle=False, 
    collate_fn=data_collator
)

print(f"DataLoaders created: {len(train_dataloader)} train batches, {len(eval_dataloader)} eval batches")


if __name__ == "__main__":
    # Standalone test: inspect a batch
    print("\n" + "="*80)
    print("STANDALONE TEST: Inspecting first batch from training data")
    print("="*80)
    
    batch = next(iter(train_dataloader))
    
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch size: {batch['input_ids'].size(0)}")
    print(f"Sequence length: {batch['input_ids'].size(1)}")
    
    # Show first example in detail
    print("\n--- Example 0 ---")
    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    attention_mask = batch["attention_mask"][0]
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Count tokens
    real_tokens = (attention_mask == 1).sum().item()
    label_tokens = (labels != -100).sum().item()
    
    print(f"\nReal tokens (non-padded): {real_tokens}")
    print(f"Supervised tokens (labels != -100): {label_tokens}")
    
    # Decode
    print("\n--- Decoded Input ---")
    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
    print(decoded_input[:500] + "..." if len(decoded_input) > 500 else decoded_input)
    
    print("\n--- Decoded Labels (only supervised part) ---")
    supervised_ids = [tok if tok != -100 else tokenizer.pad_token_id for tok in labels.tolist()]
    decoded_labels = tokenizer.decode(supervised_ids, skip_special_tokens=False)
    # Find where labels start
    first_label_idx = (labels != -100).nonzero()[0].item() if (labels != -100).any() else 0
    print(f"Labels start at token index: {first_label_idx}")
    print(decoded_labels[first_label_idx:])
    
    print("="*80)
