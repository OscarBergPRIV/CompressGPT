import torch
from peft import PeftModel  # Add this import if not already present
from transformers import GPT2Config  # Add this for loading the base config
from pretrain_attnBottleneck_withinModel import GPT2LMHeadModelCompress
from torch.utils.data import DataLoader
# Assuming eval_dataloader is imported from a custom module; replace 'your_module' with the actual module name
from dataset_alpaca import tokenizer, eval_dataloader, device, val_ds
from config import CFG_M
cfg_m = CFG_M()
import math
from tqdm import tqdm
from evaluate import load


# Path to your LoRA adapter (replace if needed)
adapter_path = "./gpt2-alpaca-final_r8_bl3_ratio16_BL_type_attention"

# Load the base config (from the original pretrained model)
base_model_name = "gpt2"  # Or "gpt2-medium", etc., based on what was used for fine-tuning
config = GPT2Config.from_pretrained(base_model_name)

# Initialize the custom model with the base config and your custom parameters
model = GPT2LMHeadModelCompress(
    config=config,
    bl_layer=cfg_m.bl_layer,
    bl_ratio=cfg_m.bl_ratio,
    BL_type=cfg_m.BL_type,
    pretrain=False
)

# Attach the LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.to(device)
# Set to eval mode
model.eval()

def evaluate(model, generate_examples=True, num_examples=3, full_metrics=False):
    """
    Evaluate model on validation dataset.
    
    Args:
        model: Model to evaluate
        generate_examples: Whether to generate and display example outputs
        num_examples: Number of examples to generate
        full_metrics: If True, compute metrics on full validation set (slower)
    
    Returns:
        Tuple of (perplexity, bleu_score, rouge_scores, bertscore_scores)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    all_predictions = []
    all_references = []
    
    print(f"\nEvaluating on validation dataset ({len(eval_dataloader)} batches)...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch_device = {k: v.to(device) for k, v in batch.items()}
            
            # Compute loss
            outputs = model(**batch_device)
            loss = outputs.loss
            
            num_tokens = (batch_device["labels"] != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Collect predictions for BLEU/ROUGE (first 50 batches or all if full_metrics)
            compute_metrics_for_batch = full_metrics or batch_idx < 2
            if compute_metrics_for_batch:
                for i in range(batch_device["input_ids"].size(0)):
                    label_ids = batch_device["labels"][i]
                    valid_label_mask = label_ids != -100
                    
                    if valid_label_mask.sum() > 0:
                        # Extract reference text
                        reference_ids = label_ids[valid_label_mask]
                        reference_text = tokenizer.decode(reference_ids, skip_special_tokens=True)
                        
                        # Find prompt (everything before response)
                        first_valid_label_idx = valid_label_mask.nonzero()[0].item()
                        prompt_ids = batch_device["input_ids"][i:i+1, :first_valid_label_idx]
                        prompt_mask = batch_device["attention_mask"][i:i+1, :first_valid_label_idx]
                        
                        # Generate prediction
                        generated = model.generate(
                            prompt_ids,
                            max_new_tokens=min(100, len(reference_ids) + 20),
                            do_sample=False,
                            attention_mask=prompt_mask,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                        prediction_ids = generated[0, prompt_ids.size(1):]
                        prediction_text = tokenizer.decode(prediction_ids, skip_special_tokens=True)
                        
                        all_predictions.append(prediction_text)
                        all_references.append(reference_text)
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    # Compute BLEU and ROUGE scores
    bleu_score = 0.0
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    bertscore_score = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if len(all_predictions) > 0 and len(all_references) > 0:
        # BLEU
        bleu_metric = load("bleu")
        bleu_result = bleu_metric.compute(
            predictions=all_predictions,
            references=[[ref] for ref in all_references]
        )
        bleu_score = bleu_result["bleu"] * 100
        
        # ROUGE
        rouge_metric = load("rouge")
        rouge_result = rouge_metric.compute(
            predictions=all_predictions,
            references=all_references
        )
        rouge_scores = {
            "rouge1": rouge_result["rouge1"] * 100,
            "rouge2": rouge_result["rouge2"] * 100,
            "rougeL": rouge_result["rougeL"] * 100,
        }
        
        # BERTScore
        print("Computing BERTScore (this may take a moment)...")
        bertscore_metric = load("bertscore")
        bert_results = bertscore_metric.compute(
            predictions=all_predictions,
            references=all_references,
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            batch_size=16,
            device=device,
        )
        bertscore_score = {
            "precision": sum(bert_results["precision"]) / len(bert_results["precision"]) * 100,
            "recall": sum(bert_results["recall"]) / len(bert_results["recall"]) * 100,
            "f1": sum(bert_results["f1"]) / len(bert_results["f1"]) * 100,
        }
    
    # Generate examples
    if generate_examples:
        print("\n" + "="*80)
        print("GENERATION EXAMPLES FROM VALIDATION SET")
        print("="*80)
        
        example_indices = [0, len(val_ds)//4, len(val_ds)//2][:num_examples]
        
        for idx in example_indices:
            example = val_ds[idx]
            instr = example["instruction"]
            inp = example.get("input", "").strip()
            ground_truth = example["output"]
            
            # Create prompt
            if inp:
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instr}

### Input:
{inp}

### Response:"""
            else:
                prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instr}

### Response:"""
            
            # Generate
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(generated_ids[0, input_ids.size(1):], skip_special_tokens=True)
            
            print(f"\n--- Example {idx+1} ---")
            print(f"INSTRUCTION: {instr}")
            if inp:
                print(f"INPUT: {inp}")
            print(f"\nGROUND TRUTH:\n{ground_truth}")
            print(f"\nMODEL PREDICTION:\n{generated_text}")
            print("-" * 80)
    print(perplexity, bleu_score, rouge_scores, bertscore_score)
    return perplexity, bleu_score, rouge_scores, bertscore_score

evaluate(model)
