"""
Main training and evaluation script for GPT-2 with compression.
Supports LoRA fine-tuning, bottleneck compression, and comprehensive metrics.
"""

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import (
    AutoModelForCausalLM,
    get_scheduler,
)
from tqdm import tqdm
import os
import math
from evaluate import load
import matplotlib.pyplot as plt

# Local imports
from modeling_GPT_compress import GPT2LMHeadModelCompress, GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from config import CFG_M
from dataset_alpaca import tokenizer, train_dataloader, eval_dataloader, model_name, device, val_ds

# Initialize configuration
cfg_m = CFG_M()
cfg_m.print_cfg()


def plot_metrics(metrics_history, save_path="./training_metrics.png"):
    """
    Plot training metrics over time.
    Creates subplots for Perplexity, BLEU, ROUGE, and BERTScore.
    """
    if len(metrics_history['steps']) == 0:
        print("No metrics to plot yet")
        return
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')
    
    steps = metrics_history['steps']
    
    # Plot 1: Perplexity (lower is better)
    ax1 = axes[0]
    ax1.plot(steps, metrics_history['perplexity'], 'b-o', linewidth=2, markersize=6, label='Perplexity')
    ax1.set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    ax1.set_title('Perplexity (Lower is Better)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: BLEU Score (higher is better)
    ax2 = axes[1]
    ax2.plot(steps, metrics_history['bleu'], 'g-o', linewidth=2, markersize=6, label='BLEU')
    ax2.set_ylabel('BLEU Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('BLEU Score (Higher is Better)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: ROUGE Scores
    ax3 = axes[2]
    ax3.plot(steps, metrics_history['rouge1'], 'r-o', linewidth=2, markersize=6, label='ROUGE-1')
    ax3.plot(steps, metrics_history['rouge2'], 'm-s', linewidth=2, markersize=6, label='ROUGE-2')
    ax3.plot(steps, metrics_history['rougeL'], 'c-^', linewidth=2, markersize=6, label='ROUGE-L')
    ax3.set_ylabel('ROUGE Score (%)', fontsize=12, fontweight='bold')
    ax3.set_title('ROUGE Scores (Higher is Better)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: BERTScore
    ax4 = axes[3]
    ax4.plot(steps, metrics_history['bertscore_f1'], color='purple', marker='o', 
             linestyle='-', linewidth=2, markersize=6, label='BERTScore-F1')
    ax4.plot(steps, metrics_history['bertscore_precision'], color='blue', marker="s", 
             linestyle="--", linewidth=2, markersize=6, label='Precision')
    ax4.plot(steps, metrics_history['bertscore_recall'], color='red', marker="^", 
             linestyle="--", linewidth=2, markersize=6, label='Recall')
    ax4.set_ylabel('BERTScore (%)', fontsize=12, fontweight='bold')
    ax4.set_title('BERTScore (Semantic Similarity)', fontsize=12)
    ax4.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to {save_path}")
    plt.close()


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
            compute_metrics_for_batch = full_metrics or batch_idx < 50
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
    
    model.train()
    return perplexity, bleu_score, rouge_scores, bertscore_score


def main():
    """Main training loop."""
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    # Model initialization
    if not cfg_m.default:
        # Use compressed model
        config = GPT2Config.from_pretrained(model_name)
        model = GPT2LMHeadModelCompress(
            config, 
            bl_layer=cfg_m.bl_layer, 
            bl_ratio=cfg_m.bl_ratio
        )
        original_model = GPT2LMHeadModel.from_pretrained(model_name)
        model.transformer.load_state_dict(original_model.transformer.state_dict(), strict=False)
        model.lm_head.weight = model.transformer.wte.weight
        del original_model  # Free memory
    else:
        # Use standard model
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Resize embeddings if needed
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    
    # LoRA configuration
    print("\nApplying LoRA...")
    from peft import LoraConfig, get_peft_model
    
    lora_config = LoraConfig(
        r=cfg_m.r_LoRA,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["transformer.bottleneck"] if not cfg_m.default else None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # FP16 setup
    use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()
        print(f"\nUsing FP16 mixed precision (CUDA compute capability: {torch.cuda.get_device_capability()})")
    else:
        scaler = None
        print("\nUsing FP32 precision")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = cfg_m.num_epochs
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_scheduler(
        "linear", 
        optimizer=optimizer, 
        num_warmup_steps=0, 
        num_training_steps=num_training_steps
    )
    
    # Metrics tracking
    metrics_history = {
        'steps': [],
        'perplexity': [],
        'bleu': [],
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'bertscore_f1': [],
        'bertscore_precision': [],
        'bertscore_recall': []
    }
    
    # Baseline evaluation
    print("\n" + "="*80)
    print("BASELINE EVALUATION")
    print("="*80)
    baseline_ppl, baseline_bleu, baseline_rouge, baseline_bertscore = evaluate(
        model, generate_examples=True, num_examples=2
    )
    
    print(f"\nBaseline Metrics:")
    print(f"  Perplexity: {baseline_ppl:.2f}")
    print(f"  BLEU: {baseline_bleu:.2f}")
    print(f"  ROUGE-1/2/L: {baseline_rouge['rouge1']:.2f} / {baseline_rouge['rouge2']:.2f} / {baseline_rouge['rougeL']:.2f}")
    print(f"  BERTScore-F1: {baseline_bertscore['f1']:.2f}")
    
    # Record baseline
    metrics_history['steps'].append(0)
    metrics_history['perplexity'].append(baseline_ppl)
    metrics_history['bleu'].append(baseline_bleu)
    metrics_history['rouge1'].append(baseline_rouge['rouge1'])
    metrics_history['rouge2'].append(baseline_rouge['rouge2'])
    metrics_history['rougeL'].append(baseline_rouge['rougeL'])
    metrics_history['bertscore_f1'].append(baseline_bertscore['f1'])
    metrics_history['bertscore_precision'].append(baseline_bertscore['precision'])
    metrics_history['bertscore_recall'].append(baseline_bertscore['recall'])
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    global_step = 0
    best_perplexity = float("inf")
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        model.train()
        
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # FP16 mixed precision training
            if use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            if step % 50 == 0:
                print(f"\nStep {global_step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Evaluation checkpoints
            if step % 500 == 0 and step > 0:
                print(f"\n{'='*80}")
                print(f"EVALUATION AT STEP {global_step}")
                print('='*80)
                
                perplexity, bleu, rouge, bertscore = evaluate(
                    model, generate_examples=True, num_examples=2
                )
                
                print(f"\nMetrics at step {global_step}:")
                print(f"  Perplexity: {perplexity:.2f}")
                print(f"  BLEU: {bleu:.2f}")
                print(f"  ROUGE-1/2/L: {rouge['rouge1']:.2f} / {rouge['rouge2']:.2f} / {rouge['rougeL']:.2f}")
                print(f"  BERTScore-F1: {bertscore['f1']:.2f}")
                
                # Record metrics
                metrics_history['steps'].append(global_step)
                metrics_history['perplexity'].append(perplexity)
                metrics_history['bleu'].append(bleu)
                metrics_history['rouge1'].append(rouge['rouge1'])
                metrics_history['rouge2'].append(rouge['rouge2'])
                metrics_history['rougeL'].append(rouge['rougeL'])
                metrics_history['bertscore_f1'].append(bertscore['f1'])
                metrics_history['bertscore_precision'].append(bertscore['precision'])
                metrics_history['bertscore_recall'].append(bertscore['recall'])
                
                # Plot metrics
                plot_metrics(metrics_history)
                
                # Save best model
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    save_dir = "./gpt2-alpaca-best"
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                    print(f"  ✓ New best model saved! (PPL: {best_perplexity:.2f})")
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION (full validation set)")
    print("="*80)
    
    final_ppl, final_bleu, final_rouge, final_bertscore = evaluate(
        model, generate_examples=True, num_examples=3, full_metrics=True
    )
    
    print(f"\nFinal Metrics:")
    print(f"  Perplexity: {final_ppl:.2f}")
    print(f"  BLEU: {final_bleu:.2f}")
    print(f"  ROUGE-1/2/L: {final_rouge['rouge1']:.2f} / {final_rouge['rouge2']:.2f} / {final_rouge['rougeL']:.2f}")
    print(f"  BERTScore-F1: {final_bertscore['f1']:.2f}")
    
    # Record final metrics
    final_step = len(train_dataloader) * num_epochs
    metrics_history['steps'].append(final_step)
    metrics_history['perplexity'].append(final_ppl)
    metrics_history['bleu'].append(final_bleu)
    metrics_history['rouge1'].append(final_rouge['rouge1'])
    metrics_history['rouge2'].append(final_rouge['rouge2'])
    metrics_history['rougeL'].append(final_rouge['rougeL'])
    metrics_history['bertscore_f1'].append(final_bertscore['f1'])
    metrics_history['bertscore_precision'].append(final_bertscore['precision'])
    metrics_history['bertscore_recall'].append(final_bertscore['recall'])
    
    # Save final model and plots
    if cfg_m.default:
        save_path = f"./metrics_default_r{cfg_m.r_LoRA}.png"
        final_dir = "./gpt2-alpaca-final-default"
    else:
        save_path = f"./metrics_r{cfg_m.r_LoRA}_bl{cfg_m.bl_layer}_ratio{cfg_m.bl_ratio}.png"
        final_dir = f"./gpt2-alpaca-final_r{cfg_m.r_LoRA}_bl{cfg_m.bl_layer}_ratio{cfg_m.bl_ratio}"
    
    plot_metrics(metrics_history, save_path=save_path)
    
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    print(f"\n{'='*80}")
    print(f"Training complete! Final model saved to {final_dir}")
    print(f"Metrics plot saved to {save_path}")
    print('='*80)


if __name__ == "__main__":
    main()