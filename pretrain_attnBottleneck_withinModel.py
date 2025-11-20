"""
GPT-2 model with compression via bottleneck layers.
Based on HuggingFace transformers implementation with custom modifications.
"""

import math
import copy
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import GPT2LMHeadModel as HF_GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2PreTrainedModel,
    GPT2Model,
    GPT2Block,
    GPT2Attention,
    GPT2MLP,
)
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.generation import GenerationMixin
from transformers import GPT2Tokenizer

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN, get_activation
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.utils import (
    ModelOutput,
    auto_docstring,
    logging,
)
import inspect

from modeling_AE_Attn import BottleneckAttention

from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import GPT2Tokenizer
from config import CFG_M

from modeling_AE_Attn import GPT2BlockSplit
from dataset_alpaca import tokenizer, train_dataloader, eval_dataloader, model_name, device, val_ds
from pretrain_attnBottleneck import CombinedLoss

# ============================================================================
# Bottleneck Modules for Compression
# ============================================================================

class AdaptiveBottleneck(nn.Module):
    """
    Simple bottleneck layer that compresses hidden states to a lower dimension.
    Uses linear projection with activation for information compression.
    """
    def __init__(self, hidden_size=768, ratio=2):
        super().__init__()
        self.inner_dim = hidden_size // ratio
        self.compress = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.decompress = nn.Linear(self.inner_dim, hidden_size, bias=False)
        self.act = nn.GELU()
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            decompressed: [batch, seq_len, hidden_size]
        """
        compressed = self.act(self.compress(hidden_states))
        decompressed = self.decompress(compressed)
        return decompressed


# this was for some experiments for gated progressive Bottleneck
class SplitBottleneck(nn.Module):
    """
    Bottleneck with learnable gating mechanism.
    Allows smooth transition between compressed and original representations.
    """
    def __init__(self, hidden_size=768, ratio=16, dropout=0.1):
        super().__init__()
        self.inner_dim = hidden_size // ratio
        self.compress = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.decompress = nn.Linear(self.inner_dim, hidden_size, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Gate value (controlled externally for progressive training)
        self.register_buffer("gate_value", torch.tensor(1.0))
        self.register_buffer("current_gate", torch.tensor(1.0))

    def forward(self, hidden_states):
        """
        Blends original and compressed representations based on gate value.
        gate=1.0: uses original hidden states
        gate=0.0: uses compressed representation
        """
        gate = self.gate_value
        self.current_gate.copy_(gate)

        compressed = self.dropout(self.act(self.compress(hidden_states)))
        decompressed = self.decompress(compressed)

        # Optional: add noise during training for regularization
        if self.training and gate > 0.7:
            compressed = compressed + torch.randn_like(compressed) * 0.05

        return gate * hidden_states + (1.0 - gate) * decompressed


# ============================================================================
# Custom GPT-2 Model with Bottleneck
# ============================================================================

class GPT2ModelCompress(GPT2PreTrainedModel):
    _supports_param_buffer_assignment = False

    def __init__(self, config, bl_layer=None, bl_ratio=2, BL_type="linear"):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

        print("Using Model w/ compression...")
        self.BL_type = BL_type
        print("Using Bottleneck type: ", self.BL_type)
        ## Bottleneck
        if bl_layer is not None:
            self.bl_layer = bl_layer
            if BL_type == "linear":
                self.bottleneck = AdaptiveBottleneck(ratio=bl_ratio) # SplitBottleneck(ratio=bl_ratio) #   AdaptiveBottleneck(ratio=bl_ratio)
            elif BL_type == "attention":
                print("Using Attention: ")
                config_BL = GPT2Config(
                    hidden_size=768,
                    num_attention_heads=16,
                    max_position_embeddings=1024,
                    scale_attn_weights=True,
                    attn_pdrop=0.1,
                    resid_pdrop=0.1,
                    activation_function="gelu_new",
                    _attn_implementation="eager"
                )
                config_BL_dec = copy.deepcopy(config_BL)
                config_BL_dec.hidden_size = int(config_BL.hidden_size / bl_ratio)
                print("config hidden shape: ", config_BL.hidden_size)
                print("config_dec hidden shape: ", config_BL_dec.hidden_size)
                self.bottleneck_enc = GPT2BlockSplit(config_BL, layer_idx=0, ratio=bl_ratio)
                self.bottleneck_dec = GPT2BlockSplit(config_BL_dec, layer_idx=1, ratio=1/bl_ratio)
    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    #@auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=self.config))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif self._attn_implementation != "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.bl_layer == i:
                #print(f"Bottleneck at layer {i}")
                if self.BL_type == "linear":
                    hidden_states = self.bottleneck(hidden_states)
                elif self.BL_type == "attention":
                    #hidden_states = self.bottleneck(hidden_states)
                    input_hs = hidden_states
                    hidden_states = self.bottleneck_enc(
                        hidden_states,
                        None, # past_key_values if not (self.gradient_checkpointing and self.training) else None,
                        cache_position,
                        causal_mask,
                        encoder_hidden_states,  # as a positional argument for gradient checkpointing
                        encoder_attention_mask=encoder_attention_mask,
                        use_cache=False,  #use_cache,
                        output_attentions=output_attentions,
                        **kwargs,
                    )
                    hidden_states = hidden_states[0]
                    hidden_states = self.bottleneck_dec(
                        hidden_states,
                        None,  # past_key_values if not (self.gradient_checkpointing and self.training) else None,
                        cache_position,
                        causal_mask,
                        encoder_hidden_states,  # as a positional argument for gradient checkpointing
                        encoder_attention_mask=encoder_attention_mask,
                        use_cache=False,  #use_cache,
                        output_attentions=output_attentions,
                        **kwargs,
                    )
                    hidden_states = hidden_states[0]
                    output_hs = hidden_states


            outputs = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
            #print("output.shape:", outputs[0].shape)

            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[2],)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), (input_hs, output_hs)


class GPT2LMHeadModelCompress(GPT2PreTrainedModel, GenerationMixin):
    """
    GPT-2 Language Model with compression bottleneck.
    Maintains compatibility with HuggingFace generation methods.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, bl_layer=None, bl_ratio=2, BL_type="linear"):
        super().__init__(config)
        print(f"Initializing GPT2LMHeadModelCompress (bl_layer={bl_layer}, bl_ratio={bl_ratio})")
        
        self.transformer = GPT2ModelCompress(config, bl_layer=bl_layer, bl_ratio=bl_ratio, BL_type=BL_type)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        """
        Forward pass for language modeling with optional labels.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get transformer outputs
        transformer_outputs, hs_pretrain = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        ), hs_pretrain


# Export standard GPT2LMHeadModel for compatibility
GPT2LMHeadModel = HF_GPT2LMHeadModel


# ============================================================================
# Standalone Testing
# ============================================================================
if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime
    from pathlib import Path
    import numpy as np
    import os
    from tqdm import tqdm
    from transformers import GPT2Config, GPT2LMHeadModel
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.nn.utils import clip_grad_norm_

    # Assuming train_dataloader, eval_dataloader, tokenizer, device are defined earlier from Alpaca prep
    # Also assuming CFG_M, CombinedLoss, GPT2LMHeadModelCompress are defined

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='collected_hidden_states/real_hidden_data_3_16')  # Unused if using Alpaca; keep for compatibility
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--train_split', type=float, default=0.9)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--mse_weight', type=float, default=1.0)
    parser.add_argument('--cosine_weight', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default='checkpoints_bottleneck')
    parser.add_argument('--model_name', type=str, default='adaptive_bottleneck')
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_config', action='store_true')
    args = parser.parse_args()

    cfg_m = CFG_M()
    print("Chosen ratio: ", cfg_m.bl_ratio)

    # SLURM: disable workers
    num_workers = args.num_workers
    if 'SLURM_JOB_ID' in os.environ:
        print("SLURM detected → num_workers=0")
        num_workers = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir) / f"_r_{cfg_m.bl_ratio}_pos_{cfg_m.bl_layer}_type_{cfg_m.BL_type}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Device: {device}")

    # Load config and models
    model_name = "gpt2"  # Or from cfg_m if defined
    config = GPT2Config.from_pretrained(model_name)
    model = GPT2LMHeadModelCompress(
        config,
        bl_layer=cfg_m.bl_layer,
        bl_ratio=cfg_m.bl_ratio,
        BL_type=cfg_m.BL_type
    ).to(device)
    original_model = GPT2LMHeadModel.from_pretrained(model_name)
    model.transformer.load_state_dict(original_model.transformer.state_dict(), strict=False)
    model.lm_head.weight = model.transformer.wte.weight  # Tie weights
    del original_model  # Free memory

    # Freeze everything except bottleneck
    for name, param in model.named_parameters():
        if "bottleneck_enc" in name or "bottleneck_dec" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer, scheduler, criterion
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    criterion = CombinedLoss(mse_weight=args.mse_weight, cosine_weight=args.cosine_weight)

    def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
        model.train()
        total_loss, total_mse, total_cos = 0, 0, 0
        num_batches = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            transformer_outputs, (hidden_before, hidden_after) = model.transformer(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            lengths = batch["attention_mask"].sum(dim=1)
            loss, mse, cos = criterion(hidden_after, hidden_before, lengths)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_mse += mse.item()
            total_cos += cos.item()
            num_batches += 1
            if step % 50 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | MSE: {mse.item():.4f} | Cos: {cos.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        return total_loss / num_batches, total_mse / num_batches, total_cos / num_batches

    def validate(model, dataloader, criterion, device):
        model.eval()
        total_loss, total_mse, total_cos = 0, 0, 0
        num_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, (hidden_before, hidden_after) = model.transformer(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                lengths = batch["attention_mask"].sum(dim=1)
                loss, mse, cos = criterion(hidden_after, hidden_before, lengths)
                total_loss += loss.item()
                total_mse += mse.item()
                total_cos += cos.item()
                num_batches += 1
        return total_loss / num_batches, total_mse / num_batches, total_cos / num_batches

    def save_checkpoint(model, optimizer, epoch, val_loss, path, args):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'args': vars(args)
        }, path)

    # Training loop
    best_val = float('inf')
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_mse, train_cos = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch)
        val_loss, val_mse, val_cos = validate(model, eval_dataloader, criterion, device)
        scheduler.step()
        print(f"\nEpoch {epoch} | Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Cos: {train_cos:.4f}) | Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, Cos: {val_cos:.4f})")
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, out_dir / f"epoch_{epoch}.pt", args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, out_dir / "best.pt", args)
            print("New best!")
    save_checkpoint(model, optimizer, args.num_epochs, val_loss, out_dir / "final.pt", args)
    print(f"Training done! Best val: {best_val:.4f}")
