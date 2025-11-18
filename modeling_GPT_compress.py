"""
GPT-2 model with compression via bottleneck layers.
Based on HuggingFace transformers implementation with custom modifications.
"""

import math
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
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import GPT2Tokenizer
from config import CFG_M


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

class AttentionBottleneck(nn.Module):
    """
    Uses cross-attention to compress sequence to fixed number of latent tokens,
    then expands back. This is sequence-length agnostic and captures global context.
    
    Based on Perceiver architecture - compresses arbitrary length to fixed latents.
    """
    def __init__(self, hidden_size=768, num_latents=32, latent_dim=None, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_latents = num_latents
        self.latent_dim = latent_dim or hidden_size // 2
        self.num_heads = num_heads
        
        # Learnable latent queries for compression
        self.latent_queries = nn.Parameter(torch.randn(1, num_latents, self.latent_dim))
        
        # Cross-attention for compression (latents attend to input)
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=self.latent_dim,
            num_heads=num_heads,
            kdim=hidden_size,
            vdim=hidden_size,
            batch_first=True
        )
        
        # Cross-attention for decompression (input positions attend to latents)
        self.decompress_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            kdim=self.latent_dim,
            vdim=self.latent_dim,
            batch_first=True
        )
        
        # Layer norms
        self.compress_norm = nn.LayerNorm(self.latent_dim)
        self.decompress_norm = nn.LayerNorm(hidden_size)
        
        # Positional encoding for decompression queries
        self.pos_encoding = None
        
    def _get_positional_encoding(self, seq_len, device):
        """Generate or retrieve cached positional encoding."""
        if self.pos_encoding is None or self.pos_encoding.size(1) < seq_len:
            # Create positional encoding
            position = torch.arange(seq_len, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.hidden_size, 2, device=device) * 
                               (-math.log(10000.0) / self.hidden_size))
            
            pe = torch.zeros(seq_len, self.hidden_size, device=device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pos_encoding = pe.unsqueeze(0)  # [1, seq_len, hidden_size]
        
        return self.pos_encoding[:, :seq_len, :]
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            reconstructed: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compression: latent queries attend to input sequence
        latents = self.latent_queries.expand(batch_size, -1, -1)
        compressed, _ = self.compress_attn(
            query=latents,
            key=hidden_states,
            value=hidden_states
        )
        compressed = self.compress_norm(compressed + latents)
        
        # Decompression: positional queries attend to compressed latents
        pos_queries = self._get_positional_encoding(seq_len, hidden_states.device)
        pos_queries = pos_queries.expand(batch_size, -1, -1)
        
        reconstructed, _ = self.decompress_attn(
            query=pos_queries,
            key=compressed,
            value=compressed
        )
        reconstructed = self.decompress_norm(reconstructed + pos_queries)
        
        return reconstructed

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

        ## Bottleneck
        if bl_layer is not None:
            self.bl_layer = bl_layer
            if BL_type == "linear":
                self.bottleneck = AdaptiveBottleneck(ratio=bl_ratio) # SplitBottleneck(ratio=bl_ratio) #   AdaptiveBottleneck(ratio=bl_ratio)
            elif BL_type == "attention":
                self.attention = AttentionBottleneck(ratio=bl_ratio)

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
                hidden_states = self.bottleneck(hidden_states)

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
        )


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
        transformer_outputs = self.transformer(
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
        )


# Export standard GPT2LMHeadModel for compatibility
GPT2LMHeadModel = HF_GPT2LMHeadModel


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":


    print("FULL COMPARISON: Official GPT2LMHeadModel vs GPT2LMHeadModelCompress\n")

    model_name = "gpt2"  # or "gpt2-medium", "gpt2-large", "gpt2-xl"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Loading {model_name}...\n")

    # =============================================================
    # 1. Load OFFICIAL model (this is the real OpenAI GPT-2)
    # =============================================================
    official_model = GPT2LMHeadModel.from_pretrained(model_name)
    official_model.eval()
    official_model.to(device)

    # =============================================================
    # 2. Load YOUR custom model with exact same weights
    # =============================================================
    bl_layer = 2
    config = GPT2Config.from_pretrained(model_name)
    my_model = GPT2LMHeadModelCompress(config, bl_layer=bl_layer, bl_ratio=2)

    # Copy weights from official model's transformer
    my_model.transformer.load_state_dict(official_model.transformer.state_dict(), strict=False)  # strict=False to enable Bottleneck layer loading
    # TIE lm_head to wte (exactly like official)
    my_model.lm_head.weight = my_model.transformer.wte.weight

    my_model.eval()
    my_model.to(device)

    print("Both models loaded and weights tied\n")

    # =============================================================
    # 3. Tokenizer
    # =============================================================
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =============================================================
    # 4. Test inputs
    # =============================================================
    # Real text (more meaningful than random)
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Random input (for shape checking)
    batch_size, seq_len = 2, 15
    random_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    print(f"Prompt: '{prompt}'")
    print(f"Input IDs shape: {input_ids.shape}\n")

    # =============================================================
    # 5. Forward pass (logits + hidden states)
    # =============================================================
    with torch.no_grad():
        # Official
        official_out = official_model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False
        )

        # Your model
        my_out = my_model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False
        )

    # =============================================================
    # 6. COMPARISONS
    # =============================================================
    print("=== COMPARISON RESULTS ===\n")

    # 6.1 Hidden states
    hidden_diff = (my_out.hidden_states[-1] - official_out.hidden_states[-1]).abs().max()
    print(f"Last hidden state max diff : {hidden_diff.item():.2e}")

    # 6.2 Logits
    logits_diff = (my_out.logits - official_out.logits).abs().max()
    print(f"Logits             max diff : {logits_diff.item():.2e}")

    # 6.3 Perplexity on random input (more sensitive)
    with torch.no_grad():
        labels = random_ids[:, 1:].contiguous()
        shift_logits_official = official_model(random_ids).logits[:, :-1, :].contiguous()
        shift_logits_my = my_model(random_ids).logits[:, :-1, :].contiguous()

        loss_official = torch.nn.functional.cross_entropy(
            shift_logits_official.view(-1, config.vocab_size),
            labels.view(-1)
        )
        loss_my = torch.nn.functional.cross_entropy(
            shift_logits_my.view(-1, config.vocab_size),
            labels.view(-1)
        )
        ppl_official = torch.exp(loss_official)
        ppl_my = torch.exp(loss_my)

    print(f"Perplexity (official) : {ppl_official.item():.4f}")
    print(f"Perplexity (yours)    : {ppl_my.item():.4f}")
    print(f"Perplexity diff       : {abs(ppl_official - ppl_my).item():.6f}")

    # 6.4 Generation (greedy)
    print("\n=== GENERATION TEST (greedy, 30 tokens) ===")
    with torch.no_grad():
        gen_official = official_model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_my = my_model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    text_official = tokenizer.decode(gen_official[0], skip_special_tokens=True)
    text_my = tokenizer.decode(gen_my[0], skip_special_tokens=True)

    print("Official:", text_official)
    print("Yours   :", text_my)
    print(f"Generated text identical? {'YES' if text_official == text_my else 'NO'}")

    # =============================================================
    # 7. FINAL VERDICT
    # =============================================================
    print("\n" + "="*60)
    if (hidden_diff < 1e-6 and
        logits_diff < 1e-6 and
        abs(ppl_official - ppl_my) < 1e-4 and
        text_official == text_my):
        print("PERFECT MATCH! Your GPT2LMHeadModelCompress is 100% identical to official GPT-2")
        print("You can now safely insert bottlenecks, quantization, or split computing!")
    else:
        print("WARNING: Differences detected!")
        if hidden_diff >= 1e-6:
            print("   → Hidden states differ")
        if logits_diff >= 1e-6:
            print("   → Logits differ")
        if abs(ppl_official - ppl_my) >= 1e-4:
            print("   → Perplexity differs")
        if text_official != text_my:
            print("   → Generated text differs")
    print("="*60)
