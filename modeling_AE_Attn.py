# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import copy

#from transformers import initialization as init
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
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from transformers import GPT2Tokenizer

def eager_attention_forward_split(module, query, key, value, attention_mask, **kwargs):
    #print(f"  [EAGER ATTENTION] Query shape: {query.shape}, Key shape: {key.shape}, Value shape: {value.shape}")
    
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    #print(f"  [EAGER ATTENTION] Attention weights shape after Q@K^T: {attn_weights.shape}")

    if module.scale_attn_weights:
        scale = value.size(-1) ** 0.5
        attn_weights = attn_weights / torch.full(
            [], scale, dtype=attn_weights.dtype, device=attn_weights.device
        )
        #print(f"  [EAGER ATTENTION] Scaled by sqrt(d_k) = {scale:.2f}")

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        layer_scale = float(module.layer_idx + 1)
        attn_weights = attn_weights / layer_scale
        #print(f"  [EAGER ATTENTION] Scaled by inverse layer idx: {layer_scale}")

    #print("attn weight.shape: ", attn_weights.shape)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min

        #print("query len, key len: ", query_length, key_length)
        #causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        #print("Causal_mask: ", causal_mask)

        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        #print(f"  [EAGER ATTENTION] Applied causal mask (query_len={query_length}, key_len={key_length})")

    if attention_mask is not None:
        # Apply the attention mask
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask
        #print(f"  [EAGER ATTENTION] Applied attention mask")
        #print("Attn weights mask: ", attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    #print(f"  [EAGER ATTENTION] Applied softmax")

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)
    #print(f"  [EAGER ATTENTION] Output shape after attention: {attn_output.shape}")

    return attn_output, attn_weights


class GPT2AttentionSplit(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, ratio=2):
        super().__init__()
        print(f"[GPT2Attention.__init__] Initializing attention layer (layer_idx={layer_idx}, is_cross_attention={is_cross_attention})")
        self.ratio = ratio
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        # Compute reduced attention output dimension
        self.attn_output_dim = int(self.embed_dim / (ratio))
        if int(self.attn_output_dim * ratio) != self.embed_dim:
            raise ValueError(f"hidden_size ({self.embed_dim}) must be divisible by attn_dim_ratio * integer. "
                             f"Got ratio={ratio}, attn_output_dim={self.attn_output_dim}")
        self.head_dim = self.attn_output_dim // self.num_heads
        if self.head_dim * self.num_heads != self.attn_output_dim:
            raise ValueError(f"attn_output_dim ({self.attn_output_dim}) must be divisible by num_heads ({self.num_heads})")

        self.split_size = self.attn_output_dim
        #############


        
        #print(f"[GPT2Attention.__init__] embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}")
        
        #if self.head_dim * self.num_heads != self.embed_dim:
        #    raise ValueError(
        #        f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
        #        f" {self.num_heads})."
        #    )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:  # Doesnt matter
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.attn_output_dim, self.embed_dim)    # New self.embed_dim


        # NEW
        #self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.attn_output_dim, self.attn_output_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = not is_cross_attention

    def _upcast_and_reordered_attn_split(self, query, key, value, attention_mask=None):
        #print(f"  [UPCAST ATTENTION] Using upcast and reordered attention")
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)
        
        #print(f"  [UPCAST ATTENTION] Scale factor: {scale_factor:.6f}")

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            #print("query len, key len: ", query_length, key_length)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
            #print(f"  [UPCAST ATTENTION] Applied causal mask")

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
            #print(f"  [UPCAST ATTENTION] Applied attention mask")

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], ...]:
        #print(f"\n[ {self.layer_idx}, input shape: {hidden_states.shape}")
        
        is_cross_attention = encoder_hidden_states is not None
        #print(f"[GPT2Attention.forward] is_cross_attention: {is_cross_attention}")
        
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask
            #print(f"[GPT2Attention.forward] Cross-attention: query from hidden_states")

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
                #print(f"[GPT2Attention.forward] Using cached key/value states")
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
                #print(f"[GPT2Attention.forward] Computing key/value from encoder_hidden_states")
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            #print(f"[GPT2Attention.forward] Self-attention: split QKV from c_attn output")
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        #print(f"[GPT2Attention.forward] Reshaped Q: {query_states.shape}, K: {key_states.shape}, V: {value_states.shape}")

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward_split
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        #print(f"[GPT2Attention.forward] Using attention implementation: {self.config._attn_implementation}")

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn_split(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        #print(f"[GPT2Attention.forward] Reshaped attention output: {attn_output.shape}")
        
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        #print(f"[GPT2Attention.forward] After projection and dropout: {attn_output.shape}")

        return attn_output, attn_weights


class GPT2MLPSPLIT(nn.Module):
    def __init__(self, intermediate_size, config, ratio=2.0):
        super().__init__()
        print(f"[GPT2MLP.__init__] Initializing MLP with intermediate_size={intermediate_size}")
        
        embed_dim = config.hidden_size
        # NEW
        embed_dim = int(embed_dim / ratio)

        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        #print(f"[GPT2MLP.forward] Input shape: {hidden_states.shape}")
        
        hidden_states = self.c_fc(hidden_states)
        #print(f"[GPT2MLP.forward] After c_fc: {hidden_states.shape}")
        
        hidden_states = self.act(hidden_states)
        #print(f"[GPT2MLP.forward] After activation")
        
        hidden_states = self.c_proj(hidden_states)
        #print(f"[GPT2MLP.forward] After c_proj: {hidden_states.shape}")
        
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2BlockSplit(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None, ratio=2.0):
        super().__init__()
        print(f"\n[GPT2Block.__init__] Initializing block {layer_idx}")
        
        hidden_size = config.hidden_size # // int(ratio)    # NEW
        inner_dim = 4 * hidden_size # config.n_inner if config.n_inner is not None else 4 * hidden_size
        print(f"[GPT2Block.__init__] hidden_size={hidden_size}, inner_dim={inner_dim}")

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2AttentionSplit(config=config, layer_idx=layer_idx, ratio=ratio)
        self.ln_2 = nn.LayerNorm(int(hidden_size /ratio), eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            print(f"[GPT2Block.__init__] Adding cross-attention")
            self.crossattention = GPT2AttentionSplit(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLPSPLIT(inner_dim, config, ratio=ratio)

    def forward(
        self,
        hidden_states: Optional[tuple[torch.FloatTensor]],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], Optional[tuple[torch.Tensor, tuple[torch.FloatTensor, ...]]]]:
        #print(f"\n{'='*60}")
        #print(f"[GPT2Block.forward] Processing block")
        #print(f"{'='*60}")
        
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        #print(f"[GPT2Block.forward] After ln_1 (pre-attention LayerNorm)")
        
        attn_output, self_attn_weights = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        # residual connection
        # NEw TODO uncomment
        hidden_states = attn_output     # + residual
        #print(f"[GPT2Block.forward] After self-attention w/o residual: {hidden_states.shape}")

        
        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            #print(f"[GPT2Block.forward] Processing cross-attention")
            
            cross_attn_output, cross_attn_weights = self.crossattention(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # residual connection TODO
            hidden_states = residual + cross_attn_output
            #print(f"[GPT2Block.forward] After cross-attention + residual: {hidden_states.shape}")

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        #print(f"[GPT2Block.forward] After ln_2 (pre-MLP LayerNorm)")
        
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states
        #print(f"[GPT2Block.forward] After MLP + residual: {hidden_states.shape}")

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
            if encoder_hidden_states is not None:
                outputs += (cross_attn_weights,)

        return outputs

class BottleneckAttention(GPT2PreTrainedModel):
    _supports_param_buffer_assignment = False

    def __init__(self, config, ratio=2.0):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        print("AT THE BEGINNIGN IN BOTTLENECK EMBED DIM: ", self.embed_dim)

        #self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        config_dec = copy.deepcopy(config)
        config_dec.hidden_size = int(config.hidden_size / ratio)
        print("config hidden shape: ", config.hidden_size)
        print("config_dec hidden shape: ", config_dec.hidden_size)
        
        self.h = nn.ModuleList([GPT2BlockSplit(config, layer_idx=0, ratio=ratio), GPT2BlockSplit(config_dec, layer_idx=1, ratio=1/ratio)])
        #self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

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

        #print("Causal mask: \n", causal_mask)

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

            hidden_states = outputs[0]
            #print("hidden states in Bottleneck after Attn: ", hidden_states.shape)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[2],)

        # NEW commented
        #hidden_states = self.ln_f(hidden_states)

        #hidden_states = hidden_states.view(output_shape)
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



if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING GPT2 COMPONENTS")
    print("="*80 + "\n")
    
    # Create config
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

    # Note num_attention_heads, ratio must align, e.g.  (hidden_size=768/ratio=6) / 16 heads == head_dim MUST BE INT
    
    # Test parameters
    batch_size = 2
    seq_length = 10
    hidden_size = config_BL.hidden_size
    ratio = 6.0

    config = GPT2Config.from_pretrained("gpt2")
    config._attn_implementation = "eager"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    texts = [
        "Hello world",
        "This is a much longer",
        "Short"
        #"Another one with different length"
    ]

    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    print("Input IDs:")
    print(inputs["input_ids"])
    print("\nAttention mask:")
    print(inputs["attention_mask"])

    batch_size, seq_len = inputs["input_ids"].shape
    hidden_size = config.hidden_size

    # Fake hidden states from layer N (e.g. after layer 5)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size) * 0.1
    print(f"\nFake hidden_states shape: {hidden_states.shape}")


    print(f"Test Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num heads: {config_BL.num_attention_heads}")
    print(f"  Head dim: {hidden_size // config_BL.num_attention_heads}")
    
    # Create dummy input
    #hidden_states = torch.randn(batch_size, seq_length, hidden_size)
    #print(f"\nInput hidden_states shape: {hidden_states.shape}")
    
    # Test GPT2Block
    print("\n" + "="*80)
    print("TESTING BottleneckAttention")
    print("="*80)
    
    bl = BottleneckAttention(config_BL, ratio=ratio)


    #block = GPT2BlockSplit(config, layer_idx=0, ratio=ratio)
    
    # Set to eval mode to disable dropout randomness
    bl.eval()
    
    with torch.no_grad():
        outputs = bl(
            inputs_embeds=hidden_states,
            attention_mask=inputs["attention_mask"],
            #output_attentions=True
        )
    
    output_hidden_states = outputs[0]
    print("Shape out Bottleneck: ", output_hidden_states.shape)
    assert output_hidden_states.shape == hidden_states.shape, "Are you fucking stupid?"

    print("\n" + "="*80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
