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

def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)

    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

    if attention_mask is not None:
        # Apply the attention mask
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, svd_r=1.0):
        super().__init__()
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

        self.svd_r = svd_r

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)


        # ADAPT FOR BETTER SVD PROJECTION
        self.WQ = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.WK = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.WV = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
                
        self.WQ_A = nn.Linear(self.embed_dim, int(self.embed_dim*svd_r), bias=False)
        self.WQ_B = nn.Linear(int(self.embed_dim*svd_r), self.embed_dim, bias=True)
        self.WK_A = nn.Linear(self.embed_dim, int(self.embed_dim*svd_r), bias=False)
        self.WK_B = nn.Linear(int(self.embed_dim*svd_r), self.embed_dim, bias=True)
        self.WV_A = nn.Linear(self.embed_dim, int(self.embed_dim*svd_r), bias=False)
        self.WV_B = nn.Linear(int(self.embed_dim*svd_r), self.embed_dim, bias=True)
        #print("WV A: ", self.WV_A)
        #print("WV B: ", self.WV_B)
        #print("WK A: ", self.WK_A)
        #print("WK B: ", self.WK_B)
        #print("WQ: ", self.WQ)
        
        
        

        ###############################################

        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = not is_cross_attention

    def translate_SVD(self):
        """
        Convert the full WK / WV matrices into SVD-factorized
        2-layer decompositions (A and B) with optional low-rank r.
        Skip Q matrix.
        
        If rank=None → use full rank (embed_dim).
        If rank=r → use truncated SVD of rank r*embed_dim.
        """
        # Determine maximum possible rank
        
        r = int(self.svd_r * self.embed_dim)
        print(f"[translate_SVD] Using rank = {r} (max = {self.embed_dim})")
        
        # Helper: factorize a (embed_dim × embed_dim) projection matrix W
        def factorize(W, b):
            # W shape: (embed_dim, embed_dim)
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # Truncate to rank r
            U = U[:, :r]          # (embed_dim, r)
            S = S[:r]             # (r,)
            Vh = Vh[:r, :]        # (r, embed_dim)
            # Fuse sqrt(S) into U and V
            sqrtS = torch.sqrt(S)
            U_fused = U * sqrtS.unsqueeze(0)      # (embed_dim, r)
            V_fused = (Vh.T * sqrtS.unsqueeze(0)) # (embed_dim, r)
            return U_fused, V_fused, b
        
        # -------------------------------------------------------
        # K projection
        # -------------------------------------------------------
        W = self.WK.weight.data.clone()
        b = self.WK.bias.data.clone()
        U_fused, V_fused, b = factorize(W, b)
        self.WK_A.weight.data = V_fused.T.clone()
        self.WK_B.weight.data = U_fused.clone()
        self.WK_B.bias.data   = b.clone()
        
        # -------------------------------------------------------
        # V projection
        # -------------------------------------------------------
        W = self.WV.weight.data.clone()
        b = self.WV.bias.data.clone()
        U_fused, V_fused, b = factorize(W, b)
        self.WV_A.weight.data = V_fused.T.clone()
        self.WV_B.weight.data = U_fused.clone()
        self.WV_B.bias.data   = b.clone()
        
        print("[translate_SVD] SVD factorization + low-rank projection complete.")



    def translate_weights(self):
        """
        Translate weights and biases from self.c_attn to the full WQ, WK, WV linear layers (not separated by heads).
        Call this after loading the pretrained weights into self.c_attn.
        """
        if not hasattr(self, 'c_attn') or self.c_attn is None:
            raise ValueError("self.c_attn must be loaded with weights first.")
        print("Translate weights to full Q/K/V projection layers")

        weight = self.c_attn.weight.data  # shape: (embed_dim, 3 * embed_dim)
        bias = self.c_attn.bias.data  # shape: (3 * embed_dim)

        q_weight = weight[:, 0:self.embed_dim]  # (embed_dim, embed_dim) -> (in, out)
        k_weight = weight[:, self.embed_dim:2 * self.embed_dim]  # (embed_dim, embed_dim)
        v_weight = weight[:, 2 * self.embed_dim:3 * self.embed_dim]  # (embed_dim, embed_dim)

        q_bias = bias[0:self.embed_dim]  # (embed_dim,)
        k_bias = bias[self.embed_dim:2 * self.embed_dim]  # (embed_dim,)
        v_bias = bias[2 * self.embed_dim:3 * self.embed_dim]  # (embed_dim,)

        # Set weights for WQ, WK, WV
        # nn.Linear weight is (out, in), so transpose the sliced weights
        self.WQ.weight.data = q_weight.T  # (embed_dim, embed_dim)
        self.WQ.bias.data = q_bias

        self.WK.weight.data = k_weight.T  # (embed_dim, embed_dim)
        self.WK.bias.data = k_bias

        self.WV.weight.data = v_weight.T  # (embed_dim, embed_dim)
        self.WV.bias.data = v_bias

        # Optionally, delete the original c_attn to save memory
        # del self.c_attn

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None):
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

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with maybe_autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

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
        is_cross_attention = encoder_hidden_states is not None
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

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            #query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            query_states = self.WQ(hidden_states)
            key_states = self.WK_B(self.WK_A(hidden_states))
            value_states = self.WV_B(self.WV_A(hidden_states))



            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)
            # Full weights
            #key_states = torch.stack([self.WK_H[i](hidden_states) for i in range(self.num_heads)], dim=1)
            #value_states = torch.stack([self.WV_H[i](hidden_states) for i in range(self.num_heads)], dim=1)
            #key_states = torch.stack(
            #    [self.WK_B[i](self.WK_A[i](hidden_states)) for i in range(self.num_heads)],
            #    dim=1
            #)

            # VALUE
            #value_states = torch.stack(
            #    [self.WV_B[i](self.WV_A[i](hidden_states)) for i in range(self.num_heads)],
            #    dim=1
            #)
            


        #query_states = query_states.view(shape_q).transpose(1, 2)
        # Full weights
        #query_states = torch.stack([self.WQ_H[i](hidden_states) for i in range(self.num_heads)], dim=1)
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)
        print("Query shapes: ", query_states.shape)
        print("Key shapes: ", key_states.shape)
        print("Value shapes: ", value_states.shape)
        
        #query_states = torch.stack(
        #    [self.WQ_B[i](self.WQ_A[i](hidden_states)) for i in range(self.num_heads)],
        #    dim=1
        #)


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
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
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
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None, svd_r=1.0):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx, svd_r=svd_r)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

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
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
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
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_output, cross_attn_weights = self.crossattention(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # residual connection
            hidden_states = residual + cross_attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
            if encoder_hidden_states is not None:
                outputs += (cross_attn_weights,)

        return outputs

class GPT2ModelCompress(GPT2PreTrainedModel):
    _supports_param_buffer_assignment = False

    def __init__(self, config, bl_layer=None, bl_ratio=2, BL_type="linear", svd_r=1.0):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i, svd_r=svd_r) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

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


class GPT2LMHeadModelSVD(GPT2PreTrainedModel, GenerationMixin):
    """
    GPT-2 Language Model with compression bottleneck.
    Maintains compatibility with HuggingFace generation methods.
    """
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, bl_layer=None, svd_r=1.00):
        super().__init__(config)
        #print(f"Initializing GPT2LMHeadModelSVD (bl_layer={bl_layer}, bl_ratio={bl_ratio})")
        
        self.transformer = GPT2ModelCompress(config, svd_r=svd_r)
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


    print("FULL COMPARISON: Official GPT2LMHeadModel vs GPT2LMHeadModelSVD\n")

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
    my_model = GPT2LMHeadModelSVD(config)

    # Copy weights from official model's transformer
    my_model.transformer.load_state_dict(official_model.transformer.state_dict(), strict=False)  # strict=False to enable Bottleneck layer loading
    # TIE lm_head to wte (exactly like official)
    my_model.lm_head.weight = my_model.transformer.wte.weight

    my_model.eval()
    my_model.to(device)

    print("Both models loaded and weights tied\n")

    # =============================================================
    # 2.1. translate model
    # =============================================================

    for i in range(12):

        my_model.transformer.h[i].attn.translate_weights()
        my_model.transformer.h[i].attn.translate_SVD(rank=1.00)

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
    prompt = "The future of evolutionary computation is"
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
