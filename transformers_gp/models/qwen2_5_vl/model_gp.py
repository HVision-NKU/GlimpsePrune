# This is the core implementation of GlimpsePrune.

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import math
import copy

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import StaticCache, DynamicCache, Cache
from transformers.utils import logging
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VisionTransformerPretrainedModel,    
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLModel,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLDecoderLayer,
    Qwen2_5_VisionRotaryEmbedding,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2RMSNorm,
    Qwen2MLP,
    Qwen2_5_VLFlashAttention2,
    Qwen2_5_VLSdpaAttention,
    Qwen2_5_VLAttention,
    repeat_kv,
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)
from transformers.activations import ACT2FN

from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward
)
    
from .configuration import *
from transformers.modeling_outputs import ModelOutput   
from warppers import debug_calls, time_logger
from utils import download_model_from_hf

logger = logging.get_logger(__name__)


def print_rank0(message):
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(message)


@debug_calls()
def debug_print(*args, **kwargs):
    print(*args, **kwargs)


def convert_2d_to_4d_mask(mask_2d: torch.Tensor, 
                          query_seq_len: int, 
                          dtype: torch.dtype = torch.float32) -> torch.Tensor:
    batch_size, key_seq_len = mask_2d.shape
    mask_4d = mask_2d.unsqueeze(1).unsqueeze(2)
    mask_4d = mask_4d.expand(batch_size, 1, query_seq_len, key_seq_len)
    mask_4d = mask_4d.to(dtype)
    inverted_mask = 1 - mask_4d
    masked_value = -torch.inf
    inverted_mask = inverted_mask.masked_fill(inverted_mask == 1, masked_value)
    return inverted_mask
    

# ---------- Fusers ----------

class BaseAttnFuser(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        raise NotImplementedError("Subclasses should implement this method.")


ATTN_FUSER_REGISTRY = {}

def register_attn_fuser():
    def decorator(cls):
        name = cls.__name__
        if name in ATTN_FUSER_REGISTRY:
            raise ValueError(f"AttnFuser {name} already registered.")
        if not issubclass(cls, BaseAttnFuser):
            raise ValueError(f"AttnFuser {name} must be a subclass of BaseAttnFuser.")
        ATTN_FUSER_REGISTRY[name] = cls
        return cls
    return decorator


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, bias):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = ACT2FN[hidden_act]
        
    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class CondSdpaAttention(nn.Module):
    def __init__(self, hidden_size, cond_size, num_heads):
        super().__init__()
        qk_size = hidden_size + cond_size
        v_size = hidden_size
        self.q_proj = nn.Linear(qk_size, qk_size, bias=False)
        self.k_proj = nn.Linear(qk_size, qk_size, bias=False)
        self.v_proj = nn.Linear(v_size, v_size, bias=False)
        self.o_proj = nn.Linear(v_size, v_size, bias=False)
        self.num_heads = num_heads

    
    def forward(self, hidden_states, cond_states, cu_seqlens, position_embeddings):
        seq_length = hidden_states.shape[0]
        if cond_states is not None:
            qk = torch.cat([hidden_states, cond_states], dim=-1)
        else:
            qk = hidden_states
        q = self.q_proj(qk).reshape(seq_length, self.num_heads, -1)
        k = self.k_proj(qk).reshape(seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
        )
        
        attn_output = attn_output.squeeze(0).transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output
    

class AttnFuserLayer(nn.Module):
    def __init__(self, hidden_size, cond_size, num_heads, hidden_act):
        super().__init__()
        self.norm1 = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.attn = CondSdpaAttention(hidden_size, cond_size, num_heads)
        self.mlp = MLP(hidden_size, hidden_size * 2, hidden_act, bias=True)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cond_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
        

@register_attn_fuser()
class AttnFuserDummy(BaseAttnFuser):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        attn_outs = []
        attn_map_mean = attn_map.mean(dim=-1)
        st = 0
        ori_attns = []
        for b, (h, w) in enumerate(attn_grid_hw):
            ed = st + h * w
            one_attn_map = attn_map_mean[st:ed]
            st = ed
            if self.config.use_attention_logits:
                one_attn_map = torch.softmax(one_attn_map, dim=-1)
            else:
                one_attn_map = torch.exp(one_attn_map)
            one_attn_map_min = one_attn_map.min()
            one_attn_map_max = one_attn_map.max()
            one_attn_map = (one_attn_map - one_attn_map_min) / (one_attn_map_max - one_attn_map_min + 1e-6)
            ori_attns.append(one_attn_map)
        ori_attns = torch.cat(ori_attns, dim=0)
        attn_outs.append(ori_attns)
        attn_outs = torch.stack(attn_outs, dim=0)
        return attn_outs
    

@register_attn_fuser()
class AttnFuserV1(BaseAttnFuser):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        attn_fuse_size = config.attn_fuse_size
        num_layers = len(config.selected_visual_layers)
        visual_cond_size = config.visual_cond_size if num_layers > 0 else 0
        num_attn_layers = len(config.selected_layers)
        num_attn_heads = config.num_attention_heads
        self.attn_in_proj = nn.Linear(num_attn_layers * num_attn_heads, attn_fuse_size)
        self.cond_in_projs = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.attn_out_projs = nn.ModuleList()
        for i in range(num_layers):
            self.cond_in_projs.append(nn.Linear(config.vision_config.hidden_size, visual_cond_size))
            self.layers.append(AttnFuserLayer(
                attn_fuse_size, visual_cond_size, config.attn_fuse_num_heads, config.attn_fuse_hidden_act
            ))
            if not self.config.deep_supervision and i < num_layers - 1:
                self.attn_out_projs.append(nn.Identity())
            else:
                self.attn_out_projs.append(nn.Linear(attn_fuse_size, 1))
        head_dim = (attn_fuse_size + visual_cond_size) // config.attn_fuse_num_heads
        assert (attn_fuse_size + visual_cond_size) % config.attn_fuse_num_heads == 0, f"attn_fuse_size {attn_fuse_size} + visual_cond_size {visual_cond_size} must be divisible by num_heads {config.attn_fuse_num_heads}"
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

    def rot_pos_emb(self, grid_hw):
        pos_ids = []
        for h, w in grid_hw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)  # [h, w]
            hpos_ids = hpos_ids.flatten()  # [h * w]
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)  # [h, w]
            wpos_ids = wpos_ids.flatten()  # [h * w]
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))  # [h * w, 2]
        pos_ids = torch.cat(pos_ids, dim=0)  # [N, 2]
        max_grid_size = grid_hw.max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb
        
    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        attn_outs = []
        if self.config.ori_attn_supervision and not self.training:
            attn_map_mean = attn_map.mean(dim=-1)
            st = 0
            ori_attns = []
            for b, (h, w) in enumerate(attn_grid_hw):
                ed = st + h * w
                one_attn_map = attn_map_mean[st:ed]
                st = ed
                if self.config.use_attention_logits:
                    one_attn_map = torch.softmax(one_attn_map, dim=-1)
                else:
                    one_attn_map = torch.exp(one_attn_map)
                one_attn_map_min = one_attn_map.min()
                one_attn_map_max = one_attn_map.max()
                one_attn_map = (one_attn_map - one_attn_map_min) / (one_attn_map_max - one_attn_map_min + 1e-6)
                ori_attns.append(one_attn_map)
            ori_attns = torch.cat(ori_attns, dim=0)
            attn_outs.append(ori_attns)
            
        attn_hiddens = self.attn_in_proj(attn_map)
        attn_hiddens = attn_hiddens[window_index, :]
        cond_hiddens = [img_embeds[window_index, :] for img_embeds in selected_image_embeds]
        rotary_pos_emb = self.rot_pos_emb(attn_grid_hw)
        rotary_pos_emb = rotary_pos_emb[window_index, :]
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        reverse_indices = torch.argsort(window_index)
        
        if self.config.attn_fuse_global:
            cu_current_seqlens = cu_seqlens // (self.config.vision_config.spatial_merge_size**2)
        else:
            cu_current_seqlens = cu_window_seqlens // (self.config.vision_config.spatial_merge_size**2)
        for layer_num, layer in enumerate(self.layers):
            curr_cond_hiddens = self.cond_in_projs[layer_num](cond_hiddens[layer_num])
            attn_hiddens = layer(attn_hiddens, curr_cond_hiddens, cu_current_seqlens, position_embeddings)
            if self.training or layer_num == len(self.layers) - 1:
                attn_out_proj = self.attn_out_projs[layer_num]
                if isinstance(attn_out_proj, nn.Identity):
                    continue  # not deep supervision
                aux_attn_out = attn_out_proj(attn_hiddens).squeeze(-1)
                aux_attn_out = aux_attn_out[reverse_indices]
                attn_outs.append(aux_attn_out)

        attn_outs = torch.stack(attn_outs, dim=0)  # [num_layers, all_seq_len]
        return attn_outs
        

@register_attn_fuser()
class AttnFuserV2(AttnFuserV1):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        attn_fuse_size = config.attn_fuse_size
        num_layers = len(config.selected_visual_layers)
        visual_cond_size = 0
        num_attn_layers = len(config.selected_layers)
        num_attn_heads = config.num_attention_heads
        self.attn_in_proj = nn.Linear(num_attn_layers * num_attn_heads, attn_fuse_size)
        self.layers = nn.ModuleList()
        self.attn_out_projs = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(AttnFuserLayer(
                attn_fuse_size, visual_cond_size, config.attn_fuse_num_heads, config.attn_fuse_hidden_act
            ))
            if not self.config.deep_supervision and i < num_layers - 1:
                self.attn_out_projs.append(nn.Identity())
            else:
                self.attn_out_projs.append(nn.Linear(attn_fuse_size, 1))
        head_dim = (attn_fuse_size + visual_cond_size) // config.attn_fuse_num_heads
        assert (attn_fuse_size + visual_cond_size) % config.attn_fuse_num_heads == 0, f"attn_fuse_size {attn_fuse_size} + visual_cond_size {visual_cond_size} must be divisible by num_heads {config.attn_fuse_num_heads}"
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)


    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        attn_outs = []
        if self.config.ori_attn_supervision and not self.training:
            attn_map_mean = attn_map.mean(dim=-1)
            st = 0
            ori_attns = []
            for b, (h, w) in enumerate(attn_grid_hw):
                ed = st + h * w
                one_attn_map = attn_map_mean[st:ed]
                st = ed
                if self.config.use_attention_logits:
                    one_attn_map = torch.softmax(one_attn_map, dim=-1)
                else:
                    one_attn_map = torch.exp(one_attn_map)
                one_attn_map_min = one_attn_map.min()
                one_attn_map_max = one_attn_map.max()
                one_attn_map = (one_attn_map - one_attn_map_min) / (one_attn_map_max - one_attn_map_min + 1e-6)
                ori_attns.append(one_attn_map)
            ori_attns = torch.cat(ori_attns, dim=0)
            attn_outs.append(ori_attns)
            
        attn_hiddens = self.attn_in_proj(attn_map)
        attn_hiddens = attn_hiddens[window_index, :]
        rotary_pos_emb = self.rot_pos_emb(attn_grid_hw)
        rotary_pos_emb = rotary_pos_emb[window_index, :]
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        reverse_indices = torch.argsort(window_index)
        
        if self.config.attn_fuse_global:
            cu_current_seqlens = cu_seqlens // (self.config.vision_config.spatial_merge_size**2)
        else:
            cu_current_seqlens = cu_window_seqlens // (self.config.vision_config.spatial_merge_size**2)
        for layer_num, layer in enumerate(self.layers):
            attn_hiddens = layer(attn_hiddens, None, cu_current_seqlens, position_embeddings)
            if self.training or layer_num == len(self.layers) - 1:
                attn_out_proj = self.attn_out_projs[layer_num]
                if isinstance(attn_out_proj, nn.Identity):
                    continue  # not deep supervision
                aux_attn_out = attn_out_proj(attn_hiddens).squeeze(-1)
                aux_attn_out = aux_attn_out[reverse_indices]
                attn_outs.append(aux_attn_out)

        attn_outs = torch.stack(attn_outs, dim=0)  # [num_layers, all_seq_len]
        return attn_outs
        

# ---------- VLM ----------


@dataclass
class Qwen2_5_VL_GP_CausalLMOutputWithPast(ModelOutput):
    logits: torch.FloatTensor = None
    le_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    input_ids: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    position_ids: Optional[torch.LongTensor] = None
    attn_grid: Optional[torch.LongTensor] = None
    image_token_mask_logits: Optional[torch.Tensor] = None
    image_token_bool_masks: Optional[torch.Tensor] = None


class Qwen2_5_VLAttention_GP(Qwen2_5_VLAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        q_indices=None,
        kv_mask=None,
        use_attention_logits=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        if output_attentions:
            attn_weights_out = attn_weights.clone()  # [bsz, nheads, q_len, k_len]
            if q_indices is not None:
                attn_weights_out = attn_weights_out[list(range(bsz)), :, q_indices, :].unsqueeze(2)  # [bsz, nheads, 1, k_len]
            if not use_attention_logits:
                attn_weights_out = nn.functional.log_softmax(attn_weights, dim=-1)
            if kv_mask is not None:
                attn_weights_out = attn_weights_out.squeeze(2)
                attn_weights_out = attn_weights_out.transpose(1, 2)
                selected_attn_weights = attn_weights_out[kv_mask]  # [N, nheads]
                kv_length = kv_mask.sum(dim=-1)
                attn_weights_out = selected_attn_weights.split(kv_length.tolist(), dim=0)  # List(bsz) of [k_select_len, nheads]
        else:
            attn_weights_out = None
        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights_out, past_key_value

class Qwen2_5_VLSdpaAttention_GP(Qwen2_5_VLSdpaAttention):
    
    def _cal_attn_weights(self,
                          query_states: torch.Tensor,
                          key_states: torch.Tensor,
                          attention_mask: Optional[torch.Tensor]=None,
                          q_indices: Optional[List[int]]=None,
                          kv_mask: Optional[torch.Tensor]=None,
                          use_attention_logits: bool = False
                          ):
        bsz, nheads, _, head_dim = query_states.size()
        selected_query_states = query_states[list(range(bsz)), :, q_indices, :].view(bsz, nheads, 1, head_dim)
        attn_weights = torch.matmul(selected_query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [bsz, nheads, 1, k_len]
        if not use_attention_logits:
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    min_dtype = torch.finfo(attn_weights.dtype).min
                    attention_mask_float = torch.full_like(attention_mask, min_dtype, dtype=attn_weights.dtype)
                    attention_mask_float.masked_fill_(attention_mask, 0.0)
                else:
                    attention_mask_float = attention_mask
                attn_weights += attention_mask_float
            attn_weights = torch.log_softmax(attn_weights, dim=-1)  # [bsz, nheads, 1, k_len]
        if kv_mask is not None:
            attn_weights = attn_weights.squeeze(2)
            attn_weights = attn_weights.transpose(1, 2)
            selected_attn_weights = attn_weights[kv_mask]  # [N, nheads]
            kv_length = kv_mask.sum(dim=-1)
            attn_weights = selected_attn_weights.split(kv_length.tolist(), dim=0)  # List(bsz) of [k_select_len, nheads]
        return attn_weights
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        q_indices=None,
        kv_mask=None,
        use_attention_logits=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if output_attentions:
            attn_weights = self._cal_attn_weights(query_states, key_states, attention_mask, 
                                                  q_indices=q_indices, kv_mask=kv_mask, use_attention_logits=use_attention_logits)
        else:
            attn_weights = None
        
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class Qwen2_5_VLFlashAttention2_GP(Qwen2_5_VLFlashAttention2):
    # @time_logger
    def _cal_attn_weights(self,
                          query_states: torch.Tensor,
                          key_states: torch.Tensor,
                          attention_mask: Optional[torch.Tensor]=None,
                          q_indices: Optional[List[int]]=None,
                          kv_mask: Optional[torch.Tensor]=None,
                          use_attention_logits: bool = False
                          ):
        # selected_query_states = query_states[:, :, -1:, :]
        bsz, nheads, _, head_dim = query_states.size()
        selected_query_states = query_states[list(range(bsz)), :, q_indices, :].view(bsz, nheads, 1, head_dim)
        attn_weights = torch.matmul(selected_query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [bsz, nheads, 1, k_len]
        if not use_attention_logits:
            if attention_mask is not None:
                attention_mask_4d = convert_2d_to_4d_mask(attention_mask, 1, dtype=attn_weights.dtype)
                attn_weights = attn_weights + attention_mask_4d  # [bsz, nheads, 1, k_len]
            attn_weights = torch.log_softmax(attn_weights, dim=-1)  # [bsz, nheads, 1, k_len]
        if kv_mask is not None:
            attn_weights = attn_weights.squeeze(2)
            attn_weights = attn_weights.transpose(1, 2)  # [bsz, k_len, nheads]
            selected_attn_weights = attn_weights[kv_mask]  # [N, nheads]
            kv_length = kv_mask.sum(dim=-1)
            attn_weights = selected_attn_weights.split(kv_length.tolist(), dim=0)  # List(bsz) of [k_select_len, nheads]
        return attn_weights
    
    def forward(self, 
                hidden_states, 
                attention_mask = None, 
                position_ids = None, 
                past_key_value = None, 
                output_attentions = False, 
                use_cache = False, 
                cache_position = None, 
                position_embeddings = None,
                q_indices = None,
                kv_mask = None,
                use_attention_logits=False):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # [bsz, n_heads, q_len, head_dim]
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # [bsz, n_heads_kv, q_len, head_dim]
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)  # [bsz, n_heads_kv, q_len, head_dim]

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout
        
        if output_attentions:
            attn_weights = self._cal_attn_weights(query_states, key_states, attention_mask, 
                                                  q_indices=q_indices, kv_mask=kv_mask, use_attention_logits=use_attention_logits)
        else:
            attn_weights = None
        
        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None
            
        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
            
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)


        return attn_output, attn_weights, past_key_value


QWEN2_5_VL_GP_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLAttention_GP,
    "sdpa": Qwen2_5_VLSdpaAttention_GP,
    "flash_attention_2": Qwen2_5_VLFlashAttention2_GP,
}


class Qwen2_5_VLDecoderLayer_GP(Qwen2_5_VLDecoderLayer):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        nn.Module.__init__(self)
        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_5_VL_GP_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        

    def forward(self, 
                hidden_states, 
                attention_mask = None, 
                position_ids = None, 
                past_key_value = None, 
                output_attentions = False, 
                use_cache = False, 
                cache_position = None, 
                position_embeddings = None,
                q_indices = None,
                kv_mask = None,
                use_attention_logits=False,
                **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            q_indices=q_indices,
            kv_mask=kv_mask,
            use_attention_logits=use_attention_logits
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    


class Qwen2_5_VLModel_GP(Qwen2_5_VLModel):
    def __init__(self, config: Qwen2_5_VLConfig):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer_GP(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class Qwen2_5_VL_GP_ForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    # fix inference (for better speed?)
    
    config_class = Qwen2_5_VL_GPConfig
    def __init__(self, config):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.model = Qwen2_5_VLModel_GP(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here
        self.reset_image_tokens_cache()
        self._init_new_modules(config)
        self.post_init()
        
    def _init_new_modules(self, config, re_init=False):
        self.config = config
        # NOTE: config of other sub-modules is not overwrited !
        

        selected_layers = self.config.selected_layers
        reduce_layer = self.config.reduce_layer  # do reduce after this layer
        le_layers = self.config.le_layers
        
        max_selected_layers = max(selected_layers) if len(selected_layers) > 0 else 0
        max_le_layers = max(le_layers) if len(le_layers) > 0 else 0
        
        assert max_le_layers < len(self.model.layers), f"max_le_layers {le_layers} must be less than the number of layers {len(self.model.layers)}"
        assert max_selected_layers <= reduce_layer, f"selected_layers {selected_layers} must be less than or equal to reduce_layer {reduce_layer}"
        
        
        if re_init:
            dtype = next(self.model.embed_tokens.parameters()).dtype
            new_module_to_device = {}
            for name, module in self.new_modules_to_be_loaded().items():
                if isinstance(module, nn.Parameter):
                    new_module_to_device[name] = module.device
                else:
                    new_module_to_device[name] = next(module.parameters()).device
            for name, module in self.new_modules_to_be_saved().items():
                if isinstance(module, nn.Parameter):
                    new_module_to_device[name] = module.device
                else:
                    new_module_to_device[name] = next(module.parameters()).device
        try:
            self.attn_fuser = ATTN_FUSER_REGISTRY[config.attn_fuse_type](config)
        except KeyError:
            raise ValueError(f"AttnFuser {config.attn_fuse_type} not found in registry. Available options: {list(ATTN_FUSER_REGISTRY.keys())}")
        
        if len(self.config.le_layers) > 0 and self.config.le_length > 0:
            self.register_parameter(
                "learnable_embeddings",
                nn.Parameter(torch.empty(len(self.config.le_layers), self.config.le_length, self.config.hidden_size))
            )
            self.le_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            if self.config.le_norm_type == "rmsnorm":
                self.le_norm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
            elif self.config.le_norm_type == "layernorm":
                self.le_norm = nn.LayerNorm(self.config.hidden_size)
            else:
                raise ValueError(f"Unsupported le_norm_type: {self.config.le_norm_type}. Supported types: 'rmsnorm', 'layernorm'.")
            self.le_dropout = nn.Dropout(self.config.le_dropout_prob)
        else:
            if hasattr(self, "learnable_embeddings"):
                del self.learnable_embeddings
            if hasattr(self, "le_proj"):
                del self.le_proj
            if hasattr(self, "le_norm"):
                del self.le_norm
            if hasattr(self, "le_dropout"):
                del self.le_dropout
        
        # convert all new modules to the same device as attn_head_weights
        if re_init:
            for name, module in self.new_modules_to_be_loaded().items():
                device = new_module_to_device[name]
                if isinstance(module, nn.Parameter):
                    module.data = module.data.to(device=device, dtype=dtype)
                else:
                    module.to(device=device, dtype=dtype)
            for name, module in self.new_modules_to_be_saved().items():
                device = new_module_to_device[name]
                if isinstance(module, nn.Parameter):
                    module.data = module.data.to(device=device, dtype=dtype)
                else:
                    module.to(device=device, dtype=dtype)
        
    
    def peft_target_modules(self):
        target_modules = {}
        for name, module in self.model.named_modules():
            actual_name = "model." + name
            if "embed_tokens" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                target_modules[actual_name] = module
        return target_modules
        
        
    def new_modules_to_be_loaded(self):
        return {}
    
    
    def new_modules_to_be_saved(self):
        rtn = {
            "attn_fuser": self.attn_fuser,
        }
        if hasattr(self, "learnable_embeddings"):
            rtn.update({
                "learnable_embeddings": self.learnable_embeddings,
                "le_proj": self.le_proj,
                "le_norm": self.le_norm,
            })
        return rtn            
            
    def _init_weights(self, module):
        super()._init_weights(module)
        # init new modules and parameters
        if hasattr(self, "learnable_embeddings"):
            nn.init.normal_(self.learnable_embeddings, 0, 0.02)

        for name, module in self.new_modules_to_be_saved().items():
            if isinstance(module, nn.Module):
                if hasattr(module, "_init_weights"):
                    module._init_weights()
                else:
                    for p_name, param in module.named_parameters():
                        if 'norm' in name or 'norm' in p_name:
                            if 'weight' in p_name:
                                nn.init.ones_(param)
                            elif 'bias' in p_name:
                                nn.init.zeros_(param)
                        elif 'proj' in name or 'proj' in p_name:
                            if 'weight' in p_name:
                                nn.init.xavier_uniform_(param)
                            elif 'bias' in p_name:
                                nn.init.zeros_(param)
                        
    
    def save_new_modules(self, save_directory):
        #save config
        self.config.save_pretrained(save_directory)
        
        # save the new modules
        is_main_process = int(os.getenv("LOCAL_RANK", "0")) == 0
        if is_main_process:
            new_states = {}
            for name, module in self.new_modules_to_be_saved().items():
                if isinstance(module, nn.Parameter):
                    new_states[name] = module.data
                else:
                    new_states[name] = module.state_dict()
            for name, module in self.new_modules_to_be_loaded().items():
                if isinstance(module, nn.Parameter):
                    new_states[name] = module.data
                else:
                    new_states[name] = module.state_dict()
            torch.save(new_states, os.path.join(save_directory, "new_modules_gp.pt"))
            print(f"new_modules of {self.__class__.__name__} saved to {os.path.join(save_directory, 'new_modules_gp.pt')}")
    
    
    def load_new_modules(self, load_directory):
        if not os.path.isdir(load_directory):
            load_directory = download_model_from_hf(load_directory)
        # load config
        config_path = os.path.join(load_directory, "config.json")
        config = self.config_class.from_json_file(config_path)
        self._init_new_modules(config, re_init=True)
        # load 
        new_modules_path = os.path.join(load_directory, "new_modules_gp.pt")
        if os.path.exists(new_modules_path):
            new_modules = torch.load(new_modules_path, weights_only=True)
            new_modules_in_ckpt = set(new_modules.keys())
            for name, module in self.new_modules_to_be_loaded().items():
                try:
                    if isinstance(module, nn.Parameter):
                        module.data.copy_(new_modules[name])
                    else:
                        module.load_state_dict(new_modules[name], strict=True)
                except Exception as e:
                    print(f"Failed to load new modules {name}: {e}")
                    raise e
                new_modules_in_ckpt.discard(name)
                print_rank0(f"new_modules of {self.__class__.__name__}.{name} loaded from {new_modules_path}")
            for name in new_modules_in_ckpt:
                module = getattr(self, name)
                try:
                    if isinstance(module, nn.Parameter):
                        module.data.copy_(new_modules[name])
                    else:
                        module.load_state_dict(new_modules[name], strict=True)
                except Exception as e:
                    print(f"Failed to load new modules {name}: {e}")
                    raise e
                print_rank0(f"new_modules of {self.__class__.__name__}.{name} loaded from {new_modules_path}")
        else:
            warnings.warn(f"new_modules_gp.pt not found in {load_directory}.")
            

    def reset_image_tokens_cache(self):
        self.todo_selection = False
        self.glimpse_return_before_selection = None
        self.reduced_input_ids = None

    
    def _check_padding_side(self, attention_mask: torch.LongTensor, default_side='right') -> str:
        """
        Determines the padding side based on the attention mask.

        Args:
            attention_mask (torch.LongTensor): Attention mask [B, L]. 1 for non-padding, 0 for padding.
            default_side (str): Default padding side to return if unclear.
            
        Returns:
            str: 'right' or 'left'.
        """
        B, L = attention_mask.shape
        if L == 0:
            return default_side

        # Check if any sequence has actual content
        has_content = attention_mask.sum(dim=1) > 0

        # If no sequence has content, padding side doesn't matter much
        if not torch.any(has_content):
            return default_side

        # Check if all sequences *with content* start with 1 (suggests right padding)
        starts_with_one = attention_mask[has_content, 0].all()
        # Check if all sequences *with content* end with 1 (suggests left padding)
        ends_with_one = attention_mask[has_content, -1].all()

        if not starts_with_one and ends_with_one:
            return 'left'
        elif starts_with_one and not ends_with_one:
            raise NotImplementedError(f"Unsupported padding side: right")
            return 'right'
        elif starts_with_one and ends_with_one:
            # Ambiguous: Could be right padding on sequences shorter than L,
            # or left padding on sequences shorter than L, or no padding needed.
            # Defaulting to 'right' is usually safer for truncation.
            # Check if any padding exists at all
            if torch.any(attention_mask[has_content] == 0):
                 warnings.warn(
                    "Could not definitively determine padding side from attention_mask; "
                    f"assuming '{default_side}'. If this is incorrect, results may be wrong.",
                    UserWarning
                 )
                 raise NotImplementedError(f"Unsupported padding side: uncontinuous")
            return default_side
        else:
            # Both start and end with 0 for some sequences, unusual padding.
            warnings.warn(
                "Attention mask suggests padding could be on both sides or is inconsistent. "
                f"Assuming '{default_side}' padding. If this is incorrect, results may be wrong.",
                 UserWarning
            )
            raise NotImplementedError(f"Unsupported padding side: both")
            return default_side
        
    def _try_add_le(self, layer_id, hidden_states, q_indices):
        """
        Add learnable embeddings to the hidden states if the layer ID matches
        the specified layer ID for adding learnable embeddings.
        """
        try:
            le_idx = self.config.le_layers.index(layer_id)
        except ValueError:
            return hidden_states
        le = self.learnable_embeddings[le_idx]
        le = le.to(device=self.le_proj.weight.device)
        le_dtype = le.dtype
        le = self.le_dropout(self.le_norm(self.le_proj(le)))
        le = le.to(dtype=le_dtype)
        
 
        bsz, seq_len, hidden_size = hidden_states.shape
        le_len = self.config.le_length
        device = hidden_states.device

        le = le.view(1, le_len, hidden_size).expand(bsz, -1, -1)
        le = le.to(device=device)

        # `q_indices` is a list, convert it to a tensor of shape (bsz, 1)
        end_indices = torch.tensor(q_indices, device=device, dtype=torch.long).unsqueeze(1) + 1
        start_indices = end_indices - le_len
        
        # Create a range of offsets [0, 1, ..., le_len-1]
        le_range = torch.arange(le_len, device=device, dtype=torch.long).unsqueeze(0)
        
        # Calculate the target sequence index for each position in the le window for each batch item.
        # Shape: (bsz, le_len)
        target_seq_indices = start_indices + le_range

        # Create a mask to filter out-of-bounds additions
        # This mask is True only for `le` elements whose target position is within [0, seq_len-1]
        # This gracefully handles cases where q_indices[i] < le_len
        mask = (target_seq_indices >= 0) & (target_seq_indices < seq_len)

        # Select only the valid `le` values and their target indices
        # `le[mask]` flattens the selected values. Shape: (num_valid, hidden_size)
        valid_le_to_add = le[mask]
        
        # We also need the batch index for each valid value
        batch_idx_tensor = torch.arange(bsz, device=device).unsqueeze(1).expand_as(mask)
        valid_batch_indices = batch_idx_tensor[mask] # Shape: (num_valid,)
        valid_target_seq_indices = target_seq_indices[mask] # Shape: (num_valid,)

        # 6. Calculate the flat 1D index for `index_add_`
        # We are adding to a view of hidden_states reshaped to (bsz * seq_len, hidden_size)
        flat_destination_indices = valid_batch_indices * seq_len + valid_target_seq_indices

        # 7. Perform the addition using `index_add_`
        # This is an efficient, in-place operation that adds elements from `valid_le_to_add`
        # to the specified `flat_destination_indices` of the flattened hidden_states.
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        hidden_states_flat.index_add_(0, flat_destination_indices, valid_le_to_add)
        
        # The modification was in-place on a view, so hidden_states is already updated.
        
        # --- End of Modified Section ---
                
        return hidden_states

        
    
    def _append_le(self, input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position):
        bsz, seq_len = input_ids.shape
        le_len = self.config.le_length
        device = inputs_embeds.device
        le_idx = self.config.le_layers.index(0)
        le = self.learnable_embeddings[le_idx]
        le = le.to(device=self.le_proj.weight.device)
        le_dtype = le.dtype
        le = self.le_dropout(self.le_norm(self.le_proj(le)))
        le = le.to(dtype=le_dtype)
        le = le.view(1, le_len, self.config.hidden_size).expand(bsz, -1, -1)
        le = le.to(device=inputs_embeds.device)
        if labels is None:
            inputs_embeds = torch.cat([inputs_embeds, le], dim=1)
            input_ids = torch.cat([input_ids, torch.full((bsz, le_len), self.config.eos_token_id, device=input_ids.device)], dim=1)
        else:
            label_mask = labels != -100
            insert_pos = label_mask.int().argmax(dim=-1)
            
            new_len = seq_len + le_len
            split_idx = insert_pos.unsqueeze(1) # Shape: (bsz, 1)

            # 1. Create the master gather_indices tensor
            # This tensor tells us where to pick elements from a combined (original + le) source
            new_indices = torch.arange(new_len, device=device).unsqueeze(0) # Shape: (1, new_len)
            
            # Indices for part 1 (before insertion)
            indices_part1 = new_indices
            # Indices for the LE part (shifted to the source's LE block)
            indices_le_part = seq_len + (new_indices - split_idx)
            # Indices for part 2 (shifted back to the source's original part)
            indices_part2 = new_indices - le_len

            # Use masks to select which index rule to apply for each position
            mask_part1 = new_indices < split_idx
            mask_le_part = (new_indices >= split_idx) & (new_indices < split_idx + le_len)

            gather_indices = torch.where(mask_part1, indices_part1, torch.where(mask_le_part, indices_le_part, indices_part2))

            # 2. Apply gather to `inputs_embeds`
            source_embeds = torch.cat([inputs_embeds, le], dim=1)
            # Expand gather_indices to match the hidden_size dimension for gathering
            gather_indices_embeds = gather_indices.unsqueeze(2).expand(-1, -1, self.config.hidden_size)
            inputs_embeds = torch.gather(source_embeds, 1, gather_indices_embeds)
            
            # 3. Apply gather to `input_ids`, `labels`, `attention_mask`
            # Create source tensors by concatenating original data with LE placeholder data
            le_ids = torch.full((bsz, le_len), self.config.eos_token_id, device=input_ids.device, dtype=input_ids.dtype)
            source_ids = torch.cat([input_ids, le_ids], dim=1)
            input_ids = torch.gather(source_ids, 1, gather_indices)
            
            le_labels = torch.full((bsz, le_len), -100, device=labels.device, dtype=labels.dtype)
            source_labels = torch.cat([labels, le_labels], dim=1)
            labels = torch.gather(source_labels, 1, gather_indices)

        # It is simpler to deal with attention_mask, position_ids and cache_position
        attention_mask = torch.cat([attention_mask, torch.ones((bsz, le_len), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
        le_pos_ids = []
        for b in range(bsz):
            one_last_pos_idx = position_ids[-1, b, -1].item()
            one_le_pos_ids = torch.arange(one_last_pos_idx + 1, one_last_pos_idx + 1 + le_len, device=position_ids.device)
            one_le_pos_ids = one_le_pos_ids.view(1, 1, -1).expand(3, -1, -1)
            le_pos_ids.append(one_le_pos_ids)
        le_pos_ids = torch.cat(le_pos_ids, dim=1)
        position_ids = torch.cat([position_ids, le_pos_ids], dim=2)
        last_cache_position = cache_position[-1].item()
        cache_position = torch.cat([cache_position, torch.arange(last_cache_position + 1, last_cache_position + 1 + le_len, device=cache_position.device)], dim=-1)
        

        return input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position
        
        
    # @time_logger
    def _decode_image_token_mask_logits(self, batched_attn_map, attn_grid, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        """
        batched_attn_map: list(bsz) of [num_tokens, num_layers, num_heads]
        grid_size: [bsz, 2] (height, width)
        selected_image_embeds: [N, hidden_size]
        """
        num_tokens_by_batch = [attn_map.shape[0] for attn_map in batched_attn_map]
        catted_attn_map = torch.cat(batched_attn_map, dim=0)  # [N, num_layers, num_heads]
        N = catted_attn_map.shape[0]
        catted_attn_map = catted_attn_map.view(N, -1)
        catted_attn_map = self.attn_fuser(catted_attn_map, attn_grid, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens)
        N = catted_attn_map.shape[-1]
        
        selected_masks = catted_attn_map.split(num_tokens_by_batch, dim=-1)  # list(bsz) of [num_layers, num_tokens]
        return selected_masks
     
    @time_logger
    def _glimpse_forward(
        self,
        input_ids,
        inputs_embeds,
        labels,
        position_ids,
        cache_position,
        attention_mask,
        past_key_values,
        use_cache,
        image_info,
        image_grid_thw,
        ref_token_masks,
        return_dict,
        delay_selection,
        use_ref_masks,
    ):
        # Step 1. Check inputs
        # Check padding left
        self._check_padding_side(attention_mask)  
        
        # Other check
        if self.model.gradient_checkpointing and self.model.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()
            
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        
        # Step 2. Append input_ids, inputs_embeds, position_ids, attention_mask, cache_position for learnable embeddings
        if not use_ref_masks and hasattr(self, "learnable_embeddings"):
            input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position = self._append_le(
                input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position)
        
        # Step 3. Prepare other inputs for decoder
        causal_mask = self.model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions=False
        )
        hidden_states = inputs_embeds
        
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)
        
        # q_indices, kv_mask decide the shape of extracted attentions
        bsz, seq_len = input_ids.shape
        if labels is None:
            q_indices = [seq_len - 1] * bsz
        else:
            label_mask = labels != -100
            q_indices = label_mask.int().argmax(dim=-1) - 1
            q_indices = q_indices.tolist()
        kv_mask = input_ids == self.config.image_token_id    
        
        
        selected_layers = tuple(self.config.selected_layers)
        if len(selected_layers) == 0:
            batched_attns = None
        else:
            batched_attns = [[None] * len(selected_layers) for _ in range(bsz)]
        
        # Step 4. Extract attentions during the forward pass of LLM 
        next_decoder_cache = None
        max_forward_layer = max(self.config.selected_layers) if len(self.config.selected_layers) > 0 else 0
        max_forward_layer = max(max_forward_layer, self.config.reduce_layer)
        if labels is not None:
            max_forward_layer = len(self.model.layers) - 1  # always forward all layers if labels are provided
        
        hidden_states_for_reduction = None
        kv_cache_for_reduction = None
        
        for layer_id, decoder_layer in enumerate(self.model.layers):
            if layer_id > 0 and not use_ref_masks and hasattr(self, "learnable_embeddings"):  # le in layer 0 has been added
                hidden_states = self._try_add_le(layer_id, hidden_states, q_indices)
            try:
                layer_pos = self.config.selected_layers.index(layer_id)
            except ValueError:
                layer_pos = None
            if layer_pos is not None and not use_ref_masks:
                _output_attentions = True
            else:
                _output_attentions = False
            
            if self.model.gradient_checkpointing and self.model.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    _output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    q_indices,
                    kv_mask,
                    self.config.use_attention_logits,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=_output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    q_indices=q_indices,
                    kv_mask=kv_mask,
                    use_attention_logits=self.config.use_attention_logits,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[-1]
            if _output_attentions:
                attn_weights = layer_outputs[1]
                for b in range(bsz):
                    batched_attns[b][layer_pos] = attn_weights[b]

            if hidden_states_for_reduction is None:
                if layer_id == self.config.reduce_layer and layer_id < len(self.model.layers) - 1:
                    if layer_id >= max_forward_layer:
                        hidden_states_for_reduction = hidden_states
                        kv_cache_for_reduction = next_decoder_cache if use_cache else None
                    else:
                        hidden_states_for_reduction = hidden_states.clone()
                        if use_cache:
                            kv_cache_for_reduction = DynamicCache()
                            for layer_idx, (key_cache, value_cache) in enumerate(next_decoder_cache):
                                kv_cache_for_reduction.update(key_cache.clone(), value_cache.clone(), layer_idx)
                        else:
                            kv_cache_for_reduction = None
            if layer_id >= max_forward_layer:
                break
        
        if max_forward_layer >= len(self.model.layers) - 1:
            hidden_states = self.model.norm(hidden_states)
        
        if hidden_states_for_reduction is None:
            hidden_states_for_reduction = hidden_states
            kv_cache_for_reduction = next_decoder_cache if use_cache else None
        
        if labels is not None:
            le_logits = self.lm_head(hidden_states).float()
            shift_le_logits = le_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_le_logits = shift_le_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_le_logits.device)
            le_loss = loss_fct(shift_le_logits, shift_labels)
            del le_logits, shift_le_logits, shift_labels
        else:
            le_loss = None
        
        del hidden_states
        del next_decoder_cache
        
        # Step 5. Decode image tokens
        if batched_attns is not None and not use_ref_masks:
            for i, one_attns in enumerate(batched_attns):
                batched_attns[i] = torch.stack(one_attns, dim=1)  # [num_tokens, num_layers, num_heads]
        attn_grid = image_grid_thw[:, 1:] // self.config.vision_config.spatial_merge_size

        if use_ref_masks:
            image_token_mask_logits = []
            for i in range(len(attn_grid)):
                image_token_mask_logits.append(torch.logit(ref_token_masks[i].float().to(device=hidden_states_for_reduction.device).view(1, -1)))
        elif self.config.use_zero_masks:
            image_token_mask_logits = []
            for i in range(len(attn_grid)):
                image_token_mask_logits.append(torch.logit(torch.zeros((1, attn_grid[i][0] * attn_grid[i][1]), device=hidden_states_for_reduction.device)))
        else:
            image_token_mask_logits = self._decode_image_token_mask_logits(batched_attns, attn_grid, **image_info)
                
        # Step 6. Trim (reducted image tokens and le tokens) hidden_states, next_cache, input_ids, attention_mask
        if not use_ref_masks and hasattr(self, "learnable_embeddings"):
            le_length = self.config.le_length
            input_ids = input_ids[:, :-le_length]
            inputs_embeds = inputs_embeds[:, :-le_length, :]
            hidden_states_for_reduction = hidden_states_for_reduction[:, :-le_length]
            # Assume past_key_values is DynamicCache
            if kv_cache_for_reduction is not None:
                # assert isinstance(next_cache, DynamicCache)
                kv_cache_for_reduction.crop(-le_length)
            position_ids = position_ids[:, :, :-le_length]
            attention_mask = attention_mask[:, :-le_length]
        
        if delay_selection:
            self.todo_selection = True
            logits = None
            rtn_dict = Qwen2_5_VL_GP_CausalLMOutputWithPast(
                    logits=logits,
                    le_loss=le_loss,
                    past_key_values=kv_cache_for_reduction,
                    hidden_states=hidden_states_for_reduction,
                    rope_deltas=self.rope_deltas,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    attn_grid=attn_grid,
                    image_token_mask_logits=image_token_mask_logits,
                )
            self.glimpse_return_before_selection = rtn_dict
            # print(rtn_dict.hidden_states.dtype)
            if return_dict:
                return rtn_dict
            else:
                return (
                    logits,
                    kv_cache_for_reduction,
                    hidden_states_for_reduction,
                    self.rope_deltas,
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    image_token_mask_logits,
                )
        else:
            reduced_info = self._reduce_tokens(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                hidden_states=hidden_states_for_reduction,
                past_key_values=kv_cache_for_reduction,
                position_ids=position_ids,
                attention_mask=attention_mask,
                image_token_mask_logits=image_token_mask_logits,
                attn_grid=attn_grid,
            )
            return self._glimpse_forward_after_reduction(**reduced_info, return_dict=return_dict, use_cache=use_cache)
            
    def _do_delayed_selection(self, override_image_token_mask_logits, return_dict, use_cache, do_clone=False):
        assert self.todo_selection, "No delayed selection to do."
        self.todo_selection = False
        if do_clone:
            self.todo_selection = True   # Can be called more than once
            cached_input_ids = self.glimpse_return_before_selection.input_ids.clone()
            cached_inputs_embeds = self.glimpse_return_before_selection.inputs_embeds.clone()
            cached_hidden_states = self.glimpse_return_before_selection.hidden_states.clone()
            cached_attention_mask = self.glimpse_return_before_selection.attention_mask.clone()
            cached_position_ids = self.glimpse_return_before_selection.position_ids.clone()
            cached_past_key_values = DynamicCache()
            for layer_idx, (key_cache, value_cache) in enumerate(self.glimpse_return_before_selection.past_key_values):
                cached_past_key_values.update(key_cache.clone(), value_cache.clone(), layer_idx)
            attn_grid = self.glimpse_return_before_selection.attn_grid
        else:
            cached_input_ids = self.glimpse_return_before_selection.input_ids
            cached_inputs_embeds = self.glimpse_return_before_selection.inputs_embeds
            cached_hidden_states = self.glimpse_return_before_selection.hidden_states
            cached_attention_mask = self.glimpse_return_before_selection.attention_mask
            cached_position_ids = self.glimpse_return_before_selection.position_ids
            cached_past_key_values = self.glimpse_return_before_selection.past_key_values
            attn_grid = self.glimpse_return_before_selection.attn_grid
            self.reset_image_tokens_cache()
        
        reduced_info = self._reduce_tokens(
            input_ids=cached_input_ids,
            inputs_embeds=cached_inputs_embeds,
            hidden_states=cached_hidden_states,
            past_key_values=cached_past_key_values,
            position_ids=cached_position_ids,
            attention_mask=cached_attention_mask,
            image_token_mask_logits=override_image_token_mask_logits,
            attn_grid=attn_grid,
            )
        return self._glimpse_forward_after_reduction(**reduced_info, return_dict=return_dict, use_cache=use_cache)

    
    def _get_remain_masks(self, input_ids, attention_mask, image_token_mask_logits, attn_grid):
        threshold = self.config.reduce_threshold
        anchor_positions = list(self.config.anchor_positions) if self.config.anchor_positions is not None else []
        min_remain_num = self.config.min_remain_num
        max_remain_ratio = self.config.max_remain_ratio
        
        # image_token_lengths = [logits.shape[1] for logits in image_token_mask_logits]
        image_token_bool_masks = []
        
        for b, one_logits in enumerate(image_token_mask_logits):
            one_image_token_prob = one_logits[-1].sigmoid()
            one_image_token_bool_mask = one_image_token_prob > threshold
            
            if max_remain_ratio is not None:
                remain_num = one_image_token_bool_mask.sum().item()
                remain_ratio = remain_num / one_image_token_bool_mask.numel()
                if remain_ratio > max_remain_ratio:
                    max_remain_num = int(max_remain_ratio * one_image_token_bool_mask.numel())
                    one_image_token_bool_indices = torch.topk(one_image_token_prob, max_remain_num, dim=-1).indices
                    one_image_token_bool_mask.zero_()
                    one_image_token_bool_mask[one_image_token_bool_indices] = True
            
            if min_remain_num is not None:
                remain_num = one_image_token_bool_mask.sum().item()
                if remain_num < min_remain_num:
                    one_image_token_bool_indices = torch.topk(one_image_token_prob, min_remain_num, dim=-1).indices
                    one_image_token_bool_mask[one_image_token_bool_indices] = True
            
            if anchor_positions:
                if attn_grid.shape[0] != len(image_token_mask_logits):
                    raise NotImplementedError("anchor positions are not supported when using multi-images input")
                one_attn_height, one_attn_width = attn_grid[b]
                for anchor_pos in anchor_positions:
                    if anchor_pos == 'tl':
                        one_image_token_bool_mask[0] = True
                    elif anchor_pos == 'tr':
                        anchor_pos_id = one_attn_width - 1
                        one_image_token_bool_mask[anchor_pos_id] = True
                    elif anchor_pos == 'bl':
                        anchor_pos_id = (one_attn_height - 1) * one_attn_width
                        one_image_token_bool_mask[anchor_pos_id] = True
                    elif anchor_pos == 'br':
                        anchor_pos_id = one_attn_height * one_attn_width - 1
                        one_image_token_bool_mask[anchor_pos_id] = True
                    else:
                        raise ValueError(f"Unknown anchor position: {anchor_pos}. Supported: tl, tr, bl, br.")
            
            image_token_bool_masks.append(one_image_token_bool_mask)
            
        # image_token_bool_masks = torch.cat(image_token_bool_masks, dim=0)
        is_image_mask = input_ids == self.config.image_token_id
        remain_masks = attention_mask.clone().bool()
        remain_masks[is_image_mask] = torch.cat(image_token_bool_masks, dim=0)
        remain_masks &= attention_mask.bool()  # Ensure remain_masks is still a valid attention mask, usually all image tokens are valid in attention_mask
        return remain_masks, image_token_bool_masks
    
        
    # @time_logger
    def _reduce_tokens(
        self,
        input_ids,
        inputs_embeds,
        hidden_states,
        past_key_values,
        position_ids,
        attention_mask,
        image_token_mask_logits,
        attn_grid,
    ):
        B = input_ids.shape[0]
        if past_key_values is not None:
            old_key_states = past_key_values.key_cache
            old_value_states = past_key_values.value_cache
        else:
            old_key_states = None
            old_value_states = None
            reduced_key_states = None
        
        remain_masks, image_token_bool_masks = self._get_remain_masks(input_ids, attention_mask, image_token_mask_logits, attn_grid)
        
        lengths = remain_masks.sum(dim=1).cpu().tolist()
        max_remain_len = max(lengths)
        repad_masks = torch.zeros((B, max_remain_len), dtype=torch.bool, device=remain_masks.device)
        for b in range(B):
            repad_masks[b, -lengths[b]:] = True
        
        reduced_hidden_states = hidden_states[remain_masks]
        reduced_attention_mask = attention_mask[remain_masks]
        reduced_position_ids = position_ids[:, remain_masks]
        reduced_input_ids = input_ids[remain_masks]
        
        if self.training or past_key_values is None:
            reduced_input_embeds = inputs_embeds[remain_masks]
        else:
            reduced_input_embeds = None
        
        if past_key_values is not None:
            reduced_key_states = []
            reduced_value_states = []
            for layer_idx, (old_k, old_v) in enumerate(zip(old_key_states, old_value_states)):
                layer_device = old_k.device
                remain_masks_on_same_device = remain_masks.to(layer_device)
                remain_masks_on_same_device = remain_masks_on_same_device.unsqueeze(1).expand(-1, old_k.shape[1], -1)  # [B, num_heads, max_remain_len]
                reduced_key_states.append(old_k[remain_masks_on_same_device])
                reduced_value_states.append(old_v[remain_masks_on_same_device])
        
        max_remain_len = max(lengths)
        
        
        # Step 3: re-pad
        if reduced_input_embeds is not None:
            padded_input_embeds = torch.zeros((B, max_remain_len, inputs_embeds.shape[-1]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            padded_input_embeds[repad_masks] = reduced_input_embeds
        else:
            padded_input_embeds = None
        pad_token_id = getattr(self.config, "pad_token_id") or 0
        padded_input_ids = torch.full((B, max_remain_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
        padded_input_ids[repad_masks] = reduced_input_ids
        padded_hidden_states = torch.zeros((B, max_remain_len, hidden_states.shape[-1]), dtype=hidden_states.dtype, device=hidden_states.device)
        padded_hidden_states[repad_masks] = reduced_hidden_states
        padded_attention_mask = torch.zeros((B, max_remain_len), dtype=attention_mask.dtype, device=attention_mask.device)
        padded_attention_mask[repad_masks] = reduced_attention_mask
        
        pos_pad_value = 1
        pos_padded_shape = list(position_ids.shape) # Get original dims structure
        pos_padded_shape[2] = max_remain_len # Set sequence length
        padded_position_ids = torch.full(tuple(pos_padded_shape), pos_pad_value, dtype=position_ids.dtype, device=position_ids.device)
        padded_position_ids[:, repad_masks] = reduced_position_ids
     
        if past_key_values is not None:
            padded_key_states = []
            padded_value_states = []
            for layer_idx, (old_k, old_v, reduced_k, reduced_v) in enumerate(zip(old_key_states, old_value_states, reduced_key_states, reduced_value_states)):
                k_shape = list(old_k.shape)
                k_shape[-2] = max_remain_len
                v_shape = list(old_v.shape)
                v_shape[-2] = max_remain_len
                repad_masks_on_same_device = repad_masks.to(old_k.device)
                repad_masks_on_same_device = repad_masks_on_same_device.unsqueeze(1).expand(-1, k_shape[1], -1)  # [B, num_heads, max_remain_len]
                new_k = torch.zeros(k_shape, dtype=old_k.dtype, device=old_k.device)
                new_k[repad_masks_on_same_device] = reduced_k
                new_v = torch.zeros(v_shape, dtype=old_v.dtype, device=old_v.device)
                new_v[repad_masks_on_same_device] = reduced_v
                padded_key_states.append(new_k)
                padded_value_states.append(new_v)
        

        if past_key_values is not None:
            # Because past_key_values is an object, we just overwrite the attribute of it directly.
            past_key_values._seen_tokens = max_remain_len
            past_key_values.key_cache = padded_key_states
            past_key_values.value_cache = padded_value_states
            
        self.reduced_input_ids = padded_input_ids

        return {
            "input_ids": padded_input_ids,
            "inputs_embeds": padded_input_embeds,
            "hidden_states": padded_hidden_states,
            "past_key_values": past_key_values,
            "position_ids": padded_position_ids,
            "attention_mask": padded_attention_mask,
            "image_token_mask_logits": image_token_mask_logits,
            "image_token_bool_masks": image_token_bool_masks,
        }
        
        
           
    def _glimpse_forward_after_reduction(
        self,
        input_ids,
        inputs_embeds,
        hidden_states,
        position_ids,
        attention_mask,
        past_key_values,
        use_cache,
        image_token_mask_logits,
        image_token_bool_masks,
        return_dict,
    ):
        max_reduced_len = attention_mask.shape[1]
        # Step 4: If reduce_layer is not the last layer, we need to finish the decoding.
        if self.config.reduce_layer < len(self.model.layers) - 1:
            cache_position = torch.arange(0, max_reduced_len, device=attention_mask.device)
            position_embeddings = self.model.rotary_emb(
                hidden_states, position_ids
            )
            casual_mask = self.model._update_causal_mask(
                attention_mask, hidden_states, cache_position, past_key_values, output_attentions=False
            )
            
            for layer_id in range(self.config.reduce_layer + 1, len(self.model.layers)):
                decoder_layer = self.model.layers[layer_id]
                if self.model.gradient_checkpointing and self.model.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        casual_mask,
                        position_ids,
                        past_key_values,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=casual_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                hidden_states = layer_outputs[0]
                if use_cache:
                    past_key_values = layer_outputs[-1]
            hidden_states = self.model.norm(hidden_states)
           
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            
            return Qwen2_5_VL_GP_CausalLMOutputWithPast(
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                rope_deltas=self.rope_deltas,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_token_mask_logits=image_token_mask_logits,
                image_token_bool_masks=image_token_bool_masks,
            )
        
        return (
            logits,
            past_key_values,
            hidden_states,
            self.rope_deltas,
            input_ids,
            inputs_embeds,
            attention_mask,
            position_ids,
            image_token_mask_logits,
        )
    
            
            
    def _scatter_image_embeds(self, input_ids, inputs_embeds, image_embeds):
        mask = input_ids == self.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds
            
    # @time_logger
    def _visual_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        hidden_states = self.visual.patch_embed(hidden_states)
        rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.visual.get_window_index(grid_thw)
        reverse_indices = torch.argsort(window_index)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.visual.spatial_merge_unit, self.visual.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.visual.spatial_merge_unit, self.visual.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        selected_visual_layers = tuple(self.config.selected_visual_layers)
        selected_hidden_states = [None] * len(selected_visual_layers)
        for layer_num, blk in enumerate(self.visual.blocks):
            if layer_num in self.visual.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            if self.visual.gradient_checkpointing and self.visual.training:
                hidden_states = self.visual._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)
                
            try:
                layer_pos = selected_visual_layers.index(layer_num)
            except ValueError:
                layer_pos = None
            # save selected visual hidden states
            if layer_pos is not None:
                one_hidden_states = hidden_states.reshape(seq_len // self.visual.spatial_merge_unit, self.visual.spatial_merge_unit, -1)
                one_hidden_states = one_hidden_states.mean(dim=1)
                selected_hidden_states[layer_pos] = one_hidden_states[reverse_indices, :]
                

        hidden_states = self.visual.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]

        
        middle_info = {
            "selected_image_embeds": selected_hidden_states,
            "window_index": window_index,
            "cu_window_seqlens": cu_window_seqlens,
            "cu_seqlens": cu_seqlens,
        }
        
        return hidden_states, middle_info

    def text_embed_forward(self, input_ids):
        return self.model.embed_tokens(input_ids)
    
    # @memory_logger
    @time_logger
    def llm_forward(self, 
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    cache_position: Optional[torch.LongTensor] = None):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        return self.lm_head(hidden_states), outputs
    
    
    @time_logger
    def llm_forward_prefilling(self, 
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_values: Optional[List[torch.FloatTensor]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None,
                    cache_position: Optional[torch.LongTensor] = None):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = outputs[0]
        return self.lm_head(hidden_states), outputs
    
            
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_token_mask_logits: Optional[List[torch.FloatTensor]] = None,
        ref_token_masks: Optional[List[torch.BoolTensor]] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        do_selection: bool = True,
        delay_selection: bool = False,
        use_ref_masks: Optional[bool] = None,
    ) -> Union[Tuple, Qwen2_5_VL_GP_CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        actual_use_ref_masks = use_ref_masks if use_ref_masks is not None else self.config.use_ref_masks

        if inputs_embeds is None and not self.todo_selection:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds, image_info = self._visual_forward(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                inputs_embeds = self._scatter_image_embeds(input_ids, inputs_embeds, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        if not self.todo_selection:
            # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
            if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
                # calculate RoPE index once per generation in the pre-fill stage only
                if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):
                    position_ids, rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        second_per_grid_ts,
                        attention_mask,
                    )
                    self.rope_deltas = rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = inputs_embeds.shape
                    delta = (
                        (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                        if cache_position is not None
                        else 0
                    )
                    position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    position_ids = position_ids.add(delta)
                    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                
        if do_selection and pixel_values is not None:   # only support single/multi image inputs now
            if image_token_mask_logits is None:   # only do prune at the prefilling time
                assert not output_attentions
                assert not output_hidden_states
                return self._glimpse_forward(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    image_info=image_info,
                    image_grid_thw=image_grid_thw,
                    ref_token_masks=ref_token_masks,
                    return_dict=return_dict,
                    delay_selection=delay_selection,
                    use_ref_masks=actual_use_ref_masks,
                )
            elif self.todo_selection:
                return self._do_delayed_selection(
                    override_image_token_mask_logits=image_token_mask_logits, 
                    return_dict=return_dict,
                    use_cache=use_cache,
                )
                
        # tmp
        # print_rank0(f"input_ids.shape: {input_ids.shape}")
        # print_rank0(f"position_ids.shape: {position_ids.shape}")
        # print_rank0(f"position_ids: {position_ids}")
        # print_rank0(f"past_key_values.get_seq_length(): {past_key_values.get_seq_length() if past_key_values is not None else None}")
        # print_rank0(f"attention_mask.shape: {attention_mask.shape if attention_mask is not None else None}")
        # print_rank0(f"self.rope_deltas: {self.rope_deltas}")
        
        if ((cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)):
            logits, outputs = self.llm_forward_prefilling(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        else:
            logits, outputs = self.llm_forward(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

        if not return_dict:
            return (
                logits,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
                self.rope_deltas,
                input_ids,
                # inputs_embeds,
                attention_mask,
                image_token_mask_logits,
            )

        return Qwen2_5_VL_GP_CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            rope_deltas=self.rope_deltas,
            input_ids=input_ids,
            # inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_token_mask_logits=image_token_mask_logits,
        )
        
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        image_token_mask_logits=None,
        ref_token_masks=None,  # new added
        do_selection=True,   # new added
        delay_selection=False,   # new added
        use_ref_masks=None,   # new added
        total_num_new_tokens=None,  # new added
        **kwargs,
    ):
        if self.reduced_input_ids is not None:
            input_ids = torch.cat((self.reduced_input_ids, input_ids[:, -total_num_new_tokens:]), dim=-1)

        if inputs_embeds is not None:
            n = input_ids.shape[1] - inputs_embeds.shape[1]
            if n > 0:
                new_inputs_embeds = self.model.embed_tokens(input_ids[:, -n:])
                inputs_embeds = torch.cat((inputs_embeds, new_inputs_embeds), dim=1)
                
        # print_rank0(f"input_ids.shape: {input_ids.shape}")
        # print_rank0(f"inputs_embeds.shape: {inputs_embeds.shape if inputs_embeds is not None else None}")
        # print_rank0(f"cache_position.shape: {cache_position.shape if cache_position is not None else None}")
        # print_rank0(f"cache_position[-1]: {cache_position[-1] if cache_position is not None else None}")
            
        model_inputs = GenerationMixin.prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )
        
        if self.reduced_input_ids is None:
            # Qwen2-5-VL position_ids are prepared with rope_deltas in forward
            model_inputs["position_ids"] = None
        
        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        model_inputs.update(
            {
                "image_token_mask_logits": image_token_mask_logits,
                "ref_token_masks": ref_token_masks,
                "do_selection": do_selection,
                "delay_selection": delay_selection,
                "use_ref_masks": use_ref_masks,
            }
        )
        return model_inputs
        
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: Qwen2_5_VL_GP_CausalLMOutputWithPast,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # Update past_key_values keeping its naming used in model code
        model_kwargs["past_key_values"] = getattr(outputs, "past_key_values")
        
        # Calculate actual attention_mask and position_ids after reduction
        model_kwargs["attention_mask"] = torch.cat(
            [outputs.attention_mask, outputs.attention_mask.new_ones((outputs.attention_mask.shape[0], num_new_tokens))], dim=-1
        )
        
        new_postion_ids = torch.repeat_interleave(outputs.position_ids[:, :, -1:], num_new_tokens, dim=-1)
        add_position_ids = torch.arange(1, num_new_tokens+1, device=new_postion_ids.device, dtype=new_postion_ids.dtype).view(1, 1, -1)
        new_postion_ids = new_postion_ids + add_position_ids
        if not model_kwargs.get("use_cache", True):
            new_postion_ids = torch.cat((outputs.position_ids, new_postion_ids), dim=-1)
        model_kwargs["position_ids"] = new_postion_ids

        input_ids_x = outputs.input_ids
        if input_ids_x is not None:
            if input_ids_x.shape[1] < model_kwargs["cache_position"].shape[0]:
                # reduction happened, we need to update past_positions
                past_positions = torch.ones_like(input_ids_x[0, :], dtype=torch.int64).cumsum(0) - 1
            else:
                past_positions = model_kwargs["cache_position"]
        else:
            past_positions = model_kwargs["cache_position"]
        
        model_kwargs["total_num_new_tokens"] = model_kwargs.get("total_num_new_tokens", 0) + num_new_tokens

        if outputs.inputs_embeds is not None:
            model_kwargs["inputs_embeds"] = outputs.inputs_embeds
        
        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = past_positions[-1:] + num_new_tokens
        else:
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))

        # pass image_token_mask_logits to indicate no need to do reduction.
        model_kwargs["image_token_mask_logits"] = outputs.image_token_mask_logits
        return model_kwargs

    
__all__ = [
    "Qwen2_5_VL_GP_CausalLMOutputWithPast",
    "Qwen2_5_VL_GP_ForConditionalGeneration",
]