from typing import List, Optional, Tuple, Union

from dataclasses import dataclass
import warnings
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    LlamaConfig, LlamaModel, LlamaForCausalLM,
)

from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
    LlamaMLP,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.generation.utils import GenerateOutput
from transformers.cache_utils import DynamicCache, Cache
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.utils import logging

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

from llava.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN
)
from warppers import time_logger
from utils import download_model_from_hf

logger = logging.get_logger(__name__)

def print_rank0(message):
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(message)

# ---------- Fusers ----------

class BaseAttnFuser(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds):
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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


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

    
    def forward(self, hidden_states, cond_states, position_embeddings):
        bsz, seq_length = hidden_states.shape[:2]
        if cond_states is not None:
            qk = torch.cat([hidden_states, cond_states], dim=-1)
        else:
            qk = hidden_states
        q = self.q_proj(qk).reshape(bsz, seq_length, self.num_heads, -1)
        k = self.k_proj(qk).reshape(bsz, seq_length, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(bsz, seq_length, self.num_heads, -1)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        
        attention_mask = None  # because grid_hw of each batch is same
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attention_mask, dropout_p=0.0
        )
        
        attn_output = attn_output.transpose(1, 2)  # [bsz, seq_length, num_heads, head_dim]
        attn_output = attn_output.reshape(bsz, seq_length, -1)  # [bsz, seq_length, hidden_size]
        attn_output = self.o_proj(attn_output)
        return attn_output



class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class AttnFuserLayer(nn.Module):
    def __init__(self, hidden_size, cond_size, num_heads, hidden_act):
        super().__init__()
        self.norm1 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.norm2 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.attn = CondSdpaAttention(hidden_size, cond_size, num_heads)
        self.mlp = MLP(hidden_size, hidden_size * 2, hidden_act, bias=True)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cond_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


@register_attn_fuser()
class AttnFuserDummy(BaseAttnFuser):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        
    def forward(self, attn_map, attn_grid_hw, selected_image_embeds):
        attn_outs = []
        attn_map_mean = attn_map.mean(dim=-1)  # bsz, h*w
        if self.config.use_attention_logits:
            attn_map_mean = torch.softmax(attn_map_mean, dim=-1)
        else:
            attn_map_mean = torch.exp(attn_map_mean)
        attn_map_min = attn_map_mean.min(dim=-1, keepdim=True).values
        attn_map_max = attn_map_mean.max(dim=-1, keepdim=True).values
        attn_map_mean = (attn_map_mean - attn_map_min) / (attn_map_max - attn_map_min + 1e-6)
        attn_outs.append(attn_map_mean)
        attn_outs = torch.stack(attn_outs, dim=1)  # [bsz, L, h*w]
        return attn_outs
    

@register_attn_fuser()
class AttnFuserV1(BaseAttnFuser):
    def __init__(self, config, vision_hidden_size=None, **kwargs):
        super().__init__(config)
        assert vision_hidden_size is not None, "vision_hidden_size must be provided for AttnFuserV1"
        self.config = config
        attn_fuse_size = config.attn_fuse_size
        num_layers = len(config.selected_visual_layers)
        visual_cond_size = config.visual_cond_size if num_layers > 0 else 0
        num_attn_heads = config.num_attention_heads
        self.attn_in_proj = nn.Linear(num_attn_heads, attn_fuse_size)
        self.cond_in_projs = nn.ModuleList()
        self.layers = nn.ModuleList()
        self.attn_out_projs = nn.ModuleList()
        for i in range(num_layers):
            self.cond_in_projs.append(nn.Linear(vision_hidden_size, visual_cond_size))
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

    def rot_pos_emb(self, bsz, h, w):
        pos_ids = []
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)  # [h, w]
        hpos_ids = hpos_ids.flatten()  # [h * w]
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)  # [h, w]
        wpos_ids = wpos_ids.flatten()  # [h * w]
        pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1))  # [h * w, 2]
        max_grid_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        rotary_pos_emb = rotary_pos_emb.unsqueeze(0).expand(bsz, -1, -1)
        return rotary_pos_emb
        
    def forward(self, attn_map, attn_grid_hw, selected_image_embeds):
        attn_outs = []
        bsz = attn_map.shape[0]
        h, w = attn_grid_hw
        if self.config.ori_attn_supervision and not self.training:
            attn_map_mean = attn_map.mean(dim=-1)  # bsz, h*w
            if self.config.use_attention_logits:
                attn_map_mean = torch.softmax(attn_map_mean, dim=-1)
            else:
                attn_map_mean = torch.exp(attn_map_mean)
            attn_map_min = attn_map_mean.min(dim=-1, keepdim=True).values
            attn_map_max = attn_map_mean.max(dim=-1, keepdim=True).values
            attn_map_mean = (attn_map_mean - attn_map_min) / (attn_map_max - attn_map_min + 1e-6)
            attn_outs.append(attn_map_mean)

        attn_hiddens = self.attn_in_proj(attn_map)
        rotary_pos_emb = self.rot_pos_emb(bsz, h, w)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        
        cond_hiddens = selected_image_embeds
        
        for layer_num, layer in enumerate(self.layers):
            curr_cond_hiddens = self.cond_in_projs[layer_num](cond_hiddens[layer_num])
            attn_hiddens = layer(attn_hiddens, curr_cond_hiddens, position_embeddings)  # [bsz, h*w, attn_fuse_size]
            if self.training or layer_num == len(self.layers) - 1:
                attn_out_proj = self.attn_out_projs[layer_num]
                if isinstance(attn_out_proj, nn.Identity):
                    continue  # not deep supervision
                aux_attn_out = attn_out_proj(attn_hiddens).squeeze(-1)
                attn_outs.append(aux_attn_out)

        attn_outs = torch.stack(attn_outs, dim=1)  # [bsz, L, h*w]
        return attn_outs


@dataclass
class LlavaGPOutputWithPast(ModelOutput):
    logits: torch.FloatTensor = None
    le_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    input_ids: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    position_ids: Optional[torch.LongTensor] = None
    grid_hw: Optional[Tuple[int, int]] = None
    image_token_mask_logits: Optional[torch.Tensor] = None
    image_token_bool_masks: Optional[torch.Tensor] = None
    

class LlavaConfig_GP(LlamaConfig):
    model_type = "llava_llama_gp"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        selected_layers=(21,),
        use_attention_logits=False,
        attn_fuse_size=256,
        selected_visual_layers=(23, 17, 11, 5),
        visual_cond_size=256,
        attn_fuse_type="AttnFuserV1",
        attn_fuse_num_heads=4,
        attn_fuse_hidden_act="silu",
        ori_attn_supervision=True,
        deep_supervision=True,
        le_layers=(0,),
        le_length=1,
        le_dropout_prob=0.0,
        reduce_threshold=0.5,
        use_ref_masks=False,
        use_zero_masks=False,
        reduce_layer=21,
        anchor_positions=(),
        min_remain_num=1,
        max_remain_ratio=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )
        self.selected_layers = selected_layers
        self.use_attention_logits = use_attention_logits
        self.attn_fuse_type = attn_fuse_type
        self.attn_fuse_size = attn_fuse_size
        self.selected_visual_layers = selected_visual_layers
        self.visual_cond_size = visual_cond_size
        self.attn_fuse_num_heads = attn_fuse_num_heads
        self.attn_fuse_hidden_act = attn_fuse_hidden_act
        self.ori_attn_supervision = ori_attn_supervision
        self.deep_supervision = deep_supervision
        self.le_layers = le_layers
        self.le_length = le_length
        self.le_dropout_prob = le_dropout_prob
        self.reduce_threshold = reduce_threshold
        self.use_ref_masks = use_ref_masks
        self.use_zero_masks = use_zero_masks
        self.reduce_layer = reduce_layer
        self.anchor_positions = anchor_positions
        self.min_remain_num = min_remain_num
        self.max_remain_ratio = max_remain_ratio
    
    
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


class LlamaFlashAttention2_GP(LlamaFlashAttention2):
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
          # [bsz, nheads, k_len]
        if not use_attention_logits:
            if attention_mask is not None:
                attention_mask_4d = convert_2d_to_4d_mask(attention_mask, 1, dtype=attn_weights.dtype)
                attn_weights = attn_weights + attention_mask_4d  # [bsz, nheads, k_len]
            attn_weights = torch.log_softmax(attn_weights, dim=-1)  # [bsz, nheads, k_len]
        if kv_mask is not None:
            attn_weights = attn_weights.squeeze(2)
            attn_weights = attn_weights.transpose(1, 2)  # [bsz, k_len, nheads]
            selected_attn_weights = attn_weights[kv_mask]  # [N, nheads]
            kv_length = kv_mask.sum(dim=-1)
            attn_weights = selected_attn_weights.split(kv_length.tolist(), dim=0)  # List(bsz) of [k_select_len, nheads]
        return attn_weights
        
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        q_indices: Optional[List[int]] = None,
        kv_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        # NOTE: During prefilling stage, the seq_len of kv cache in the layers after pruning will be less than the max position_ids
        # So we use the max position id as seq_len
        kv_seq_len = position_ids.max().item() + 1
        
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
    
        if output_attentions:
            attn_weights = self._cal_attn_weights(query_states, key_states, attention_mask, 
                                                  q_indices=q_indices, kv_mask=kv_mask, use_attention_logits=self.config.use_attention_logits)
        else:
            attn_weights = None
    
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        dropout_rate = self.attention_dropout if self.training else 0.0
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

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )
        
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value
    

LLAMA_ATTENTION_CLASSES_GP = {
    "flash_attention_2": LlamaFlashAttention2_GP,
}


class LlamaDecoderLayer_GP(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES_GP[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    

class LlamaModel_GP(LlamaModel):
    _no_split_modules = ["LlamaDecoderLayer_GP"]
    
    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer_GP(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    
    
    
class LlavaLlamaModel_GP(LlavaMetaModel, LlamaModel_GP):
    config_class = LlavaConfig_GP

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel_GP, self).__init__(config)


class LlavaLlamaForCausalLM_GP(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig_GP

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel_GP(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.todo_selection = False
        self.glimpse_return_before_selection = None
        # vision_hidden_size = self.get_model().vision_tower.hidden_size
        self._init_new_modules(config, re_init=False)
        # Initialize weights and apply final processing
        self.post_init()

    def _init_new_modules(self, config, re_init=False):
        self.config = config
        # NOTE: config of other sub-modules is not overwrited !
        vision_hidden_size = self.get_model().vision_tower.hidden_size
        selected_layers = self.config.selected_layers
        reduce_layer = self.config.reduce_layer  # do reduce after this layer
        le_layers = self.config.le_layers
        
        max_selected_layers = max(selected_layers) if len(selected_layers) > 0 else 0
        max_le_layers = max(le_layers) if len(le_layers) > 0 else 0
        
        # assert max_le_layers <= max_selected_layers, f"max_le_layers {le_layers} must be less than or equal to max_selected_layers {selected_layers}"
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
            self.attn_fuser = ATTN_FUSER_REGISTRY[config.attn_fuse_type](config, vision_hidden_size=vision_hidden_size)
        except KeyError:
            raise ValueError(f"AttnFuser {config.attn_fuse_type} not found in registry. Available options: {list(ATTN_FUSER_REGISTRY.keys())}")
        
        self.register_parameter(
            "learnable_embeddings",
            nn.Parameter(torch.empty(len(self.config.le_layers), self.config.le_length, self.config.hidden_size))
        )
        self.le_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.le_norm = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.le_dropout = nn.Dropout(self.config.le_dropout_prob)
        
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
        return {
            "attn_fuser": self.attn_fuser,
            "learnable_embeddings": self.learnable_embeddings,
            "le_proj": self.le_proj,
            "le_norm": self.le_norm,
        }
        
    def _init_weights(self, module):
        super()._init_weights(module)
        # init new modules and parameters
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

    def get_model(self):
        return self.model
    
    def encode_images(self, images):
        vision_tower = self.get_model().get_vision_tower()
        image_forward_outs = vision_tower.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        selected_visual_layers = tuple(self.config.selected_visual_layers)
        selected_image_embeds = [image_forward_outs.hidden_states[i][:, 1:].to(images.dtype) for i in selected_visual_layers]
        image_info = {
            "selected_image_embeds": selected_image_embeds  # [bsz, num_seqs, hidden_size]
        }
        image_features = vision_tower.feature_select(image_forward_outs).to(images.dtype)
        image_features = self.get_model().mm_projector(image_features)
        return image_features, image_info

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        if type(images) is list or images.ndim == 5:
            raise NotImplementedError()
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features, image_info = self.encode_images(images)  # B,N,C

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_ids = []
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_input_ids.append(cur_input_ids)
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_ids = []
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_ids.append(cur_input_ids_noim[i])
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), IMAGE_TOKEN_INDEX, device=cur_input_ids.device, dtype=cur_input_ids.dtype))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_input_ids = torch.cat(cur_new_input_ids)

            new_input_ids.append(cur_new_input_ids)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_input_ids_padded = torch.zeros((batch_size, max_len), dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_input_ids, cur_new_embed, cur_new_labels) in enumerate(zip(new_input_ids, new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'left') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    new_input_ids_padded[i, -cur_len:] = cur_new_input_ids[:cur_len]
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    new_input_ids_padded[i, :cur_len] = cur_new_input_ids[:cur_len]
                    
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
            new_labels[new_labels == IMAGE_TOKEN_INDEX] = IGNORE_INDEX  # remove image token labels

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return new_input_ids_padded, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_info

    def _update_causal_mask(self, attention_mask, inputs_embeds, past_key_values):
        batch_size, seq_length = attention_mask.shape
        past_key_values_length = past_key_values.get_usable_length(seq_length) if past_key_values is not None else 0
        
        if self.model._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self.model._use_sdpa:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            causal_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            causal_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
        return causal_mask

    def _append_le(self, input_ids, inputs_embeds, labels, position_ids, attention_mask):
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
            label_mask = labels != IGNORE_INDEX
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
            
            # 3. Apply gather to `input_ids`, `labels`
            # Create source tensors by concatenating original data with LE placeholder data
            le_ids = torch.full((bsz, le_len), self.config.eos_token_id, device=input_ids.device, dtype=input_ids.dtype)
            source_ids = torch.cat([input_ids, le_ids], dim=1)
            input_ids = torch.gather(source_ids, 1, gather_indices)
            
            le_labels = torch.full((bsz, le_len), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)
            source_labels = torch.cat([labels, le_labels], dim=1)
            labels = torch.gather(source_labels, 1, gather_indices)

        # It is simpler to deal with attention_mask, position_ids
        attention_mask = torch.cat([attention_mask, torch.ones((bsz, le_len), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
        le_pos_ids = []
        for b in range(bsz):
            one_last_pos_idx = position_ids[b, -1].item()
            one_le_pos_ids = torch.arange(one_last_pos_idx + 1, one_last_pos_idx + 1 + le_len, device=position_ids.device)
            one_le_pos_ids = one_le_pos_ids.unsqueeze(0)
            le_pos_ids.append(one_le_pos_ids)
        le_pos_ids = torch.cat(le_pos_ids, dim=0)  # B, le_len
        position_ids = torch.cat([position_ids, le_pos_ids], dim=1)

        return input_ids, inputs_embeds, labels, position_ids, attention_mask
    
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
        return hidden_states
    
    
    def _crop_kv_cache(self, kv_cache: DynamicCache, max_length: int):
        if max_length < 0:
            max_length = kv_cache.get_seq_length() - abs(max_length)

        if kv_cache.get_seq_length() <= max_length:
            return

        kv_cache.seen_tokens = max_length
        for idx in range(len(kv_cache.key_cache)):
            if kv_cache.key_cache[idx].numel():
                kv_cache.key_cache[idx] = kv_cache.key_cache[idx][..., :max_length, :]
                kv_cache.value_cache[idx] = kv_cache.value_cache[idx][..., :max_length, :]
                
    def _decode_image_token_mask_logits(self, batched_attn_map, grid_hw, selected_image_embeds):
        return self.attn_fuser(batched_attn_map, grid_hw, selected_image_embeds)
        
    def _get_remain_masks(self, input_ids, attention_mask, image_token_mask_logits, grid_hw):
        threshold = self.config.reduce_threshold
        anchor_positions = list(self.config.anchor_positions) if self.config.anchor_positions is not None else []
        min_remain_num = self.config.min_remain_num
        max_remain_ratio = self.config.max_remain_ratio
        
        image_token_bool_masks = []

        last_image_token_mask_logits = image_token_mask_logits[:, -1]  # B,N

        for b, one_logits in enumerate(last_image_token_mask_logits):
            one_image_token_prob = one_logits.sigmoid()  # [N,]
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
                one_attn_height, one_attn_width = grid_hw
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
            
        is_image_mask = input_ids == IMAGE_TOKEN_INDEX
        remain_masks = attention_mask.clone().bool()
        remain_masks[is_image_mask] = torch.cat(image_token_bool_masks, dim=0)
        remain_masks &= attention_mask.bool()  # Ensure remain_masks is still a valid attention mask, usually all image tokens are valid in attention_mask
        return remain_masks, image_token_bool_masks
    
    def _reduce_tokens(
        self,
        input_ids,
        inputs_embeds,
        hidden_states,
        past_key_values,
        position_ids,
        attention_mask,
        image_token_mask_logits,
        grid_hw,
    ):
        # Step 1
        B = input_ids.shape[0]
        if past_key_values is not None:
            old_key_states = past_key_values.key_cache
            old_value_states = past_key_values.value_cache
        else:
            old_key_states = None
            old_value_states = None
            reduced_key_states = None
        
        remain_masks, image_token_bool_masks = self._get_remain_masks(input_ids, attention_mask, image_token_mask_logits, grid_hw)
        
        lengths = remain_masks.sum(dim=1).cpu().tolist()
        max_remain_len = max(lengths)
        repad_masks = torch.zeros((B, max_remain_len), dtype=torch.bool, device=remain_masks.device)
        for b in range(B):
            repad_masks[b, -lengths[b]:] = True
        
        # Step 2. 
        reduced_hidden_states = hidden_states[remain_masks]
        reduced_attention_mask = attention_mask[remain_masks]
        reduced_position_ids = position_ids[remain_masks]
        reduced_input_ids = input_ids[remain_masks]
        
        if self.training:
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
        if self.training:
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
        pos_padded_shape[1] = max_remain_len # Set sequence length
        padded_position_ids = torch.full(tuple(pos_padded_shape), pos_pad_value, dtype=position_ids.dtype, device=position_ids.device)
        padded_position_ids[repad_masks] = reduced_position_ids
     
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
            past_key_values.seen_tokens = max_remain_len
            past_key_values.key_cache = padded_key_states
            past_key_values.value_cache = padded_value_states

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
        # Step 4: If reduce_layer is not the last layer, we need to finish the decoding.
        if self.config.reduce_layer < len(self.model.layers) - 1:
            causal_mask = self._update_causal_mask(attention_mask, hidden_states, past_key_values)
            for layer_id in range(self.config.reduce_layer + 1, len(self.model.layers)):
                decoder_layer = self.model.layers[layer_id]
                if self.model.gradient_checkpointing and self.model.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        use_cache,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        use_cache=use_cache,
                    )
                hidden_states = layer_outputs[0]
                if use_cache:
                    past_key_values = layer_outputs[-1]
            hidden_states = self.model.norm(hidden_states)
        
        logits = self.lm_head(hidden_states)
        
        if return_dict:
            return LlavaGPOutputWithPast(
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_token_mask_logits=image_token_mask_logits,
                image_token_bool_masks=image_token_bool_masks,
            )
        else:
            return (
                logits,
                past_key_values,
                hidden_states,
                input_ids,
                inputs_embeds,
                attention_mask,
                position_ids,
                image_token_mask_logits,
                image_token_bool_masks,
            )


    def _glimpse_forward(
        self,
        input_ids,
        inputs_embeds,
        labels,
        attention_mask,
        position_ids,
        past_key_values,
        ref_token_masks,
        use_cache,
        image_info,
        grid_hw,
        return_dict,
        delay_selection,
        use_ref_masks,
    ):
        # Step 1. Check inputs
        assert getattr(self.config, "tokenizer_padding_side", "left") == "left"
        
        if self.model.gradient_checkpointing and self.model.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
                
        batch_size, seq_length = input_ids.shape[:2]
        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
            
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
        
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            
        # Step 2. Append input_ids, inputs_embeds, position_ids, attention_mask for learnable embeddings
        if not use_ref_masks:
            input_ids, inputs_embeds, labels, position_ids, attention_mask = self._append_le(
                input_ids, inputs_embeds, labels, position_ids, attention_mask
            )
        
        # Step 3. Prepare other inputs for decoder
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, past_key_values)
        hidden_states = inputs_embeds
        
        batch_size, seq_length = input_ids.shape[:2]
        # q_indices, kv_mask decide the shape of extracted attentions
        if labels is None:
            q_indices = [seq_length - 1] * batch_size
        else:
            label_mask = labels != -100
            q_indices = label_mask.int().argmax(dim=-1) - 1
            q_indices = q_indices.tolist()
        kv_mask = input_ids == IMAGE_TOKEN_INDEX
        
        selected_layers = tuple(self.config.selected_layers)
        if len(selected_layers) == 0:
            batched_attns = None
        else:
            batched_attns = [[None] * len(selected_layers) for _ in range(batch_size)]
        
        # Step 4. Extract attentions during the forward pass of LLM 
        next_decoder_cache = None
        max_forward_layer = max(self.config.selected_layers) if len(self.config.selected_layers) > 0 else 0
        max_forward_layer = max(max_forward_layer, self.config.reduce_layer)
        if labels is not None:
            max_forward_layer = len(self.model.layers) - 1  # always forward all layers if labels are provided
        
        hidden_states_for_reduction = None
        kv_cache_for_reduction = None
        
        
        for layer_id, decoder_layer in enumerate(self.model.layers):
            if layer_id > 0 and not use_ref_masks:  # le in layer 0 has been added
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
                    q_indices,
                    kv_mask,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=_output_attentions,
                    use_cache=use_cache,
                    q_indices=q_indices,
                    kv_mask=kv_mask
                )
            
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache = layer_outputs[-1]
            if _output_attentions:
                attn_weights = layer_outputs[1]
                for b in range(batch_size):
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
            loss_fct = nn.CrossEntropyLoss()
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
            batched_attns = torch.stack(batched_attns, dim=0)  # [bsz, num_tokens, num_layers, num_heads]
            batched_attns = batched_attns.flatten(2)  # [bsz, num_tokens, num_layers * num_heads]
            
        if use_ref_masks:
            image_token_mask_logits = []
            for b in range(batch_size):
                image_token_mask_logits.append(torch.logit(ref_token_masks[b].float().to(device=hidden_states_for_reduction.device).view(1, -1)))
        elif self.config.use_zero_masks:
            image_token_mask_logits = []
            for b in range(batch_size):
                image_token_mask_logits.append(torch.logit(torch.zeros((1, grid_hw[0] * grid_hw[1]), device=hidden_states_for_reduction.device)))
        else:
            image_token_mask_logits = self._decode_image_token_mask_logits(batched_attns, grid_hw, **image_info)

        # Step 6. Trim (reducted image tokens and le tokens) hidden_states, next_cache, input_ids, attention_mask
        if not use_ref_masks:
            le_length = self.config.le_length
            input_ids = input_ids[:, :-le_length]
            inputs_embeds = inputs_embeds[:, :-le_length, :]
            hidden_states_for_reduction = hidden_states_for_reduction[:, :-le_length]
            # Assume past_key_values is DynamicCache
            if kv_cache_for_reduction is not None:
                try:
                    kv_cache_for_reduction.crop(-le_length)
                except AttributeError:
                    self._crop_kv_cache(kv_cache_for_reduction, -le_length)
                
            position_ids = position_ids[:, :-le_length]
            attention_mask = attention_mask[:, :-le_length]

        if delay_selection:
            self.todo_selection = True
            logits = None
            rtn_dict = LlavaGPOutputWithPast(
                    logits=logits,
                    le_loss=le_loss,
                    past_key_values=kv_cache_for_reduction,
                    hidden_states=hidden_states_for_reduction,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    grid_hw=grid_hw,
                    image_token_mask_logits=image_token_mask_logits,
                )
            self.glimpse_return_before_selection = rtn_dict

            if return_dict:
                return rtn_dict
            else:
                return (
                    logits,
                    le_loss,
                    kv_cache_for_reduction,
                    hidden_states_for_reduction,
                    input_ids,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    grid_hw,
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
                grid_hw=grid_hw,
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
            grid_hw = self.glimpse_return_before_selection.grid_hw
        else:
            cached_input_ids = self.glimpse_return_before_selection.input_ids
            cached_inputs_embeds = self.glimpse_return_before_selection.inputs_embeds
            cached_hidden_states = self.glimpse_return_before_selection.hidden_states
            cached_attention_mask = self.glimpse_return_before_selection.attention_mask
            cached_position_ids = self.glimpse_return_before_selection.position_ids
            cached_past_key_values = self.glimpse_return_before_selection.past_key_values
            grid_hw = self.glimpse_return_before_selection.grid_hw
            self.reset_image_tokens_cache()
        
        reduced_info = self._reduce_tokens(
            input_ids=cached_input_ids,
            inputs_embeds=cached_inputs_embeds,
            hidden_states=cached_hidden_states,
            past_key_values=cached_past_key_values,
            position_ids=cached_position_ids,
            attention_mask=cached_attention_mask,
            image_token_mask_logits=override_image_token_mask_logits,
            grid_hw=grid_hw
        )
        return self._glimpse_forward_after_reduction(**reduced_info, return_dict=return_dict, use_cache=use_cache)

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
                    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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
        return self.llm_forward(
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
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        image_info: Optional[List[torch.FloatTensor]] = None,
        image_token_mask_logits: Optional[List[torch.FloatTensor]] = None,
        ref_token_masks: Optional[List[torch.BoolTensor]] = None,
        return_dict: Optional[bool] = None,
        do_selection: bool = True,
        delay_selection: bool = False,
        use_ref_masks: Optional[bool] = None,
    ) -> Union[Tuple, LlavaGPOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                image_info,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        actual_use_ref_masks = use_ref_masks if use_ref_masks is not None else self.config.use_ref_masks
        
        grid_h = grid_w = self.get_vision_tower().num_patches_per_side
        grid_hw = (grid_h, grid_w)
        
        if do_selection:
            if image_token_mask_logits is None:
                # TODO: Make sure this should be called only if multimodal input
                assert not output_attentions
                assert not output_hidden_states
                return self._glimpse_forward(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    image_info=image_info,
                    grid_hw=grid_hw,
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
        
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            logits, outputs = self.llm_forward_prefilling(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            logits, outputs = self.llm_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        if return_dict:
            return LlavaGPOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_token_mask_logits=image_token_mask_logits,
            )
        
        return (
            logits,
            outputs.past_key_values,
            outputs.hidden_states,
            input_ids,
            inputs_embeds,
            attention_mask,
            position_ids,
            image_token_mask_logits,
        )
            

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                image_info,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
            kwargs["image_info"] = image_info
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        input_length = input_ids.shape[1]
        generated_ids = super().generate(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        generated_ids = generated_ids[:, input_length:]
        return generated_ids

    def prepare_inputs_for_generation(self, 
                                      input_ids, 
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      images=None,
                                      image_sizes=None,
                                      attention_mask=None,
                                      position_ids=None,
                                      ref_token_masks=None,
                                      do_selection=None,
                                      use_ref_masks=None,
                                      image_token_mask_logits=None,
                                      image_info=None,
                                      delay_selection=None,
                                      input_ids_x=None,
                                      **kwargs):

        if input_ids_x is not None:
            next_token = input_ids[:, -1:]
            input_ids = torch.cat([input_ids_x, next_token], dim=-1)
        
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"), 
        }
        
        if inputs_embeds is not None:
            inputs['inputs_embeds'] = inputs_embeds        
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if position_ids is not None:
            inputs['position_ids'] = position_ids
        if ref_token_masks is not None:
            inputs['ref_token_masks'] = ref_token_masks
        if do_selection is not None:
            inputs['do_selection'] = do_selection
        if image_token_mask_logits is not None:
            inputs['image_token_mask_logits'] = image_token_mask_logits
        if image_info is not None:
            inputs['image_info'] = image_info
        if delay_selection is not None:
            inputs['delay_selection'] = delay_selection
        if use_ref_masks is not None:
            inputs['use_ref_masks'] = use_ref_masks 

        return inputs
    
    def _update_model_kwargs_for_generation(self, 
                                            outputs: LlavaGPOutputWithPast, 
                                            model_kwargs, 
                                            is_encoder_decoder = False, 
                                            standardize_cache_format = False):
        super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            standardize_cache_format=standardize_cache_format,
        )
        
        # update position ids
        num_new_tokens = 1
        
        model_kwargs["input_ids_x"] = outputs.input_ids
        
        # Calculate actual attention_mask and position_ids after reduction
        model_kwargs["attention_mask"] = torch.cat(
            [outputs.attention_mask, outputs.attention_mask.new_ones((outputs.attention_mask.shape[0], num_new_tokens))], dim=-1
        )
        
        new_postion_ids = torch.repeat_interleave(outputs.position_ids[:, -1:], 
                                                  repeats=num_new_tokens, 
                                                  dim=-1)
        add_position_ids = torch.arange(1, num_new_tokens+1, device=new_postion_ids.device, dtype=new_postion_ids.dtype)
        new_postion_ids = new_postion_ids + add_position_ids
        model_kwargs['position_ids'] = new_postion_ids
        
        model_kwargs["image_token_mask_logits"] = outputs.image_token_mask_logits
        
        model_kwargs.pop("inputs_embeds", None)
        
        return model_kwargs
        