import contextlib
import inspect
import math
import os
import sys
import warnings
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, GenerationConfig
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import ModelOutput
from transformers.activations import ACT2FN

from .configuration import InternVL2_5_GPConfig


try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
    _FLASH_ATTN_AVAILABLE = True
except Exception:
    _flash_attn_varlen_func = None
    _FLASH_ATTN_AVAILABLE = False


def _dynamic_cache_layer_past(past_key_value, layer_idx):
    if not isinstance(past_key_value, DynamicCache) or layer_idx is None:
        return past_key_value, False
    if layer_idx < len(past_key_value.key_cache) and past_key_value.key_cache[layer_idx].numel():
        return (past_key_value.key_cache[layer_idx], past_key_value.value_cache[layer_idx]), True
    return None, True


def _dynamic_cache_store_layer(cache, layer_idx, present_key_value):
    if present_key_value is None or layer_idx is None:
        return cache
    key_states, value_states = present_key_value
    old_len = 0
    if layer_idx < len(cache.key_cache) and cache.key_cache[layer_idx].numel():
        old_len = cache.key_cache[layer_idx].shape[-2]
        cache.key_cache[layer_idx] = key_states
        cache.value_cache[layer_idx] = value_states
    else:
        while len(cache.key_cache) < layer_idx:
            cache.key_cache.append(torch.tensor([], device=key_states.device, dtype=key_states.dtype))
            cache.value_cache.append(torch.tensor([], device=value_states.device, dtype=value_states.dtype))
        if len(cache.key_cache) == layer_idx:
            cache.key_cache.append(key_states)
            cache.value_cache.append(value_states)
        else:
            cache.key_cache[layer_idx] = key_states
            cache.value_cache[layer_idx] = value_states
    if layer_idx == 0:
        cache._seen_tokens += key_states.shape[-2] - old_len
    return cache


def _with_rotary_min_seq_len(attn_module, min_seq_len):
    if min_seq_len is None or min_seq_len <= 0:
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def manager():
        rotary = attn_module.rotary_emb
        original_forward = rotary.forward

        def forward_with_min_seq_len(x, seq_len=None):
            if seq_len is None:
                seq_len = min_seq_len
            else:
                seq_len = max(int(seq_len), min_seq_len)
            return original_forward(x, seq_len=seq_len)

        rotary.forward = forward_with_min_seq_len
        try:
            yield
        finally:
            rotary.forward = original_forward

    return manager()


def _compute_gp_side_attention(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    layer_past_key_value=None,
):
    """Compute only the GP selective attention rows; never feeds the value path."""
    extras = getattr(self, "_gp_extras", None)
    if not extras or not extras.get("is_selected") or extras.get("q_indices") is None:
        return None

    rearrange = self._gp_helpers["rearrange"]
    apply_rotary_pos_emb = self._gp_helpers["apply_rotary_pos_emb"]
    repeat_kv = self._gp_helpers["repeat_kv"]

    bsz, _, _ = hidden_states.size()
    qkv_states = self.wqkv(hidden_states)
    qkv_states = rearrange(
        qkv_states, "b q (h gs d) -> b q h gs d",
        gs=2 + self.num_key_value_groups, d=self.head_dim,
    )
    query_states = qkv_states[..., : self.num_key_value_groups, :]
    query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
    key_states = qkv_states[..., -2, :]

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if layer_past_key_value is not None:
        kv_seq_len += layer_past_key_value[0].shape[-2]
    if position_ids is not None and position_ids.numel() > 0:
        kv_seq_len = max(kv_seq_len, int(position_ids.max().item()) + 1)

    cos, sin = self.rotary_emb(key_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if layer_past_key_value is not None:
        key_states = torch.cat([layer_past_key_value[0], key_states], dim=2)

    key_states_full = repeat_kv(key_states, self.num_key_value_groups)
    kv_len = key_states_full.shape[-2]
    batch_idx = torch.arange(bsz, device=query_states.device)
    q_idx_t = torch.as_tensor(extras["q_indices"], device=query_states.device, dtype=torch.long)
    q_sel = query_states[batch_idx, :, q_idx_t, :].unsqueeze(2)
    attn_slim = torch.matmul(q_sel, key_states_full.transpose(2, 3))
    attn_slim = attn_slim / math.sqrt(self.head_dim)

    use_attention_logits = bool(extras.get("use_attention_logits"))
    side_mask = extras.get("causal_mask")
    if side_mask is None:
        side_mask = attention_mask
    if not use_attention_logits and side_mask is not None and side_mask.dim() == 4:
        mask_slim = side_mask[batch_idx, :, q_idx_t, :kv_len].unsqueeze(2)
        attn_slim = attn_slim + mask_slim
    if not use_attention_logits:
        attn_slim = F.log_softmax(attn_slim, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return attn_slim


def _patched_internlm2_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    **kwargs,
):
    """Wrapper around native InternLM2 attention.

    The normal value path is delegated to the original InternVL attention forward
    unchanged. GP only computes an optional slim attention row as a side output.
    """
    layer_idx = getattr(self, "_gp_layer_idx", None)
    layer_past_key_value, is_dynamic = _dynamic_cache_layer_past(past_key_value, layer_idx)
    main_attention_mask = attention_mask
    if bool(getattr(self, "_gp_use_flash_attn", False)) and main_attention_mask is not None and main_attention_mask.dim() == 2:
        expected_kv_len = hidden_states.shape[1]
        if layer_past_key_value is not None:
            expected_kv_len += layer_past_key_value[0].shape[-2]
        if main_attention_mask.shape[-1] != expected_kv_len:
            main_attention_mask = main_attention_mask[:, -expected_kv_len:]
        if not bool((main_attention_mask == 0).any().item()):
            main_attention_mask = None
    attn_weights_out = (
        _compute_gp_side_attention(
            self,
            hidden_states,
            attention_mask=main_attention_mask,
            position_ids=position_ids,
            layer_past_key_value=layer_past_key_value,
        )
        if output_attentions
        else None
    )

    min_rotary_seq_len = None
    if position_ids is not None and position_ids.numel() > 0:
        min_rotary_seq_len = int(position_ids.max().item()) + 1

    original_forward = self._gp_original_forward
    with _with_rotary_min_seq_len(self, min_rotary_seq_len):
        attn_output, _native_attn, present_key_value = original_forward(
            hidden_states=hidden_states,
            attention_mask=main_attention_mask,
            position_ids=position_ids,
            past_key_value=layer_past_key_value,
            output_attentions=False,
            use_cache=use_cache,
            **kwargs,
        )

    if is_dynamic and use_cache:
        present_key_value = _dynamic_cache_store_layer(
            past_key_value, layer_idx, present_key_value
        )

    return attn_output, attn_weights_out, present_key_value


class InternVLAttnFuserMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_size = len(config.selected_layers) * config.num_attention_heads
        if in_size <= 0:
            raise ValueError("InternVLAttnFuserMLP requires at least one selected layer.")
        self.net = nn.Sequential(
            nn.Linear(in_size, config.attn_fuse_size),
            ACT2FN[config.attn_fuse_hidden_act],
            nn.Linear(config.attn_fuse_size, 1),
        )

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        return self.net(attn_map).squeeze(-1).unsqueeze(0)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_vision(q, k, cos, sin):
    # q,k: [N, num_heads, head_dim]; cos,sin: [N, head_dim]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    q_e = (q * cos) + (_rotate_half(q) * sin)
    k_e = (k * cos) + (_rotate_half(k) * sin)
    return q_e, k_e


class _Vision2DRotaryEmbedding(nn.Module):
    """Per-token (h, w) rotary, half head_dim per axis (Qwen2.5-VL vision style)."""

    def __init__(self, dim, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen, device):
        t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
        return torch.outer(t, self.inv_freq.to(device))  # [seqlen, dim//2]


class _CondSdpaAttention(nn.Module):
    """QK = concat(hidden, cond); V = hidden. Per-segment SDPA via cu_seqlens."""

    def __init__(self, hidden_size, cond_size, num_heads):
        super().__init__()
        qk_size = hidden_size + cond_size
        v_size = hidden_size
        if qk_size % num_heads != 0:
            raise ValueError(f"(attn_fuse_size + cond_size)={qk_size} must be divisible by num_heads={num_heads}.")
        if v_size % num_heads != 0:
            raise ValueError(f"attn_fuse_size={v_size} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.q_proj = nn.Linear(qk_size, qk_size, bias=False)
        self.k_proj = nn.Linear(qk_size, qk_size, bias=False)
        self.v_proj = nn.Linear(v_size, v_size, bias=False)
        self.o_proj = nn.Linear(v_size, v_size, bias=False)

    def forward(self, hidden_states, cond_states, cu_seqlens, position_embeddings):
        L = hidden_states.shape[0]
        qk = torch.cat([hidden_states, cond_states], dim=-1) if cond_states is not None else hidden_states
        q = self.q_proj(qk).reshape(L, self.num_heads, -1)
        k = self.k_proj(qk).reshape(L, self.num_heads, -1)
        v = self.v_proj(hidden_states).reshape(L, self.num_heads, -1)
        cos, sin = position_embeddings
        q, k = _apply_rope_vision(q, k, cos, sin)
        q = q.transpose(0, 1).unsqueeze(0)  # [1, H, L, D_qk]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)  # [1, H, L, D_v]
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        if len(lengths) == 1:
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)[0]
        else:
            splits = [torch.split(t, lengths, dim=2) for t in (q, k, v)]
            outs = [F.scaled_dot_product_attention(qs, ks, vs, dropout_p=0.0)[0]
                    for qs, ks, vs in zip(*splits)]
            attn_output = torch.cat(outs, dim=1)
        attn_output = attn_output.transpose(0, 1).reshape(L, -1)
        return self.o_proj(attn_output)


class _CondFA2Attention(nn.Module):
    """Same I/O contract as _CondSdpaAttention, dispatches to flash_attn_varlen_func.

    Workaround for FA2's qk_dim==v_dim constraint: pad V with a constant dummy column
    of width `cond_size`, then trim from output.
    """

    def __init__(self, hidden_size, cond_size, num_heads):
        super().__init__()
        qk_size = hidden_size + cond_size
        v_size = hidden_size
        if qk_size % num_heads != 0:
            raise ValueError(f"(attn_fuse_size + cond_size)={qk_size} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.cond_size = cond_size
        self.q_proj = nn.Linear(qk_size, qk_size, bias=False)
        self.k_proj = nn.Linear(qk_size, qk_size, bias=False)
        self.v_proj = nn.Linear(v_size, v_size, bias=False)
        self.o_proj = nn.Linear(v_size, v_size, bias=False)
        if cond_size > 0:
            self.register_buffer(
                "dummy_value", torch.ones(1, cond_size), persistent=False
            )

    def forward(self, hidden_states, cond_states, cu_seqlens, position_embeddings):
        L = hidden_states.shape[0]
        qk = torch.cat([hidden_states, cond_states], dim=-1) if cond_states is not None else hidden_states
        q = self.q_proj(qk).reshape(L, self.num_heads, -1)
        k = self.k_proj(qk).reshape(L, self.num_heads, -1)
        v = self.v_proj(hidden_states)
        if self.cond_size > 0:
            v = torch.cat([v, self.dummy_value.to(v.dtype).expand(L, -1)], dim=-1)
        v = v.reshape(L, self.num_heads, -1)
        cos, sin = position_embeddings
        q, k = _apply_rope_vision(q, k, cos, sin)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        attn_output = _flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        ).reshape(L, -1)
        if self.cond_size > 0:
            attn_output = attn_output[..., : -self.cond_size]
        return self.o_proj(attn_output)


class _AttnFuserLayer(nn.Module):
    def __init__(self, hidden_size, cond_size, num_heads, hidden_act, use_flash_attn=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        if use_flash_attn and _FLASH_ATTN_AVAILABLE:
            self.attn = _CondFA2Attention(hidden_size, cond_size, num_heads)
        else:
            if use_flash_attn and not _FLASH_ATTN_AVAILABLE:
                warnings.warn("attn_fuse_use_flash_attn=True but flash_attn unavailable; using SDPA.")
            self.attn = _CondSdpaAttention(hidden_size, cond_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            ACT2FN[hidden_act],
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states, cond_states, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cond_states, cu_seqlens, position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class InternVLAttnFuserV1(nn.Module):
    """Multi-layer cond-attention fuser with 2D RoPE, aligned with Qwen AttnFuserV1.

    - Stacks `len(selected_visual_layers)` layers (config knob; default = 1).
    - Each layer has independent norm1/norm2 (pre-LN attention + pre-LN MLP).
    - 2D RoPE over (h, w) per tile grid.
    - Per-layer cond projection and per-layer output head (deep supervision optional).
    - The placeholder window_index/cu_window_seqlens args are accepted but ignored.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        attn_in_size = len(config.selected_layers) * config.num_attention_heads
        if attn_in_size <= 0:
            raise ValueError("InternVLAttnFuserV1 requires at least one selected layer.")
        attn_fuse_size = config.attn_fuse_size
        num_layers = max(1, len(config.selected_visual_layers))
        visual_cond_size = config.visual_cond_size if num_layers > 0 else 0
        num_heads = config.attn_fuse_num_heads
        head_dim = (attn_fuse_size + visual_cond_size) // num_heads
        if (attn_fuse_size + visual_cond_size) % num_heads != 0:
            raise ValueError(
                f"attn_fuse_size+visual_cond_size={attn_fuse_size + visual_cond_size} "
                f"must be divisible by attn_fuse_num_heads={num_heads}."
            )
        if head_dim % 2 != 0:
            raise ValueError(f"per-head qk dim={head_dim} must be even for RoPE.")

        self.attn_in_proj = nn.Linear(attn_in_size, attn_fuse_size)
        self.cond_in_projs = nn.ModuleList([
            nn.Linear(config.hidden_size, visual_cond_size) for _ in range(num_layers)
        ])
        use_fa2 = bool(getattr(config, "attn_fuse_use_flash_attn", False))
        self.layers = nn.ModuleList([
            _AttnFuserLayer(attn_fuse_size, visual_cond_size, num_heads,
                            config.attn_fuse_hidden_act, use_flash_attn=use_fa2)
            for _ in range(num_layers)
        ])
        deep = bool(getattr(config, "deep_supervision", False))
        self.attn_out_projs = nn.ModuleList([
            nn.Linear(attn_fuse_size, 1) if (deep or i == num_layers - 1) else nn.Identity()
            for i in range(num_layers)
        ])
        self.rotary_pos_emb = _Vision2DRotaryEmbedding(head_dim // 2)
        self.num_layers = num_layers

    def _rope_for_grid(self, grid_hw, device):
        # Each tile is its own h×w grid; build per-token (h_pos, w_pos) ids.
        pos_ids = []
        for h, w in grid_hw.tolist():
            hpos = torch.arange(h, device=device).unsqueeze(1).expand(-1, w).flatten()
            wpos = torch.arange(w, device=device).unsqueeze(0).expand(h, -1).flatten()
            pos_ids.append(torch.stack([hpos, wpos], dim=-1))
        pos_ids = torch.cat(pos_ids, dim=0)  # [N, 2]
        max_grid = int(grid_hw.max().item()) if grid_hw.numel() > 0 else 0
        if max_grid <= 0:
            zero = torch.zeros((pos_ids.shape[0], 0), device=device)
            return zero, zero
        rope_full = self.rotary_pos_emb(max_grid, device)  # [max, dim/2]
        rope = rope_full[pos_ids].flatten(1)  # [N, dim]
        emb = torch.cat([rope, rope], dim=-1)  # [N, 2*dim] = head_dim
        return emb.cos(), emb.sin()

    def forward(self, attn_map, attn_grid_hw, selected_image_embeds, window_index, cu_seqlens, cu_window_seqlens):
        del window_index  # placeholder for InternVL (no ViT windowing)
        device = attn_map.device
        attn_hidden = self.attn_in_proj(attn_map)  # [N, attn_fuse_size]
        # attn_fuse_global=True → whole-image attention across tiles of a sample.
        # attn_fuse_global=False → tile-local attention (each 448x448 tile is its own segment).
        cu = cu_seqlens if self.config.attn_fuse_global else cu_window_seqlens
        if cond_count := len(selected_image_embeds):
            cond_feats = [selected_image_embeds[min(i, cond_count - 1)] for i in range(self.num_layers)]
        else:
            cond_feats = [None] * self.num_layers
        position_embeddings = self._rope_for_grid(attn_grid_hw, device)

        attn_outs = []
        for i, layer in enumerate(self.layers):
            cond_i = self.cond_in_projs[i](cond_feats[i].to(attn_hidden.dtype)) if cond_feats[i] is not None else None
            attn_hidden = layer(attn_hidden, cond_i, cu, position_embeddings)
            head = self.attn_out_projs[i]
            if isinstance(head, nn.Identity):
                continue
            logits = head(attn_hidden).squeeze(-1)  # [N]
            attn_outs.append(logits)
        return torch.stack(attn_outs, dim=0)  # [num_out_heads, N]


ATTN_FUSER_REGISTRY = {
    "AttnFuserMLP": InternVLAttnFuserMLP,
    "AttnFuserV1": InternVLAttnFuserV1,
}


@dataclass
class InternVL2_5_GPCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    le_loss: Optional[torch.Tensor] = None
    image_token_mask_logits: Optional[list[torch.Tensor]] = None
    image_token_bool_masks: Optional[list[torch.Tensor]] = None
    input_ids: Optional[torch.Tensor] = None
    inputs_embeds: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None


class InternVL2_5_GP_ForConditionalGeneration(nn.Module):
    config_class = InternVL2_5_GPConfig

    def __init__(self, base_model, gp_config=None, tokenizer=None):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = self._coerce_config(gp_config, base_model.config)
        self.img_context_token_id = getattr(base_model, "img_context_token_id", None)
        self.num_image_token = base_model.num_image_token
        self.reset_image_tokens_cache()
        self._init_new_modules(self.config)
        self._layer_kwargs_cache = None
        self._patch_internlm2_attentions()

    @classmethod
    def _coerce_config(cls, gp_config, base_config):
        if gp_config is None:
            gp_config = cls.config_class()
        elif not isinstance(gp_config, cls.config_class):
            gp_config = cls.config_class.from_dict(gp_config.to_dict())

        llm_config = base_config.llm_config
        vision_config = base_config.vision_config
        gp_config.llm_config = llm_config.to_dict() if hasattr(llm_config, "to_dict") else llm_config
        gp_config.vision_config = SimpleNamespace(
            hidden_size=llm_config.hidden_size,
            spatial_merge_size=1,
        )
        gp_config.num_attention_heads = llm_config.num_attention_heads
        gp_config.hidden_size = llm_config.hidden_size
        gp_config.rms_norm_eps = getattr(llm_config, "rms_norm_eps", 1e-5)
        gp_config.vocab_size = llm_config.vocab_size
        gp_config.pad_token_id = getattr(llm_config, "pad_token_id", None)
        gp_config.eos_token_id = getattr(llm_config, "eos_token_id", None)
        gp_config.internvl_vision_hidden_size = vision_config.hidden_size
        return gp_config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, tokenizer=None, **kwargs):
        # GP attention is patched at runtime, so we don't strictly need eager. If caller
        # passes `attn_implementation='flash_attention_2'`, propagate to InternVL's
        # `use_flash_attn` flag so the base loads consistent attribute layout.
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("low_cpu_mem_usage", True)
        attn_impl = kwargs.pop("attn_implementation", None)
        if attn_impl is None and config is not None:
            attn_impl = getattr(config, "attn_implementation", None)
            if attn_impl is None:
                attn_impl = getattr(config, "_attn_implementation", None)
        if attn_impl is not None:
            kwargs.setdefault("use_flash_attn", str(attn_impl).lower() == "flash_attention_2")
        else:
            kwargs.setdefault("use_flash_attn", False)
        # Newer transformers' caching_allocator_warmup iterates model._tp_plan without
        # guarding against None (the class-level default for trust_remote_code models).
        import transformers.modeling_utils as _mu
        _orig_warmup = getattr(_mu, "caching_allocator_warmup", None)
        if _orig_warmup is not None:
            def _safe_warmup(model, device_map, factor=2):
                if model._tp_plan is None:
                    model._tp_plan = {}
                return _orig_warmup(model, device_map, factor=factor)
            _mu.caching_allocator_warmup = _safe_warmup
        try:
            base_model = AutoModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        finally:
            if _orig_warmup is not None:
                _mu.caching_allocator_warmup = _orig_warmup
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                use_fast=False,
            )
        instance = cls(base_model, gp_config=config, tokenizer=tokenizer)
        if attn_impl is not None:
            # Re-patch with FA2 flag now that we know the requested impl.
            instance.config.attn_implementation = attn_impl
            instance._patch_internlm2_attentions()
        return instance

    @property
    def device(self):
        return self.base_model.device

    @property
    def language_model(self):
        return self.base_model.language_model

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    @property
    def generation_config(self):
        return self.language_model.generation_config

    def _inner_lm(self):
        return self.language_model.model

    def _final_norm(self):
        return self._inner_lm().norm

    def _output_head(self):
        for attr in ("output", "lm_head"):
            head = getattr(self.language_model, attr, None)
            if head is not None:
                return head
        raise RuntimeError("Cannot locate output head on language_model.")

    def _patch_internlm2_attentions(self):
        """Wrap each InternLM2Attention.forward with GP selective-attention side output.

        Sets `_gp_layer_idx` (for DynamicCache routing) and `_gp_helpers` (rearrange,
        apply_rotary_pos_emb, repeat_kv pulled from the trust_remote_code module).
        The wrapped forward delegates the normal attention output to the native InternVL
        attention implementation, including flash_attention_2 when requested.
        """
        layers = self._inner_lm().layers
        if len(layers) == 0:
            return
        attn_mod = sys.modules.get(type(layers[0].attention).__module__)
        if attn_mod is None:
            warnings.warn("InternLM2 module not found; skipping attention patch.")
            return
        helpers = {
            "rearrange": getattr(attn_mod, "rearrange"),
            "apply_rotary_pos_emb": getattr(attn_mod, "apply_rotary_pos_emb"),
            "repeat_kv": getattr(attn_mod, "repeat_kv"),
        }
        ai = str(getattr(self.config, "attn_implementation", "eager") or "eager").lower()
        use_fa2 = (ai == "flash_attention_2") and _FLASH_ATTN_AVAILABLE
        if ai == "flash_attention_2" and not _FLASH_ATTN_AVAILABLE:
            warnings.warn("attn_implementation='flash_attention_2' requested but flash_attn unavailable; falling back to SDPA.")
        for layer_idx, layer in enumerate(layers):
            attn = layer.attention
            module_is_fa2 = hasattr(attn, "_flash_attention_forward")
            attn._gp_layer_idx = layer_idx
            attn._gp_helpers = helpers
            attn._gp_extras = None
            attn._gp_use_flash_attn = bool(module_is_fa2 or use_fa2)
            if not hasattr(attn, "_gp_original_forward"):
                attn._gp_original_forward = attn.forward
            attn.forward = MethodType(_patched_internlm2_attention_forward, attn)

    def _set_gp_extras(self, q_indices, selected_set, padding_mask=None, causal_mask=None):
        """Stash per-layer GP context that the patched attention reads."""
        layers = self._inner_lm().layers
        use_attn_logits = bool(getattr(self.config, "use_attention_logits", False))
        for layer_idx, layer in enumerate(layers):
            layer.attention._gp_extras = {
                "is_selected": layer_idx in selected_set,
                "q_indices": q_indices,
                "use_attention_logits": use_attn_logits,
                "padding_mask": padding_mask,
                "causal_mask": causal_mask,
            }

    def _set_gp_padding_mask(self, padding_mask):
        """Update only the padding_mask field on all layers (used for phase 2)."""
        for layer in self._inner_lm().layers:
            extras = layer.attention._gp_extras
            if extras is None:
                layer.attention._gp_extras = {
                    "is_selected": False,
                    "q_indices": None,
                    "use_attention_logits": False,
                    "padding_mask": padding_mask,
                    "causal_mask": None,
                }
            else:
                extras["padding_mask"] = padding_mask

    def _clear_gp_extras(self):
        for layer in self._inner_lm().layers:
            layer.attention._gp_extras = None

    def _layer_call(self, decoder_layer, hidden_states, **kwargs):
        if self._layer_kwargs_cache is None:
            sig = inspect.signature(decoder_layer.forward)
            self._layer_kwargs_cache = set(sig.parameters.keys())
        accepted = self._layer_kwargs_cache
        filtered = {k: v for k, v in kwargs.items() if k in accepted and v is not None}
        return decoder_layer(hidden_states, **filtered)

    def _init_new_modules(self, config, re_init=False):
        self.config = config
        if len(config.selected_layers) == 0:
            warnings.warn("InternVL2.5 GP initialized with no selected layers; masks will be zero/ref only.")
        try:
            self.attn_fuser = ATTN_FUSER_REGISTRY[config.attn_fuse_type](config)
        except KeyError as exc:
            raise ValueError(
                f"AttnFuser {config.attn_fuse_type} not found. "
                f"Available: {list(ATTN_FUSER_REGISTRY.keys())}"
            ) from exc
        if len(config.le_layers) > 0 and config.le_length > 0:
            self.learnable_embeddings = nn.Parameter(
                torch.empty(len(config.le_layers), config.le_length, config.hidden_size)
            )
            nn.init.normal_(self.learnable_embeddings, 0.0, 0.02)
            self.le_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.le_norm = nn.LayerNorm(config.hidden_size)
            self.le_dropout = nn.Dropout(config.le_dropout_prob)
        base_dtype = next(self.base_model.parameters()).dtype
        for module in self.new_modules_to_be_saved().values():
            if isinstance(module, nn.Module):
                module.to(device=self.device, dtype=base_dtype)
            elif isinstance(module, nn.Parameter):
                module.data = module.data.to(device=self.device, dtype=base_dtype)

    def new_modules_to_be_saved(self):
        modules = {"attn_fuser": self.attn_fuser}
        if hasattr(self, "learnable_embeddings"):
            modules.update({
                "learnable_embeddings": self.learnable_embeddings,
                "le_proj": self.le_proj,
                "le_norm": self.le_norm,
            })
        return modules

    def save_new_modules(self, save_directory, state_dict=None):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        new_states = {}
        for name, module in self.new_modules_to_be_saved().items():
            if isinstance(module, nn.Parameter):
                new_states[name] = module.detach().cpu()
            else:
                new_states[name] = module.state_dict()
        torch.save(new_states, os.path.join(save_directory, "new_modules_gp.pt"))
        print(f"new_modules of {self.__class__.__name__} saved to {os.path.join(save_directory, 'new_modules_gp.pt')}")

    def load_new_modules(self, load_directory):
        config = self.config_class.from_pretrained(load_directory)
        config = self._coerce_config(config, self.base_model.config)
        self._init_new_modules(config, re_init=True)
        path = os.path.join(load_directory, "new_modules_gp.pt")
        if not os.path.exists(path):
            warnings.warn(f"new_modules_gp.pt not found in {load_directory}.")
            return
        new_states = torch.load(path, map_location="cpu", weights_only=True)
        for name, module in self.new_modules_to_be_saved().items():
            if name not in new_states:
                continue
            if isinstance(module, nn.Parameter):
                module.data.copy_(new_states[name].to(device=module.device, dtype=module.dtype))
            else:
                module.load_state_dict(new_states[name], strict=True)
        print(f"Loaded InternVL2.5 GP new modules from {path}")

    def reset_image_tokens_cache(self):
        self.todo_selection = False
        self.reduced_input_ids = None

    @contextlib.contextmanager
    def disable_adapter(self):
        yield

    def text_embed_forward(self, input_ids):
        return self.get_input_embeddings()(input_ids)

    def _ensure_img_context_token_id(self, input_ids=None):
        if self.img_context_token_id is not None:
            return
        if self.tokenizer is None:
            raise ValueError("InternVL2.5 GP requires tokenizer or img_context_token_id.")
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.base_model.img_context_token_id = self.img_context_token_id

    def _build_inputs_embeds(self, input_ids, pixel_values=None, visual_features=None):
        self._ensure_img_context_token_id(input_ids)
        input_embeds = self.get_input_embeddings()(input_ids).clone()
        if pixel_values is None and visual_features is None:
            return input_embeds, None
        vit_embeds = visual_features if visual_features is not None else self.base_model.extract_feature(pixel_values)
        bsz, seq_len, hidden = input_embeds.shape
        flat_embeds = input_embeds.reshape(bsz * seq_len, hidden)
        selected = input_ids.reshape(-1) == self.img_context_token_id
        if selected.sum().item() == 0:
            raise ValueError("No <IMG_CONTEXT> tokens found in input_ids.")
        vit_flat = vit_embeds.reshape(-1, hidden).to(flat_embeds.device, dtype=flat_embeds.dtype)
        n_token = int(selected.sum().item())
        if vit_flat.shape[0] < n_token:
            raise ValueError(f"Visual features shorter than IMG_CONTEXT tokens: {vit_flat.shape[0]} < {n_token}")
        flat_embeds[selected] = vit_flat[:n_token]
        return flat_embeds.reshape(bsz, seq_len, hidden), vit_flat[:n_token]

    def _visual_token_slices(self, input_ids):
        visual_mask = input_ids == self.img_context_token_id
        slices = []
        for b in range(input_ids.shape[0]):
            idx = torch.nonzero(visual_mask[b], as_tuple=False).flatten()
            slices.append(idx)
        return visual_mask, slices

    def _get_query_indices(self, attention_mask, labels=None):
        bsz, seq_len = attention_mask.shape
        if labels is not None:
            label_mask = labels != -100
            return (label_mask.int().argmax(dim=-1) - 1).clamp_min(0).tolist()
        return [seq_len - 1] * bsz

    def _make_attn_grid(self, num_patches_list, device):
        grids = []
        for num_patches in num_patches_list:
            for _ in range(int(num_patches)):
                grids.append([16, 16])
        if len(grids) == 0:
            return torch.zeros((0, 2), dtype=torch.long, device=device)
        return torch.tensor(grids, dtype=torch.long, device=device)

    def _make_tile_cu_seqlens(self, num_patches_list, device):
        offsets = [0]
        cur = 0
        for num_patches in num_patches_list:
            for _ in range(int(num_patches)):
                cur += self.num_image_token
                offsets.append(cur)
        return torch.tensor(offsets, dtype=torch.int32, device=device)

    def _make_sample_cu_seqlens(self, num_patches_list, device):
        offsets = [0]
        cur = 0
        for num_patches in num_patches_list:
            cur += int(num_patches) * self.num_image_token
            offsets.append(cur)
        return torch.tensor(offsets, dtype=torch.int32, device=device)

    def _decode_image_token_mask_logits(self, captured_attns, input_ids, attention_mask, num_patches_list,
                                         visual_token_embeds, labels=None):
        if len(self.config.selected_layers) == 0:
            return None
        visual_mask, visual_indices = self._visual_token_slices(input_ids)
        per_sample_maps = []
        for b, idx in enumerate(visual_indices):
            layer_maps = []
            for layer_id in self.config.selected_layers:
                attn = captured_attns[layer_id][b]  # slim path: [H, 1, K]; legacy: [H, Q, K]
                if attn.dim() == 3 and attn.shape[1] == 1:
                    layer_maps.append(attn[:, 0, idx].transpose(0, 1))
                else:
                    # Legacy fallback when selective path is disabled.
                    q_idx = self._get_query_indices(attention_mask, labels=labels)[b]
                    layer_maps.append(attn[:, q_idx, idx].transpose(0, 1))
            per_sample_maps.append(torch.cat(layer_maps, dim=-1))
        attn_map = torch.cat(per_sample_maps, dim=0)
        device = attn_map.device
        total_visual = attn_map.shape[0]
        window_index = torch.arange(total_visual, device=device)
        attn_grid = self._make_attn_grid(num_patches_list, device=device)
        cu_seqlens = self._make_sample_cu_seqlens(num_patches_list, device=device)
        cu_window_seqlens = self._make_tile_cu_seqlens(num_patches_list, device=device)
        selected_image_embeds = [visual_token_embeds]
        logits = self.attn_fuser(
            attn_map,
            attn_grid,
            selected_image_embeds,
            window_index,
            cu_seqlens,
            cu_window_seqlens,
        )
        split_lengths = [idx.numel() for idx in visual_indices]
        return list(logits.split(split_lengths, dim=-1))

    def _apply_token_constraints(self, one_prob):
        one_prob = one_prob.flatten()
        if one_prob.numel() == 0:
            return torch.zeros_like(one_prob, dtype=torch.bool)
        if self.config.fixed_remain_ratio is not None:
            k = int(float(self.config.fixed_remain_ratio) * one_prob.numel())
            k = max(1, min(one_prob.numel(), k))
            idx = torch.topk(one_prob, k).indices
            mask = torch.zeros_like(one_prob, dtype=torch.bool)
            mask[idx] = True
            return mask
        mask = one_prob > self.config.reduce_threshold
        if self.config.max_remain_ratio is not None:
            max_k = max(1, min(one_prob.numel(), int(float(self.config.max_remain_ratio) * one_prob.numel())))
            if int(mask.sum().item()) > max_k:
                idx = torch.topk(one_prob, max_k).indices
                mask.zero_()
                mask[idx] = True
        if self.config.min_remain_num is not None and int(mask.sum().item()) < int(self.config.min_remain_num):
            k = max(1, min(one_prob.numel(), int(self.config.min_remain_num)))
            idx = torch.topk(one_prob, k).indices
            mask[idx] = True
        return mask

    def _bool_masks_from_logits(self, image_token_mask_logits):
        return [self._apply_token_constraints(logits[-1].sigmoid()) for logits in image_token_mask_logits]

    def _ref_logits(self, ref_token_masks, device):
        logits = []
        for mask in ref_token_masks:
            if mask is None:
                logits.append(None)
            else:
                prob = mask.float().to(device).reshape(1, -1).clamp(1e-6, 1 - 1e-6)
                logits.append(torch.logit(prob))
        return logits

    def _build_causal_mask(self, attention_mask, inputs_embeds, cache_position, past_key_values):
        inner = self._inner_lm()
        if hasattr(inner, "_update_causal_mask"):
            try:
                return inner._update_causal_mask(
                    attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions=False,
                )
            except TypeError:
                pass
        bsz, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype
        kv_len = (past_key_values.get_seq_length() if past_key_values is not None else 0) + seq_len if (cache_position is None) else int(cache_position[-1].item()) + 1
        causal = torch.full((seq_len, kv_len), torch.finfo(dtype).min, device=device, dtype=dtype)
        diag = torch.arange(seq_len, device=device)
        offset = kv_len - seq_len
        for i in range(seq_len):
            causal[i, : offset + i + 1] = 0
        causal = causal[None, None, :, :].expand(bsz, 1, seq_len, kv_len).contiguous()
        if attention_mask is not None and attention_mask.dim() == 2:
            pad_mask = (attention_mask == 0)[:, None, None, :].expand(bsz, 1, seq_len, attention_mask.shape[-1])
            if pad_mask.shape[-1] == kv_len:
                causal = causal.masked_fill(pad_mask, torch.finfo(dtype).min)
        return causal

    @staticmethod
    def _native_flash_attention_mask(attention_mask):
        """Match InternVL native flash path: all-valid masks are passed as None."""
        if attention_mask is not None and bool((attention_mask == 0).any().item()):
            return attention_mask
        return None

    def _reduce_state(self, input_ids, inputs_embeds, hidden_states, past_key_values,
                       attention_mask, position_ids, image_token_bool_masks):
        """Drop image tokens not in image_token_bool_masks; left-pad to max remaining length.

        Crops the per-layer KV cache the same way and resets _seen_tokens.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        _, visual_indices = self._visual_token_slices(input_ids)
        remain = attention_mask.bool().clone()
        for b, idx in enumerate(visual_indices):
            remain[b, idx] = image_token_bool_masks[b].to(remain.device)

        lengths = [int(remain[b].sum().item()) for b in range(bsz)]
        max_len = max(lengths) if lengths else 0
        repad = torch.zeros((bsz, max_len), dtype=torch.bool, device=device)
        for b in range(bsz):
            repad[b, max_len - lengths[b]:] = True

        pad_id = self.config.pad_token_id or 0
        new_ids = torch.full((bsz, max_len), pad_id, dtype=input_ids.dtype, device=device)
        new_ids[repad] = input_ids[remain]

        new_hidden = torch.zeros((bsz, max_len, hidden_states.shape[-1]),
                                 dtype=hidden_states.dtype, device=hidden_states.device)
        new_hidden[repad.to(hidden_states.device)] = hidden_states[remain.to(hidden_states.device)]

        if inputs_embeds is not None:
            new_embeds = torch.zeros((bsz, max_len, inputs_embeds.shape[-1]),
                                     dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            new_embeds[repad.to(inputs_embeds.device)] = inputs_embeds[remain.to(inputs_embeds.device)]
        else:
            new_embeds = None

        new_attn = torch.zeros((bsz, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        new_attn[repad.to(attention_mask.device)] = 1

        if position_ids is not None:
            new_pos = torch.ones((bsz, max_len), dtype=position_ids.dtype, device=position_ids.device)
            new_pos[repad.to(position_ids.device)] = position_ids[remain.to(position_ids.device)]
        else:
            new_pos = None

        if past_key_values is not None and hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
            new_keys, new_vals = [], []
            for old_k, old_v in zip(past_key_values.key_cache, past_key_values.value_cache):
                # old_k: [B, num_kv_heads, seq, head_dim]
                rmask = remain.to(old_k.device).unsqueeze(1).expand(-1, old_k.shape[1], -1)
                pmask = repad.to(old_k.device).unsqueeze(1).expand(-1, old_k.shape[1], -1)
                k_shape = list(old_k.shape); k_shape[-2] = max_len
                v_shape = list(old_v.shape); v_shape[-2] = max_len
                new_k = torch.zeros(k_shape, dtype=old_k.dtype, device=old_k.device)
                new_v = torch.zeros(v_shape, dtype=old_v.dtype, device=old_v.device)
                new_k[pmask] = old_k[rmask]
                new_v[pmask] = old_v[rmask]
                new_keys.append(new_k)
                new_vals.append(new_v)
            past_key_values.key_cache = new_keys
            past_key_values.value_cache = new_vals
            past_key_values._seen_tokens = max_len

        self.reduced_input_ids = new_ids

        return {
            "input_ids": new_ids,
            "inputs_embeds": new_embeds,
            "hidden_states": new_hidden,
            "attention_mask": new_attn,
            "position_ids": new_pos,
            "past_key_values": past_key_values,
        }

    def _continue_after_reduction(self, hidden_states, attention_mask, position_ids,
                                   past_key_values, use_cache, start_layer):
        inner = self._inner_lm()
        layers = inner.layers
        n_layers = len(layers)
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        cache_position = torch.arange(0, seq_len, device=device)
        causal_mask = self._build_causal_mask(attention_mask, hidden_states, cache_position, past_key_values)
        # Refresh 2D padding_mask in extras so FA2 varlen path sees the reduced sequence's mask.
        self._set_gp_padding_mask(attention_mask)
        for layer_id in range(start_layer, n_layers):
            layer_attention = layers[layer_id].attention
            layer_mask = (
                self._native_flash_attention_mask(attention_mask)
                if bool(getattr(layer_attention, "_gp_use_flash_attn", False))
                else causal_mask
            )
            layer_out = self._layer_call(
                layers[layer_id], hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_out[0]
        hidden_states = self._final_norm()(hidden_states)
        return hidden_states, past_key_values

    def _append_le(self, input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position):
        """Insert learnable embeddings (layer 0) into the sequence before the first label position."""
        bsz, seq_len = input_ids.shape
        le_len = self.config.le_length
        device = inputs_embeds.device
        le_idx = self.config.le_layers.index(0)
        le = self.learnable_embeddings[le_idx]
        le = le.to(device=self.le_proj.weight.device)
        le_dtype = le.dtype
        le = self.le_dropout(self.le_norm(self.le_proj(le))).to(dtype=le_dtype)
        le = le.view(1, le_len, self.config.hidden_size).expand(bsz, -1, -1).to(device=device)

        eos_id = self.config.eos_token_id or 0

        if labels is None:
            inputs_embeds = torch.cat([inputs_embeds, le], dim=1)
            input_ids = torch.cat([
                input_ids,
                torch.full((bsz, le_len), eos_id, device=device, dtype=input_ids.dtype)
            ], dim=1)
            le_token_mask = torch.cat([
                torch.zeros((bsz, seq_len), device=device, dtype=torch.bool),
                torch.ones((bsz, le_len), device=device, dtype=torch.bool),
            ], dim=1)
        else:
            label_mask = labels != -100
            insert_pos = label_mask.int().argmax(dim=-1)
            new_len = seq_len + le_len
            split_idx = insert_pos.unsqueeze(1)
            new_indices = torch.arange(new_len, device=device).unsqueeze(0)
            indices_le_part = seq_len + (new_indices - split_idx)
            indices_part2 = new_indices - le_len
            mask_part1 = new_indices < split_idx
            mask_le_part = (new_indices >= split_idx) & (new_indices < split_idx + le_len)
            gather_indices = torch.where(
                mask_part1, new_indices,
                torch.where(mask_le_part, indices_le_part, indices_part2)
            )

            source_embeds = torch.cat([inputs_embeds, le], dim=1)
            inputs_embeds = torch.gather(
                source_embeds, 1,
                gather_indices.unsqueeze(2).expand(-1, -1, self.config.hidden_size)
            )

            le_ids = torch.full((bsz, le_len), eos_id, device=device, dtype=input_ids.dtype)
            input_ids = torch.gather(torch.cat([input_ids, le_ids], dim=1), 1, gather_indices)

            le_labels = torch.full((bsz, le_len), -100, device=device, dtype=labels.dtype)
            labels = torch.gather(torch.cat([labels, le_labels], dim=1), 1, gather_indices)

            source_le_mask = torch.cat([
                torch.zeros((bsz, seq_len), device=device, dtype=torch.bool),
                torch.ones((bsz, le_len), device=device, dtype=torch.bool),
            ], dim=1)
            le_token_mask = torch.gather(source_le_mask, 1, gather_indices)

        attention_mask = torch.cat([
            attention_mask,
            torch.ones((bsz, le_len), device=device, dtype=attention_mask.dtype)
        ], dim=1)

        if position_ids is not None:
            last_pos = position_ids[:, -1]
            le_pos = torch.stack([
                torch.arange(int(last_pos[b].item()) + 1, int(last_pos[b].item()) + 1 + le_len, device=device)
                for b in range(bsz)
            ])
            position_ids = torch.cat([position_ids, le_pos], dim=1)

        last_cache = int(cache_position[-1].item())
        cache_position = torch.cat([
            cache_position,
            torch.arange(last_cache + 1, last_cache + 1 + le_len, device=device)
        ])

        return input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position, le_token_mask

    @staticmethod
    def _trim_tensor_by_keep(tensor, keep):
        if tensor is None:
            return None
        bsz = keep.shape[0]
        new_len = int(keep.sum(dim=1)[0].item())
        return tensor[keep.to(tensor.device)].view(bsz, new_len, *tensor.shape[2:])

    @staticmethod
    def _trim_cache_by_keep(past_key_values, keep, new_len):
        if past_key_values is None or not hasattr(past_key_values, "key_cache") or len(past_key_values.key_cache) == 0:
            return past_key_values
        new_keys, new_vals = [], []
        for old_k, old_v in zip(past_key_values.key_cache, past_key_values.value_cache):
            rmask = keep.to(old_k.device).unsqueeze(1).expand(-1, old_k.shape[1], -1)
            new_k = old_k[rmask].view(old_k.shape[0], old_k.shape[1], new_len, old_k.shape[-1])
            new_v = old_v[rmask].view(old_v.shape[0], old_v.shape[1], new_len, old_v.shape[-1])
            new_keys.append(new_k)
            new_vals.append(new_v)
        past_key_values.key_cache = new_keys
        past_key_values.value_cache = new_vals
        past_key_values._seen_tokens = new_len
        return past_key_values

    def _trim_le_tokens(self, le_token_mask, input_ids, inputs_embeds, labels, hidden_states,
                        past_key_values, attention_mask, position_ids):
        keep = ~le_token_mask
        if not bool(le_token_mask.any().item()):
            return input_ids, inputs_embeds, labels, hidden_states, past_key_values, attention_mask, position_ids

        input_ids = self._trim_tensor_by_keep(input_ids, keep)
        inputs_embeds = self._trim_tensor_by_keep(inputs_embeds, keep)
        labels = self._trim_tensor_by_keep(labels, keep)
        hidden_states = self._trim_tensor_by_keep(hidden_states, keep)
        attention_mask = self._trim_tensor_by_keep(attention_mask, keep)
        position_ids = self._trim_tensor_by_keep(position_ids, keep)

        new_len = input_ids.shape[1]
        past_key_values = self._trim_cache_by_keep(past_key_values, keep, new_len)

        return input_ids, inputs_embeds, labels, hidden_states, past_key_values, attention_mask, position_ids

    def _try_add_le(self, layer_id, hidden_states, q_indices):
        """Add learnable embeddings in-place to hidden states at the specified le_layer."""
        try:
            le_idx = self.config.le_layers.index(layer_id)
        except ValueError:
            return hidden_states
        le = self.learnable_embeddings[le_idx]
        le = le.to(device=self.le_proj.weight.device)
        le_dtype = le.dtype
        le = self.le_dropout(self.le_norm(self.le_proj(le))).to(dtype=le_dtype)

        bsz, seq_len, hidden_size = hidden_states.shape
        le_len = self.config.le_length
        device = hidden_states.device
        le = le.view(1, le_len, hidden_size).expand(bsz, -1, -1).to(device=device)

        end_indices = torch.tensor(q_indices, device=device, dtype=torch.long).unsqueeze(1) + 1
        start_indices = end_indices - le_len
        le_range = torch.arange(le_len, device=device).unsqueeze(0)
        target_seq_indices = start_indices + le_range
        mask = (target_seq_indices >= 0) & (target_seq_indices < seq_len)
        valid_le = le[mask]
        batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand_as(mask)
        valid_batch = batch_idx[mask]
        valid_target = target_seq_indices[mask]
        flat_indices = valid_batch * seq_len + valid_target
        hidden_states.view(-1, hidden_size).index_add_(0, flat_indices, valid_le)
        return hidden_states

    def _glimpse_forward(
        self,
        input_ids,
        inputs_embeds,
        visual_token_embeds,
        attention_mask,
        position_ids,
        past_key_values,
        labels,
        use_cache,
        num_patches_list,
        ref_token_masks,
        image_token_mask_logits,
        actual_use_ref_masks,
        delay_selection,
        do_selection,
    ):
        inner = self._inner_lm()
        layers = inner.layers
        n_layers = len(layers)
        bsz, seq_len = input_ids.shape
        device = inputs_embeds.device

        selected_layers = tuple(self.config.selected_layers)
        selected_set = set(selected_layers)
        reduce_layer = int(self.config.reduce_layer)
        reduce_layer = max(0, min(reduce_layer, n_layers - 1))

        need_attn = (
            do_selection
            and image_token_mask_logits is None
            and not (actual_use_ref_masks and ref_token_masks is not None)
            and not self.config.use_zero_masks
            and len(selected_layers) > 0
        )

        # Determine how far Phase 1 needs to go.
        max_phase1_layer = reduce_layer
        if need_attn:
            max_phase1_layer = max(max_phase1_layer, max(selected_layers))
        # If selection is disabled, this wrapper must behave like the base model:
        # run the full decoder and return logits. Likewise, labels need a full
        # forward for LM loss when selection is not delayed, or when LE loss is needed.
        run_full_no_selection = not do_selection
        has_le = hasattr(self, "learnable_embeddings") and len(getattr(self.config, "le_layers", [])) > 0
        run_full_for_loss = labels is not None and (not delay_selection or has_le)
        if run_full_no_selection or run_full_for_loss:
            max_phase1_layer = n_layers - 1
        max_phase1_layer = min(max_phase1_layer, n_layers - 1)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past_seen = past_key_values.get_seq_length() if (
            past_key_values is not None and hasattr(past_key_values, "get_seq_length")
        ) else 0
        cache_position = torch.arange(past_seen, past_seen + seq_len, device=device)
        if position_ids is None:
            if past_seen == 0 and attention_mask is not None and attention_mask.dim() == 2:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            else:
                position_ids = cache_position.unsqueeze(0).expand(bsz, -1)

        le_token_mask = None
        # Insert learnable embeddings at layer 0 position (when not using ref masks).
        if has_le and not actual_use_ref_masks and 0 in self.config.le_layers:
            input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position, le_token_mask = self._append_le(
                input_ids, inputs_embeds, labels, position_ids, attention_mask, cache_position
            )
            bsz, seq_len = input_ids.shape

        causal_mask = self._build_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values)

        hidden_states = inputs_embeds
        captured_attns = {} if (need_attn and selected_set) else None
        captured_hidden = None
        captured_cache = None

        # Tell the patched attention which layers should emit slim attn rows and where to gather Q.
        # Always pass the native mask shape to the main attention path. The 4D
        # causal mask is stashed only for GP's side attention rows.
        if need_attn:
            q_indices_for_attn = self._get_query_indices(attention_mask, labels=labels)
            self._set_gp_extras(
                q_indices_for_attn,
                selected_set,
                padding_mask=attention_mask,
                causal_mask=causal_mask,
            )
        else:
            self._set_gp_extras([0] * bsz, set(), padding_mask=attention_mask, causal_mask=causal_mask)

        q_indices_for_le = self._get_query_indices(attention_mask, labels=labels) if has_le else None

        for layer_id in range(0, max_phase1_layer + 1):
            if has_le and not actual_use_ref_masks and layer_id > 0:
                hidden_states = self._try_add_le(layer_id, hidden_states, q_indices_for_le)
            out_attn = bool(need_attn and (layer_id in selected_set))
            layer_attention = layers[layer_id].attention
            layer_mask = (
                self._native_flash_attention_mask(attention_mask)
                if bool(getattr(layer_attention, "_gp_use_flash_attn", False))
                else causal_mask
            )
            layer_out = self._layer_call(
                layers[layer_id], hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=out_attn,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_out[0]
            if out_attn:
                captured_attns[layer_id] = layer_out[1]
            # Snapshot at reduce_layer.
            if (
                captured_hidden is None
                and layer_id == reduce_layer
                and reduce_layer < n_layers - 1
                and not run_full_no_selection
                and not run_full_for_loss
                and not delay_selection
            ):
                if max_phase1_layer == reduce_layer:
                    captured_hidden = hidden_states
                    captured_cache = past_key_values
                else:
                    captured_hidden = hidden_states.clone()
                    if past_key_values is not None and hasattr(past_key_values, "key_cache"):
                        cloned = DynamicCache()
                        for li, (k, v) in enumerate(zip(past_key_values.key_cache, past_key_values.value_cache)):
                            cloned.update(k.clone(), v.clone(), li)
                        captured_cache = cloned
                    else:
                        captured_cache = None

        # Done capturing attn; clear is_selected so phase 2 / decode skip slim path,
        # but keep padding_mask available for FA2 varlen on phase 2.
        for layer in self._inner_lm().layers:
            extras = layer.attention._gp_extras
            if extras is not None:
                extras["is_selected"] = False
                extras["q_indices"] = None

        # Compute image_token_mask_logits.
        if image_token_mask_logits is None and do_selection:
            if actual_use_ref_masks and ref_token_masks is not None:
                image_token_mask_logits = self._ref_logits(ref_token_masks, device)
            elif self.config.use_zero_masks:
                _, visual_indices = self._visual_token_slices(input_ids)
                image_token_mask_logits = [
                    torch.full((1, idx.numel()), -20.0, device=device, dtype=hidden_states.dtype)
                    for idx in visual_indices
                ]
            elif need_attn:
                if visual_token_embeds is None:
                    visual_token_embeds = inputs_embeds[input_ids == self.img_context_token_id]
                image_token_mask_logits = self._decode_image_token_mask_logits(
                    captured_attns, input_ids, attention_mask, num_patches_list, visual_token_embeds, labels=labels,
                )
        image_token_bool_masks = (
            self._bool_masks_from_logits(image_token_mask_logits) if image_token_mask_logits is not None else None
        )

        le_loss = None
        logits = None

        if run_full_no_selection or run_full_for_loss:
            # We've already gone through all layers; compute logits for normal
            # generation and optionally LM loss.
            hidden_states = self._final_norm()(hidden_states)
            logits = self._output_head()(hidden_states)
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]).float()
                shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
                le_loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        if le_token_mask is not None:
            input_ids, inputs_embeds, labels, hidden_states, past_key_values, attention_mask, position_ids = self._trim_le_tokens(
                le_token_mask=le_token_mask,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            if captured_hidden is not None:
                original_keep = ~le_token_mask
                new_len = input_ids.shape[1]
                if captured_cache is past_key_values:
                    captured_cache = past_key_values
                else:
                    captured_cache = self._trim_cache_by_keep(captured_cache, original_keep, new_len)
                if captured_hidden.shape[1] != input_ids.shape[1]:
                    captured_hidden = self._trim_tensor_by_keep(captured_hidden, original_keep)
            bsz, seq_len = input_ids.shape

        if delay_selection:
            # Training-style return: don't actually reduce. Trainer reads image_token_mask_logits.
            self.todo_selection = True
            return InternVL2_5_GPCausalLMOutputWithPast(
                loss=le_loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                le_loss=le_loss,
                image_token_mask_logits=image_token_mask_logits,
                image_token_bool_masks=image_token_bool_masks,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        if not do_selection or image_token_bool_masks is None or run_full_for_loss:
            # Either no selection requested, or we've fully forwarded for SFT loss; return as-is.
            return InternVL2_5_GPCausalLMOutputWithPast(
                loss=le_loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                le_loss=le_loss,
                image_token_mask_logits=image_token_mask_logits,
                image_token_bool_masks=image_token_bool_masks,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        # Inference path: reduce + finish remaining layers.
        if captured_hidden is None:
            captured_hidden = hidden_states
            captured_cache = past_key_values
        reduced = self._reduce_state(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            hidden_states=captured_hidden,
            past_key_values=captured_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_token_bool_masks=image_token_bool_masks,
        )

        if reduce_layer < n_layers - 1:
            cont_hidden, cont_cache = self._continue_after_reduction(
                hidden_states=reduced["hidden_states"],
                attention_mask=reduced["attention_mask"],
                position_ids=reduced["position_ids"],
                past_key_values=reduced["past_key_values"],
                use_cache=use_cache,
                start_layer=reduce_layer + 1,
            )
        else:
            cont_hidden = self._final_norm()(reduced["hidden_states"])
            cont_cache = reduced["past_key_values"]

        cont_logits = self._output_head()(cont_hidden)

        return InternVL2_5_GPCausalLMOutputWithPast(
            loss=le_loss,
            logits=cont_logits,
            past_key_values=cont_cache,
            hidden_states=cont_hidden,
            le_loss=le_loss,
            image_token_mask_logits=image_token_mask_logits,
            image_token_bool_masks=image_token_bool_masks,
            input_ids=reduced["input_ids"],
            inputs_embeds=reduced["inputs_embeds"],
            attention_mask=reduced["attention_mask"],
            position_ids=reduced["position_ids"],
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        num_patches_list=None,
        ref_token_masks=None,
        labels=None,
        inputs_embeds=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        do_selection=True,
        delay_selection=False,
        use_ref_masks=None,
        image_token_mask_logits=None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape, dtype=torch.long)
        if inputs_embeds is None:
            inputs_embeds, visual_token_embeds = self._build_inputs_embeds(input_ids, pixel_values=pixel_values)
        else:
            visual_token_embeds = None
        actual_use_ref_masks = self.config.use_ref_masks if use_ref_masks is None else use_ref_masks
        return self._glimpse_forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            visual_token_embeds=visual_token_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            num_patches_list=num_patches_list,
            ref_token_masks=ref_token_masks,
            image_token_mask_logits=image_token_mask_logits,
            actual_use_ref_masks=actual_use_ref_masks,
            delay_selection=delay_selection,
            do_selection=do_selection,
        )

    def _sample_next(self, logits, generation_config):
        do_sample = bool(getattr(generation_config, "do_sample", False))
        temperature = float(getattr(generation_config, "temperature", 1.0) or 1.0)
        if do_sample and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        num_patches_list=None,
        ref_token_masks=None,
        image_token_mask_logits=None,
        generation_config=None,
        do_selection=True,
        use_ref_masks=None,
        max_new_tokens=None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape, dtype=torch.long)
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(self.language_model.config)
        max_new = max_new_tokens
        if max_new is None:
            max_new = getattr(generation_config, "max_new_tokens", None) or getattr(generation_config, "max_length", None) or 256
        eos_token_id = kwargs.get("eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(generation_config, "eos_token_id", None)
        pad_token_id = getattr(generation_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id if isinstance(eos_token_id, int) else 0

        # Prefill with mid-network reduction (or trivial if do_selection=False).
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            ref_token_masks=ref_token_masks,
            image_token_mask_logits=image_token_mask_logits,
            do_selection=do_selection,
            use_ref_masks=use_ref_masks,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        prefix_ids = outputs.input_ids
        running_attn = outputs.attention_mask
        prefix_position_ids = outputs.position_ids
        bsz = prefix_ids.shape[0]
        device = prefix_ids.device
        next_position_ids = None
        if prefix_position_ids is not None:
            next_position_ids = prefix_position_ids[:, -1:] + 1

        # First sampled token from prefill last-position logits.
        first_logits = outputs.logits[:, -1, :]
        next_token = self._sample_next(first_logits, generation_config)
        generated = [next_token]
        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        if isinstance(eos_token_id, int):
            finished |= (next_token == eos_token_id)

        for _ in range(int(max_new) - 1):
            if finished.all():
                break
            running_attn = torch.cat(
                [running_attn, running_attn.new_ones((bsz, 1))], dim=1
            )
            step = self.language_model(
                input_ids=next_token.unsqueeze(-1),
                attention_mask=running_attn,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_logits = step.logits[:, -1, :]
            next_token = self._sample_next(next_logits, generation_config)
            if isinstance(eos_token_id, int):
                next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
                finished |= (next_token == eos_token_id)
            generated.append(next_token)
            past_key_values = step.past_key_values
            if next_position_ids is not None:
                next_position_ids = next_position_ids + 1

        gen_ids = torch.stack(generated, dim=1)
        return torch.cat([prefix_ids, gen_ids], dim=1)


__all__ = [
    "InternVL2_5_GPCausalLMOutputWithPast",
    "InternVL2_5_GP_ForConditionalGeneration",
]
