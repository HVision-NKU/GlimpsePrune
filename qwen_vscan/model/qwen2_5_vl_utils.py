import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union, List

from flash_attn.layers.rotary import apply_rotary_emb  # noqa

import time

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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    
    cos = cos.to(q.device)
    sin = sin.to(q.device)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def token_merging(image_embeds, keep_indices, scaling=1):
    """
    Merges non-retained tokens with their nearest retained tokens based on cosine similarity.

    Args:
        image_embeds (Tensor): Tensor of shape (N, D) where N is the number of tokens, and D is the feature dimension.
        keep_indices (Tensor): Tensor of shape (T, ), where T is the number of retained tokens.

    Returns:
        merged_features (Tensor): Tensor of shape (T, D) where T is the number of retained tokens
                                and D is the feature dimension. The merged features are
                                the average of the retained token and the non-retained tokens.
    """
    N, D = image_embeds.shape
    T = len(keep_indices)
    
    keep_index_mask = torch.zeros(N, dtype=torch.bool, device=image_embeds.device)
    keep_index_mask[keep_indices] = True
    
    retained_tokens = image_embeds[keep_index_mask, :] # [T, D]
    non_retained_tokens = image_embeds[~keep_index_mask, :] # [N - T, D]
    # print(retained_tokens.shape, non_retained_tokens.shape)
    
    cosine_sim = F.cosine_similarity(non_retained_tokens.unsqueeze(1), retained_tokens.unsqueeze(0), dim=2)
    nearest_token_indices = cosine_sim.argmax(dim=1) # [N - T]
    # print(nearest_token_indices)
    
    merged_features = torch.zeros_like(retained_tokens) # [T, D]
    merged_features += retained_tokens * scaling
    
    expanded_indices = nearest_token_indices # [N - T]
    merged_features.scatter_add_(0, expanded_indices.unsqueeze(-1).expand(-1, D), non_retained_tokens)
    
    merge_count = torch.zeros(T, device=image_embeds.device, dtype=torch.int) # [T]
    merge_count.scatter_add_(0, expanded_indices, torch.ones_like(expanded_indices, dtype=merge_count.dtype))
    merged_features /= (scaling + merge_count.unsqueeze(1))
    
    return merged_features

def window_selection(attn_weights, num_keep_tokens, image_grid_thw, window_size=4):
    """
    Selects the top num_keep_tokens tokens with the highest attention weights. 
    The tokens are selected from non-overlapping windows of size window_size x window_size.
    Args:
        attn_weights (Tensor): Tensor of shape (N, ) where N is the number of tokens.
        num_keep_tokens (int): The number of tokens to keep.
        window_size (int): The size of the window to select the tokens from.
    """
    # start_time = time.time()
    
    token_h, token_w = image_grid_thw[0, 1] // 2, image_grid_thw[0, 2] // 2
    assert token_h * token_w == attn_weights.shape[0], "The number of tokens in the window is not equal to num_keep_tokens"
    
    num_windows_h = (token_h / window_size).floor().int()
    num_windows_w = (token_w / window_size).floor().int()
    num_windows = num_windows_h * num_windows_w
    extra_h = token_h - (num_windows_h * window_size)
    extra_w = token_w - (num_windows_w * window_size)

    # attn_weights = attn_weights.view(token_h, token_w)
    # if extra_h > 0:
    #     attn_weights = attn_weights[:num_windows_h * window_size, :]
    # if extra_w > 0:
    #     attn_weights = attn_weights[:, :num_windows_w * window_size]
    # attn_weights = attn_weights.view(num_windows_h, window_size, num_windows_w, window_size)
    # attn_weights = attn_weights.permute(0, 2, 1, 3).reshape(num_windows_h * num_windows_w, window_size * window_size)

    sorted_indices = torch.argsort(attn_weights, dim=0, descending=True)
    window_counter = torch.zeros(token_h, token_w, device=attn_weights.device, dtype=torch.int)
    total_counter = 0
    
    # Calculate the limit of the number of tokens to keep in each window
    limit = (num_keep_tokens / num_windows).ceil().int()
    

    keep_indices = torch.zeros(num_keep_tokens, device=attn_weights.device, dtype=torch.int)
    for index in sorted_indices:
        x = (index // token_w) // window_size
        y = (index % token_w) // window_size
        if x == num_windows_h:
            x = num_windows_h - 1
        if y == num_windows_w:
            y = num_windows_w - 1
        x = x.int()
        y = y.int()
        if window_counter[x, y] < limit:
            window_counter[x, y] += 1
            keep_indices[total_counter] = index
            total_counter += 1
        if total_counter >= num_keep_tokens:
            break
        
    # end_time = time.time()
    # print(f"Window selection time: {end_time - start_time:.4f} seconds")

    assert total_counter == num_keep_tokens, "The number of tokens to keep is not equal to num_keep_tokens"
    return keep_indices
    
    
    
    
    
    
    
