from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)

class Qwen2_5_VL_GPConfig(Qwen2_5_VLConfig):
    model_type = "qwen2_5_vl_gp"

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=8192,
        intermediate_size=29568,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=80,
        attention_dropout=0.0,
        vision_config=None,
        rope_scaling=None,
        selected_layers=(),
        use_attention_logits=False,
        attn_fuse_size=256,
        selected_visual_layers=(8,),
        visual_cond_size=256,
        attn_fuse_type="AttnFuserV1",
        attn_fuse_num_heads=4,
        attn_fuse_hidden_act="silu",
        attn_fuse_global=False,
        ori_attn_supervision=True,
        deep_supervision=True,
        le_layers=(0,),
        le_length=1,
        le_dropout_prob=0.0,
        le_norm_type="rmsnorm",
        reduce_threshold=0.5,
        use_ref_masks=False,
        use_zero_masks=False,
        reduce_layer=1000,
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
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            attention_dropout=attention_dropout,
            vision_config=vision_config,
            rope_scaling=rope_scaling,
            **kwargs,
        )
        self.selected_layers = selected_layers
        self.use_attention_logits = use_attention_logits
        self.attn_fuse_type = attn_fuse_type
        self.attn_fuse_size = attn_fuse_size
        self.attn_fuse_global = attn_fuse_global
        self.selected_visual_layers = selected_visual_layers
        self.visual_cond_size = visual_cond_size
        self.attn_fuse_num_heads = attn_fuse_num_heads
        self.attn_fuse_hidden_act = attn_fuse_hidden_act
        self.ori_attn_supervision = ori_attn_supervision
        self.deep_supervision = deep_supervision
        self.le_layers = le_layers
        self.le_length = le_length
        self.le_dropout_prob = le_dropout_prob
        self.le_norm_type = le_norm_type
        self.reduce_threshold = reduce_threshold
        self.use_ref_masks = use_ref_masks
        self.use_zero_masks = use_zero_masks
        self.reduce_layer = reduce_layer
        self.anchor_positions = anchor_positions
        self.min_remain_num = min_remain_num
        self.max_remain_ratio = max_remain_ratio
        
__all__ = [
    "Qwen2_5_VL_GPConfig",
]