from transformers import PretrainedConfig


class InternVL2_5_GPConfig(PretrainedConfig):
    model_type = "internvl2_5_gp"

    def __init__(
        self,
        selected_layers=(21,),
        reduce_layer=21,
        selected_visual_layers=(-1,),
        use_attention_logits=False,
        attn_fuse_size=256,
        visual_cond_size=512,
        attn_fuse_type="AttnFuserV1",
        attn_fuse_num_heads=4,
        attn_fuse_hidden_act="silu",
        attn_fuse_global=False,
        attn_fuse_use_flash_attn=False,
        ori_attn_supervision=False,
        deep_supervision=False,
        le_layers=tuple(range(18)),
        le_length=1,
        le_dropout_prob=0.0,
        le_norm_type="rmsnorm",
        reduce_threshold=0.5,
        use_ref_masks=False,
        use_zero_masks=False,
        min_remain_num=1,
        max_remain_ratio=None,
        fixed_remain_ratio=None,
        vision_config=None,
        llm_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.selected_layers = tuple(selected_layers)
        self.reduce_layer = reduce_layer
        self.selected_visual_layers = tuple(selected_visual_layers)
        self.use_attention_logits = use_attention_logits
        self.attn_fuse_size = attn_fuse_size
        self.visual_cond_size = visual_cond_size
        self.attn_fuse_type = attn_fuse_type
        self.attn_fuse_num_heads = attn_fuse_num_heads
        self.attn_fuse_hidden_act = attn_fuse_hidden_act
        self.attn_fuse_global = attn_fuse_global
        self.attn_fuse_use_flash_attn = attn_fuse_use_flash_attn
        self.ori_attn_supervision = ori_attn_supervision
        self.deep_supervision = deep_supervision
        self.le_layers = tuple(le_layers)
        self.le_length = le_length
        self.le_dropout_prob = le_dropout_prob
        self.le_norm_type = le_norm_type
        self.reduce_threshold = reduce_threshold
        self.use_ref_masks = use_ref_masks
        self.use_zero_masks = use_zero_masks
        self.min_remain_num = min_remain_num
        self.max_remain_ratio = max_remain_ratio
        self.fixed_remain_ratio = fixed_remain_ratio
        self.vision_config = vision_config
        self.llm_config = llm_config

    def to_dict(self):
        output = super().to_dict()
        vision_config = output.get("vision_config")
        if hasattr(vision_config, "__dict__"):
            output["vision_config"] = dict(vision_config.__dict__)
        llm_config = output.get("llm_config")
        if hasattr(llm_config, "to_dict"):
            output["llm_config"] = llm_config.to_dict()
        elif hasattr(llm_config, "__dict__"):
            output["llm_config"] = dict(llm_config.__dict__)
        return output


__all__ = [
    "InternVL2_5_GPConfig",
]
