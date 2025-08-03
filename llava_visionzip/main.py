from .utils import CLIP_EncoderLayer_forward, CLIPAttention_forward, apply_info
from .clip_encoder import CLIPVisionTower_VisionZip
from .llava_arch import prepare_inputs_labels_for_multimodal_visionzip, encode_images_visionzip, encode_images_visionzip_multi, restore_image_features_sorted

def visionzip(model, dominant=191, contextual=30):

    apply_info(model.model.vision_tower.vision_tower, dominant_num=dominant-1, contextual_num=contextual)


    from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention

    CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_VisionZip.forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip


    return model
