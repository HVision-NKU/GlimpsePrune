import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
processor.tokenizer.padding_side = "left"

# --- Two separate conversations, each with one video ---
messages_batch = [
    # Sample 1
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "examples/space_woaudio.mp4",
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ],
    # Sample 2
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "examples/sample_video.mp4",
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": "What is happening in this video?"},
            ],
        }
    ],
]

# Process each sample
texts, all_image_inputs, all_video_inputs = [], [], []
all_video_kwargs_list = []

for msgs in messages_batch:
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(msgs, return_video_kwargs=True)
    texts.append(text)
    all_image_inputs.append(image_inputs)
    all_video_inputs.append(video_inputs)
    all_video_kwargs_list.append(video_kwargs)

# Merge video_kwargs: list values get concatenated, scalar values must agree
merged_video_kwargs = {}
if all_video_kwargs_list:
    for key in all_video_kwargs_list[0]:
        vals = [vk[key] for vk in all_video_kwargs_list]
        if isinstance(vals[0], list):
            merged_video_kwargs[key] = [item for v in vals for item in v]
        else:
            merged_video_kwargs[key] = vals[0]

# Flatten image/video inputs for the processor
flat_images = [img for imgs in all_image_inputs if imgs for img in imgs]
flat_videos = [vid for vids in all_video_inputs if vids for vid in vids]

inputs = processor(
    text=texts,
    images=flat_images if flat_images else None,
    videos=flat_videos if flat_videos else None,
    padding=True,
    return_tensors="pt",
    **merged_video_kwargs,
)
inputs = inputs.to("cuda")

print("input_ids shape:", inputs.input_ids.shape)

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

for i, text in enumerate(output_text):
    print(f"\n=== Sample {i+1} ===")
    print(text)
