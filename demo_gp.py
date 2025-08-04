import gradio as gr
import torch
import argparse
from transformers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from threading import Thread
import traceback
import sys
from io import StringIO
from PIL import Image
import os

# Ensure transformers_gp is in the python path
# import sys
# sys.path.append('path/to/your/project')

from transformers_gp.models.qwen2_5_vl import (
    Qwen2_5_VL_GP_ForConditionalGeneration,
    Qwen2_5_VL_GP_Processor
)


def apply_mask_on_image(image: Image.Image,
                        mask: torch.Tensor,
                        alpha: float = 0.4,
                        color: tuple = (0, 255, 0),
                        patch_size: int = 28) -> Image.Image:
    """
    Apply a boolean mask on the image.
    """
    if image is None or mask is None:
        return None
        
    mask_h, mask_w = mask.shape
    target_h = mask_h * patch_size
    target_w = mask_w * patch_size
    
    # Resize image and mask
    image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    mask = torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=(target_h, target_w),
        mode='nearest'
    ).squeeze(0).squeeze(0).bool()

    # Apply mask with color and alpha
    mask_image = Image.new('RGBA', image.size, color + (0,))
    alpha_data = (mask * 255 * alpha).byte().cpu().numpy()
    alpha_channel = Image.fromarray(alpha_data, mode='L')
    mask_image.putalpha(alpha_channel)
    blended_image = Image.alpha_composite(image.convert('RGBA'), mask_image)
    return blended_image.convert('RGB')

# ---------------- Model and Processor Loading ----------------

parser = argparse.ArgumentParser(description="Qwen2.5-VL-GP Gradio Demo")
parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="Base model to load.")
parser.add_argument("--new_modules_dir", type=str, default="output/qwen2_5_7b_gp",
                    help="Directory containing new modules for GlimpsePrune.")
parser.add_argument("--share", action='store_true',
                    help="Share the Gradio app publicly.")
args = parser.parse_args()

BASE_MODEL = args.base_model
NEW_MODULES_DIR = args.new_modules_dir

print("Loading base model...")
model = Qwen2_5_VL_GP_ForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": "cuda:0"},
)

processor = Qwen2_5_VL_GP_Processor.from_pretrained(
    BASE_MODEL,
)

# Load GP-specific modules
print(f"Loading new modules from {NEW_MODULES_DIR}...")
model.load_new_modules(NEW_MODULES_DIR)


model.eval()
print("Model and processor loaded successfully.")

# ---------------- Core Gradio Demo Logic ----------------

def stream_chat_gp(image, question, temperature, top_p, max_new_tokens, enable_gp, show_mask, max_ratio, threshold):
    """
    A generator function for streaming chat responses with Qwen2.5VL-GP,
    including optional mask visualization and exception capturing.
    """
    # Initialize outputs
    generated_text = ""
    log_output = ""
    image_token_bool_masks = None
    masked_image_result = None

    # Check if inputs are valid
    if image is None:
        yield "Please upload an image first.", "Error: No image provided.", None
        return
    if not question or not question.strip():
        yield "Please enter a question.", "Error: No question provided.", None
        return

    log_stream = StringIO()
    original_stderr = sys.stderr
    sys.stderr = log_stream

    try:
        if isinstance(image, str):
            image = Image.open(image)
        image_pil = image.convert("RGB")
        
        # Set GP-specific parameters on the model
        model.config.max_remain_ratio = max_ratio
        model.config.reduce_threshold = threshold

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # --- Mask Generation (if requested) ---
        # Only generate and display the mask if GP is enabled and the user requests it.
        if enable_gp and show_mask:
            try:
                with torch.inference_mode():
                    # This forward pass calculates the token mask.
                    # do_selection must be True to get the mask.
                    outputs = model(**inputs, do_selection=True)
                    image_token_bool_masks = outputs.image_token_bool_masks
                    del outputs
                
                image_grid_thw = inputs.image_grid_thw
                attn_hw = image_grid_thw[0, 1:] // 2
                image_token_bool_mask = image_token_bool_masks[0].reshape(
                    attn_hw[0].item(), attn_hw[1].item()
                )
                
                # Create the visual mask image
                masked_image_result = apply_mask_on_image(image_pil.copy(), image_token_bool_mask)
            except Exception as e:
                log_output += f"Error during mask generation: {str(e)}\n"
            finally:
                torch.cuda.empty_cache()  # Clear GPU memory after mask generation


        # --- Text Generation ---
        model.reset_image_tokens_cache()  # Reset cache before generation
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            do_selection=enable_gp,  # Enable or disable Glimpse Prune (GP) based on the checkbox.
        )
        
        if image_token_bool_masks is not None:
            generation_kwargs['ref_token_masks'] = image_token_bool_masks
            generation_kwargs['use_ref_masks'] = True
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        yield generated_text, log_output + log_stream.getvalue(), masked_image_result

        for new_text in streamer:
            generated_text += new_text
            current_log = log_stream.getvalue()
            # yield generated_text, log_output + current_log, masked_image_result  # NOTE: return masked_image_result each time is slow!
            yield generated_text, log_output + current_log, gr.update()
            
    except Exception as e:
        tb_str = traceback.format_exc()
        final_log = log_stream.getvalue() + "\n" + tb_str
        yield generated_text, final_log, masked_image_result
    finally:
        sys.stderr = original_stderr
        final_log = log_stream.getvalue()
        torch.cuda.empty_cache()
        if final_log:
            yield generated_text, log_output + final_log, gr.update()

# ---------------- Gradio UI Layout ----------------

def update_mask_checkbox_visibility(enable_gp_status):
    """Update the mask checkbox based on the GlimpsePrune enabled status."""
    if not enable_gp_status:
        # When GP is disabled, also disable and uncheck the 'Show Mask' option.
        return gr.update(value=False, interactive=False)
    else:
        # When GP is enabled, re-enable the 'Show Mask' option.
        return gr.update(interactive=True)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1><center>Qwen2.5-VL-GP Demo</center></h1>")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            question_input = gr.Textbox(label="Enter Question", lines=2)
            
            with gr.Accordion("Generation Parameters", open=True):
                temperature_slider = gr.Slider(minimum=0, maximum=2, value=0.7, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(minimum=0, maximum=1, value=0.9, step=0.1, label="Top P")
                max_new_tokens_number = gr.Number(value=1024, label="Max New Tokens", precision=0)

            with gr.Accordion("GlimpsePrune Parameters", open=True):
                enable_gp_checkbox = gr.Checkbox(label="Enable GlimpsePrune", value=True)
                show_mask_checkbox = gr.Checkbox(label="Show Image Token Mask (Slower)", value=True)
                max_ratio_slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="Max Remain Ratio")
                mask_threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Reduce Threshold")
            
            run_button = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Model Output", interactive=False, lines=15)
            masked_image_output = gr.Image(type="pil", label="Image with Token Mask", interactive=False)
            log_output = gr.Textbox(label="Logs", interactive=False, lines=5)

    gr.Examples(
        examples=[
            ["examples/people.png", "What kind of a tie is the groom wearing?"],
            ["examples/food.png", "What are the red slices on the pizza?"],
            ["examples/cat.png", "What is above the cat?"],
            ["examples/shop.jpg", "From the information on that advertising board, what is the type of this shop?"],
            ["examples/street.jpg", "Based on the information of that white banner on the left, what activity is going to be held?"],
            ["examples/table.webp", "Where to buy a mug like this based on its logo?"],
            ["examples/court.png", "Tell me the number of that player who is shooting."],
        ],
        inputs=[image_input, question_input],
        label="Examples"
    )

    # Bind the change event of 'enable_gp_checkbox' to control 'show_mask_checkbox'.
    enable_gp_checkbox.change(
        fn=update_mask_checkbox_visibility,
        inputs=enable_gp_checkbox,
        outputs=show_mask_checkbox
    )

    # Adjusted the order of the 'inputs' list to correctly match the 'stream_chat_gp' function parameters.
    run_button.click(
        stream_chat_gp,
        inputs=[
            image_input, 
            question_input, 
            temperature_slider, 
            top_p_slider, 
            max_new_tokens_number,
            enable_gp_checkbox,
            show_mask_checkbox,
            max_ratio_slider,
            mask_threshold_slider,
        ],
        outputs=[output_text, log_output, masked_image_output]
    )

demo.launch(share=args.share)
