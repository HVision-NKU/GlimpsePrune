import os
import argparse
import datasets
import gradio as gr
from PIL import Image, ImageDraw # Import ImageDraw
import ast # For ast.literal_eval

# --- Global variables to store data and args ---
ref_generate_dataset_global = None
generate_dataset_global = None
glimpse_dataset_global = None # This might not be strictly needed if glimpse info (iou, ratio) is in compare_gen.jsonl
                              # and glimpse images are separate. For now, keeping it if it holds iou/ratio.
filtered_indices_global = []
args_global = None

# --- Modified data mapper ---
def cot_bench_generate_dataset_mapper(one_data, args_for_mapper):
    query = one_data["conversations"][0]["value"].replace("Please provide the bounding box coordinate of the region that can help you answer the question better.", "").strip()
    query = query.replace("<image>\n", "")
    answer = one_data['conversations'][-1]['value']
    img_path = os.path.join(args_for_mapper.img_dir, one_data["image"][0])
    one_data["query"] = query
    one_data["answer"] = answer
    one_data["img_path"] = img_path
    
    if args_for_mapper.use_box:
        try:
            # Assuming the bbox string is in one_data["image"][1]
            # Example: "some_other_info###[10,20,100,120]"
            bbox_str_part = one_data["image"][1].split('###')[-1] # Get the part after ###
            bbox = ast.literal_eval(bbox_str_part)
            if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
                one_data["bbox"] = bbox   # xyxy
            else:
                print(f"Warning: Parsed bbox is not a valid list of 4 coordinates: {bbox} from {one_data['image'][1]}")
                one_data["bbox"] = None
        except (IndexError, SyntaxError, ValueError, TypeError) as e:
            print(f"Warning: Could not parse bbox from '{one_data.get('image', [None, ''])[1]}': {e}")
            one_data["bbox"] = None # Ensure bbox key exists but is None if parsing fails
    return one_data


def vstar_bench_generate_dataset_mapper(one_data, args_for_mapper):
    query = one_data.get("query", one_data["text"])
    answer = one_data["label"]
    img_path = os.path.join("vstar_bench", one_data["image"])
    one_data["query"] = query
    one_data["answer"] = answer
    one_data["img_path"] = img_path
    if args_for_mapper.use_box:
        pass
    return one_data

    

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Gradio Demo for COT Benchmark Comparison")
    parser.add_argument("--ref_generate_path", type=str, default="results/compare_ref_gen.jsonl", help="Path to the reference generation file")
    parser.add_argument("--generate_path", type=str, default="results/compare_gen.jsonl", help="Path to the generation file")
    parser.add_argument("--glimpse_path", type=str, default=None, help="Optional: Path to the glimpse file (for iou/ratio)")
    parser.add_argument("--img_dir", type=str, default="datas", help="Path to the image directory")
    
    # New arguments
    parser.add_argument("--use_box", action="store_true", help="If set, tries to parse and draw bounding box from data.")
    parser.add_argument("--glimpse_dir", type=str, default=None, help="Directory containing glimpse mask PNGs (e.g., 0.png, 1.png).")
    parser.add_argument("--color", type=str, default="#00FF00", help="Color for the glimpse mask overlay (default: green).")
    parser.add_argument("--alpha", type=float, default=0.4, help="Alpha value for the glimpse mask overlay (default: 0.4).")

    parser.add_argument("--filter_gen_gt_ref", action="store_true", help="Filter for gen_score > ref_score")
    parser.add_argument("--filter_gen_lt_ref", action="store_true", help="Filter for gen_score < ref_score")
    parser.add_argument("--filter_iou_lt", type=float, default=None, help="Filter for iou < threshold")
    parser.add_argument("--filter_ratio_lt", type=float, default=None, help="Filter for ratio < threshold")
    parser.add_argument("--filter_logic", type=str, default="ALL_ACTIVE",
                        help="Logic for combining filters (e.g., 'F_GEN_GT_REF AND (F_IOU_LT OR NOT F_RATIO_LT)'). "
                             "Use F_GEN_GT_REF, F_GEN_LT_REF, F_IOU_LT, F_RATIO_LT as placeholders. "
                             "'ALL_ACTIVE' means AND all active filters. 'ANY_ACTIVE' means OR all active filters. "
                             "Default is ALL_ACTIVE.")
    
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Gradio server name")
    parser.add_argument("--server_port", type=int, default=7860, help="Gradio server port")

    return parser.parse_args()

# --- Data Loading and Filtering (largely unchanged, mapper will handle new fields) ---
def load_and_filter_data(args):
    global ref_generate_dataset_global, generate_dataset_global, glimpse_dataset_global, filtered_indices_global, args_global
    args_global = args

    print("Loading datasets...")
    assert os.path.isfile(args.ref_generate_path)
    assert os.path.isfile(args.generate_path)
    try:
        ref_generate_dataset_global = datasets.load_dataset("json", data_files=args.ref_generate_path, split="train")
        generate_dataset_global = datasets.load_dataset("json", data_files=args.generate_path, split="train")
        if args.glimpse_path and os.path.isfile(args.glimpse_path): # For IoU/Ratio data
            glimpse_dataset_global = datasets.load_dataset("json", data_files=args.glimpse_path, split="train")
            print(f"Loaded glimpse dataset (for iou/ratio) with {len(glimpse_dataset_global)} items.")
        else:
            glimpse_dataset_global = None
            print("No glimpse dataset path (for iou/ratio) provided or file not found.")
    except Exception as e:
        print(f"Error loading dataset files: {e}")
        return False

    print("Mapping datasets...")
    # The mapper now also processes 'use_box'
    if 'vstar' in args.generate_path:
        mapper_func = vstar_bench_generate_dataset_mapper
    else:
        mapper_func = cot_bench_generate_dataset_mapper
    ref_generate_dataset_global = ref_generate_dataset_global.map(mapper_func, fn_kwargs={"args_for_mapper": args})
    generate_dataset_global = generate_dataset_global.map(mapper_func, fn_kwargs={"args_for_mapper": args})
    if glimpse_dataset_global: # If separate file for IoU/Ratio
        glimpse_dataset_global = glimpse_dataset_global.map(mapper_func, fn_kwargs={"args_for_mapper": args})


    assert len(ref_generate_dataset_global) == len(generate_dataset_global), \
        f"Mismatch: ref_generate {len(ref_generate_dataset_global)}, generate {len(generate_dataset_global)}"
    if glimpse_dataset_global:
        assert len(ref_generate_dataset_global) == len(glimpse_dataset_global), \
            f"Mismatch: ref_generate {len(ref_generate_dataset_global)}, glimpse {len(glimpse_dataset_global)}"

    print(f"Total items before filtering: {len(ref_generate_dataset_global)}")
    # Filtering logic (remains the same)
    filtered_indices_global = []
    for i in range(len(ref_generate_dataset_global)):
        one_ref = ref_generate_dataset_global[i]
        one_gen = generate_dataset_global[i]
        # Determine which item to get iou/ratio from.
        # If glimpse_dataset_global exists, use it. Otherwise, try to get from one_gen.
        item_for_glimpse_metrics = {}
        if glimpse_dataset_global and i < len(glimpse_dataset_global):
            item_for_glimpse_metrics = glimpse_dataset_global[i]
        elif 'iou' in one_gen or 'ratio' in one_gen: # Check if gen data itself has these
             item_for_glimpse_metrics = one_gen


        conditions_met = {}
        active_filter_flags = []

        if args.filter_gen_gt_ref:
            active_filter_flags.append("F_GEN_GT_REF")
            try:
                conditions_met["F_GEN_GT_REF"] = float(one_gen.get("score", 0)) > float(one_ref.get("score", 0))
            except (ValueError, TypeError): conditions_met["F_GEN_GT_REF"] = False
        elif args.filter_gen_lt_ref: # If gen < ref filter is active
            active_filter_flags.append("F_GEN_LT_REF")
            try:
                conditions_met["F_GEN_LT_REF"] = float(one_gen.get("score", 0)) < float(one_ref.get("score", 0))
            except (ValueError, TypeError): conditions_met["F_GEN_LT_REF"] = False
        

        if args.filter_iou_lt is not None: # Check if IoU filter is active
            active_filter_flags.append("F_IOU_LT")
            try:
                conditions_met["F_IOU_LT"] = float(item_for_glimpse_metrics.get("iou", float('inf'))) < args.filter_iou_lt
            except (ValueError, TypeError): conditions_met["F_IOU_LT"] = False
        elif "F_IOU_LT" in args.filter_logic: conditions_met["F_IOU_LT"] = True


        if args.filter_ratio_lt is not None: # Check if Ratio filter is active
            active_filter_flags.append("F_RATIO_LT")
            try:
                conditions_met["F_RATIO_LT"] = float(item_for_glimpse_metrics.get("ratio", float('inf'))) < args.filter_ratio_lt
            except (ValueError, TypeError): conditions_met["F_RATIO_LT"] = False
        elif "F_RATIO_LT" in args.filter_logic: conditions_met["F_RATIO_LT"] = True
        
        passes_filter = True 
        # ... (rest of the filtering logic is identical to previous version)
        if not active_filter_flags and args.filter_logic not in ["ALL_ACTIVE", "ANY_ACTIVE"]:
             eval_logic_str = args.filter_logic
             for p in ["F_GEN_GT_REF", "F_IOU_LT", "F_RATIO_LT", "F_GEN_LT_REF"]:
                 if p not in conditions_met: conditions_met[p] = True
                 eval_logic_str = eval_logic_str.replace(p, str(conditions_met[p]))
             eval_logic_str = eval_logic_str.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
             try:
                 if eval_logic_str.strip() and not all(c in "TrueFalseandornot() " for c in eval_logic_str): passes_filter = eval(eval_logic_str)
                 elif not eval_logic_str.strip(): passes_filter = True
             except Exception as e:
                 print(f"Warning: Error evaluating filter logic '{args.filter_logic}' -> '{eval_logic_str}': {e}. Item {i} excluded."); passes_filter = False
        elif active_filter_flags:
            if args.filter_logic == "ALL_ACTIVE": passes_filter = all(conditions_met[flag] for flag in active_filter_flags)
            elif args.filter_logic == "ANY_ACTIVE": passes_filter = any(conditions_met[flag] for flag in active_filter_flags)
            else:
                eval_logic_str = args.filter_logic
                for flag in ["F_GEN_GT_REF", "F_IOU_LT", "F_RATIO_LT", "F_GEN_LT_REF"]: eval_logic_str = eval_logic_str.replace(flag, str(conditions_met.get(flag, True)))
                eval_logic_str = eval_logic_str.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
                try: passes_filter = eval(eval_logic_str)
                except Exception as e: print(f"Warning: Error evaluating filter logic '{args.filter_logic}' -> '{eval_logic_str}': {e}. Item {i} excluded."); passes_filter = False
        
        if passes_filter:
            filtered_indices_global.append(i)
    
    num_filtered = len(filtered_indices_global)
    print(f"Filtering complete. {num_filtered} items match the criteria.")
    if num_filtered == 0: print("Warning: No items match the filter criteria. The UI will be empty.")
    return True


# --- Gradio UI Helper Functions (Major Update) ---
def get_data_for_ui_index(ui_idx):
    # Updated default outputs: main_img, query, gt_ans, ref_resp, ref_score, gen_resp, gen_score, gen_iou, gen_ratio, glimpse_overlay_img, status
    default_outputs = [None, "N/A", "N/A", "N/A", None, "N/A", None, None, None, None, "No data loaded or matches filters."]
    
    if not filtered_indices_global or ui_idx < 0 or ui_idx >= len(filtered_indices_global):
        return default_outputs

    actual_idx = filtered_indices_global[ui_idx] # This is the original index in the dataset
    
    ref_item = ref_generate_dataset_global[actual_idx]
    gen_item = generate_dataset_global[actual_idx]
    
    # Determine item for glimpse metrics (iou, ratio)
    item_for_glimpse_metrics = {}
    if glimpse_dataset_global and actual_idx < len(glimpse_dataset_global):
        item_for_glimpse_metrics = glimpse_dataset_global[actual_idx]
    elif 'iou' in gen_item or 'ratio' in gen_item:
        item_for_glimpse_metrics = gen_item


    # --- Main Image Loading and Optional BBox Drawing ---
    pil_image = None
    img_path = ref_item.get("img_path")
    if img_path and os.path.exists(img_path):
        try:
            pil_image = Image.open(img_path).convert("RGB")
            if args_global.use_box and ref_item.get("bbox"):
                # Draw bounding box
                # Make a copy to draw on, so the original pil_image can be used for glimpse overlay
                img_with_box = pil_image.copy() 
                draw = ImageDraw.Draw(img_with_box)
                bbox = ref_item["bbox"]
                draw.rectangle(bbox, outline="red", width=3)
                pil_image = img_with_box # pil_image now refers to the image with the box
        except Exception as e:
            print(f"Warning: Could not open/process image at {img_path}: {e}")
            pil_image = None 
    elif img_path:
        print(f"Warning: Image not found at {img_path}")

    # --- Glimpse Mask Overlay ---
    glimpse_overlay_image = None
    if args_global.glimpse_dir and pil_image: # Only proceed if glimpse_dir is set AND main image loaded
        mask_filename = f"{actual_idx}.png" # Assumes naming based on original dataset index
        mask_path = os.path.join(args_global.glimpse_dir, mask_filename)
        
        if os.path.exists(mask_path):
            try:
                glimpse_mask = Image.open(mask_path).convert('L') # Ensure grayscale
                
                # Resize mask to original image dimensions
                resized_mask = glimpse_mask.resize(pil_image.size, Image.NEAREST)
                
                # Create a colored overlay (e.g., semi-transparent green)
                # Base image needs to be RGBA for alpha_composite
                base_rgba_image = pil_image.copy().convert("RGBA") if pil_image.mode != "RGBA" else pil_image.copy()

                # Create overlay layer
                # convert hex color to RGBA
                mask_color = Image.new("RGB", (1, 1), args_global.color)
                mask_color = mask_color.getpixel((0, 0)) # Get the RGBA tuple from the single pixel
                alpha = int(args_global.alpha * 255) # Convert alpha from 0-1 to 0-255
                
                # The resized_mask (L mode) acts as the alpha for the solid_color_for_mask
                # We need to ensure the mask is suitable to be an alpha channel.
                # If mask values are 0-255, they can be used directly.
                # Let's make a binary alpha mask from resized_mask for clarity:
                # if pixel > threshold, alpha = mask_color[3], else alpha = 0
                
                alpha_mask_for_overlay = Image.new('L', resized_mask.size)
                alpha_data = []
                for pixel_val in resized_mask.getdata():
                    if pixel_val > 128: # Threshold for mask being "on"
                        alpha_data.append(alpha) # Use specified alpha
                    else:
                        alpha_data.append(0) # Fully transparent
                alpha_mask_for_overlay.putdata(alpha_data)

                # Create the colored part of the overlay
                colored_mask_region = Image.new("RGBA", base_rgba_image.size)
                colored_mask_region_data = []
                for alpha_val in alpha_mask_for_overlay.getdata():
                    if alpha_val > 0:
                        colored_mask_region_data.append((mask_color[0], mask_color[1], mask_color[2], alpha_val))
                    else:
                        colored_mask_region_data.append((0,0,0,0))
                colored_mask_region.putdata(colored_mask_region_data)
                
                # Composite the colored mask region onto the base image
                glimpse_overlay_image = Image.alpha_composite(base_rgba_image, colored_mask_region)

            except Exception as e:
                print(f"Warning: Could not load or process glimpse mask {mask_path}: {e}")
                glimpse_overlay_image = None # Fallback to None
        else:
            print(f"Warning: Glimpse mask not found: {mask_path}")
            glimpse_overlay_image = pil_image.copy() if pil_image else None # Show original if mask missing but dir specified

    elif args_global.glimpse_dir and not pil_image:
        print(f"Warning: Main image for item {actual_idx} not loaded, cannot create glimpse overlay.")
        glimpse_overlay_image = None


    query = ref_item.get("query", "N/A")
    gt_answer = ref_item.get("answer", "N/A")

    ref_response = ref_item.get("response", ref_item.get("value", "N/A"))
    ref_score = ref_item.get("score")
    try: ref_score = float(ref_score) if ref_score is not None else None
    except: ref_score = None

    gen_response = gen_item.get("response", gen_item.get("value", "N/A"))
    gen_score = gen_item.get("score")
    try: gen_score = float(gen_score) if gen_score is not None else None
    except: gen_score = None
    
    gen_iou = None
    gen_ratio = None
    if item_for_glimpse_metrics:
        gen_iou = item_for_glimpse_metrics.get("iou")
        try: gen_iou = float(gen_iou) if gen_iou is not None else None
        except: gen_iou = None
        
        gen_ratio = item_for_glimpse_metrics.get("ratio")
        try: gen_ratio = float(gen_ratio) if gen_ratio is not None else None
        except: gen_ratio = None

    status_text = f"Displaying item {ui_idx + 1} of {len(filtered_indices_global)} (Original Index: {actual_idx})"
    
    return pil_image, query, gt_answer, ref_response, ref_score, gen_response, gen_score, gen_iou, gen_ratio, glimpse_overlay_image, status_text


def update_ui(ui_idx):
    return get_data_for_ui_index(ui_idx)

def next_item(current_ui_idx):
    if not filtered_indices_global: return 0 
    new_idx = (current_ui_idx + 1) % len(filtered_indices_global)
    return new_idx

def prev_item(current_ui_idx):
    if not filtered_indices_global: return 0
    new_idx = (current_ui_idx - 1 + len(filtered_indices_global)) % len(filtered_indices_global)
    return new_idx

def go_to_item_ui(target_ui_idx_str, current_ui_idx):
    if not filtered_indices_global: return 0
    try:
        target_ui_idx = int(target_ui_idx_str) -1 
        if 0 <= target_ui_idx < len(filtered_indices_global): return target_ui_idx
    except ValueError: pass 
    return current_ui_idx

# --- Main Gradio App ---
def main():
    cli_args = parse_args() # cli_args will be available here
    
    if not load_and_filter_data(cli_args): # args_global is set inside this
        print("Failed to load or filter data. Exiting.")
        return

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        current_ui_idx_state = gr.State(0) 

        gr.Markdown("# CoT Benchmark Comparison Tool")
        status_bar = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            # This will display the image, potentially with a bbox drawn on it
            img_display = gr.Image(label="Image / Image with BBox", type="pil", height=400, interactive=False)
            with gr.Column():
                query_display = gr.Textbox(label="Query", lines=3, interactive=False)
                gt_answer_display = gr.Textbox(label="Ground Truth Answer", lines=3, interactive=False)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Reference Model")
                ref_response_display = gr.Textbox(label="Reference Response", lines=6, interactive=False)
                ref_score_display = gr.Number(label="Reference Score", interactive=False)
            
            with gr.Column(scale=2): # Make generated column wider to accommodate glimpse image
                gr.Markdown("## Generated Model")
                with gr.Row():
                    with gr.Column(scale=1):
                        gen_response_display = gr.Textbox(label="Generated Response", lines=6, interactive=False)
                        gen_score_display = gr.Number(label="Generated Score", interactive=False)
                        # IoU/Ratio display logic (based on glimpse_path for metrics, not glimpse_dir for images)
                        if cli_args.glimpse_path or True: # Show even if no glimpse_path, will be None
                            gen_iou_display = gr.Number(label="Generated IoU", interactive=False)
                            gen_ratio_display = gr.Number(label="Generated Ratio", interactive=False)
                        else: # Fallback if you strictly want to hide them
                            gen_iou_display = gr.Number(label="Generated IoU (N/A)", interactive=False, visible=False)
                            gen_ratio_display = gr.Number(label="Generated Ratio (N/A)", interactive=False, visible=False)
                    
                    with gr.Column(scale=1):
                        # Glimpse Overlay Image, visible only if glimpse_dir is provided
                        glimpse_overlay_img_display = gr.Image(
                            label="Glimpse Mask Overlay", 
                            type="pil", 
                            height=300, # Adjust height as needed
                            interactive=False,
                            visible=bool(cli_args.glimpse_dir) # Control visibility
                        )
                        if not cli_args.glimpse_dir: # Create a dummy placeholder if not visible for output matching
                            glimpse_overlay_img_display = gr.Image(visible=False)


        with gr.Row():
            prev_btn = gr.Button("⬅️ Previous")
            next_btn = gr.Button("Next ➡️")
            with gr.Row():
                idx_input = gr.Textbox(label="Go to (1-based):", scale=1)
                go_btn = gr.Button("Go", scale=0)
        
        # Define outputs for update_ui - ensure order matches get_data_for_ui_index return
        outputs = [
            img_display, query_display, gt_answer_display,
            ref_response_display, ref_score_display,
            gen_response_display, gen_score_display,
            gen_iou_display, gen_ratio_display,
            glimpse_overlay_img_display, # New output
            status_bar
        ]

        prev_btn.click(prev_item, inputs=[current_ui_idx_state], outputs=[current_ui_idx_state])
        next_btn.click(next_item, inputs=[current_ui_idx_state], outputs=[current_ui_idx_state])
        go_btn.click(go_to_item_ui, inputs=[idx_input, current_ui_idx_state], outputs=[current_ui_idx_state])
        
        current_ui_idx_state.change(update_ui, inputs=[current_ui_idx_state], outputs=outputs)
        demo.load(lambda: update_ui(current_ui_idx_state.value), inputs=None, outputs=outputs)

    print(f"Launching Gradio app on {cli_args.server_name}:{cli_args.server_port}")
    demo.launch(share=cli_args.share, server_name=cli_args.server_name, server_port=cli_args.server_port)

if __name__ == "__main__":
    def create_dummy_data_extended():
        os.makedirs("results", exist_ok=True)
        os.makedirs("datas/dummy_images", exist_ok=True)
        os.makedirs("datas/dummy_glimpses", exist_ok=True) # For glimpse masks

        # Create dummy images and glimpse masks
        for i in range(5):
            # Main image
            img = Image.new('RGB', (200, 150), color = (i*20, 100, 150-i*20))
            d = ImageDraw.Draw(img)
            d.text((10,10), f"Image {i}", fill=(255,255,0))
            img.save(f"datas/dummy_images/img_{i}.png")

            # Glimpse mask (smaller, grayscale)
            mask_size = (50, 37) # Smaller than main image
            glimpse_mask = Image.new('L', mask_size, color=0) # Black background
            d_mask = ImageDraw.Draw(glimpse_mask)
            if i % 2 == 0: # Draw something on even masks
                 d_mask.ellipse([5,5, mask_size[0]-5, mask_size[1]-5], fill=200) # Gray ellipse
            else: # Draw a line on odd masks
                 d_mask.line([0,0, mask_size[0], mask_size[1]], fill=180, width=5)
            glimpse_mask.save(f"datas/dummy_glimpses/{i}.png")


        # JSONL data
        # image field: first element is path, second is "other_info###[bbox_coords_str]"
        # Example bbox: "[x1, y1, x2, y2]"
        bboxes_example = [
            "[10, 10, 80, 70]",
            "[20, 30, 120, 100]",
            "[5, 5, 50, 50]",
            "[0, 0, 199, 149]", # Full image
            "[40, 40, 150, 120]"
        ]

        with open("results/compare_ref_gen.jsonl", "w") as f:
            for i in range(5):
                data = {
                    "conversations": [
                        {"from": "human", "value": f"<image>\nThis is query {i} for reference."},
                        {"from": "gpt", "value": f"This is GT answer {i}."}
                    ],
                    "image": [f"dummy_images/img_{i}.png", f"info_ref_{i}###{bboxes_example[i]}"],
                    "id": f"ref_{i}", "response": f"Reference response {i}", "score": i * 0.2 + 0.5
                }
                f.write(str(data).replace("'", "\"") + "\n")

        with open("results/compare_gen.jsonl", "w") as f:
            for i in range(5):
                data = {
                     "conversations": [
                        {"from": "human", "value": f"<image>\nThis is query {i} for generation."}, # Query might be same or different
                        {"from": "gpt", "value": f"This is GT answer {i}."} # GT Answer should be consistent
                    ],
                    "image": [f"dummy_images/img_{i}.png", f"info_gen_{i}###{bboxes_example[i]}"], # BBox info might be same or from this file
                    "id": f"gen_{i}", "response": f"Generated response {i}", "score": (4-i) * 0.2 + 0.4,
                    "iou": i * 0.1 + 0.3, "ratio": (4-i) * 0.1 + 0.2 # Adding iou/ratio here directly
                }
                f.write(str(data).replace("'", "\"") + "\n")
        
        # No separate compare_glimpse.jsonl in this dummy setup, iou/ratio are in compare_gen.jsonl
        # If you have a separate one, create it.
        
        print("Dummy data created in 'results/', 'datas/dummy_images/', and 'datas/dummy_glimpses/'.")
        print("You can run the script now. Example usages:")
        print("python your_script_name.py")
        print("python your_script_name.py --use_box")
        print("python your_script_name.py --glimpse_dir datas/dummy_glimpses")
        print("python your_script_name.py --use_box --glimpse_dir datas/dummy_glimpses --filter_iou_lt 0.4")
    
    # Uncomment to create dummy data
    # create_dummy_data_extended()
    
    main()