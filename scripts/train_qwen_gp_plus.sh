ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"

port=${PORT:-29606}

base_model="Qwen/Qwen2.5-VL-7B-Instruct"
config="train_configs/qwen2_5_7b_gp/qwen2_5_7b_gp_lora_ft.yaml"
adapter_dir="output/qwen2_5_7b_gp_lora_ft_0801"

mask_config="train_configs/qwen2_5_7b_gp/qwen2_5_7b_gp_plus.yaml"
output_dir="output/qwen2_5_7b_gp_plus_0801"

DEBUG_CALLS_N=160 \
accelerate launch \
    --num_processes $ngpus \
    --main_process_port $port \
    train_qwen_gp.py \
    --config "$config"

# if not successful, exit with error
if [ $? -ne 0 ]; then
    echo "Training failed for config: $config"
    exit 1
fi

accelerate launch \
    --num_processes 2 \
    --main_process_port $port \
    train_qwen_gp.py \
    --config "$mask_config"

# if not successful, exit with error
if [ $? -ne 0 ]; then
    echo "Training failed for mask config: $mask_config"
    exit 1
fi


BASE_MODEL=$base_model bash scripts/infer_qwen_gp_cot.sh $output_dir
BASE_MODEL=$base_model DO_GLIMPSE=1 bash scripts/infer_qwen_gp_cot.sh $output_dir
BASE_MODEL=$base_model MAX_REMAIN_RATIO=0.111 bash scripts/infer_qwen_gp_cot.sh $output_dir
BASE_MODEL=$base_model DO_GLIMPSE=1 MAX_REMAIN_RATIO=0.111 bash scripts/infer_qwen_gp_cot.sh $output_dir
