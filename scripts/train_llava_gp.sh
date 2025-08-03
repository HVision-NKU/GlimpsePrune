ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"

port=${PORT:-12345}


configs=( \
"train_configs/llava1_5_7b_gp/llava1_5_7b_gp.yaml" \
"train_configs/llava1_5_13b_gp/llava1_5_13b_gp.yaml" \
)


output_dirs=( \
"output/llava1_5_7b_gp_0801" \
"output/llava1_5_13b_gp_0801" \
)

base_model=${BASE_MODEL:-'liuhaotian/llava-v1.5-13b'}


for i in "${!configs[@]}"; do
    config=${configs[$i]}
    output_dir=${output_dirs[$i]}
    echo "Running with config: $config"

    accelerate launch \
        --num_processes $ngpus \
        --main_process_port $port \
        train_llava_gp.py \
        --config "$config"

    train_ok=$?
    if [ $train_ok -ne 0 ]; then
        echo "Training failed for config: $config"
        continue
    fi

    BASE_MODEL=$base_model bash infer_llava_gp_cot.sh $output_dir
    BASE_MODEL=$base_model DO_GLIMPSE=1 bash infer_llava_gp_cot.sh $output_dir
    BASE_MODEL=$base_model MAX_REMAIN_RATIO=0.111 bash infer_llava_gp_cot.sh $output_dir
    BASE_MODEL=$base_model MAX_REMAIN_RATIO=0.111 DO_GLIMPSE=1 bash infer_llava_gp_cot.sh $output_dir
done



