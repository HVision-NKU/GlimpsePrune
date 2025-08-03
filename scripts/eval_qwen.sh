ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"

export LMMS_EVAL_PLUGINS="my_lmms_eval"

base_model=${BASE_MODEL:-'Qwen/Qwen2.5-VL-3B-Instruct'}
base_model_suffix=$(basename $base_model)
base_output_path=${BASE_OUTPUT_PATH:-"result/${base_model_suffix}/lmms_eval"}
port=${PORT:-29500}
attn_implementation=${ATTN_IMPL:-"flash_attention_2"}


eval_list=( \
"vqav2_val_lite" \
"gqa" \
"vizwiz_vqa_val" \
"scienceqa_img" \
"pope" \
"mme" \
"mmbench_en_test" \
"mmbench_cn_test" \
"seedbench" \
"vstar_bench" \
)

for task in ${eval_list[@]}
do
    output_path=${base_output_path}/${task}

    if [ -d "$output_path" ]; then
        echo "Output path $output_path already exists. Skipping evaluation for task: $task"
        continue
    fi

    echo "Evaluating task: $task"
    accelerate launch --num_processes=$ngpus --main_process_port=$port -m lmms_eval \
        --model qwen2_5_vl \
        --model_args=pretrained=${base_model},attn_implementation=${attn_implementation} \
        --tasks $task \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples
done





