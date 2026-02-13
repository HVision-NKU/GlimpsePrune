#!/bin/bash
ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"

export LMMS_EVAL_PLUGINS="my_lmms_eval"

if [ -z "$1" ]; then
    echo "Usage: $0 <new_modules_dir (e.g., result/xxx)> [adapter_dir]"
    echo "Optional env: TASKS=\"vqav2_val_lite,gqa,...\" to override default task list"
    echo "Optional env: BATCH_SIZE=1 to override default batch size"
    echo "Optional env: MAX_PIXELS=1605632 to override processor max_pixels"
    echo "Optional env: RERUN=1 to re-run even if output exists (will delete existing task dir)"
    exit 1
fi

new_modules_dir=$1
adapter_dir=${2:-""}

base_model=${BASE_MODEL:-"Qwen/Qwen2.5-VL-3B-Instruct"}
min_remain_num=${MIN_REMAIN_NUM:-""}
max_remain_ratio=${MAX_REMAIN_RATIO:-""}
attn_implementation=${ATTN_IMPL:-"flash_attention_2"}
adapter_merge=${ADAPTER_MERGE:-1}
port=${PORT:-29501}
batch_size=${BATCH_SIZE:-1}
max_pixels=${MAX_PIXELS:-0}
rerun=${RERUN:-0}

MORE_ARGS=""
PATH_SUFFIX=""

if [[ "$adapter_dir" == "" ]]; then
    if [[ "$new_modules_dir" == "output/"* ]]; then
        suffix_path="${new_modules_dir#output/}"
        base_output_path="result/$suffix_path"
    else
        # echo "Error: new_modules_dir ('$new_modules_dir') does not start with 'output/'."
        # exit 1
        suffix_path=$(basename "$new_modules_dir")
        base_output_path="result/${suffix_path}"
    fi
else
    if [[ "$adapter_dir" == "output/"* ]]; then
        suffix_path="${adapter_dir#output/}"
        base_output_path="result/$suffix_path"
    else
        suffix_path=$(basename "$adapter_dir")
        base_output_path="result/${suffix_path}"
    fi

    if [[ $adapter_merge -eq 1 ]]; then
        MORE_ARGS="${MORE_ARGS},adapter_merge=True"
        base_output_path="${base_output_path}_merge"
    fi

    if [[ "$new_modules_dir" != "$adapter_dir" ]]; then
        if [[ "$new_modules_dir" == "output/"* ]]; then
            suffix_path="${new_modules_dir#output/}"
            base_output_path="${base_output_path}/${suffix_path}"
        else
            suffix_path=$(basename "$new_modules_dir")
            base_output_path="${base_output_path}/${suffix_path}"
        fi
    fi
    MORE_ARGS="${MORE_ARGS},adapter_dir=${adapter_dir}"
fi

base_output_path=${base_output_path}/lmms_eval


if [[ -n "$min_remain_num" ]]; then
    MORE_ARGS="${MORE_ARGS},min_remain_num=$min_remain_num"
    PATH_SUFFIX="${PATH_SUFFIX}_min_${min_remain_num}"
fi

if [[ -n "$max_remain_ratio" ]]; then
    MORE_ARGS="${MORE_ARGS},max_remain_ratio=$max_remain_ratio"
    PATH_SUFFIX="${PATH_SUFFIX}_max_${max_remain_ratio}"
fi

if [[ "$attn_implementation" != "flash_attention_2" ]]; then
    MORE_ARGS="${MORE_ARGS},attn_implementation=$attn_implementation"
    PATH_SUFFIX="${PATH_SUFFIX}_${attn_implementation}"
else
    MORE_ARGS="${MORE_ARGS},attn_implementation=flash_attention_2"
fi

if [[ "$max_pixels" != 0 && -n "$max_pixels" ]]; then
    MORE_ARGS="${MORE_ARGS},max_pixels=$max_pixels"
    PATH_SUFFIX="${PATH_SUFFIX}_maxp-${max_pixels}"
fi

base_output_path=${base_output_path}${PATH_SUFFIX}

echo "Input (new_modules_dir): $new_modules_dir"
echo "Output (base_output_path): $base_output_path"
echo "More args: $MORE_ARGS"
echo "Batch size: $batch_size"
echo "Rerun: $rerun"


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

tasks_override=${TASKS:-""}
if [[ -n "$tasks_override" ]]; then
    tasks_override=${tasks_override//,/ }
    read -ra eval_list <<< "$tasks_override"
fi

if [[ ${#eval_list[@]} -eq 0 ]]; then
    echo "Error: task list is empty."
    exit 1
fi

for task in "${eval_list[@]}"
do
    output_path=${base_output_path}/${task}

    if [ -d "$output_path" ]; then
        if [[ "$rerun" == "1" ]]; then
            if [[ -n "$output_path" && "$output_path" == "$base_output_path"/* ]]; then
                echo "Output path $output_path already exists. RERUN=1, deleting and re-running task: $task"
                rm -rf -- "$output_path"
            else
                echo "Error: refusing to delete unexpected output_path: '$output_path'"
                exit 1
            fi
        else
        echo "Output path $output_path already exists. Skipping evaluation for task: $task"
        continue
        fi
    fi

    echo "Evaluating task: $task"
    accelerate launch --num_processes=$ngpus --main_process_port=$port -m lmms_eval \
        --model qwen2_5_vl_gp \
        --model_args "pretrained=${base_model},new_modules_dir=${new_modules_dir}${MORE_ARGS}" \
        --tasks $task \
        --batch_size $batch_size \
        --output_path ${output_path} \
        --log_samples
done
