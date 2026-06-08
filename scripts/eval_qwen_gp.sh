#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)

if [[ -f "${repo_root}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${repo_root}/.env" >/dev/null 2>&1
fi

count_visible_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local count=0
        local visible_gpus=()
        IFS=',' read -ra visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
        for gpu in "${visible_gpus[@]}"; do
            if [[ -n "${gpu//[[:space:]]/}" ]]; then
                count=$((count + 1))
            fi
        done
        echo "$count"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
        return
    fi

    echo 0
}

ngpus=$(count_visible_gpus)
num_processes=${NUM_PROCESSES:-$ngpus}

if ! [[ "$num_processes" =~ ^[0-9]+$ ]]; then
    echo "Error: NUM_PROCESSES must be a positive integer, got '$num_processes'."
    exit 1
fi

if [[ "$num_processes" -lt 1 ]]; then
    echo "Error: no visible GPUs found. Set CUDA_VISIBLE_DEVICES or NUM_PROCESSES."
    exit 1
fi

echo "Number of visible GPUs: $ngpus"
echo "Data parallel processes: $num_processes"

export LMMS_EVAL_PLUGINS="my_lmms_eval"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-40960}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <new_modules_dir (e.g., result/xxx)> [adapter_dir]"
    echo "Optional env: TASKS=\"vqav2_val_lite,gqa,...\" to override default task list"
    echo "Optional env: BATCH_SIZE=1 to override default batch size"
    echo "Optional env: MAX_PIXELS=1605632 to override processor max_pixels"
    echo "Optional env: NUM_PROCESSES=N to override automatic visible GPU count"
    echo "Optional env: RERUN=1 to re-run even if output exists (will delete existing task dir)"
    exit 1
fi

new_modules_dir=$1
adapter_dir=${2:-""}

base_model=${BASE_MODEL:-"Qwen/Qwen2.5-VL-3B-Instruct"}
min_remain_num=${MIN_REMAIN_NUM:-""}
max_remain_ratio=${MAX_REMAIN_RATIO:-""}
enable_frame_redundancy_merge=${ENABLE_FRAME_REDUNDANCY_MERGE:-0}
frame_redundancy_pooling_mode=${FRAME_REDUNDANCY_POOLING_MODE:-""}
frame_redundancy_min_keep_ratio=${FRAME_REDUNDANCY_MIN_KEEP_RATIO:-""}
frame_redundancy_min_keep_tokens=${FRAME_REDUNDANCY_MIN_KEEP_TOKENS:-""}
frame_redundancy_similarity_threshold=${FRAME_REDUNDANCY_SIMILARITY_THRESHOLD:-""}
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

if [[ $enable_frame_redundancy_merge -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS},enable_frame_redundancy_merge=True"
    PATH_SUFFIX="${PATH_SUFFIX}_frm"
fi

if [[ -n "$frame_redundancy_pooling_mode" ]]; then
    MORE_ARGS="${MORE_ARGS},frame_redundancy_pooling_mode=$frame_redundancy_pooling_mode"
    PATH_SUFFIX="${PATH_SUFFIX}_${frame_redundancy_pooling_mode}"
fi

if [[ -n "$frame_redundancy_min_keep_ratio" ]]; then
    MORE_ARGS="${MORE_ARGS},frame_redundancy_min_keep_ratio=$frame_redundancy_min_keep_ratio"
    PATH_SUFFIX="${PATH_SUFFIX}_frmr-${frame_redundancy_min_keep_ratio}"
fi

if [[ -n "$frame_redundancy_min_keep_tokens" ]]; then
    MORE_ARGS="${MORE_ARGS},frame_redundancy_min_keep_tokens=$frame_redundancy_min_keep_tokens"
    PATH_SUFFIX="${PATH_SUFFIX}_frmt-${frame_redundancy_min_keep_tokens}"
fi

if [[ -n "$frame_redundancy_similarity_threshold" ]]; then
    MORE_ARGS="${MORE_ARGS},frame_redundancy_similarity_threshold=$frame_redundancy_similarity_threshold"
    PATH_SUFFIX="${PATH_SUFFIX}_tau-${frame_redundancy_similarity_threshold}"
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
output_root="${repo_root}/${base_output_path}"

echo "Input (new_modules_dir): $new_modules_dir"
echo "Output (base_output_path): $base_output_path"
echo "Output root: $output_root"
echo "More args: $MORE_ARGS"
echo "Batch size: $batch_size"
echo "Rerun: $rerun"
echo "DECORD_EOF_RETRY_MAX: $DECORD_EOF_RETRY_MAX"
echo "Launch port: $port"


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
    output_path="${output_root}/${task}"

    if [[ -d "$output_path" ]]; then
        if [[ "$rerun" == "1" ]]; then
            if [[ -n "$output_path" && "$output_path" == "$output_root"/* ]]; then
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

    mkdir -p "${output_path}"
    echo "Evaluating task: $task"

    launch_args=(--num_processes "$num_processes" --main_process_port "$port")
    if [[ "$num_processes" -gt 1 ]]; then
        launch_args=(--multi_gpu "${launch_args[@]}")
    fi

    env \
        -u RANK \
        -u WORLD_SIZE \
        -u LOCAL_RANK \
        -u LOCAL_WORLD_SIZE \
        -u GROUP_RANK \
        -u GROUP_WORLD_SIZE \
        -u MASTER_ADDR \
        -u MASTER_PORT \
        accelerate launch "${launch_args[@]}" -m lmms_eval \
        --model qwen2_5_vl_gp \
        --model_args "pretrained=${base_model},new_modules_dir=${new_modules_dir}${MORE_ARGS}" \
        --tasks $task \
        --batch_size $batch_size \
        --output_path "${output_path}" \
        --log_samples
done
