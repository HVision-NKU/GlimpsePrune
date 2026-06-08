script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)

if [[ -f "${repo_root}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${repo_root}/.env" >/dev/null 2>&1
fi

ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"


new_modules_dir=$1
adapter_dir=${2:-""}

brief=${BRIEF:-0}
use_ref=${USE_REF:-0}
use_zero_masks=${USE_ZERO_MASKS:-0}
do_glimpse=${DO_GLIMPSE:-0}
attn_implementation=${ATTN_IMPL:-""}
save_masks=${SAVE_MASKS:-0}
port=${PORT:-12345}
base_model=${BASE_MODEL:-'Qwen/Qwen2.5-VL-3B-Instruct'}
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-1}

if ! [[ "$batch_size_per_device" =~ ^[0-9]+$ ]] || [[ "$batch_size_per_device" -le 0 ]]; then
    echo "Error: BATCH_SIZE_PER_DEVICE must be a positive integer, got: '$batch_size_per_device'"
    exit 1
fi
min_remain_num=${MIN_REMAIN_NUM:-""}
max_remain_ratio=${MAX_REMAIN_RATIO:-""}
fixed_remain_ratio=${FIXED_REMAIN_RATIO:-""}
vip_use_fa=${VIP_USE_FA:-0}
num_samples=${NUM_SAMPLES:-0}
time_logger=${TIME_LOGGER:-0}
memory_logger=${MEMORY_LOGGER:-0}
warmup_iters=${WARMUP_ITERS:-0}
no_cache=${NO_CACHE:-0}
adapter_merge=${ADAPTER_MERGE:-1}
min_pixels=${MIN_PIXELS:-0}
max_pixels=${MAX_PIXELS:-0}
reduce_layer=${REDUCE_LAYER:-""}
enable_frame_redundancy_merge=${ENABLE_FRAME_REDUNDANCY_MERGE:-0}
frame_redundancy_pooling_mode=${FRAME_REDUNDANCY_POOLING_MODE:-""}
frame_redundancy_min_keep_ratio=${FRAME_REDUNDANCY_MIN_KEEP_RATIO:-""}
frame_redundancy_min_keep_tokens=${FRAME_REDUNDANCY_MIN_KEEP_TOKENS:-""}
frame_redundancy_similarity_threshold=${FRAME_REDUNDANCY_SIMILARITY_THRESHOLD:-""}
tasks_override=${TASKS:-""}

score_func="vllm_qwen_2_5_32b_int8"
score_batch=32
vllm_env=${VLLM_ENV:-"gp_qwen"}

MORE_ARGS=""
PATH_SUFFIX=""

if [[ "$adapter_dir" == "" ]]; then
    if [[ "$new_modules_dir" == "output/"* ]]; then
        suffix_path="${new_modules_dir#output/}"
        base_output_path="result/$suffix_path"
    else
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
        MORE_ARGS="${MORE_ARGS} --adapter_merge"
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
    MORE_ARGS="${MORE_ARGS} --adapter_dir ${adapter_dir}"
fi


base_output_path=${base_output_path}/viscot_bench



if [[ $brief -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --brief"
    PATH_SUFFIX="_brief"
fi

if [[ "$batch_size_per_device" -gt 1 ]]; then
    PATH_SUFFIX="${PATH_SUFFIX}_bs-${batch_size_per_device}"
fi

if [[ $use_ref -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --use_ref_masks --use_box"
    PATH_SUFFIX="_use_ref"
fi

if [[ $use_zero_masks -eq 1 ]]; then
    if [[ $use_ref -eq 1 ]]; then
        echo "Error: USE_ZERO_MASKS cannot be used with USE_REF."
        exit 1
    fi
    MORE_ARGS="${MORE_ARGS} --use_zero_masks"
    PATH_SUFFIX="${PATH_SUFFIX}_zero-mask"
fi


if [[ $do_glimpse -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --do_func_name glimpse"
    # assert not use_ref
    if [[ $use_ref -eq 1 ]]; then
        echo "Error: --do_func_name glimpse cannot be used with --use_ref_masks"
        exit 1
    fi
    if [[ $save_masks -eq 1 ]]; then
        MORE_ARGS="${MORE_ARGS} --save_masks"
    fi
else
    if [[ $save_masks -eq 1 ]]; then
        echo "Error: --save_masks can only be used with --do_func_name glimpse"
        exit 1
    fi
fi

if [[ -n "$attn_implementation" ]]; then
    MORE_ARGS="${MORE_ARGS} --attn_implementation ${attn_implementation}"
    PATH_SUFFIX="${PATH_SUFFIX}_${attn_implementation}"
fi

if [[ -n "$min_remain_num" ]]; then
    if ! [[ "$min_remain_num" =~ ^[0-9]+$ ]]; then
        echo "Error: MIN_REMAIN_NUM must be a non-negative integer, got: '$min_remain_num'"
        exit 1
    fi
    MORE_ARGS="${MORE_ARGS} --min_remain_num ${min_remain_num}"
    PATH_SUFFIX="${PATH_SUFFIX}_min_${min_remain_num}"
fi

if [[ -n "$fixed_remain_ratio" && -n "$max_remain_ratio" ]]; then
    echo "Error: FIXED_REMAIN_RATIO and MAX_REMAIN_RATIO cannot be set at the same time."
    exit 1
fi

if [[ -n "$max_remain_ratio" ]]; then
    max_remain_ratio=$(python -c "print(${max_remain_ratio})")
    echo "max_remain_ratio: $max_remain_ratio"
    MORE_ARGS="${MORE_ARGS} --max_remain_ratio ${max_remain_ratio}"
    PATH_SUFFIX="${PATH_SUFFIX}_max_${max_remain_ratio}"
fi

if [[ -n "$fixed_remain_ratio" ]]; then
    fixed_remain_ratio=$(python -c "print(${fixed_remain_ratio})")
    echo "fixed_remain_ratio: $fixed_remain_ratio"
    MORE_ARGS="${MORE_ARGS} --fixed_remain_ratio ${fixed_remain_ratio}"
    PATH_SUFFIX="${PATH_SUFFIX}_fixed_${fixed_remain_ratio}"
fi

if [[ $num_samples -ne 0 ]]; then
    MORE_ARGS="${MORE_ARGS} --num_samples ${num_samples}"
fi

if [[ $time_logger -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --enable_time_logger"
    PATH_SUFFIX="${PATH_SUFFIX}_time"
    if [[ $warmup_iters -ne 0 ]]; then
        MORE_ARGS="${MORE_ARGS} --warmup_iters ${warmup_iters}"
        PATH_SUFFIX="${PATH_SUFFIX}_warmup-${warmup_iters}"
    fi
fi

if [[ $memory_logger -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --enable_memory_logger"
    PATH_SUFFIX="${PATH_SUFFIX}_memory"
fi

if [[ $no_cache -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --use_cache False"
    PATH_SUFFIX="${PATH_SUFFIX}_no-cache"
fi

if [[ $min_pixels -ne 0 ]]; then
    MORE_ARGS="${MORE_ARGS} --min_pixels ${min_pixels}"
    PATH_SUFFIX="${PATH_SUFFIX}_minp-${min_pixels}"
fi

if [[ $max_pixels -ne 0 ]]; then
    MORE_ARGS="${MORE_ARGS} --max_pixels ${max_pixels}"
    PATH_SUFFIX="${PATH_SUFFIX}_maxp-${max_pixels}"
fi

if [[ -n "$reduce_layer" ]]; then
    MORE_ARGS="${MORE_ARGS} --reduce_layer ${reduce_layer}"
    PATH_SUFFIX="${PATH_SUFFIX}_l-${reduce_layer}"
fi

if [[ $enable_frame_redundancy_merge -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --enable_frame_redundancy_merge"
    PATH_SUFFIX="${PATH_SUFFIX}_frm"
fi

if [[ -n "$frame_redundancy_pooling_mode" ]]; then
    MORE_ARGS="${MORE_ARGS} --frame_redundancy_pooling_mode ${frame_redundancy_pooling_mode}"
    PATH_SUFFIX="${PATH_SUFFIX}_${frame_redundancy_pooling_mode}"
fi

if [[ -n "$frame_redundancy_min_keep_ratio" ]]; then
    frame_redundancy_min_keep_ratio=$(python -c "print(${frame_redundancy_min_keep_ratio})")
    MORE_ARGS="${MORE_ARGS} --frame_redundancy_min_keep_ratio ${frame_redundancy_min_keep_ratio}"
    PATH_SUFFIX="${PATH_SUFFIX}_frmr-${frame_redundancy_min_keep_ratio}"
fi

if [[ -n "$frame_redundancy_min_keep_tokens" ]]; then
    MORE_ARGS="${MORE_ARGS} --frame_redundancy_min_keep_tokens ${frame_redundancy_min_keep_tokens}"
    PATH_SUFFIX="${PATH_SUFFIX}_frmt-${frame_redundancy_min_keep_tokens}"
fi

if [[ -n "$frame_redundancy_similarity_threshold" ]]; then
    frame_redundancy_similarity_threshold=$(python -c "print(${frame_redundancy_similarity_threshold})")
    MORE_ARGS="${MORE_ARGS} --frame_redundancy_similarity_threshold ${frame_redundancy_similarity_threshold}"
    PATH_SUFFIX="${PATH_SUFFIX}_tau-${frame_redundancy_similarity_threshold}"
fi


if [[ -n "$tasks_override" ]]; then
    tasks_override=${tasks_override//,/ }
    read -ra tasks <<< "$tasks_override"
else
    tasks=( \
    "cub" \
    "docvqa" \
    "dude" \
    "flickr30k" \
    "gqa" \
    "infographicsvqa" \
    "openimages" \
    "sroie" \
    "textcap" \
    "textvqa" \
    "visual7w" \
    "vsr"
    )
fi

if [[ ${#tasks[@]} -eq 0 ]]; then
    echo "Error: task list is empty."
    exit 1
fi

datasets_str=$(IFS=, ; echo "${tasks[*]}")


output_path=${base_output_path}${PATH_SUFFIX}

echo "Using new modules dir: $new_modules_dir"
if [[ -n "$adapter_dir" ]]; then
    echo "Using adapter dir: $adapter_dir"
fi
echo "Output path: $output_path"
echo "Tasks: ${datasets_str}"
echo "MORE_ARGS: $MORE_ARGS"


torchrun --nnodes=1 --nproc_per_node=$ngpus --master_port=$port \
    -m viscot_eval.infer_cot \
    --model_type qwen2_5_vl_gp \
    --base_model $base_model \
    --new_modules_dir $new_modules_dir \
    --batch_size_per_device $batch_size_per_device \
    --output_dir ${output_path} \
    --dataset ${datasets_str} \
    $MORE_ARGS

if [[ $? -ne 0 ]]; then
    echo "Error: Inference failed."
    exit 1
fi

if [[ $do_glimpse -eq 1 ]]; then
    exit 0
fi

result_paths=""
for task in "${tasks[@]}"; do
    result_path=${output_path}/${task}_generate.jsonl
    result_paths="${result_paths} ${result_path}"
done

if [[ -n "${CONDA_HOME:-}" && -f "${CONDA_HOME}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_HOME}/bin/activate"
fi

conda activate "$vllm_env"
echo "Using VLLM environment: $vllm_env"

python -m viscot_eval.cal_cot_score \
    --result-jsonl $result_paths \
    --mapper cot_bench \
    --score-func $score_func \
    --batch-size $score_batch \
    --max-num-seqs $score_batch \
    --max-model-len 2048 \
    --tensor-parallel-size $ngpus
