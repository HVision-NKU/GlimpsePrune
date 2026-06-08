set -e

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

if [[ -n "${CONDA_HOME:-}" && -f "${CONDA_HOME}/bin/conda" ]]; then
    eval "$("${CONDA_HOME}/bin/conda" shell.bash hook)"
fi

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
    conda activate "$CONDA_ENV_NAME"
    echo "Using inference environment: $CONDA_ENV_NAME"
fi

new_modules_dir=${1:-${NEW_MODULES_DIR:-""}}
if [[ -z "$new_modules_dir" && -z "${NEW_MODULES_CONFIG:-}" && "${USE_REF:-0}" -ne 1 && "${USE_ZERO_MASKS:-0}" -ne 1 ]]; then
    echo "Usage: $0 <new_modules_dir>, set NEW_MODULES_CONFIG, or set USE_REF=1/USE_ZERO_MASKS=1 for mask-only selection."
    exit 1
fi

ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Number of GPUs: $ngpus"

base_model=${BASE_MODEL:-'OpenGVLab/InternVL2_5-8B'}
base_model_suffix=$(basename "$base_model")
suffix_source="${new_modules_dir:-${NEW_MODULES_CONFIG:-internvl2_5_gp_default}}"
suffix_path=$(basename "$suffix_source")
base_output_path=${BASE_OUTPUT_PATH:-"result/${suffix_path}/${base_model_suffix}/viscot_bench"}
port=${PORT:-29500}
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-1}
score_func="vllm_qwen_2_5_32b_int8"
score_batch=32
vllm_env=${VLLM_ENV:-"gp_qwen"}
skip_score=${SKIP_SCORE:-0}

brief=${BRIEF:-0}
attn_implementation=${ATTN_IMPL:-"flash_attention_2"}
num_samples=${NUM_SAMPLES:-0}
time_logger=${TIME_LOGGER:-0}
memory_logger=${MEMORY_LOGGER:-0}
warmup_iters=${WARMUP_ITERS:-0}
min_pixels=${MIN_PIXELS:-0}
max_pixels=${MAX_PIXELS:-0}
do_glimpse=${DO_GLIMPSE:-0}
max_remain_ratio=${MAX_REMAIN_RATIO:-""}
fixed_remain_ratio=${FIXED_REMAIN_RATIO:-""}
use_ref=${USE_REF:-0}
use_zero_masks=${USE_ZERO_MASKS:-0}
min_remain_num=${MIN_REMAIN_NUM:-""}
reduce_layer=${REDUCE_LAYER:-""}

model_type="internvl2_5_gp"
MORE_ARGS=""
PATH_SUFFIX=""

if [[ -n "$new_modules_dir" ]]; then
    MORE_ARGS="${MORE_ARGS} --new_modules_dir ${new_modules_dir}"
elif [[ -n "${NEW_MODULES_CONFIG:-}" ]]; then
    MORE_ARGS="${MORE_ARGS} --new_modules_config ${NEW_MODULES_CONFIG}"
fi
if [[ $brief -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --brief"
    PATH_SUFFIX="${PATH_SUFFIX}_brief"
fi
if [[ "$batch_size_per_device" -gt 1 ]]; then
    PATH_SUFFIX="${PATH_SUFFIX}_bs-${batch_size_per_device}"
fi
if [[ -n "$attn_implementation" ]]; then
    MORE_ARGS="${MORE_ARGS} --attn_implementation ${attn_implementation}"
    PATH_SUFFIX="${PATH_SUFFIX}_${attn_implementation}"
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
if [[ $min_pixels -ne 0 ]]; then
    MORE_ARGS="${MORE_ARGS} --min_pixels ${min_pixels}"
    PATH_SUFFIX="${PATH_SUFFIX}_minp-${min_pixels}"
fi
if [[ $max_pixels -ne 0 ]]; then
    MORE_ARGS="${MORE_ARGS} --max_pixels ${max_pixels}"
    PATH_SUFFIX="${PATH_SUFFIX}_maxp-${max_pixels}"
fi
if [[ -n "$max_remain_ratio" ]]; then
    MORE_ARGS="${MORE_ARGS} --max_remain_ratio ${max_remain_ratio}"
    PATH_SUFFIX="${PATH_SUFFIX}_max-${max_remain_ratio}"
fi
if [[ -n "$fixed_remain_ratio" ]]; then
    MORE_ARGS="${MORE_ARGS} --fixed_remain_ratio ${fixed_remain_ratio}"
    PATH_SUFFIX="${PATH_SUFFIX}_fixed-${fixed_remain_ratio}"
fi
if [[ $do_glimpse -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --do_func_name glimpse"
    PATH_SUFFIX="${PATH_SUFFIX}_glimpse"
fi
if [[ $use_ref -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --use_ref_masks --use_box"
    PATH_SUFFIX="${PATH_SUFFIX}_use_ref"
fi
if [[ $use_zero_masks -eq 1 ]]; then
    if [[ $use_ref -eq 1 ]]; then
        echo "Error: USE_ZERO_MASKS cannot be used with USE_REF."
        exit 1
    fi
    MORE_ARGS="${MORE_ARGS} --use_zero_masks"
    PATH_SUFFIX="${PATH_SUFFIX}_use_zero"
fi
if [[ -n "$min_remain_num" ]]; then
    if ! [[ "$min_remain_num" =~ ^[0-9]+$ ]]; then
        echo "Error: MIN_REMAIN_NUM must be a non-negative integer, got: '$min_remain_num'"
        exit 1
    fi
    MORE_ARGS="${MORE_ARGS} --min_remain_num ${min_remain_num}"
    PATH_SUFFIX="${PATH_SUFFIX}_min_${min_remain_num}"
fi
if [[ -n "$reduce_layer" ]]; then
    MORE_ARGS="${MORE_ARGS} --reduce_layer ${reduce_layer}"
    PATH_SUFFIX="${PATH_SUFFIX}_l-${reduce_layer}"
fi

tasks_override=${TASKS:-""}
if [[ -n "$tasks_override" ]]; then
    tasks_override=${tasks_override//,/ }
    read -ra tasks <<< "$tasks_override"
else
    tasks=("cub" "docvqa" "dude" "flickr30k" "gqa" "infographicsvqa" "openimages" "sroie" "textcap" "textvqa" "visual7w" "vsr")
fi
datasets_str=$(IFS=, ; echo "${tasks[*]}")
output_path=${base_output_path}${PATH_SUFFIX}

torchrun --nproc_per_node=$ngpus --nnodes=1 --master_port=$port \
    -m viscot_eval.infer_cot \
    --model_type $model_type \
    --base_model $base_model \
    --batch_size_per_device $batch_size_per_device \
    --output_dir ${output_path} \
    --dataset ${datasets_str} \
    ${MORE_ARGS}

result_paths=""
task_name="generate"
if [[ $do_glimpse -eq 1 ]]; then
    task_name="glimpse"
fi
for task in "${tasks[@]}"; do
    result_task=$task
    if [[ $num_samples -ne 0 ]]; then
        result_task="${task}_${num_samples}"
    fi
    result_path=${output_path}/${result_task}_${task_name}.jsonl
    echo "$result_path"
    result_paths="${result_paths} ${result_path}"
done

if [[ $skip_score -eq 1 || $do_glimpse -eq 1 ]]; then
    echo "Skipping VLLM scoring."
    exit 0
fi

if [[ -n "${CONDA_HOME:-}" && -f "${CONDA_HOME}/bin/conda" ]]; then
    eval "$("${CONDA_HOME}/bin/conda" shell.bash hook)"
fi
conda activate "$vllm_env"
echo "Using VLLM environment: $vllm_env"

python -m viscot_eval.cal_cot_score \
    --result-jsonl $result_paths \
    --mapper cot_bench \
    --score-func $score_func \
    --batch-size $score_batch \
    --max-num-seqs $score_batch \
    --max-model-len 4096
