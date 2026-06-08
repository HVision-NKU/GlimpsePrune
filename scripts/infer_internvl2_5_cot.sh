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

ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"


base_model=${BASE_MODEL:-'OpenGVLab/InternVL2_5-8B'}
base_model_suffix=$(basename $base_model)
base_output_path=${BASE_OUTPUT_PATH:-"result/${base_model_suffix}/viscot_bench"}
port=${PORT:-29500}
batch_size_per_device=${BATCH_SIZE_PER_DEVICE:-1}

if ! [[ "$batch_size_per_device" =~ ^[0-9]+$ ]] || [[ "$batch_size_per_device" -le 0 ]]; then
    echo "Error: BATCH_SIZE_PER_DEVICE must be a positive integer, got: '$batch_size_per_device'"
    exit 1
fi

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

model_type="internvl2_5"

MORE_ARGS=""
PATH_SUFFIX=""

if [ $brief -eq 1 ]; then
    MORE_ARGS="${MORE_ARGS} --brief"
    PATH_SUFFIX="_brief"
fi

if [ "$batch_size_per_device" -gt 1 ]; then
    PATH_SUFFIX="${PATH_SUFFIX}_bs-${batch_size_per_device}"
fi

if [ -n "$attn_implementation" ]; then
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

tasks_override=${TASKS:-""}

echo "MORE_ARGS: $MORE_ARGS"
echo "PATH_SUFFIX: $PATH_SUFFIX"


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

torchrun --nproc_per_node=$ngpus --nnodes=1 --master_port=$port \
    -m viscot_eval.infer_cot \
    --model_type $model_type \
    --base_model $base_model \
    --batch_size_per_device $batch_size_per_device \
    --output_dir ${output_path} \
    --dataset ${datasets_str} \
    ${MORE_ARGS}


result_paths=""
for task in "${tasks[@]}"; do
    result_task=$task
    if [[ $num_samples -ne 0 ]]; then
        result_task="${task}_${num_samples}"
    fi
    result_path=${output_path}/${result_task}_generate.jsonl
    echo $result_path
    result_paths="${result_paths} ${result_path}"
done

if [[ $skip_score -eq 1 ]]; then
    echo "SKIP_SCORE=1, skipping VLLM scoring."
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
    --max-model-len 2048
