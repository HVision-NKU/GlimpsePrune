ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"


base_model=${BASE_MODEL:-'liuhaotian/llava-v1.5-7b'}
base_model_suffix=$(basename $base_model)
base_model_suffix=$(echo $base_model_suffix | tr '[:upper:]' '[:lower:]')
base_model_suffix=${base_model_suffix/-instruct/}

base_output_path="result/divprune_${base_model_suffix}/viscot_bench"
port=${PORT:-29500}

score_func="vllm_qwen_2_5_32b_int8"
score_batch=32
vllm_env=${VLLM_ENV:-"gp_qwen"}

brief=${BRIEF:-0}
attn_implementation=${ATTN_IMPL:-""}
tasks_override=${TASKS:-""}

BASELINE=${BASELINE:-"Ours"}
LAYER_INDEX=${LAYER_INDEX:-"0"}
SUBSET_RATIO=${SUBSET_RATIO:-"0.111"}


MORE_ARGS=""
PATH_SUFFIX=""

if [ $brief -eq 1 ]; then
    MORE_ARGS="${MORE_ARGS} --brief"
    PATH_SUFFIX="_brief"
fi

if [ -n "$attn_implementation" ]; then
    MORE_ARGS="${MORE_ARGS} --attn_implementation ${attn_implementation}"
    PATH_SUFFIX="${PATH_SUFFIX}_${attn_implementation}"
fi

if [[ "$BASELINE" != "Ours" ]]; then
    PATH_SUFFIX="${PATH_SUFFIX}_baseline-${BASELINE}"
fi

if [[ "$LAYER_INDEX" != "0" ]]; then
    PATH_SUFFIX="${PATH_SUFFIX}_layer-${LAYER_INDEX}"
fi

if [[ "$SUBSET_RATIO" != "0.111" ]]; then
    PATH_SUFFIX="${PATH_SUFFIX}_subset-${SUBSET_RATIO}"
fi


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

BASELINE="${BASELINE}" \
LAYER_INDEX="${LAYER_INDEX}" \
SUBSET_RATIO="${SUBSET_RATIO}" \
torchrun --nproc_per_node=$ngpus --nnodes=1 --master_port=$port \
    -m viscot_eval.infer_cot \
    --model_type llava_divprune \
    --base_model $base_model \
    --batch_size_per_device 1 \
    --output_dir ${output_path} \
    --dataset ${datasets_str} \
    ${MORE_ARGS}


result_paths=""
for task in "${tasks[@]}"; do
    result_path=${output_path}/${task}_generate.jsonl
    echo $result_path
    result_paths="${result_paths} ${result_path}"
done


if [ -n "$vllm_env" ]; then
    if [[ -n "${CONDA_HOME:-}" && -f "${CONDA_HOME}/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "${CONDA_HOME}/bin/activate"
    fi
    conda activate "$vllm_env"
    echo "Using VLLM environment: $vllm_env"
fi


python -m viscot_eval.cal_cot_score \
    --result-jsonl $result_paths \
    --mapper cot_bench \
    --score-func $score_func \
    --batch-size $score_batch \
    --max-num-seqs $score_batch \
    --max-model-len 2048
