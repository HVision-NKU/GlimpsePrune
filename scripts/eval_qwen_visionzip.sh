ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"

export LMMS_EVAL_PLUGINS="my_lmms_eval"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-40960}"

base_model=${BASE_MODEL:-'Qwen/Qwen2.5-VL-3B-Instruct'}
base_model_suffix=$(basename $base_model)
base_model_suffix=$(echo $base_model_suffix | tr '[:upper:]' '[:lower:]')
base_model_suffix=${base_model_suffix/-instruct/}

base_output_path="result/visionzip_${base_model_suffix}"
echo "Base output path: $base_output_path"

port=${PORT:-29500}
batch_size=${BATCH_SIZE:-1}
max_pixels=${MAX_PIXELS:-0}
rerun=${RERUN:-0}



attn_implementation=${1:-""}

dominant_ratio=${DOMINANT_RATIO:-0}
contextual_ratio=${CONTEXTUAL_RATIO:-0}

# get port from env
port=${PORT:-29500}


MORE_ARGS=""
PATH_SUFFIX=""


if [ -n "$attn_implementation" ]; then
    MORE_ARGS="${MORE_ARGS},attn_implementation=${attn_implementation}"
    PATH_SUFFIX="${PATH_SUFFIX}_${attn_implementation}"
else
    MORE_ARGS="${MORE_ARGS},attn_implementation=flash_attention_2"
fi

if [[ $dominant_ratio != 0 ]]; then
    PATH_SUFFIX="${PATH_SUFFIX}_dominant-${dominant_ratio}"
else
    dominant_ratio=0.65
fi

if [[ $contextual_ratio != 0 ]]; then
    PATH_SUFFIX="${PATH_SUFFIX}_contextual-${contextual_ratio}"
else
    contextual_ratio=0.05
fi


MORE_ARGS="${MORE_ARGS},dominant_ratio=${dominant_ratio},contextual_ratio=${contextual_ratio}"

if [[ "$max_pixels" != 0 && -n "$max_pixels" ]]; then
    MORE_ARGS="${MORE_ARGS},max_pixels=${max_pixels}"
    PATH_SUFFIX="${PATH_SUFFIX}_maxp-${max_pixels}"
fi

echo "MORE_ARGS: $MORE_ARGS"
echo "PATH_SUFFIX: $PATH_SUFFIX"
echo "Batch size: $batch_size"
echo "Rerun: $rerun"
echo "DECORD_EOF_RETRY_MAX: $DECORD_EOF_RETRY_MAX"

base_output_path="${base_output_path}${PATH_SUFFIX}/lmms_eval"

echo "Base output path: $base_output_path"


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
        --model qwen2_5_vl_visionzip \
        --model_args=pretrained=${base_model},attn_implementation=flash_attention_2${MORE_ARGS} \
        --tasks $task \
        --batch_size $batch_size \
        --output_path $output_path \
        --verbosity=DEBUG
done

