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
do_glimpse=${DO_GLIMPSE:-0}
attn_implementation=${ATTN_IMPL:-""}
save_masks=${SAVE_MASKS:-0}
port=${PORT:-12345}
base_model=${BASE_MODEL:-'Qwen/Qwen2.5-VL-3B-Instruct'}
min_remain_num=${MIN_REMAIN_NUM:-0}
max_remain_ratio=${MAX_REMAIN_RATIO:-""}
num_samples=${NUM_SAMPLES:-0}
time_logger=${TIME_LOGGER:-0}
memory_logger=${MEMORY_LOGGER:-0}
no_cache=${NO_CACHE:-0}
adapter_merge=${ADAPTER_MERGE:-1}

score_func="vllm_qwen_2_5_32b_int8"
score_batch=32

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

if [[ $use_ref -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --use_ref_masks --use_box"
    PATH_SUFFIX="_use_ref"
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

if [[ $min_remain_num -ne 0 ]]; then
    MORE_ARGS="${MORE_ARGS} --min_remain_num ${min_remain_num}"
    PATH_SUFFIX="${PATH_SUFFIX}_min_${min_remain_num}"
fi

if [[ -n "$max_remain_ratio" ]]; then
    max_remain_ratio=$(python -c "print(${max_remain_ratio})")
    echo "max_remain_ratio: $max_remain_ratio"
    MORE_ARGS="${MORE_ARGS} --max_remain_ratio ${max_remain_ratio}"
    PATH_SUFFIX="${PATH_SUFFIX}_max_${max_remain_ratio}"
fi


if [[ $num_samples -ne 0 ]]; then
    MORE_ARGS="${MORE_ARGS} --num_samples ${num_samples}"
fi

if [[ $time_logger -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --enable_time_logger"
    PATH_SUFFIX="${PATH_SUFFIX}_time"
fi

if [[ $memory_logger -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --enable_memory_logger"
    PATH_SUFFIX="${PATH_SUFFIX}_memory"
fi

if [[ $no_cache -eq 1 ]]; then
    MORE_ARGS="${MORE_ARGS} --use_cache False"
    PATH_SUFFIX="${PATH_SUFFIX}_no-cache"
fi



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

datasets_str=""
for task in "${tasks[@]}"; do
    datasets_str="${datasets_str},${task}"
done


output_path=${base_output_path}${PATH_SUFFIX}

echo "Using new modules dir: $new_modules_dir"
if [[ -n "$adapter_dir" ]]; then
    echo "Using adapter dir: $adapter_dir"
fi
echo "Output path: $output_path"
echo "MORE_ARGS: $MORE_ARGS"


torchrun --nnodes=1 --nproc_per_node=$ngpus --master_port=$port \
    -m viscot_eval.infer_cot \
    --model_type qwen2_5_vl_gp \
    --base_model $base_model \
    --new_modules_dir $new_modules_dir \
    --batch_size_per_device 1 \
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

VLLM_USE_V1=0 \
python -m viscot_eval.cal_cot_score \
    --result-jsonl $result_paths \
    --mapper cot_bench \
    --score-func $score_func \
    --batch-size $score_batch \
    --tensor-parallel-size $ngpus \
    --max-num-seqs $score_batch \
    --max-model-len 2048