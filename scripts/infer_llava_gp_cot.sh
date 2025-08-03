#!/bin/bash
ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"

new_modules_dir=$1

brief=${BRIEF:-0}
use_ref=${USE_REF:-0}
do_glimpse=${DO_GLIMPSE:-0}
attn_implementation=${ATTN_IMPL:-""}
save_masks=${SAVE_MASKS:-0}
port=${PORT:-29500}
base_model=${BASE_MODEL:-'liuhaotian/llava-v1.5-7b'}
base_model_suffix=$(basename $base_model)
min_remain_num=${MIN_REMAIN_NUM:-0}
max_remain_ratio=${MAX_REMAIN_RATIO:-""}
num_samples=${NUM_SAMPLES:-0}

MORE_ARGS=""
PATH_SUFFIX=""

if [[ "$new_modules_dir" == "output/"* ]]; then
    suffix_path="${new_modules_dir#output/}"
    base_output_path="result/$suffix_path"
    MORE_ARGS="$MORE_ARGS --new_modules_dir ${new_modules_dir}"
else
    suffix_path=$(basename "$new_modules_dir")
    base_output_path="result/${suffix_path}"
    MORE_ARGS="$MORE_ARGS --new_modules_dir ${new_modules_dir}"
fi


base_output_path=${base_output_path}/viscot_bench



score_func="vllm_qwen_2_5_32b_int8"
score_batch=32
vllm_env=${VLLM_ENV:-""}


if [ $brief -eq 1 ]; then
    MORE_ARGS="${MORE_ARGS} --brief"
    PATH_SUFFIX="_brief"
fi

if [ $use_ref -eq 1 ]; then
    MORE_ARGS="${MORE_ARGS} --use_ref_masks --use_box"
    PATH_SUFFIX="_use_ref"
fi


if [ $do_glimpse -eq 1 ]; then
    MORE_ARGS="${MORE_ARGS} --do_func_name glimpse"
    # assert not use_ref
    if [ $use_ref -eq 1 ]; then
        echo "Error: --do_func_name glimpse cannot be used with --use_ref_masks"
        exit 1
    fi
    if [ $save_masks -eq 1 ]; then
        MORE_ARGS="${MORE_ARGS} --save_masks"
    fi
else
    if [ $save_masks -eq 1 ]; then
        echo "Error: --save_masks can only be used with --do_func_name glimpse"
        exit 1
    fi
fi

if [ -n "$attn_implementation" ]; then
    MORE_ARGS="${MORE_ARGS} --attn_implementation ${attn_implementation}"
    PATH_SUFFIX="${PATH_SUFFIX}_${attn_implementation}"
fi

if [[ $min_remain_num -ne 0 ]]; then
    MORE_ARGS="${MORE_ARGS} --min_remain_num ${min_remain_num}"
    PATH_SUFFIX="${PATH_SUFFIX}_min_${min_remain_num}"
fi

if [ -n "$max_remain_ratio" ]; then
    max_remain_ratio=$(python -c "print(${max_remain_ratio})")
    echo "max_remain_ratio: $max_remain_ratio"
    MORE_ARGS="${MORE_ARGS} --max_remain_ratio ${max_remain_ratio}"
    PATH_SUFFIX="${PATH_SUFFIX}_max_${max_remain_ratio}"
fi


if [[ $num_samples -ne 0 ]]; then
    MORE_ARGS="${MORE_ARGS} --num_samples ${num_samples}"
fi

echo "MORE_ARGS: $MORE_ARGS"
echo "PATH_SUFFIX: $PATH_SUFFIX"



tasks=( \
# "cub" \
# "docvqa" \
# "dude" \
# "flickr30k" \
# "gqa" \
# "infographicsvqa" \
# "openimages" \
# "sroie" \
# "textcap" \
"textvqa" \
# "visual7w" \
# "vsr"
)

datasets_str=""
for task in "${tasks[@]}"; do
    datasets_str="${datasets_str},${task}"
done

output_path=${base_output_path}${PATH_SUFFIX}

torchrun --nproc_per_node=$ngpus --nnodes=1 --master_port=$port \
    -m viscot_eval.infer_cot \
    --model_type llava_gp \
    --base_model $base_model \
    --batch_size_per_device 1 \
    --output_dir ${output_path} \
    --dataset ${datasets_str} \
    ${MORE_ARGS}


if [ $do_glimpse -eq 1 ]; then
    exit 0
fi


result_paths=""
for task in "${tasks[@]}"; do
    result_path=${output_path}/${task}_generate.jsonl
    echo $result_path
    result_paths="${result_paths} ${result_path}"
done


if [ -n "$vllm_env" ]; then
    CONDA_HOME=$(conda info --base)
    source $CONDA_HOME/bin/activate
    conda activate $vllm_env
    echo "Using VLLM environment: $vllm_env"
fi


VLLM_USE_V1=0 \
python -m viscot_eval.cal_cot_score \
    --result-jsonl $result_paths \
    --mapper cot_bench \
    --score-func $score_func \
    --batch-size $score_batch \
    --tensor-parallel-size $ngpus \
    --max-num-seqs $score_batch \
    --max-model-len 2048

