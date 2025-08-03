ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"

base_model=${BASE_MODEL:-'Qwen/Qwen2.5-VL-3B-Instruct'}
base_model_suffix=$(basename $base_model)
base_model_suffix=$(echo $base_model_suffix | tr '[:upper:]' '[:lower:]')
base_model_suffix=${base_model_suffix/-instruct/}

base_output_path="result/vscan_${base_model_suffix}/viscot_bench"
score_func="vllm_qwen_2_5_32b_int8"
score_batch=32


brief=${BRIEF:-0}
do_glimpse=${DO_GLIMPSE:-0}
attn_implementation=${ATTN_IMPL:-""}
save_masks=${SAVE_MASKS:-0}

layer_list=${LAYER_LIST:-""}
image_token_ratio=${IMAGE_TOKEN_RATIO:-""}
image_token_ratio_list=${IMAGE_TOKEN_RATIO_LIST:-""}

# get port from env
port=${PORT:-29500}


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

if [ $do_glimpse -eq 1 ]; then
    MORE_ARGS="${MORE_ARGS} --do_func_name glimpse"
    if [ $save_masks -eq 1 ]; then
        MORE_ARGS="${MORE_ARGS} --save_masks"
    fi
else
    if [ $save_masks -eq 1 ]; then
        echo "Error: --save_masks can only be used with --do_func_name glimpse"
        exit 1
    fi
fi

if [ -n "$layer_list" ]; then
    layer_list_str=$(python -c "print('-'.join(map(str, ${layer_list})))")
    PATH_SUFFIX="${PATH_SUFFIX}_layers-${layer_list_str}"
else
    layer_list="[18]"
fi

if [ -n "$image_token_ratio" ]; then
    image_token_ratio=$(python -c "print(${image_token_ratio})")
    PATH_SUFFIX="${PATH_SUFFIX}_image-${image_token_ratio}"
else
    image_token_ratio=0.167
fi

if [ -n "$image_token_ratio_list" ]; then
    image_token_ratio_list_str=$(python -c "print('-'.join(map(str, ${image_token_ratio_list})))")
    PATH_SUFFIX="${PATH_SUFFIX}_ratios-${image_token_ratio_list_str}"
else
    image_token_ratio_list="[0.333]"
fi


MORE_ARGS="${MORE_ARGS} --layer_list ${layer_list} --image_token_ratio ${image_token_ratio} --image_token_ratio_list ${image_token_ratio_list}"

echo "MORE_ARGS: $MORE_ARGS"
echo "PATH_SUFFIX: $PATH_SUFFIX"

if [[ $image_token_ratio == 1* ]]; then
echo "using full image tokens"
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
else
tasks=( \
"cub" \
# "docvqa" \
# "dude" \
"flickr30k" \
"gqa" \
# "infographicsvqa" \
# "openimages" \
# "sroie" \
"textcap" \
"textvqa" \
"visual7w" \
"vsr"
)
fi


datasets_str=""
for task in "${tasks[@]}"; do
    datasets_str="${datasets_str},${task}"
done

output_path=${base_output_path}${PATH_SUFFIX}

echo "Output path: ${output_path}"

torchrun --nproc_per_node=$ngpus --nnodes=1 --master_port=$port \
    -m viscot_eval.infer_cot \
    --model_type qwen2_5_vl_vscan \
    --base_model $base_model \
    --batch_size_per_device 1 \
    --output_dir ${output_path} \
    --dataset ${datasets_str} \
    ${MORE_ARGS}


if [ $do_glimpse -eq 1 ]; then
    exit
fi


result_paths=""
for task in "${tasks[@]}"; do
    result_path=${output_path}/${task}_generate.jsonl
    echo $result_path
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
