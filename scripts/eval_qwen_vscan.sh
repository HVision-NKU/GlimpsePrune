ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# if CUDA_VISIBLE_DEVICES is set, use it
if [ ! -z $CUDA_VISIBLE_DEVICES ]; then
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

echo "Number of GPUs: $ngpus"

export LMMS_EVAL_PLUGINS="my_lmms_eval"

base_model=${BASE_MODEL:-'Qwen/Qwen2.5-VL-3B-Instruct'}
base_model_suffix=$(basename $base_model)
base_model_suffix=$(echo $base_model_suffix | tr '[:upper:]' '[:lower:]')
base_model_suffix=${base_model_suffix/-instruct/}

base_output_path="result/vscan_${base_model_suffix}"
echo "Base output path: $base_output_path"

port=${PORT:-29500}



attn_implementation=${1:-""}

layer_list=${LAYER_LIST:-""}
image_token_ratio=${IMAGE_TOKEN_RATIO:-""}
image_token_ratio_list=${IMAGE_TOKEN_RATIO_LIST:-""}

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

if [ -n "$layer_list" ]; then
    layer_list_str=$(python -c "print('-'.join(map(str, ${layer_list})))")
    PATH_SUFFIX="${PATH_SUFFIX}_layers-${layer_list_str}"
else
    layer_list_str="14"
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
    image_token_ratio_list_str="0.333"
fi


MORE_ARGS="${MORE_ARGS},layer_list=${layer_list_str},image_token_ratio=${image_token_ratio},image_token_ratio_list=${image_token_ratio_list_str}"

echo "MORE_ARGS: $MORE_ARGS"
echo "PATH_SUFFIX: $PATH_SUFFIX"

base_output_path="${base_output_path}${PATH_SUFFIX}/lmms_eval"

echo "Base output path: $base_output_path"

# eval_list=( \
# "vstar_bench" \
# "gqa_lite" \
# "vqav2_val_lite" \
# "vizwiz_vqa_val_lite" \
# "textvqa_val_lite" \
# "textvqa_val" \
# "mmmu_val" \
# "scienceqa_img" \
# "pope_full"
# "docvqa_val_lite" \
# "chartqa_lite" \
# "ai2d_lite" \
# "mmvet" \
# "vstar_bench" \
# )

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


for task in ${eval_list[@]}
do
    output_path=${base_output_path}/${task}

    if [ -d "$output_path" ]; then
        echo "Output path $output_path already exists. Skipping evaluation for task: $task"
        continue
    fi

    echo "Evaluating task: $task"
    accelerate launch --num_processes=$ngpus --main_process_port=$port -m lmms_eval \
        --model qwen2_5_vl_vscan \
        --model_args=pretrained=${base_model},attn_implementation=flash_attention_2${MORE_ARGS} \
        --tasks $task \
        --batch_size 1 \
        --output_path $output_path \
        --verbosity=DEBUG
done





