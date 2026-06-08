config=${1:-"train_configs/internvl2_5_8b_gp/internvl2_5_8b_gp.yaml"}

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
    echo "Using training environment: $CONDA_ENV_NAME"
fi

ngpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    ngpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
port=${PORT:-29501}

torchrun \
    --nproc_per_node=$ngpus \
    --nnodes=1 \
    --master_port=$port \
    train_internvl2_5_gp.py \
    --config "$config"

