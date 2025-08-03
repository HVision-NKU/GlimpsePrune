VLLM_USE_V1=0 \
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8 \
    --tensor-parallel-size 2 \
    --max-num-seqs 4 \
    --max-model-len 2048 \
    --uvicorn-log-level warning
