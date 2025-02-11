set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma_2b.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

export OMP_NUM_THREADS=1
# Shift the arguments so $@ refers to the rest
shift 2

PYTHON_PATH=$(which python)
echo "PYTHON_PATH: $PYTHON_PATH"

# torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
$PYTHON_PATH -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/tir-sft-iter3-raw/train.parquet \
    data.val_files=$HOME/data/tir-sft-iter3-raw/test.parquet \
    data.prompt_key=prompt \
    data.response_key=solution \
    data.train_batch_size=8 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=16384 \
    model.fsdp_config.cpu_offload=False \
    model.fsdp_config.offload_params=False \
    model.enable_gradient_checkpointing=False \
    model.partial_pretrain=meta-llama/Llama-3.2-3B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=tir-sft-rl-prep \
    trainer.experiment_name=sft_tir_rl_prep \
    trainer.total_epochs=4 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@ \
    optim.lr=1e-5 \