# Tested in 4 GPUs

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_gemma_2b.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
hdfs_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.prompt_key=prompt \
    data.response_key=extra_info \
    +data.prompt_dict_keys=['content'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=32 \
    model.partial_pretrain=google/gemma-2b-it \
    trainer.default_hdfs_dir=$hdfs_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-gemma-2b-it \
    trainer.total_epochs=3 \
    trainer.logger=['console','wandb'] $@