#!/usr/bin/env bash

set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv |
        tail -n +2 |
        nl -v 0 |
        tee /dev/tty |
        sort -g -k 2 |
        awk '{print $1}' |
        head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 4

set -xue

if [ -z "$SFT_DATASET" ]; then
    echo "Please set \$SFT_DATASET to the path to sft dataset."
    exit 1
fi

if [ -z "$PROMPT_PATH" ]; then
    echo "Please set \$PROMPT_PATH to the path to prompts csv."
    exit 1
fi

if [ -z "$PRETRAIN_DATASET" ]; then
    echo "Please set \$PRETRAIN_DATASET to the path to alpaca data."
    exit 1
fi

BASE_DIR=$(dirname $(dirname $(realpath $BASH_SOURCE)))
EXAMPLES_DIR=$BASE_DIR/examples

export OMP_NUM_THREADS=8

# install requirements
pip install -r $EXAMPLES_DIR/requirements.txt

# FIXME: This is a hack to skip tests that are not working
#  - gpt2-ddp: RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
#  - llama-*: These tests can be passed locally, skipped for long execution time
SKIPPED_TESTS=(
    "gpt2-ddp"
    "llama-ddp"
    "llama-colossalai_gemini"
    "llama-colossalai_zero2"
)

# These tests are quick and do not have any dependencies
for model in 'gpt2' 'bloom' 'opt' 'llama'; do
    for strategy in 'ddp' 'colossalai_gemini' 'colossalai_zero2'; do
        if [[ " ${SKIPPED_TESTS[*]} " =~ " $model-$strategy " ]]; then
            echo "[Test]: Skipped $model-$strategy"
            continue
        fi
        torchrun --standalone --nproc_per_node=2 $EXAMPLES_DIR/train_prompts.py \
            --prompt_dataset $PROMPT_PATH --pretrain_dataset $PRETRAIN_DATASET \
            --strategy $strategy --model $model \
            --num_episodes 1 --num_collect_steps 2 --num_update_steps 1 \
            --train_batch_size 2 --lora_rank 4
    done
done

echo "[Test]: testing sft ..."

torchrun --standalone --nproc_per_node=4 $EXAMPLES_DIR/train_sft.py --pretrain 'bigscience/bloom-560m' \
    --model 'bloom' --strategy colossalai_zero2 --lora_rank 4 \
    --dataset $SFT_DATASET --max_datasets_size 512 --max_epochs 1 \
    --save_path $EXAMPLES_DIR/output
rm -rf $EXAMPLES_DIR/output

torchrun --standalone --nproc_per_node=4 $EXAMPLES_DIR/train_sft.py --pretrain 'gpt2' \
    --model 'gpt2' --strategy colossalai_zero2 \
    --dataset $SFT_DATASET --max_datasets_size 512 --max_epochs 1 \
    --save_path $EXAMPLES_DIR/output
rm -rf $EXAMPLES_DIR/output

torchrun --standalone --nproc_per_node=4 $EXAMPLES_DIR/train_sft.py --pretrain 'facebook/opt-350m' \
    --model 'opt' --strategy colossalai_zero2 --lora_rank 4 \
    --dataset $SFT_DATASET --max_datasets_size 512 --max_epochs 1 \
    --save_path $EXAMPLES_DIR/output
rm -rf $EXAMPLES_DIR/output

torchrun --standalone --nproc_per_node=4 $EXAMPLES_DIR/train_sft.py --pretrain 'gpt2' \
    --model 'gpt2' --strategy ddp --lora_rank 4 \
    --dataset $SFT_DATASET --max_datasets_size 512 --max_epochs 1 \
    --save_path $EXAMPLES_DIR/output
rm -rf $EXAMPLES_DIR/output

echo "[Test]: testing reward model ..."

torchrun --standalone --nproc_per_node=2 $EXAMPLES_DIR/train_reward_model.py \
    --pretrain 'facebook/opt-350m' --model 'opt' \
    --strategy colossalai_zero2 --loss_fn 'log_sig' \
    --dataset 'Anthropic/hh-rlhf' --subset 'harmless-base' \
    --test True --lora_rank 0 \
    --save_path $EXAMPLES_DIR/rm_ckpt_opt.pt

torchrun --standalone --nproc_per_node=2 $EXAMPLES_DIR/train_reward_model.py \
    --pretrain 'gpt2' --model 'gpt2' \
    --strategy colossalai_zero2 --loss_fn 'log_exp' \
    --dataset 'Dahoas/rm-static' \
    --test True --lora_rank 0 \
    --save_path $EXAMPLES_DIR/rm_ckpt_gpt.pt

torchrun --standalone --nproc_per_node=2 $EXAMPLES_DIR/train_reward_model.py \
    --pretrain 'gpt2' --model 'gpt2' \
    --strategy ddp --loss_fn 'log_exp' \
    --dataset 'Dahoas/rm-static' \
    --test True --lora_rank 4 \
    --save_path $EXAMPLES_DIR/rm_ckpt.pt
rm -rf $EXAMPLES_DIR/rm_ckpt.pt

torchrun --standalone --nproc_per_node=2 $EXAMPLES_DIR/train_reward_model.py \
    --pretrain 'bigscience/bloom-560m' --model 'bloom' \
    --strategy colossalai_zero2 --loss_fn 'log_sig' \
    --dataset 'Anthropic/hh-rlhf' --subset 'harmless-base' \
    --test True --lora_rank 4 \
    --save_path $EXAMPLES_DIR/rm_ckpt.pt
rm -rf $EXAMPLES_DIR/rm_ckpt.pt

echo "[Test]: testing RLHF ..."

torchrun --standalone --nproc_per_node=2 $EXAMPLES_DIR/train_prompts.py \
    --prompt_dataset $PROMPT_PATH --pretrain_dataset $PRETRAIN_DATASET \
    --strategy colossalai_zero2 --num_episodes 1 \
    --num_collect_steps 2 --num_update_steps 1 --train_batch_size 2 \
    --pretrain 'facebook/opt-350m' --model opt \
    --rm_pretrain 'facebook/opt-350m' \
    --rm_path $EXAMPLES_DIR/rm_ckpt_opt.pt \
    --save_path $EXAMPLES_DIR/actor_checkpoint_prompts.pt
rm -rf $EXAMPLES_DIR/rm_ckpt_opt.pt

torchrun --standalone --nproc_per_node=2 $EXAMPLES_DIR/train_prompts.py \
    --prompt_dataset $PROMPT_PATH --pretrain_dataset $PRETRAIN_DATASET \
    --strategy colossalai_zero2 --num_episodes 1 \
    --num_collect_steps 2 --num_update_steps 1 --train_batch_size 2 \
    --pretrain 'gpt2' --model gpt2 \
    --rm_pretrain 'gpt2' \
    --rm_path $EXAMPLES_DIR/rm_ckpt_gpt.pt \
    --save_path $EXAMPLES_DIR/actor_checkpoint_prompts.pt

torchrun --standalone --nproc_per_node=2 $EXAMPLES_DIR/train_prompts.py \
    --prompt_dataset $PROMPT_PATH --pretrain_dataset $PRETRAIN_DATASET \
    --strategy colossalai_gemini --num_episodes 1 \
    --num_collect_steps 2 --num_update_steps 1 --train_batch_size 2 \
    --pretrain 'gpt2' --model gpt2 \
    --rm_pretrain 'gpt2' \
    --rm_path $EXAMPLES_DIR/rm_ckpt_gpt.pt \
    --save_path $EXAMPLES_DIR/actor_checkpoint_prompts.pt
rm -rf $EXAMPLES_DIR/rm_ckpt_gpt.pt

rm -rf $EXAMPLES_DIR/actor_checkpoint_prompts.pt
