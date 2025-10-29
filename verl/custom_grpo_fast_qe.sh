#!/usr/bin/env bash
set -euxo pipefail

# === 让 Ray/子进程继承到 Torch & conda 的动态库路径 ===
export TORCH_LIB_DIR="$(python - <<'PY'
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

########## 0) 环境与项目基础 ##########
export JB_LIGHTWEIGHT=0    # 关闭轻量模式
export CUDA_VISIBLE_DEVICES=0,1,2,3
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 取消注释
RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export NCCL_DEBUG=WARN # INFO
export NCCL_DEBUG_SUBSYS=GRAPH,COLL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# 尽量先别禁用 NCCL 能力（之前这些容易把通信逼到慢路径，导致更容易卡在捕获点）
# unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
# unset NCCL_IB_DISABLE
# 避免复杂直连/跨架构 peer 访问引发的奇怪等待读写
export NCCL_P2P_DISABLE=1
# 或者完全走 SHM（本机）禁 IB（如果没有 IB，就禁用）
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

## 超参数›
gpu_count=4
# gpu_count=8 # 取消注释
export WORD_QE_MODE="only"

## 我的根目录，取消注释
MY_DIR="/ltstorage/home/4xin"
# MY_DIR="/mnt/workspace/xintong/xjf"

# wandb：换成你自己的 key / 项目名 / entity
export WANDB_API_KEY=a28f5c63c96f3bdc885978f31f4619b48811cff7
export WANDB_PROJECT="qwen2.5_3b_grpo_mt"
export WANDB_NAME="xcomet_${WORD_QE_MODE}_1024"
export WANDB_ENTITY="jetfan-universit-t-hamburg"
export WANDB_MODE=online
# 可选：指定 wandb 的本地缓存目录
export WANDB_DIR="${MY_DIR}/MT_Grpo"

########## 1) 下载奖励指标模型 ##########
# 设定固定的保存路径，之后需要 取消注释
COMET_DIR="/mnt/data1/users/4xin/hf/hub/wmt23-cometkiwi-da-xl"
WORD_QE_DIR="/mnt/data1/users/4xin/hf/hub/XCOMET-XL"
# COMET_DIR="${MY_DIR}/models/wmt23-cometkiwi-da-xl"
# WORD_QE_DIR="${MY_DIR}/models/XCOMET-XL"
export COMET_CKPT="${COMET_DIR}/checkpoints/model.ckpt"
export WORD_QE_CKPT="${WORD_QE_DIR}/checkpoints/model.ckpt"

python3 "${MY_DIR}/MT_Grpo/scripts/download_comet_ckpts.py" \
    --comet_dir "${COMET_DIR}" \
    --word_qe_dir "${WORD_QE_DIR}"

########## 2) 预处理数据 ##########
# 数据路径
DATA_DIR="${MY_DIR}/MT_Grpo/data"
train_file_path="${DATA_DIR}/train/parquet/train_base_enzh_zhen.parquet"
test_file_path="${DATA_DIR}/test/parquet/test_base_enzh_zhen.parquet"

python3 "${DATA_DIR}/process_data.py" \
    --train_files "${DATA_DIR}/train/json/train_zhen_6565.jsonl" "${DATA_DIR}/train/json/train_enzh_6565.jsonl" \
    --test_files "${DATA_DIR}/test/json/wmt23_zhen.jsonl" "${DATA_DIR}/test/json/wmt24_enzh.jsonl" \
    --tokenizer_path Qwen/Qwen2.5-3B \
    --template_type "base" \
    --train_output_file ${train_file_path} \
    --test_output_file ${test_file_path}

########## 3) 训练超参 ##########
# runs目录保存log和验证输出
runs_path="${MY_DIR}/MT_Grpo/runs/qwen2.5_3b_grpo_mt/xcomet_${WORD_QE_MODE}"
# 训练中周期性保存的模型参数(每 40 step)
checkpoints_path="${MY_DIR}/MT_Grpo/checkpoints/qwen2.5_3b_grpo_mt/xcomet_${WORD_QE_MODE}/trainer_npus_${gpu_count}"
# 自定义奖励计算代码路径
custom_reward_function_path="${MY_DIR}/MT_Grpo/verl/comet_reward_batch_with_ray.py"

# 安全创建所需的所有目录，防止运行报错
mkdir -p "${runs_path}/val"
mkdir -p "${checkpoints_path}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${train_file_path}" \
    data.val_files="${test_file_path}" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    reward_model.enable=True \
    reward_model.micro_batch_size_per_gpu=4 \
    custom_reward_function.path="${custom_reward_function_path}" \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=batch \
    trainer.val_before_train=False \
    trainer.logger=[wandb] \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${WANDB_NAME}" \
    trainer.n_gpus_per_node="${gpu_count}" \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.validation_data_dir="${runs_path}/val" \
    trainer.default_local_dir="${checkpoints_path}" \
    trainer.total_epochs=1 $@ 2>&1 | tee "${runs_path}/custom_grpo_fast_qe.log"