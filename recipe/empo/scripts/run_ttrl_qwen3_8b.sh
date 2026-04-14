#!/usr/bin/env bash
set -x



# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK:-0}"
# export NCCL_NVLS_ENABLE=False
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"


#export SWANLAB_API_KEY="i3fSd5Un24j2mdCeT4P4T"
#export SWANLAB_LOG_DIR="path/to/resource"
#export SWANLAB_MODE="local"
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"

export VLLM_USE_V1=1

project_name='TEMPO-Qwen3-TTT'

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=3e-4
clip_ratio_high=5e-4

# EMPO
target="entropy"

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 16))
enable_overlong_buffer=False
overlong_buffer_len=1024
overlong_penalty_factor=0.0

loss_mode="gspo"
loss_agg_mode="seq-mean-token-mean"

enable_filter_groups=False
confidence_lower_bound=0.3
confidence_upper_bound=1.0
filter_groups_metric=score
max_num_gen_batches=10

train_prompt_bsz=16
gen_prompt_bsz=$((train_prompt_bsz * 1))
train_prompt_mini_bsz=16
n_resp_per_prompt=8

exp_name="EMPO-Qwen3-8B"

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
NNODES=${NNODES:-1}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"path/to/resource"}
MODEL_PATH=${MODEL_PATH:-"path/to/model"}
CRITIC_PATH=${CRITIC_PATH:-"path/to/resource"}
CKPTS_DIR=${CKPTS_DIR:-"./checkpoints/${project_name}/${exp_name}"}

train_files=${train_files:-"path/to/train_data.parquet"}
test_data_dir=${TEST_DATA_DIR:-"path/to/test_data"}
aime_2024="${test_data_dir}/aime_2024.parquet"
aime_2025="${test_data_dir}/aime_2025.parquet"
amobench="${test_data_dir}/amobench.parquet"
beyond_aime="${test_data_dir}/beyond_aime.parquet"

test_files="['$aime_2024', '$aime_2025']"


# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

# Mathematically equivalent
use_dynamic_bsz=true
offload=True


HYDRA_FULL_ERROR=1 python3 -m recipe.empo.src.main_dapo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.use_shm=False \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=64 \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.confidence_lower_bound=${confidence_lower_bound} \
    algorithm.filter_groups.confidence_upper_bound=${confidence_upper_bound} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0. \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=empo \
    reward_model.reward_kwargs.reward_type=${target} \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.use_xverify=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.test_freq=4 \
    trainer.save_freq=32 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=1000 \
    trainer.total_training_steps=30000 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_from_path=null \
    trainer.resume_mode=disable