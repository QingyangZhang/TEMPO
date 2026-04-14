#!/usr/bin/env bash
set -x



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
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

#export WANDB_MODE=online
#export WANDB_API_KEY="3033da39c2bf837949ca77dbd720af778de6515d"
export WANDB_DIR="${WANDB_DIR:-./wandb}"

export VLLM_USE_V1=1

project_name='TEMPO-PassAtK-Eval'

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

# Algorithm settings
adv_estimator=grpo
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=1.0

# Generation settings
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 16))

# Pass@K settings: n_resp_per_prompt must be >= max K you want to evaluate
# With n=16, you get pass@1, pass@2, pass@4, pass@8, pass@16
n_resp_per_prompt=32

# Batch settings
train_batch_size=256
train_prompt_mini_bsz=64

# Performance settings
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
critic_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
offload=True

exp_name="Eval-PassAtK-${DATE}:${TIME_TAG}"

# Model path (change as needed)
MODEL_PATH=${MODEL_PATH:-"path/to/actor_model"}

CKPTS_DIR=${CKPTS_DIR:-"./checkpoints/${project_name}/${exp_name}"}

train_files=${train_files:-"path/to/train_data.parquet"}
# Test datasets
test_data_dir=${TEST_DATA_DIR:-"path/to/test_data"}
#aime_2024="${test_data_dir}/aime_2024.parquet"
aime_2025="${test_data_dir}/aime_2025.parquet"
aime_2026="${test_data_dir}/aime_2026.parquet"
#beyond_aime="${test_data_dir}/beyond_aime.parquet"
olymmath="${test_data_dir}/olymmath.parquet"

aime_2024="${test_data_dir}/aime_2024_olmo_prompt.parquet"
#aime_2025="${test_data_dir}/aime_2025_olmo_prompt.parquet"
#beyond_aime="${test_data_dir}/beyond_aime_olmo_prompt.parquet"

# Default: evaluate on AIME 2026. Change test_files to evaluate other datasets.
test_files="['$aime_2024']"
#test_files="['$beyond_aime']"
#test_files="['$olymmath']"
# Parse CLI overrides
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2;;
        --n_samples) n_resp_per_prompt="$2"; shift 2;;
        --test_files) test_files="$2"; shift 2;;
        --train_batch_size) train_batch_size="$2"; shift 2;;
        --mini_bsz) train_prompt_mini_bsz="$2"; shift 2;;
        --exp_name) exp_name="$2"; shift 2;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1 $2"; shift 2;;
    esac
done

echo "=== Pass@K Evaluation ==="
echo "Model: $MODEL_PATH"
echo "N samples per prompt: $n_resp_per_prompt"
echo "Test files: $test_files"
echo "Experiment: $exp_name"

# Run evaluation (val_only mode, no training)
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=${train_batch_size} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.filter_overlong_prompts=true \
    data.val_batch_size=128 \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    critic.enable=False \
    reward_model.enable=False \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.validation_data_dir=null \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable \
    $EXTRA_ARGS

echo ""
echo "=== Pass@K evaluation complete ==="
echo "$MODEL_PATH"
