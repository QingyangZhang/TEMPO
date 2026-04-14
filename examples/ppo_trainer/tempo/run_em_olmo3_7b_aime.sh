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
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

export WANDB_MODE=offline
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"

export VLLM_USE_V1=1

project_name='TEMPO-OlMO3-TTT'

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

adv_estimator=em_token

rollout_engine=vllm
rollout_mode=sync # can be async to speedup large scale xps
shuffle_dataset=true
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.00

loss_mode="gspo"
loss_agg_mode="seq-mean-token-mean"

clip_ratio_low=3e-4
clip_ratio_high=5e-4

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 16))

train_batch_size=256
train_prompt_mini_bsz=64
n_resp_per_prompt=8

exp_name="Olmo3-7B-EM-AIME-token-level-fixed-critic-${DATE}:${TIME_TAG}"

MODEL_PATH=${MODEL_PATH:-"path/to/actor_model"}
CRITIC_PATH=${CRITIC_PATH:-"path/to/critic_model"}
CKPTS_DIR=${CKPTS_DIR:-"./checkpoints/${project_name}/${exp_name}"}

train_files=${train_files:-"path/to/train_data.parquet"}
test_data_dir=${TEST_DATA_DIR:-"path/to/test_data"}
aime_2024="${test_data_dir}/aime_2024.parquet"
aime_2025="${test_data_dir}/aime_2025.parquet"
aime_2026="${test_data_dir}/aime_2026.parquet"
amobench="${test_data_dir}/amobench.parquet"
beyond_aime="${test_data_dir}/beyond_aime.parquet"
answerbench="${test_data_dir}/answerbench.parquet"
olymmath="${test_data_dir}/olymmath.parquet"
aime_2026_olmo="${test_data_dir}/aime_2026_olmo_prompt.parquet"
olymmath_olmo="${test_data_dir}/olymmath_olmo_prompt.parquet"
beyond_aime_olmo="${test_data_dir}/beyond_aime_olmo_prompt.parquet"

test_files="['$aime_2024', '$aime_2025']"

#test_files="['$olymmath']"

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1.0

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
critic_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
offload=True
# gen_tp=4

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
#     -- python3 -m recipe.dapo.src.main_dapo \
HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_ppo_em \
    algorithm.adv_estimator=${adv_estimator} \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.max_num_gen_batches=10 \
    +algorithm.gemma=1.0 \
    algorithm.lam=1.0 \
    algorithm.norm_adv_by_std_in_grpo=True \
    reward_model.reward_manager=physics \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.train_batch_size=${train_batch_size} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.shuffle=$shuffle_dataset \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.filter_overlong_prompts=true \
    data.val_batch_size=128 \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.name=${rollout_engine} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
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
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    critic.self_critic=True \
    critic.enable=True \
    critic.model.path="${CRITIC_PATH}" \
    critic.checkpoint.load_contents=['model'] \
    critic.optim.lr=0.0 \
    critic.cliprange_value=0.1 \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=$critic_max_token_len_per_gpu \
    critic.ulysses_sequence_parallel_size=$sp_size \
    critic.model.fsdp_config.param_offload=$offload \
    critic.model.fsdp_config.optimizer_offload=$offload \
    critic.use_dynamic_bsz=${use_dynamic_bsz} \
    critic.optim.lr_warmup_steps=0 \
    critic.optim.warmup_style=constant \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=False \
    trainer.validation_data_dir=null \
    trainer.test_freq=4 \
    trainer.save_freq=32 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    trainer.total_training_steps=20480 \
    +trainer.iter_ttrl=true \
    +trainer.load_critic_only=false \
    +trainer.critic_warmup_steps=0 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_from_path=null \
    trainer.resume_mode=disable
