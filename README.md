# TEMPO: Scaling Test-time Training for Large Reasoning Models

<div align="center">

**TEMPO** is a semi-supervised reinforcement learning framework for scaling test-time training (TTT) of large reasoning models. It alternates between a **Critic Recalibration** step (E-step) on labeled data and a **Policy Refinement** step (M-step) on unlabeled data, enabling effective use of both labeled and unlabeled problem sets.

</div>

---

## News

- **[2026-04]** TEMPO code released.

---

## Introduction

Test-time training (TTT) with reinforcement learning (RL) improves LLM reasoning by rolling out and learning from model-generated solutions. However, standard online RL (e.g., GRPO, DAPO) requires reward labels for every training prompt, which limits scalability when labeled data is scarce.

**TEMPO** frames TTT as an Expectation-Maximization (EM) problem:

- **E-step (Critic Recalibration):** A value critic is updated on a small labeled dataset D_L to estimate solution quality.
- **M-step (Policy Refinement):** The policy is updated on a larger unlabeled dataset D_U using critic-assigned advantages, without needing ground-truth rewards.

This allows TEMPO to leverage far more training prompts than labeled-only methods while maintaining the exploration diversity that purely outcome-supervised methods sacrifice.

---

## Key Results

### Main Benchmark Results (pass@1, greedy)

| Model | Benchmark | Base | TEMPO | Gain |
|-------|-----------|------|-------|------|
| OLMo3-7B | AIME 2024 | 33.0 | 51.1 | +18.1 |
| OLMo3-7B | AIME 2025 | 26.3 | 37.0 | +10.7 |
| OLMo3-7B | BeyondAIME | 17.6 | 24.5 | +6.9 |
| Qwen3-8B | AIME 2024 | 26.3 | 42.7 | +16.4 |
| Qwen3-8B | AIME 2025 | 25.4 | 40.8 | +15.4 |
| Qwen3-14B | AIME 2024 | 42.3 | 65.8 | +23.5 |
| Qwen3-14B | AIME 2025 | 37.1 | 44.6 | +7.5 |

Results for OLMo3-7B reported at 256 training steps; Qwen3-14B at 224 steps (training continues to improve beyond these checkpoints).

### Diversity Preservation

Unlike TTRL and EMPO, which improve mean accuracy at the cost of solution diversity (pass@k degrades), **TEMPO preserves and improves both mean accuracy and pass@k**. This reflects a fundamentally different learning dynamic: TEMPO does not collapse the model's exploration capability.

---

## Installation

TEMPO is built on top of [verl](https://github.com/volcengine/verl). Install the base dependencies first:

```bash
git clone https://github.com/QingyangZhang/TEMPO.git
cd TEMPO
pip install -e .
pip install -r requirements-cuda.txt
```

Additional requirements for vLLM rollout:

```bash
pip install vllm>=0.8.0
```

---

## Quick Start

Training scripts are located in `examples/ppo_trainer/tempo/`.

### OLMo3-7B on AIME

```bash
export MODEL_PATH=/path/to/actor_model
export CRITIC_PATH=/path/to/critic_model
export CKPTS_DIR=./checkpoints/tempo-olmo3-7b
export train_files=/path/to/train_data.parquet
export TEST_DATA_DIR=/path/to/test_data

bash examples/ppo_trainer/tempo/run_em_olmo3_7b_aime.sh
```

### Qwen3-14B on AIME

```bash
export MODEL_PATH=/path/to/actor_model
export CRITIC_PATH=/path/to/critic_model
export CKPTS_DIR=./checkpoints/tempo-qwen3-14b
export train_files=/path/to/train_data.parquet
export TEST_DATA_DIR=/path/to/test_data

bash examples/ppo_trainer/tempo/run_em_qwen3_14b_aime.sh
```

### Evaluation

```bash
# Evaluate pass@k
bash examples/ppo_trainer/tempo/eval_pass_k.sh

# Evaluate pass@1 (greedy)
bash examples/ppo_trainer/tempo/eval.sh
```

### Key Training Arguments

| Argument | Description |
|----------|-------------|
| `algorithm.adv_estimator=em_token` | Use token-level EM advantage estimator |
| `+algorithm.filter_groups.enable=True` | Enable semi-supervised group filtering |
| `critic.self_critic=True` | Enable critic recalibration (E-step) |
| `critic.enable=True` | Enable critic model |
| `+trainer.iter_ttrl=true` | Enable iterative TTT loop |
| `data.train_files` | Labeled training data (parquet) |
| `data.val_files` | Validation data (list of parquets) |

---

## Algorithm Overview

```
for each training step:
    # E-step: Critic Recalibration
    collect labeled rollouts from D_L
    update critic on labeled data with reward supervision

    # M-step: Policy Refinement
    collect unlabeled rollouts from D_U
    compute advantages using recalibrated critic
    update policy via GSPO/PPO loss
```

The labeled dataset D_L provides reward signal to recalibrate the critic; the unlabeled dataset D_U provides a much larger and more diverse set of training prompts for the policy. The critic bridges the two, acting as a soft reward model over unlabeled data.

---

## Acknowledgements

TEMPO is built on top of [verl](https://github.com/volcengine/verl) by ByteDance. We thank the verl team for their open-source RL training infrastructure.

This work was conducted at Tianjin University, Tongyi Lab, and Shanghai AI Lab.
