#!/bin/bash
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# ============================================================================
# Trajectory-generation training.
# ----------------------------------------------------------------------------
# Fine-tunes on all-atom MD trajectories (Atlas + MISATO + MDposit). The model
# learns to roll out continuous-time trajectories under the temporal attention
# mechanism. Active losses: per-frame diffusion + RMSF / relative-RMSF /
# local-RMSF / ACF / ensemble + TICA-dynamics.
#
# Requires processed trajectory data (Atlas via scripts/data_prep; MISATO &
# MDposit via the compressed codec in scripts/codec) and precomputed embeddings
# (scripts/encode_embeddings.sh). See docs/data_and_training.md.
# ============================================================================
set -euo pipefail

# ---- environment (edit to match your install) ----
export LAYERNORM_TYPE=fast_layernorm
export CUTLASS_PATH=${CUTLASS_PATH:?set CUTLASS_PATH to your cutlass checkout}
export CUDA_HOME=${CUDA_HOME:?set CUDA_HOME}
export USE_DEEPSPEED_EVO_ATTTENTION=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT_SECOND=7200
export TRITON_CACHE_DIR=/tmp/triton_cache_$$
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ---- data roots (see configs/configs_data.py) ----
export BIOKINEMA_ATLAS_ROOT=${BIOKINEMA_ATLAS_ROOT:?set BIOKINEMA_ATLAS_ROOT}
export BIOKINEMA_UNBINDING_ROOT=${BIOKINEMA_UNBINDING_ROOT:?set BIOKINEMA_UNBINDING_ROOT (MISATO/MDposit/unbinding)}
export BIOKINEMA_DATA_ROOT=${BIOKINEMA_DATA_ROOT:?set BIOKINEMA_DATA_ROOT}

# Initialization checkpoint (Protenix base model_v0.2.0.pt, or a prior BioKinema checkpoint).
checkpoint_path=${BIOKINEMA_INIT_CKPT:?set BIOKINEMA_INIT_CKPT to the init checkpoint}

USE_WANDB="${USE_WANDB:-false}"
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29501}

# ---- recipe (defaults = "sqrt": complexes + short MD). Override via env for the
#      "beta=0.25" recipe (long single-chain MD), e.g.:
#   TRAIN_SETS=MSR-CATH2,MSR-CATH1,MSR-octapeptide \
#   TEST_SETS=MSR-CATH2 SAMPLE_WEIGHTS=0.756,0.095,0.148 BETA=0.25 \
#   RUN_NAME=BioKinema_beta0.25 bash train_trajectory.sh
#   (also export BIOKINEMA_MSR_ROOT and BIOKINEMA_MSM_CACHES for the MSR sets).
RUN_NAME="${RUN_NAME:-BioKinema_atlas+misato+mdposit}"
TRAIN_SETS="${TRAIN_SETS:-misato_train,atlas_train,mdposit}"
TEST_SETS="${TEST_SETS:-misato_val,atlas_val}"
SAMPLE_WEIGHTS="${SAMPLE_WEIGHTS:-1,1,1}"
BETA="${BETA:-0.5}"
EVAL_FIRST="${EVAL_FIRST:-true}"

torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=127.0.0.1 \
         --master_port=$MASTER_PORT \
         ./runner/train.py \
        --run_name "${RUN_NAME}" \
        --seed 42 \
        --base_dir ./output_all \
        --dtype bf16 \
        --project protenix \
        --use_wandb ${USE_WANDB} \
        --diffusion_batch_size 1 \
        --diffusion_chunk_size None \
        --eval_interval 500 \
        --log_interval 10 \
        --checkpoint_interval 1000 \
        --ema_decay 0.999 \
        --train_crop_size 800 \
        --max_steps 100000 \
        --warmup_steps 200 \
        --lr 0.0001 \
        --iters_to_accumulate 4 \
        --sample_diffusion.N_step 20 \
        --sample_diffusion.noise_scale_lambda 1.75 \
        --sample_diffusion.step_scale_eta 1.5 \
        --sample_diffusion.N_sample 1 \
        --model.diffusion_module.beta ${BETA} \
        --data.train_sets ${TRAIN_SETS} \
        --data.test_sets ${TEST_SETS} \
        --data.train_sampler.train_sample_weights ${SAMPLE_WEIGHTS} \
        --data.num_dl_workers 16 \
        --loss.weight.alpha_velocity 0.0 \
        --loss.weight.alpha_rmsf 1 \
        --loss.weight.alpha_rel_rmsf 8 \
        --loss.weight.alpha_local_rmsf 8 \
        --loss.weight.alpha_lig_bond 0.0 \
        --loss.weight.alpha_center 0.0 \
        --loss.weight.alpha_ensemble 0.1 \
        --loss.weight.alpha_acf 8 \
        --loss.weight.alpha_tica_dynamics 0.25 \
        --model.diffusion_module.causal_mask false \
        --load_checkpoint_path ${checkpoint_path} \
        --load_params_only true \
        --eval_first ${EVAL_FIRST} \
        --load_strict false \
        "$@"
