#!/bin/bash

uv run python scripts/policy/run_ppo_single_hand_retargeting.py \
    --eval=True \
    --exp_name=marker_pen_env_wf_ema0.5_noguide_r0.01 \
    --save_video=True \
    --num_runs_per_traj=1
