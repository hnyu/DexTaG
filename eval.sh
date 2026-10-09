#!/bin/bash

uv run python scripts/policy/run_ppo_single_hand_retargeting.py \
    --eval=True \
    --exp_name=single_mark_pen \
    --object_name=marker_pen \
    --save_video=True \
    --num_runs_per_traj=1 \