#!/bin/bash

uv run python scripts/policy/run_ppo_single_hand_retargeting.py \
    --eval=True \
    --exp_name=single_marker_pen_env_wf \
    --object_name=marker_pen \
    --env_name=single_hand_retargeting_tactile_map_xarm_wf \
    --save_video=True \
    --num_runs_per_traj=1
