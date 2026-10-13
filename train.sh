#!/bin/bash

# 16G VRAM can only spawn 800 envs
uv run python scripts/policy/run_ppo_single_hand_retargeting.py \
    --exp_name=single_marker_pen_env_wf \
    --object_name=marker_pen \
    --num_envs=800 \
    --env_name=single_hand_retargeting_tactile_map_xarm_wf \
    --runner.total_iterations=30000 \
    --resume_checkpoint=latest
