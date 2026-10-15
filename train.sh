#!/bin/bash

# 16G VRAM can only spawn 800 envs
uv run python scripts/policy/run_ppo_single_hand_retargeting.py \
    --exp_name=single_marker_pen_env_wf_ema0.5 \
    --object_name=marker_pen \
    --num_envs=800 \
    --env_name=single_hand_retargeting_tactile_map_xarm_wf \
    --env.robot_args.action_ema=0.5 \
    --runner.total_iterations=30000 \
    --resume_checkpoint=latest
