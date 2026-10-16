#!/bin/bash

#--resume_checkpoint=latest

ACT_EMA=0.1

EXP_NAME=marker_pen_env_wf_ema${ACT_EMA}_noguide

cd "$(dirname "$0")"

# 16G VRAM can only spawn 800 envs
uv run python scripts/policy/run_ppo_single_hand_retargeting.py \
    --exp_name=${EXP_NAME} \
    --object_name=marker_pen \
    --num_envs=700 \
    --env_name=single_hand_retargeting_tactile_map_xarm_wf \
    --env.robot_args.action_ema=${ACT_EMA} \
    --runner.total_iterations=60000 \
