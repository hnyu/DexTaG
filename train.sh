#!/bin/bash

#--resume_checkpoint=latest

ACT_EMA=0.5

TIME=$(date +%Y%m%d_%H%M%S)
EXP_NAME=marker_pen_ema${ACT_EMA}_noguide_start100_r0.01-${TIME}

cd "$(dirname "$0")"

# 16G VRAM can only spawn 800 envs
uv run python scripts/policy/run_ppo_single_hand_retargeting.py \
    --exp_name=${EXP_NAME} \
    --object_name=marker_pen \
    --num_envs=600 \
    --env_name=single_hand_retargeting_tactile_map_xarm_wf \
    --env.robot_args.action_ema=${ACT_EMA} \
    --runner.total_iterations=60000 \
    --algo.reward_scale=0.01 \
    --env.randomize_starting_step_before=100 \
