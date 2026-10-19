#!/bin/bash

#--resume_checkpoint=latest

ACT_EMA=0.2

# Random 6-digit run id so repeated runs of the same config get distinct exp names
RUN_ID=$(LC_ALL=C tr -dc '0-9' </dev/urandom | head -c 6)
EXP_NAME=marker_pen_ema${ACT_EMA}_noguide_mass0.01_r0.01_ft0.5-${RUN_ID}

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
    --env.object_config.mass=0.01 \
    --reward_args.FingertipPositionTrackingReward.scale=25 \
    --reward_args.FingertipRotationTrackingReward.scale=17.5 \
