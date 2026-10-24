#!/bin/bash

#--resume_checkpoint=latest

ACT_EMA=0.1

# Random 6-digit run id so repeated runs of the same config get distinct exp names
RUN_ID=$(LC_ALL=C tr -dc '0-9' </dev/urandom | head -c 6)
EXP_NAME=hammer_ema${ACT_EMA}_r0.01_ft0.5_noperturb_filtered-${RUN_ID}

cd "$(dirname "$0")"

# 16G VRAM can only spawn 800 envs
uv run python scripts/policy/run_ppo_single_hand_retargeting.py \
    --exp_name=${EXP_NAME} \
    --object_name=hammer \
    --num_envs=500 \
    --env_name=single_hand_retargeting_tactile_map_xarm_wf \
    --env.robot_args.action_ema=${ACT_EMA} \
    --runner.total_iterations=60000 \
    --algo.reward_scale=0.01 \
    --env.object_config.perturbation_force_scale=0.0 \
    --env.object_config.perturbation_torque_scale=0.0 \
    --reward_args.FingertipPositionTrackingReward.scale=25 \
    --reward_args.FingertipRotationTrackingReward.scale=17.5 \
