#!/bin/bash


uv run python scripts/policy/run_ppo_single_hand_retargeting.py \
    --exp_name=single_marker_pen_env4096 \
    --object_name=marker_pen \
    --num_envs=800 \
    --runner.total_iterations=30000 \
    --resume_checkpoint=latest
