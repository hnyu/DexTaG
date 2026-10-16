#!/bin/bash

#--resume_checkpoint=latest

EXP_NAME=single_marker_pen_env_wf_ema0.1


cd "$(dirname "$0")"

# Snapshot the code being run (src, scripts, train.sh, eval.sh) before launching. The run folder
# (logs/<exp_name>/<timestamp>/) is created by the training script, so the snapshot is moved there
# once the run writes its configs.
mkdir -p "logs/${EXP_NAME}"
SNAPSHOT="logs/${EXP_NAME}/.DexTaG_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$SNAPSHOT" --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
    src scripts train.sh eval.sh




# 16G VRAM can only spawn 800 envs
uv run python scripts/policy/run_ppo_single_hand_retargeting.py \
    --exp_name=${EXP_NAME} \
    --object_name=marker_pen \
    --num_envs=800 \
    --env_name=single_hand_retargeting_tactile_map_xarm_wf \
    --env.robot_args.action_ema=0.1 \
    --runner.total_iterations=30000 &




TRAIN_PID=$!
trap 'kill -INT $TRAIN_PID 2>/dev/null' INT TERM

# Move the snapshot into the run folder once it has written its configs (fresh or resumed run).
# A resumed run overwrites the earlier snapshot.
while kill -0 $TRAIN_PID 2>/dev/null; do
    CONFIG=$(find "logs/${EXP_NAME}" -mindepth 3 -maxdepth 3 -path '*/configs/env_args.yaml' \
        -newer "$SNAPSHOT" 2>/dev/null | head -n 1)
    if [ -n "$CONFIG" ]; then
        DEST="$(dirname "$(dirname "$CONFIG")")/DexTaG.tar.gz"
        mv -f "$SNAPSHOT" "$DEST" && echo "Code snapshot saved to ${DEST}"
        break
    fi
    sleep 2
done
[ -e "$SNAPSHOT" ] && echo "Training exited before creating a run folder; snapshot left at ${SNAPSHOT}"

wait $TRAIN_PID
