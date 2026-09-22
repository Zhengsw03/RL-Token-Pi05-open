#!/usr/bin/env bash
#
# SO-101 teleoperation launcher
#
# Works around two pitfalls that come up repeatedly:
#   1. Every failed lerobot connection leaves an orphan Rerun viewer holding
#      port 9876 at 100% CPU, so the next run never opens a new window (it just
#      shows no image). Root cause: lerobot_teleoperate.py calls
#      teleop.connect()/robot.connect() outside the try/finally, so when connect
#      raises, shutdown_visualization() in the finally block never runs.
#   2. --display_data=true requires --robot.cameras; without cameras no images can
#      be captured and Rerun does not even create the image panel (Spatial2DView
#      is only generated from image_paths).
#
# Usage:
#   bash run_so101_teleop.sh
#
# Overridable environment variables:
#   FOLLOWER_PORT=... LEADER_PORT=... TOP_CAM=0 WRIST_CAM=3 FPS=30
#
# Note: camera indices drift (video0,1,2,3 was observed to become video0,1,3,4),
#       and these two cameras share the same USB serial number
#       (200901010001), so by-id paths cannot tell them apart; use
#       /dev/v4l/by-path/ paths when they must be pinned.
#       Confirm the indices first with `lerobot-find-cameras opencv`.
#
set -uo pipefail

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
TOP_CAM="${TOP_CAM:-0}"
WRIST_CAM="${WRIST_CAM:-3}"
FPS="${FPS:-30}"

# ---------- 1. Clean up orphan Rerun viewers ----------
if pgrep -f 'rerun_cli/rerun' >/dev/null 2>&1; then
    echo "==> Found a leftover Rerun viewer, cleaning up"
    pkill -f 'rerun_cli/rerun' 2>/dev/null
    sleep 1
    if pgrep -f 'rerun_cli/rerun' >/dev/null 2>&1; then
        echo "    Still running, forcing termination"
        pkill -9 -f 'rerun_cli/rerun' 2>/dev/null
        sleep 1
    fi
    echo "    Cleaned up (port 9876 released)"
else
    echo "==> No leftover Rerun viewer"
fi

# ---------- 2. Serial-port permission preflight ----------
echo "==> Checking serial ports"
fail=0
for p in "$FOLLOWER_PORT" "$LEADER_PORT"; do
    if [ ! -e "$p" ]; then
        echo "    [FAIL] $p does not exist (arm not plugged in?)" >&2
        fail=1
        continue
    fi
    if [ ! -r "$p" ] || [ ! -w "$p" ]; then
        echo "    [FAIL] $p is not readable/writable" >&2
        ls -l "$p" >&2
        fail=1
        continue
    fi
    echo "    [OK] $p"
done

if [ "$fail" -ne 0 ]; then
    echo >&2
    echo "Fix permissions with:  sudo bash $(dirname "$0")/fix_so101_permission.sh" >&2
    exit 1
fi

# ---------- 3. Launch ----------
echo "==> Starting teleoperate (top=video$TOP_CAM, wrist=video$WRIST_CAM, fps=$FPS)"
echo

exec lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id=so101_follower \
    --robot.cameras="{top: {type: opencv, index_or_path: $TOP_CAM, width: 640, height: 480, fps: $FPS}, wrist: {type: opencv, index_or_path: $WRIST_CAM, width: 640, height: 480, fps: $FPS}}" \
    --teleop.type=so101_leader \
    --teleop.port="$LEADER_PORT" \
    --teleop.id=so101_leader \
    --fps="$FPS" \
    --display_data=true
