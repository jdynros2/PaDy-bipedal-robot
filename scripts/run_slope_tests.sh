#!/usr/bin/env bash
# run_slope_tests.sh — Run PaDy on 5 slope angles and collect CSVs.
#
# Usage:  bash run_slope_tests.sh
#
# Each test launches the full analysis stack (Gazebo + analyser + bridge),
# waits for the robot to fall (detected via CSV) or timeout, then kills
# everything and copies the CSV for that slope.
#
# Output CSVs land in ~/ros2_ws/data/slope_tests/

set -eo pipefail

DATA_DIR="$HOME/ros2_ws/data/slope_tests"
mkdir -p "$DATA_DIR"

SLOPES=("3p00" "3p25" "3p50" "3p75" "4p00")
LABELS=("3.00" "3.25" "3.50" "3.75" "4.00")

# Common launch args (from user's tuned run)
COMMON_ARGS=(
    kick_torque:=0.2
    kick_torque_right:=-0.2
    kick_follow_torque:=0
    body_force:=0
    hip_push_torque:=0
    hip_push_start_time:=0
    hip_push_stop_time:=12.0
    spawn_x:=0.34
    spawn_pitch:=0.42
    spawn_roll:=-0.2385
    spawn_yaw:=0.1
)

TIMEOUT_SEC=60   # max wall-clock seconds per test after sim starts

source /opt/ros/jazzy/setup.bash
source "$HOME/ros2_ws/install/setup.bash"

cleanup() {
    echo "  → Killing all processes..."
    # Kill the launch process group
    kill -- -$1 2>/dev/null || true
    sleep 1
    # Belt-and-suspenders: kill by name
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "ruby.*gz" 2>/dev/null || true
    pkill -9 -f "gait_analyser" 2>/dev/null || true
    pkill -9 -f "parameter_bridge" 2>/dev/null || true
    pkill -9 -f "robot_state_publisher" 2>/dev/null || true
    pkill -9 -f "continuous_hip_push" 2>/dev/null || true
    pkill -9 -f "knee_lock" 2>/dev/null || true
    pkill -9 -f "yaw_corrector" 2>/dev/null || true
    pkill -9 -f "ros2.bag.record" 2>/dev/null || true
    sleep 3
}

for i in "${!SLOPES[@]}"; do
    slope="${SLOPES[$i]}"
    label="${LABELS[$i]}"
    world="slope_${slope}deg.sdf"
    echo ""
    echo "========================================"
    echo "  TEST ${label}° — world: ${world}"
    echo "========================================"

    # Note the newest CSV before launching so we can find the new one
    BEFORE_CSV=$(ls -t "$HOME/ros2_ws/data/run_"*.csv 2>/dev/null | head -1 || true)

    # Launch in its own process group so we can kill everything cleanly
    setsid ros2 launch pady_robot analysis.launch.py \
        world:="${world}" \
        "${COMMON_ARGS[@]}" &>/dev/null &
    LAUNCH_PID=$!

    echo "  → Waiting for sim to start..."
    sleep 18  # Gazebo + spawn + unpause takes ~15-18s

    # Wait for a NEW csv to appear
    echo "  → Watching for CSV and fall..."
    START=$SECONDS
    FALLEN=0
    while (( SECONDS - START < TIMEOUT_SEC )); do
        # Find the newest CSV that's different from BEFORE_CSV
        NEW_CSV=$(ls -t "$HOME/ros2_ws/data/run_"*.csv 2>/dev/null | head -1 || true)
        if [[ -n "$NEW_CSV" && "$NEW_CSV" != "$BEFORE_CSV" ]]; then
            # Check if fall_detected column has a 1
            if tail -5 "$NEW_CSV" 2>/dev/null | grep -q ",1$"; then
                echo "  → Fall detected at wall-clock $((SECONDS - START))s"
                FALLEN=1
                sleep 1  # let CSV flush final rows
                break
            fi
        fi
        sleep 1
    done

    if [[ $FALLEN -eq 0 ]]; then
        echo "  → Timeout (${TIMEOUT_SEC}s) — no fall detected"
    fi

    cleanup $LAUNCH_PID

    # Copy the CSV
    NEW_CSV=$(ls -t "$HOME/ros2_ws/data/run_"*.csv 2>/dev/null | head -1 || true)
    if [[ -n "$NEW_CSV" && "$NEW_CSV" != "$BEFORE_CSV" ]]; then
        DEST="${DATA_DIR}/slope_${slope}deg.csv"
        cp "$NEW_CSV" "$DEST"
        ROWS=$(wc -l < "$DEST")
        echo "  → Saved: ${DEST} (${ROWS} rows)"
    else
        echo "  → WARNING: No new CSV found for this run!"
    fi

    echo "  → Waiting before next test..."
    sleep 5
done

echo ""
echo "========================================"
echo "  ALL TESTS COMPLETE"
echo "  Results in: ${DATA_DIR}/"
echo "========================================"
ls -lh "${DATA_DIR}/"
