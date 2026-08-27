#!/bin/bash
# Auto-generated. Runs livox_camera_calib once per virtual view.
# Each view has its own initial extrinsic, so they cannot share a run.
set -e
LAUNCH="/calib/configs/calib_view.launch"

echo "=== view 00 ==="
roslaunch "$LAUNCH" config:=/calib/configs/calib_00.yaml

echo "=== view 01 ==="
roslaunch "$LAUNCH" config:=/calib/configs/calib_01.yaml

echo "=== view 02 ==="
roslaunch "$LAUNCH" config:=/calib/configs/calib_02.yaml

echo "=== view 03 ==="
roslaunch "$LAUNCH" config:=/calib/configs/calib_03.yaml

echo "=== view 04 ==="
roslaunch "$LAUNCH" config:=/calib/configs/calib_04.yaml

echo "=== view 05 ==="
roslaunch "$LAUNCH" config:=/calib/configs/calib_05.yaml

echo "All views done. Results in:"
echo "  /calib/configs/results"
