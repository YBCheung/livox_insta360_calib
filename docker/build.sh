#!/bin/bash
# Build the ROS1 calibration image. Run once on the laptop.
set -e

IMAGE="${IMAGE:-livox-camera-calib:noetic}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building $IMAGE (clones Livox-SDK, livox_ros_driver, livox_camera_calib)..."
echo "First build takes a while -- it compiles the SDK and the catkin workspace."
echo

docker build -t "$IMAGE" "$HERE"

echo
echo "Done: $IMAGE"
echo "Next:  bash $HERE/run.sh"
