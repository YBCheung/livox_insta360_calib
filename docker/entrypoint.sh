#!/bin/bash
# Sources ROS1 + the catkin workspace, then hands over to the requested command.
set -e

source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

# roslaunch and rosrun need a writable HOME for ~/.ros; the container may run as the
# host user (see run.sh --user) whose uid has no /etc/passwd entry here.
export HOME="${HOME:-/tmp}"
mkdir -p "$HOME/.ros" 2>/dev/null || { export HOME=/tmp; mkdir -p /tmp/.ros; }

if [ ! -d /calib/data/views ]; then
    echo "WARNING: /calib/data/views not found."
    echo "         Mount the calib bundle at /calib (see run.sh) and make sure steps"
    echo "         1-3 (capture, guess, cut) were done on the ROS2 side first."
    echo
fi

if [ "$1" = "bash" ] || [ -z "$1" ]; then
    cat <<'EOF'
=======================================================================
 livox_camera_calib (ROS1 Noetic) -- calibration container
=======================================================================
 Bundle mounted at /calib.  Everything below runs in here.

 1. Generate one config pair per virtual view:

      python3 /calib/calib_make_configs.py \
          --calib-repo /catkin_ws/src/livox_camera_calib

 2. Run all views:

      bash /calib/configs/run_all.sh

    ...or one at a time:

      roslaunch /calib/configs/calib_view.launch \
          config:=/calib/configs/calib_00.yaml

 3. Combine into the panorama-frame T_cam_lidar:

      python3 /calib/calib_prepare_views.py compose \
          --view-yaw 0 --extrinsic /calib/configs/results/extrinsic_00.txt \
          ... --out /calib/data/T_cam_lidar.txt

 Results land in /calib/configs/results, which is on the host.

 Param names this build actually reads:  cat /opt/calib_param_names.txt
=======================================================================
EOF
fi

exec "$@"
