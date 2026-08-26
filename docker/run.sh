#!/bin/bash
# Start the ROS1 calibration container with the calib bundle mounted at /calib.
#
#   bash run.sh              interactive shell (prints the steps)
#   bash run.sh auto         generate configs + run every view, unattended
#
# Runs as the host user so generated configs and results are NOT root-owned on the
# host -- the usual reason a calib directory ends up needing chown afterwards.
set -e

IMAGE="${IMAGE:-livox-camera-calib:noetic}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="${BUNDLE:-$(cd "$HERE/.." && pwd)}"   # the calib/ directory

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image $IMAGE not found. Build it first:"
    echo "    bash $HERE/build.sh"
    exit 1
fi

if [ ! -d "$BUNDLE/data/views" ]; then
    echo "ERROR: $BUNDLE/data/views not found."
    echo "Copy the whole calib/ bundle from the drone, including data/."
    exit 1
fi

# X11 so the calibrator's OpenCV windows and rviz can display.
XSOCK=/tmp/.X11-unix
XARGS=()
if [ -n "$DISPLAY" ] && [ -d "$XSOCK" ]; then
    XARGS=(-e "DISPLAY=$DISPLAY" -v "$XSOCK:$XSOCK:rw")
    xhost +local:docker >/dev/null 2>&1 || \
        echo "note: 'xhost +local:docker' failed; GUI windows may not appear."
else
    echo "note: no DISPLAY -- running headless. Visualisation windows will be skipped."
fi

CMD=(bash)
if [ "$1" = "auto" ]; then
    CMD=(bash -lc '
        set -e
        python3 /calib/calib_make_configs.py --calib-repo /catkin_ws/src/livox_camera_calib
        bash /calib/configs/run_all.sh
        echo
        echo "All views done. Now run compose (see the printed command above)."')
fi

echo "Bundle : $BUNDLE  ->  /calib"
echo "Image  : $IMAGE"
echo

exec docker run -it --rm \
    --name livox_calib \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    "${XARGS[@]}" \
    -v "$BUNDLE:/calib" \
    "$IMAGE" "${CMD[@]}"
