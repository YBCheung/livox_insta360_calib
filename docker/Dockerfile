# ROS1 Noetic image for hku-mars/livox_camera_calib.
#
# The calibrator is catkin-only, so it cannot run in the ROS2 flight container. This
# image is standalone and offline -- it never touches the drone. Build it once on the
# laptop, mount the calib bundle, run the calibration, copy the numbers back.
#
# livox_camera_calib depends on livox_ros_driver (for CustomMsg headers, needed at
# build time even though we feed it .pcd/.png directly), which in turn needs
# Livox-SDK. Both are built here.

FROM ros:noetic-perception

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        libceres-dev libeigen3-dev libopencv-dev \
        libapr1-dev \
        ros-noetic-cv-bridge \
        ros-noetic-pcl-ros \
        ros-noetic-pcl-conversions \
        ros-noetic-rviz \
        python3-pip python3-numpy python3-yaml \
        libgl1-mesa-glx libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# --- Livox-SDK (prerequisite of livox_ros_driver) ---
RUN git clone --depth 1 https://github.com/Livox-SDK/Livox-SDK.git /opt/Livox-SDK \
    && cmake -S /opt/Livox-SDK -B /opt/Livox-SDK/build \
    && cmake --build /opt/Livox-SDK/build -j"$(nproc)" \
    && cmake --install /opt/Livox-SDK/build \
    && ldconfig

# --- catkin workspace ---
WORKDIR /catkin_ws
RUN mkdir -p src \
    && git clone --depth 1 https://github.com/Livox-SDK/livox_ros_driver.git \
         src/livox_ros_driver \
    && git clone --depth 1 https://github.com/hku-mars/livox_camera_calib.git \
         src/livox_camera_calib

RUN . /opt/ros/noetic/setup.sh \
    && catkin_make -DCMAKE_BUILD_TYPE=Release -j"$(nproc)"

# Record which rosparam names the build actually reads, so a schema mismatch with the
# generated configs shows up immediately rather than as a silent all-zeros calibration.
RUN grep -rhoE '(param|getParam)[^;]*"[a-zA-Z_/]+"' \
        src/livox_camera_calib/src src/livox_camera_calib/include 2>/dev/null \
        | grep -oE '"[a-zA-Z_/]+"' | sort -u > /opt/calib_param_names.txt || true

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV CALIB_BUNDLE=/calib
WORKDIR /calib
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
