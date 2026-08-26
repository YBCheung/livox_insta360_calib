# Insta360 ONE X5 ↔ Livox MID-360 extrinsic calibration

Self-contained bundle. Produces the `T_cam_lidar` that
`src/realflight_modules/FAST_LIO_ROS2/config/mid360_insta360.yaml` needs for
`projection_model: "equirectangular"`.

Nothing here runs during flight — this is an offline, one-time procedure.

## Why the virtual-view detour

`livox_camera_calib` assumes a **pinhole** camera. An equirectangular panorama is not
one, so it cannot be fed in directly.

The bridge: a gnomonic ("rectilinear") cut out of a panorama is a *mathematically
exact* pinhole image — straight lines stay straight, and there is no lens distortion
to model. Better still, because we synthesize the view ourselves, its intrinsics are
known in closed form and never need calibrating. Only the extrinsic is solved for.

Each view looks in a different direction, so **each has its own extrinsic**. They are
calibrated separately and then composed back into the panorama frame. This is also
why `multi_calib` must not be used: it solves for one shared extrinsic across all its
inputs, which is only valid for several scenes seen from the *same* camera pose.

## Frames

Both use **x = forward, y = left, z = up**.

| | Meaning |
|---|---|
| LiDAR frame | the reference — raw, as captured (mount tilt included); see "T_cam_lidar must match what the colorizer actually does" below |
| Panorama frame | x = the direction at the image's horizontal centre column |

Calibrators report in the **OpenCV** convention (x = right, y = down, z = forward).
`calib_prepare_views.py` converts; you never do that algebra by hand.

## Split across two machines

| Step | Machine |
|---|---|
| 1–3 capture, guess, cut | ROS2 (the Pi) |
| 4 calibrate | **ROS1** — `livox_camera_calib` is catkin |
| 5 compose | either (needs only numpy) |

Copy this whole `calib/` directory to the ROS1 machine after step 3.

---

## 1. Capture (ROS2)

With the Livox driver and `src/insta360/insta360_ros_publisher.py` running, hold the
rig **completely still**:

```bash
python3 calib/calib_capture.py --seconds 15 --out calib/data
```

MID-360 is a non-repetitive scanner — one frame is far too sparse. Integrating while
stationary is what builds density, and lets scans concatenate directly in the LiDAR
frame with no SLAM involved.

Aim at strong vertical depth edges (building corners, poles, trunks) at **5–20 m**.
Avoid close targets: the ONE X5's dual-lens stitch has 2–3 cm of parallax under ~2 m.

## 2. Initial guess (ROS2)

Measure the camera origin in LiDAR coords with a ruler, and describe the mounting:

```bash
python3 calib/calib_prepare_views.py guess \
    --translation -0.11 0.0 0.03 \
    --rpy -90 0 3 \
    --views 6 --out-dir calib/data/guesses
```

Signs: `yaw>0` swings the camera front **left**; `pitch>0` tilts it **up**;
`roll>0` lifts its **left** side. Check the echoed "camera forward, in LiDAR coords"
matches your mounting.

Only a seed — anything within ~5–10° works. If the camera faces the same way as the
LiDAR, omit `--rpy` entirely. Unsure of the angles? Measure them from the data:

```bash
python3 calib/calib_prepare_views.py bearing calib/data/panorama.png \
    --pair 1740 660  8.00 0.00 0.00 \
    --pair  900 640 -3.00 4.00 1.20
```

Each `--pair` is `U V X Y Z`: a landmark's panorama pixel and its LiDAR-frame
coordinates from `cloud.pcd`. Two or more (spread ~120° apart) solve full
yaw/pitch/roll.

## 3. Cut the views (ROS2)

```bash
python3 calib/calib_prepare_views.py cut calib/data/panorama.png \
    --out-dir calib/data/views --fov 60 --views 6 --size 480
```

60° rather than 90°: gnomonic corner magnification is `1/cos²θ`, which is 3.0× at a
square 90° view's corners but only 1.67× at 60°. `--size 480` keeps centre sampling
near 1:1 against the panorama's 8 px/deg.

**Now copy `calib/` to the ROS1 machine.**

## 4. Calibrate (ROS1, in Docker)

`livox_camera_calib` is catkin-only, so it can't run in the ROS2 flight container.
`docker/` builds a self-contained Noetic image with Livox-SDK, `livox_ros_driver`
and the calibrator already compiled.

Copy this whole `calib/` directory to the laptop, then:

```bash
bash calib/docker/build.sh    # once; clones and compiles everything
bash calib/docker/run.sh      # mounts the bundle at /calib, drops you in a shell
```

Inside the container:

```bash
python3 /calib/calib_make_configs.py --calib-repo /catkin_ws/src/livox_camera_calib
bash /calib/configs/run_all.sh
```

Or unattended, doing both in one go:

```bash
bash calib/docker/run.sh auto
```

Generate the configs **inside the container** — `livox_camera_calib` reads absolute
paths from its YAML, so generating them on the Pi would bake in Pi paths. Everything
under `/calib` is the mounted bundle, so results appear on the host in
`calib/configs/results/`.

The container runs as your host user (`--user $(id -u):$(id -g)`), so nothing it
writes ends up root-owned.

Your clone is left untouched: the generated `calib_view.launch` takes the config as
an argument, unlike the repo's own `calib.launch` which hardcodes
`config/calib.yaml`.

If the calibrator's rosparam names ever drift from what the generator emits, the
image records the real ones at build time — `cat /opt/calib_param_names.txt`.

Per view it writes two files, because the tool splits its settings:

- `calib_NN.yaml` — rosparam: image/pcd/result paths + intrinsics
- `config_NN.yaml` — OpenCV FileStorage: **`ExtrinsicMat`**, this view's initial guess

If a view struggles:

| Symptom | Fix |
|---|---|
| too few edges found | lower `Canny.gray_threshold` in that `config_NN.yaml` (views are resampled, so edges are slightly softer than a native camera's) |
| rough search wanders | regenerate with `--no-rough` |
| indoor scene | `--scene indoor` |
| one view won't converge | drop it in step 5 |

## 5. Compose

```bash
python3 calib/calib_prepare_views.py compose \
    --view-yaw 0   --extrinsic configs/results/extrinsic_00.txt \
    --view-yaw 60  --extrinsic configs/results/extrinsic_01.txt \
    ... \
    --out calib/data/T_cam_lidar.txt
```

**Read the spread it reports.** Views should agree to ~1–3 cm (the dual-lens
baseline). More than that, or >2° rotation, means a view didn't converge — drop it
and re-run rather than averaging it in.

## 6. Apply

Paste the printed `T_cam_lidar: [...]` into `mid360_insta360.yaml`, then:

```bash
./shfile/insta360_run.sh fusion
```

Inspect `/cloud_registered_color` in RViz. Colour boundaries should sit crisply *on*
depth edges. Smearing tells you the axis: horizontal → yaw, vertical → pitch, growing
with distance → translation.

To disable the fusion without touching SLAM: `colorizer:=false`.

## Before you calibrate: T_cam_lidar must match what the colorizer actually does

`lidar_camera_colorizer_node.cpp` projects **raw** `/livox/lidar` points straight
through `T_cam_lidar` (`p_c = r_cl_ * p_l + t_cl_`) with **no real-time attitude
correction** — odometry is only used for an unrelated feature. That fixes the frame
this calibration must target: `calib_capture.py` saves `cloud.pcd` in the **raw,
un-leveled LiDAR frame** by default, mount tilt and all.

This matters because the Insta360 stitches its panorama **gravity-locked**
(FlowState/horizon-lock ON) — its "up" is gravity, not the camera body — while the
LiDAR has no such correction in this pipeline. If the LiDAR is mounted with a fixed
tilt relative to the (level) camera (e.g. pitched forward), that tilt is real
geometry the calibrated `T_cam_lidar` needs to encode, not an error to remove.
Capturing indoors with the **rig held level** is exactly how to isolate it cleanly:
raw LiDAR tilt in that capture *is* the mount tilt, nothing else mixed in.

`calib_capture.py` always measures gravity from `/livox/imu` (a stationary
accelerometer reads pure `+g` "up" in the LiDAR's own frame) and writes
`data/<capture>/gravity.txt` as a diagnostic — e.g. to confirm the measured tilt
matches a known mounting angle. It does **not** apply that measurement to
`cloud.pcd` unless you pass `--level`, which is only for a *different* downstream
consumer that does its own real-time attitude correction (this project's colorizer
doesn't, so don't use it here).

**The unavoidable consequence:** since `T_cam_lidar` is a single static matrix and
the colorizer never corrects for attitude, it's only accurate near the attitude it
was calibrated at. Calibrate level (as above) and expect colorization to degrade
gradually as real flight attitude departs from level — that's the system's actual
behavior, not a calibration bug. Fixing it fully would mean teaching the colorizer
to compose `T_cam_lidar` with the odometry it already subscribes to but currently
only uses for YOLO 3D accumulation.

## Files

| | |
|---|---|
| `insta360_views.py` | projection math — panorama↔bearing, view extraction, extrinsic composition. Shared with the future YOLO split-view node. |
| `calib_capture.py` | grabs one panorama + a dense raw-frame cloud, measuring tilt via `/livox/imu` as a diagnostic (ROS2) |
| `calib_prepare_views.py` | `bearing` / `guess` / `cut` / `compose` |
| `calib_make_configs.py` | generates `livox_camera_calib` configs (run on ROS1) |
| `docker/` | ROS1 Noetic image + build/run scripts for the calibrator |
| `data/` | captured panorama, cloud, views, guesses |
| `configs/` | generated calibrator configs + results |
