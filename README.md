# Insta360 ONE X5 ↔ Livox MID-360 extrinsic calibration

Self-contained bundle. Produces the `T_cam_lidar` that
`src/realflight_modules/FAST_LIO_ROS2/config/mid360_insta360.yaml` needs for
`projection_model: "equirectangular"`.

The calibration itself is offline and one-time. The tools that *consume* the
result — the extrinsic check and the bbox→object matchers — do run on the live
rig; see **Running the live tools** below.

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

## Visualizing before you trust a result

Two standalone tools, usable at any point — on a seed before you've even run the
calibrator, or on a solved result afterward. Neither needs `livox_camera_calib` or
ROS; both only need `numpy` (+ `Pillow` for the first), so they run equally well on
the Pi or the laptop, inside or outside the container.

**1. Project the cloud onto one view's image, via any extrinsic file:**

```bash
python3 calib/project_pointcloud_to_view.py --data-dir calib/data/calib_data_yard_1 --view 5
```

Colors the cloud by depth and draws it over that view's cut image — a flat 2D
overlay, one view at a time. Defaults to that view's seed guess
(`data/guesses/guess_NN_*.txt`); pass `--extrinsic path/to/extrinsic_05.txt` to check
a solved result instead, against the exact same image, so seed and solved are
directly comparable. `--min-depth` drops near-field self-occlusion from the drone
body (default 0.3 m); `--max-depth` caps the far end.

**2. View a whole dataset's camera FOVs against the point cloud, in 3D:**

```bash
python3 calib/visualize_seed_fov.py --dataset calib_data_yard_1
```

Writes one self-contained `.html` (point cloud + all 6 seed FOVs baked in) that
opens directly in a browser — no server, nothing else to install. Pick a single
view to extend its FOV boundary and a bright centre beam to real room/yard scale, so
it visibly slices through the actual structure instead of a small decorative marker.
Also has a "Level to gravity" toggle (a RANSAC ground-plane fit, independent of
`calib_capture.py`'s own IMU-based measurement) for a quick visual gut-check of how
tilted a capture was. Re-run any time the dataset changes — it just overwrites the
output file.

## 6. Apply

Paste the printed `T_cam_lidar: [...]` into `mid360_insta360.yaml`, then:

```bash
./shfile/insta360_run.sh fusion
```

Inspect `/cloud_registered_color` in RViz. Colour boundaries should sit crisply *on*
depth edges. Smearing tells you the axis: horizontal → yaw, vertical → pitch, growing
with distance → translation.

To disable the fusion without touching SLAM: `colorizer:=false`.

## Downstream: matching a 2D detection bbox to a point-cloud object

Not calibration itself — a downstream consumer of a solved `T_cam_lidar` (or any
per-view `extrinsic_NN.txt`): given a 2D detection box in one view (e.g. a YOLO
bbox), find the matching object in the point cloud and report its center/bottom/top
3D coordinates. Same pinhole math as `project_pointcloud_to_view.py`, run in
reverse — points are projected into the image and filtered by which ones land
inside the box, rather than the whole cloud being drawn over it.

**Pipeline** (`bench_bbox_to_tree.py`):

1. **Frustum select** — project the whole cloud, keep points landing inside the
   bbox's pixel range and in front of the camera.
2. **Depth-mode gate** — histogram the candidates' range, keep a MAD window around
   the dominant peak. Drops background/foreground sitting at a different range than
   the object, which a loose box otherwise pulls in along the same pixel columns.
3. **Connected-component clustering** — 6-connected voxel-grid union-find (no
   scipy/sklearn dependency) splits the depth-gated points into distinct physical
   objects. A box frequently straddles the target *and* a background surface at a
   similar range (e.g. a wall panel behind a chair) that only shares one gated blob
   by luck.
4. **Cluster selection** — default to the largest cluster (correct for a reasonably
   tight box); only override it when that cluster's centroid is clearly off-center
   in the image *and* a substantially-sized alternative sits clearly near the box
   center. Picking by raw size alone latches onto a big background surface; picking
   by centrality alone loses a large, correctly-shaped object to a stray noise
   cluster that happens to be more central. Comparable-sized fragments near the
   winner (e.g. a chair's separated leg/armrest returns) are merged back in.
5. **Keypoints** — median center, 1st/99th-percentile top/bottom along the up axis
   (not min/max — a couple of stray points shouldn't set the extent).

**Tools:**

| | |
|---|---|
| `bench_bbox_to_tree.py` | fixed-bbox timing benchmark, no YOLO — measures the matching cost in isolation |
| `visualize_bbox_match.py` | static PNG: bbox + candidate/gated/cluster points + center/bottom/top markers, for one fixed bbox |
| `interactive_bbox_match.py` | Tkinter UI — drag a box on the view image with the mouse, see the match live |

```bash
python3 calib/interactive_bbox_match.py --data-dir calib/data/calib_indoor_level --view 5 \
    --extrinsic calib/configs/results/extrinsic_05.txt
```

**Use a solved extrinsic, not a seed guess** (`--extrinsic path/to/extrinsic_NN.txt`).
A seed is only accurate to ~5–10°, which is enough to sanity-check a calibration
visually but not enough for step 1's frustum select to land on the right object.

**Real-time note:** the expensive step is the whole-cloud projection in step 1
(~23 ms for ~1M points on a laptop CPU, single-threaded) — everything downstream
(mask/gate/cluster/keypoints) runs in a few ms on the resulting few-thousand-point
candidate set. For a fixed camera pose, project once and cache it; only rerun the
cheap downstream steps per new bbox, rather than reprojecting the whole cloud on
every detection. That split is what keeps a single-view 10 Hz match loop plausible
on something like a Raspberry Pi 5, where a Hailo-8/Coral-class accelerator running
the detector itself leaves the CPU free for this part — not yet benchmarked on
that hardware, so treat it as a target to validate rather than a measured number.

## Running the live tools (ROS 2, on the rig)

Three tools run against the live rig instead of a capture. Each opens the camera
**directly** with `cv2.VideoCapture` (no image topic in the loop) and claims the Hailo
accelerator, so **only one may run at a time** — and none of them while
`insta360_run.sh camera` or `fusion` is up, since that publisher holds `/dev/video0`
too.

| tool | answer is in | question it answers |
|---|---|---|
| `live_view_overlay.py` | — | is my extrinsic right? whole cloud painted by depth over the live views |
| `live_cluster_match.py` | LiDAR frame | why did this object end up *there*? one detection taken apart, stage by stage |
| `global_bbox_match.py` | **world frame** | where is this object globally? tilt-independent; optional RViz markers |

### Container

Use the runtime image that is already there — `livox-360-yolo-ego:latest`. It carries
ROS 2 Humble, `rclpy`, OpenCV, `hailo_platform` and the Livox messages, and it runs
privileged with host networking, so `/dev/hailo0` and `/dev/video0` are both reachable
from inside. **Nothing here needs a new image built or started.**

> `docker/` in this directory is unrelated: it is the ROS 1 Noetic image for
> `livox_camera_calib`, used by step 4 and nothing else.

Attach to a container that is already up:

```bash
docker ps                          # pick one running livox-360-yolo-ego:latest
docker exec -it <name> bash
```

…or start one:

```bash
docker run -it --rm --privileged --network=host \
    --env DISPLAY=:0 --env QT_X11_NO_MITSHM=1 \
    --env ROS_DOMAIN_ID=0 --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --volume /home/yibo-rpi/yibo/vins_ego_ros2:/home/local \
    --device /dev/ttyAMA0 --group-add dialout --group-add video \
    livox-360-yolo-ego:latest \
    bash -c 'cd /home/local && source install/setup.bash && exec bash'
```

`docker exec` bypasses the image's entrypoint, so a shell opened that way must source
both prefixes itself — skip them and `rclpy` is simply not importable:

```bash
source /opt/ros/humble/setup.bash
source /home/local/install/setup.bash
cd /home/local/src/livox_insta360_calib
```

### Prerequisites

`./shfile/livox_imu.sh` (from `/home/local`) brings up the Livox driver **and**
FAST-LIO2, which is both of the topics the live tools need:

| topic | needed by | note |
|---|---|---|
| `/livox/lidar` | all three | `CustomMsg`, parsed straight from the CDR bytes |
| `/Odometry` | `global_bbox_match.py` only | the camera pose **is** this topic; without it nothing can be placed in world coordinates |

Check first — both should read ~10 Hz:

```bash
ros2 topic hz /livox/lidar
ros2 topic hz /Odometry
```

### Models

Both of these are compiled for HAILO8L and load on the Pi's accelerator:

    src/hailo/yolov8s.hef
    src/insta360/best_hailo_model/yolov8s_h8l.hef

`src/insta360/best_hailo_model/best.hef` is compiled for **HAILO8**, not 8L, and will
not load. When a HEF is rejected, compare `hailortcli parse-hef <hef>` against
`hailortcli fw-control identify`.

### Commands

**Extrinsic check** — no detector, so it does not touch the accelerator:

```bash
python3 live_view_overlay.py --view 240 --view 300 \
    --extrinsic data/T_cam_lidar_indoor_level.txt
```

**One detection taken apart, LiDAR frame:**

```bash
python3 live_cluster_match.py --view 240 \
    --yolo /home/local/src/hailo/yolov8s.hef \
    --translation 0.18 0.0 -0.13 --rpy 90 0 -23
```

**Object keypoints in world coordinates** — tilt-independent, and the one to use in
flight:

```bash
python3 global_bbox_match.py --view 240 \
    --yolo /home/local/src/hailo/yolov8s.hef \
    --print --publish-markers
```

Headless (no X): add `--no-display`, which writes a PNG into `live_snapshots/` every
2 s instead. Keys while running: `q` quit, `s` snapshot, `l` faint full cloud,
`p` pause.

The mounting arguments default to this rig, so `--translation` only needs overriding
if the camera moves. `global_bbox_match.py` derives its single angle from
`data/T_cam_lidar_indoor_level.txt` + `data/calib_indoor_level/gravity.txt` and prints
it at startup (`+86.79°`) together with the residual camera tilt (`0.57°`) — which is
the error floor of the whole gravity-lock assumption, 0.10 m at 10 m.

### Markers

`--publish-markers` publishes `visualization_msgs/MarkerArray` on `/yolo_objects` in
the `world` frame at ~8 Hz: a bottom→top line, three spheres coloured to match the
on-screen C/B/T, and a text label. Each array leads with `DELETEALL`, so a frame with
fewer detections than the last leaves nothing stale behind. In RViz set the fixed
frame to `world` and add a MarkerArray display on `/yolo_objects`.

### Things that will bite

| symptom | cause |
|---|---|
| `cannot open capture device 0` | something else holds the camera: a previous run, or `insta360_run.sh camera`/`fusion` |
| `cannot load <hef>` | the accelerator is held by another process, or the HEF is HAILO8 rather than HAILO8L |
| `No module named 'rclpy'` | the two `source` lines were skipped after `docker exec` |
| `N frame(s) had no usable pose` | `/Odometry` is not publishing, or the stamp offset moved past `--max-pose-age` |
| points lag or lead the image as you rotate | tune `--cam-latency` (default 0.05 s); at 90 °/s, 50 ms is 4.5° |
| `raw CustomMsg parse rejected` | the CDR point stride is neither `n*20` nor `n*20 - 1` — the fallback is correct but costs 134 ms per message instead of 1.6 ms |

Message stamps on this rig are **not** wall clock: the Livox timebase is not
disciplined to the host, and the offset moves between runs (observed −1.04 s,
+0.045 s, +0.105 s). `global_bbox_match.py` measures it from every odometry message
and prints the median at startup, so nothing needs configuring — but a `stamp offset`
line far from zero is expected, not a fault. The camera stream carries no stamp at
all, which is why frames are placed by wall-clock arrival and `--cam-latency` exists.

## Offline: replaying a bag on a laptop

`bag_bbox_match.py` is `global_bbox_match.py` with the four inputs swapped — same pose
chain, same `fast` matcher, same world-frame output, no Hailo and no camera:

| | on the rig | on a laptop |
|---|---|---|
| detector | HEF on a Hailo-8L | **Ultralytics `.pt`** on CPU or CUDA |
| panorama | `cv2.VideoCapture` | `/insta360/image_raw/compressed` |
| cloud | `/livox/lidar` live | `/livox/lidar` from the bag |
| pose | `/Odometry` live | `/Odometry` from the bag |

It reads the bag **directly** with `rosbag2_py` — do not `ros2 bag play` alongside it.
Reading rather than replaying buys three things the live tool cannot have:

- **every pose is interpolated, never extrapolated.** All of `/Odometry` is loaded in a
  first pass before any image is touched, so a frame's pose is always bracketed by two
  real samples. Live, odometry for an instant arrives *after* it.
- **the LiDAR window is centred on the frame.** Offline there is lookahead, so a frame
  matches against `[t-w/2, t+w/2]` instead of the trailing `[t-w, t]` a live loop is
  stuck with — same point budget, half the mean temporal offset. `--trailing-window`
  restores live behaviour when the point is to reproduce it.
- **stamps, not arrival times.** Every message carries its own stamp and
  `insta360_ros_publisher.py` stamps in the grab thread, so association is
  stamp-to-stamp throughout. None of the live tool's wall-clock offset machinery runs.

### Which cloud topic, and why not the easy one

`/livox/lidar`, not `/cloud_registered` — even though the latter is already in the
world frame and would need no pose at all. Measured on `kuusamo/manual_l2`:

| topic | points per message (median) |
|---|---|
| `/livox/lidar` | **19,968** |
| `/cloud_registered` | 675 |

`/cloud_registered` is `feats_down_body`, voxel-filtered at `mapping.filter_size_surf`
(0.5 m), so a person at 5 m survives as a handful of points — below `--min-points`
before the matcher even starts. It only becomes usable with `dense_publish_en: true`
in the FAST-LIO config, which publishes `feats_undistort` instead.

The CDR bytes `rosbag2` returns are byte-identical to what a raw subscription
delivers, so `LidarBuffer`'s numpy parser reads them directly: **1.6 ms** per message
against 134 ms for `deserialize_message` — ~3 s versus ~5 minutes over a 212 s bag. It
also means `livox_ros_driver2` need **not** be built on the laptop; only the fallback
path would need the message class.

### Requirements

ROS 2 (for `rosbag2_py`, `rclpy`, `sensor_msgs`, `nav_msgs`) plus:

```bash
pip install ultralytics          # pulls torch; add a CUDA build for --device cuda
```

### Commands

```bash
python3 bag_bbox_match.py --bag ~/kuusamo/manual_l2 --view 240 --yolo yolov8s.pt
```

A slice of a long bag, on the GPU, with the results tabulated:

```bash
python3 bag_bbox_match.py --bag ~/kuusamo/manual_l2 --ring 6 --yolo-all-views \
    --yolo yolo11s.pt --device cuda \
    --start 60 --duration 40 --every 4 --csv objects.csv
```

`--start`/`--duration` slice by bag time, `--every N` subsamples frames, `--rate X`
throttles to X times real time for watching, `--no-display` for batch. Keys: `q` quit,
`space` pause, `s` snapshot, `l` faint full cloud.

`--csv` writes one row per matched detection — `stamp, view, cls, label, conf, n_box,
n_obj, range_m`, then `top/central/bottom` and the camera origin, all as world XYZ.

The run ends with `N frame(s) processed, D detection(s), M object(s) located`, which
separates the two ways to get nothing: `D=0` is the detector, `M=0` with `D>0` is the
matcher.

### What outdoor bags look like

Measured on `manual_l2`, and worth knowing before reading a disappointing result:

- **~72% of LiDAR points are `(0,0,0)`** — no return, because most of a 360° field of
  view from a flying drone is sky. 20k points per message become ~5.6k real ones. The
  tag filter is *not* what does this (it dropped 13 points in 1.2M); the finite and
  non-zero mask is. Nothing is wrong when the in-view count looks thin.
- **The airframe is in the panorama.** The 360 camera sees the drone's own arms, legs
  and props, and YOLO fires on them — a propeller scored 0.55 as `person`, and the
  matcher dutifully returned coordinates for it. Restrict with `--classes`, raise
  `--min-depth` above the airframe, and choose view yaws that look away from the arms.
- **Confidence has to come down** for distant small objects, and false positives come
  up with it. At `--yolo-conf 0.15` on `yolov8n.pt` a forest yields `person`,
  `fire hydrant` and `bear` on trees.
- Sanity check the world frame by watching one static object across frames: a treeline
  held x ≈ −9.1…−9.6 m over 30 s of flight while y swept 13.8→30 m.

### Where the MID-360 can actually see — pick the view before blaming the matcher

Two hard limits decide whether a view gets any points at all. Both were measured on
`manual_l2`, and either one alone produces `D detections, 0 located`.

**Elevation.** The MID-360 returns nothing below −7°:

    elevation in the LiDAR/body frame:  min −7.0°   p50 +5.3°   max +50.5°
        -15..-10 deg        0
        -10.. -5 deg    12945  #####          <- hard floor at exactly -7.0 deg
         -5.. +0 deg   132925  ############################################################

So a view pitched −45° looks 38° below anything the sensor can ever return, and gets
exactly zero points — not a calibration fault. It also means the ground is only seen
in a **ring at ≈ 8.1 × altitude** (h / tan 7°): at 2.75 m that is ~22 m out. Anything
directly below the drone is invisible to this LiDAR, however good the detection is.

**Azimuth.** Coverage is wildly uneven — mean points in a fov60 view, by view yaw and
pitch:

    yaw \ pitch    -30     -20     -10       0     +10     +20
      0           2294    2397    2380    2337    2224    1909
     60             59     103     129     141     129     103
     90              3       7       8       8       8       7     <- ~1000x null
    120             16      17      18      20      18      17
    180           1294    1413    1428    1339    1155     891
    270           4281    4408    4372    4142    3531    2156     <- best
    300           3826    4078    4082    3854    3322    2073

The null spanning yaw 60–120 is the airframe occluding the sensor. Pitch barely
matters by comparison; **yaw is the choice that decides whether the tool works.**

Measured end to end with `yolo11n-tuesday.pt` (one class, `Sapling`) over the same
30 s slice:

| view | detections | located |
|---|---|---|
| `--view 90 -45 0 60` | 127 | **0** |
| `--view 90 -20 0 60` and −10/0/+10 | 153 | **0** |
| `--view 270 -20 0 60` | 280 | **222 (79%)** |

At yaw 270 the matched saplings sit at 9.6–18 m — the ground ring, as predicted — and
their heights come out sensible (top −0.52 / bottom −1.31 = 0.79 m; top +0.17 /
bottom −1.78 = 1.95 m). Near saplings directly below the drone still match nothing,
because no LiDAR return exists for them.

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

`global_bbox_match.py` does exactly that for the bbox→object path (not for the
colorizer): it drops `T_cam_lidar` entirely and builds camera↔world from `/Odometry`
each frame, so attitude cannot enter the chain. What makes that possible is visible
in the calibration data itself — the solved extrinsic's camera *up* axis sits 0.57°
from gravity up even though the LiDAR was 22.75° off level at capture, so the whole
matrix collapses to one yaw angle plus a lever arm. The `roll -23` in the
`--rpy 90 0 -23` seed is not a mounting roll at all; it is the LiDAR's own tilt.

## Files

| | |
|---|---|
| `insta360_views.py` | projection math — panorama↔bearing, view extraction, extrinsic composition. Shared with the future YOLO split-view node. |
| `calib_capture.py` | grabs one panorama + a dense raw-frame cloud, measuring tilt via `/livox/imu` as a diagnostic (ROS2) |
| `calib_prepare_views.py` | `bearing` / `guess` / `cut` / `compose` |
| `calib_make_configs.py` | generates `livox_camera_calib` configs (run on ROS1) |
| `project_pointcloud_to_view.py` | 2D check: project the cloud onto one view's image via any extrinsic file |
| `visualize_seed_fov.py` | 3D check: writes a self-contained HTML viewer of a dataset's camera FOVs against its point cloud |
| `bench_bbox_to_tree.py` | bbox → point-cloud object match: pipeline + fixed-bbox timing benchmark |
| `visualize_bbox_match.py` | static PNG of a bbox match (candidates/gated/cluster points + keypoints) |
| `interactive_bbox_match.py` | mouse-driven UI for the bbox match, live on one view |
| `bbox_match.py` | the two isolation methods (`fast`, `cluster`) + the shared keypoint extraction |
| `bbox_matching_paper_notes.md` | measured comparison of the two methods, and runtime benchmarks |
| `hailo_yolo.py` | Hailo-8L YOLO wrapper (HEF load, inference, drawing) |
| `live_view_overlay.py` | **live**: cloud over the live views, for judging an extrinsic on the rig |
| `live_cluster_match.py` | **live**: YOLO → `cluster` match in the LiDAR frame, staged rendering |
| `global_bbox_match.py` | **live**: YOLO → object keypoints in WORLD coordinates, tilt-independent; optional RViz markers |
| `bag_bbox_match.py` | **offline**: the same, from a ros2 bag with an Ultralytics `.pt` — laptop, no Hailo |
| `docker/` | ROS1 Noetic image + build/run scripts for the calibrator (step 4 only — not the runtime) |
| `data/` | captured panorama, cloud, views, guesses |
| `configs/` | generated calibrator configs + results |
