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

## Tilt-robust matching: correcting for the Insta360's gravity-lock

The consequence above is fixed for the **offline bbox-matching tools**
(`project_pointcloud_to_view.py`, `bench_bbox_to_tree.py`, `visualize_bbox_match.py`,
`interactive_bbox_match.py` — not the live colorizer, which is outside this repo).

The insight: a calibrated extrinsic is only exactly correct at the one attitude it
was solved at (rig level) *because* at that attitude the Insta360's gravity-lock had
nothing to correct — the calibrated rotation and the rigid camera↔LiDAR mount happen
to coincide. At any other tilt they don't: the panorama keeps re-leveling itself to
true vertical, but the raw LiDAR cloud tilts right along with the rig. Livox's own
IMU can measure that same tilt (it's exactly what `gravity.txt` already records), so
the same re-leveling rotation the camera applied to itself can be re-derived and
folded into the extrinsic before projecting.

Pass a **fresh** `gravity.txt` (the rig's tilt *right now*, not the calibration-time
one) via `--gravity` on any of the four matching tools:

```bash
python3 calib/interactive_bbox_match.py --data-dir calib/data/calib_indoor_level --view 5 \
    --extrinsic calib/configs/results/extrinsic_05.txt \
    --gravity calib/data/current_tilt/gravity.txt
```

`--gravity` pointing at the *same* dataset's own `gravity.txt` should report a
correction of only a degree or so (self-consistency: the extrinsic was solved
against that very tilt). A meaningfully different tilt shifts the reported
correction angle roughly one-to-one with how far the rig has moved off level.

**Getting a fresh `up_measured`:**
- **Stationary** (hovering/held still): a quick `calib_capture.py --seconds 3 --out
  <dir>` grabs enough `/livox/imu` samples for `gravity.txt` alone (its panorama
  capture can be ignored). This is what `up_measured` already is — a stationary
  accelerometer reading pure `+g`.
- **Moving:** a raw accelerometer reads net (gravity + vehicle) acceleration, so
  averaging it in flight is wrong. Use a LiDAR-inertial odometry estimate instead
  (e.g. FAST-LIO's orientation, whose world frame is itself gravity-aligned at
  init) — rotate the world's up axis into the current LiDAR body frame with it and
  feed that in as `up_measured`. Not wired up in this repo yet; `--gravity` just
  needs *a* current up vector in the raw LiDAR frame, from whichever source is live.

The math lives in `insta360_views.gravity_correct_extrinsic()`; the CLI plumbing is
`project_pointcloud_to_view.apply_gravity_correction()`, reused by all four tools.

## Files

| | |
|---|---|
| `insta360_views.py` | projection math — panorama↔bearing, view extraction, extrinsic composition, gravity-lock correction. Shared with the future YOLO split-view node. |
| `calib_capture.py` | grabs one panorama + a dense raw-frame cloud, measuring tilt via `/livox/imu` as a diagnostic (ROS2) |
| `calib_prepare_views.py` | `bearing` / `guess` / `cut` / `compose` |
| `calib_make_configs.py` | generates `livox_camera_calib` configs (run on ROS1) |
| `project_pointcloud_to_view.py` | 2D check: project the cloud onto one view's image via any extrinsic file |
| `visualize_seed_fov.py` | 3D check: writes a self-contained HTML viewer of a dataset's camera FOVs against its point cloud |
| `bench_bbox_to_tree.py` | bbox → point-cloud object match: pipeline + fixed-bbox timing benchmark |
| `visualize_bbox_match.py` | static PNG of a bbox match (candidates/gated/cluster points + keypoints) |
| `interactive_bbox_match.py` | mouse-driven UI for the bbox match, live on one view |
| `docker/` | ROS1 Noetic image + build/run scripts for the calibrator |
| `data/` | captured panorama, cloud, views, guesses |
| `configs/` | generated calibrator configs + results |
