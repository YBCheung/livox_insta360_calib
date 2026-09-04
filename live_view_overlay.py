#!/usr/bin/env python3
"""Live Insta360 virtual views with the Livox cloud projected onto them.

Opens the ONE X5 directly with cv2.VideoCapture (no image topic in the loop), cuts
an arbitrary set of rectilinear views out of each panorama frame, and paints the
current LiDAR points into every view using one T_cam_lidar. This is the real-time
counterpart of project_pointcloud_to_view.py: same math, but on the live stream, so
an extrinsic can be judged while the rig is in front of you instead of after a
capture/solve round trip.

Views are arbitrary -- any count, each with its own yaw/pitch/roll/fov. The two the
calibration bundle calls view 4 and view 5 are just yaw 240 and 300:

    python3 live_view_overlay.py --view 240 --view 300 \
        --translation 0.18 0.0 -0.13 --rpy 90 0 -23

  --translation  camera ORIGIN in LiDAR coords, metres (what a ruler measures)
  --rpy          camera mounting, degrees: yaw>0 swings its front LEFT,
                 pitch>0 tilts it UP, roll>0 lifts its LEFT side
  --view         YAW [PITCH [ROLL [FOV]]] -- repeat once per view

Both those flags mean exactly what they mean in calib_prepare_views.py guess, so a
seed that looks right here is the seed to hand the calibrator. Once livox_camera_calib
has solved it, check the refined result the same way with
--extrinsic <4x4 panorama-frame T_cam_lidar>, which is the matrix mid360_insta360.yaml
carries.

What a good extrinsic looks like: depth edges in the cloud land on intensity edges in
the image -- a pole's points stay on the pole through a full sweep, a wall's points
stop at the wall's corner. Points sitting a consistent few pixels to one side is a
rotation error; error that grows as things come closer is the lever arm.

With --yolo <hef> a Hailo detector runs on the selected views, and each box is
matched against the cloud to give the object's top / central / bottom point in the
LiDAR frame. --match chooses how the object is separated from its background:
'fast' (nearest range run) or 'cluster' (depth-mode gate + 3D voxel components);
'both' runs each on the same box and draws them together -- white for fast, cyan
for cluster -- which is the only honest way to compare them, since the camera and
the accelerator are each exclusive to one process. See bbox_matching_paper_notes.md
for the two algorithms and their measured differences.

Keys: q / Esc quit, s save a snapshot, p toggle the panorama overlay,
      [ / ] shrink or grow the drawn points.

Cost, measured on a Pi 5 with two 480 px views: ~33% of one core at the default
--fps 10, ~19% at --fps 5. Two things dominate, in this order:

  decoding    ~23 ms of CPU per frame for a 2880x1440 MJPEG, single-threaded and
              irreducible, so --fps (a cap on how many frames are decoded at all,
              not a target) is the biggest lever by far. The camera offers 30 fps
              and its other MJPEG mode is 1920x1080 -- 16:9, not a panorama --
              so there is no cheaper resolution to fall back to.
  remapping   ~6 ms of CPU per extra view. --cubic doubles that for sharper
              cut-outs; bilinear is plenty for judging alignment.

The LiDAR side used to dominate everything: rclpy builds a Python object per point
when it deserialises a CustomMsg -- 166 ms for a 20k-point message, i.e. 166% of a
core at 10 Hz, spent before the callback is entered. The subscription is therefore
raw and the CDR bytes are parsed with numpy instead (1.6 ms, a hundredfold less);
--no-lidar-raw restores the old path for comparison. Remaining knobs: --point-stride
thins the cloud, and --every N keeps the video at full rate while reprojecting less
often.
"""

import argparse
import math
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bbox_match as bm
import insta360_views as iv
from calib_prepare_views import rpy_to_matrix


# --------------------------------------------------------------------------- views

class View:
    """One virtual rectilinear camera cut out of the panorama."""

    def __init__(self, yaw, pitch, roll, fov, size):
        self.yaw, self.pitch, self.roll, self.fov = yaw, pitch, roll, fov
        self.size = size
        self.origin = (0, 0)  # where this view sits in the canvas; set by ViewCanvas
        self.out_hw = (size, size)
        self.fx, self.fy, self.cx, self.cy = iv.virtual_view_intrinsics(fov, self.out_hw)
        # Columns are this view's OpenCV axes in the panorama frame, so its transpose
        # takes a panorama-frame point into the view's camera frame.
        self.r_pano_cv = iv.view_rotation(yaw, pitch, roll).astype(np.float32)
        # Half-angle to the view's corner, for a cheap cone cull before projecting.
        half_diag = math.degrees(math.atan(math.hypot(size / 2.0 / self.fx,
                                                      size / 2.0 / self.fy)))
        self.cos_limit = math.cos(math.radians(min(half_diag + 2.0, 89.9)))

    @property
    def label(self):
        s = f"yaw{self.yaw:+.0f} pitch{self.pitch:+.0f}"
        if self.roll:
            s += f" roll{self.roll:+.0f}"
        return s + f" fov{self.fov:.0f}"

    def sampling_maps(self, pano_wh):
        """The (x, y) panorama sample position of every pixel of this view."""
        return iv.build_view_maps(self.fov, self.yaw, self.pitch, self.out_hw,
                                  pano_wh, self.roll)

    def set_extrinsic(self, r_pano_lidar):
        """Fold the mounting rotation in: LiDAR axes -> this view's camera axes.

        p_cam = d @ lidar_to_cv, where d is a point relative to the CAMERA but still
        on the LiDAR's axes. Folding the two rotations into one matrix means a point
        that survives the cull is rotated once, not twice, and the optical axis in
        LiDAR coordinates falls out as its third column -- which is exactly the
        vector the cull needs.
        """
        self.lidar_to_cv = np.ascontiguousarray(r_pano_lidar.T @ self.r_pano_cv,
                                                dtype=np.float32)
        self.axis_lidar = np.ascontiguousarray(self.lidar_to_cv[:, 2])

    def unproject(self, u, v, rng, cam_origin):
        """View pixels + range -> points back in the LiDAR frame.

        The exact inverse of project(): a pixel fixes a bearing, the range fixes how
        far along it, and lidar_to_cv is orthonormal so its transpose undoes the
        rotation. Nothing extra has to be carried through the per-frame path for this
        -- (u, v, range) is a complete description of a projected point -- so the
        cost is paid only for the handful of points inside a box.
        """
        rays = np.stack([(u - self.cx) / self.fx,
                         (v - self.cy) / self.fy,
                         np.ones_like(u)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return (rays * np.asarray(rng)[:, None]) @ self.lidar_to_cv.T + cam_origin

    def project(self, d, rng, near):
        """Camera-relative LiDAR-axes points -> (u, v, range) landing in this view.

        Culling happens HERE, in the LiDAR's own axes, before any point is rotated:
        one dot product against the optical axis rejects everything outside the view
        cone, and only the survivors pay the full transform. Rotating all N points
        first and culling afterwards costs about a third more (9.4 ms vs 6.9 ms for
        an 80k-point window).

        `d` must already be camera-relative (p_lidar - camera origin): the cone's
        apex is the camera, ~20 cm off the LiDAR, and `rng` must be its norm, which
        is also the true range from the camera since rotation preserves length.
        `near` is the shared range mask, computed once per frame for all views.
        """
        empty = (np.empty(0), np.empty(0), np.empty(0))
        if len(d) == 0:
            return empty

        inside = near & ((d @ self.axis_lidar) >= self.cos_limit * rng)
        cam = d[inside] @ self.lidar_to_cv
        rng = rng[inside]
        if len(cam) == 0:
            return empty

        z = cam[:, 2]
        front = z > 1e-3
        cam, rng, z = cam[front], rng[front], z[front]
        u = self.fx * cam[:, 0] / z + self.cx
        v = self.fy * cam[:, 1] / z + self.cy
        return u, v, rng


class ViewCanvas:
    """Every view's sampling map fused into ONE remap that fills the whole window.

    At tile size, cv2.remap is dominated by per-call setup and thread dispatch, so a
    single call covering the grid costs measurably less CPU than one call per view
    (11.3 ms vs 14.6 ms for two 480 px views on a Pi 5) and writes the tiles straight
    into the canvas, which also retires the per-frame hstack.
    """

    def __init__(self, views, max_cols, interpolation):
        self.views = views
        self.interp = interpolation
        self.size = views[0].size
        self.cols = max(1, min(max_cols, len(views)))
        self.rows = -(-len(views) // self.cols)
        for i, vw in enumerate(views):
            vw.origin = ((i // self.cols) * self.size, (i % self.cols) * self.size)
        # Cells past the last view: their maps sample pixel (0,0), so blank them after.
        self.blanks = [((i // self.cols) * self.size, (i % self.cols) * self.size)
                       for i in range(len(views), self.rows * self.cols)]
        self.pano_wh = None
        self.maps = None

    def build(self, pano_wh):
        h, w = self.rows * self.size, self.cols * self.size
        mx = np.zeros((h, w), np.float32)
        my = np.zeros((h, w), np.float32)
        for vw in self.views:
            oy, ox = vw.origin
            a, b = vw.sampling_maps(pano_wh)
            mx[oy:oy + self.size, ox:ox + self.size] = a
            my[oy:oy + self.size, ox:ox + self.size] = b
        # Fixed-point maps: remap's CV_16SC2 path is about twice as cheap as float32
        # maps, and its 1/32-pixel quantisation is invisible at this sampling density.
        self.maps = cv2.convertMaps(mx, my, cv2.CV_16SC2)
        self.pano_wh = pano_wh

    def render(self, panorama):
        pano_wh = (panorama.shape[1], panorama.shape[0])
        if self.pano_wh != pano_wh:
            self.build(pano_wh)
        canvas = cv2.remap(panorama, self.maps[0], self.maps[1], self.interp,
                           borderMode=cv2.BORDER_WRAP)
        for oy, ox in self.blanks:
            canvas[oy:oy + self.size, ox:ox + self.size] = 0
        return canvas

    def tile(self, canvas, view):
        """A writable window onto one view's pixels -- painting it paints the canvas."""
        oy, ox = view.origin
        return canvas[oy:oy + self.size, ox:ox + self.size]


def parse_views(args):
    specs = []
    for raw in args.view or []:
        if len(raw) > 4:
            raise SystemExit("--view takes YAW [PITCH [ROLL [FOV]]], at most 4 numbers")
        yaw = raw[0]
        pitch = raw[1] if len(raw) > 1 else args.pitch
        roll = raw[2] if len(raw) > 2 else args.roll
        fov = raw[3] if len(raw) > 3 else args.fov
        specs.append((yaw, pitch, roll, fov))

    if args.ring:
        # The same evenly spaced ring calib_prepare_views.py cut produces, so view
        # indices line up with the files in data/*/views/.
        for i in range(args.ring):
            specs.append((args.yaw_start + i * 360.0 / args.ring,
                          args.pitch, args.roll, args.fov))
    if not specs:
        raise SystemExit("no views requested -- pass --view YAW ... or --ring N")
    return [View(y, p, r, f, args.size) for y, p, r, f in specs]


# ---------------------------------------------------------------------- extrinsic

def load_extrinsic_file(path):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if line:
                vals.extend(float(v) for v in line.replace(',', ' ').split())
    arr = np.array(vals, dtype=float)
    if arr.size == 16:
        T = arr.reshape(4, 4)
    elif arr.size == 12:
        T = np.eye(4)
        T[:3, :4] = arr.reshape(3, 4)
    else:
        raise SystemExit(f"{path}: expected 12 or 16 numbers, got {arr.size}")
    return T[:3, :3], T[:3, 3]


def build_extrinsic(args):
    """T_cam_lidar in the PANORAMA frame: p_pano = R @ p_lidar + t."""
    if args.extrinsic:
        r, t = load_extrinsic_file(args.extrinsic)
        print(f"extrinsic  : {args.extrinsic} (panorama frame, as given)")
        return r, t

    # rpy_to_matrix returns the camera's axes in LiDAR coords; the lidar->camera
    # transform is its transpose. --translation is the camera ORIGIN in LiDAR coords,
    # so the transform's translation is -R @ origin, not the offset itself.
    q_cam = rpy_to_matrix(*args.rpy)
    r = q_cam.T
    t = -r @ np.asarray(args.translation, dtype=float)
    print(f"extrinsic  : --translation {args.translation} --rpy {args.rpy}")
    print(f"             camera forward in LiDAR coords {np.round(q_cam[:, 0], 4)}, "
          f"up {np.round(q_cam[:, 2], 4)}")
    print(f"             T_cam_lidar translation {np.round(t, 4)}")
    return r, t


# --------------------------------------------------------------------------- lidar

class LidarBuffer:
    """Sliding window of recent scans, in the raw LiDAR frame.

    MID-360 is a non-repetitive scanner: one 10 Hz message is far too sparse to judge
    an extrinsic by eye, so a fraction of a second is accumulated. That is only honest
    while the rig is roughly still -- there is no motion compensation here, so a
    longer window smears the cloud once the rig moves.
    """

    def __init__(self, window_s, stride, filter_tags):
        self.window_s = window_s
        self.stride = max(1, stride)
        self.filter_tags = filter_tags
        self.scans = deque()
        self.lock = threading.Lock()
        self.n_msgs = 0
        self.raw_ok = True          # cleared for good if the byte layout surprises us
        self._raw_verified = False
        self.custom_msg_type = None
        self.dropped = 0

    def add(self, xyz):
        now = time.monotonic()
        with self.lock:
            self.scans.append((now, xyz))
            self.n_msgs += 1
            while self.scans and now - self.scans[0][0] > self.window_s:
                self.scans.popleft()

    def snapshot(self):
        with self.lock:
            cutoff = time.monotonic() - self.window_s
            arrays = [a for t, a in self.scans if t >= cutoff]
        if not arrays:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(arrays, axis=0)

    # --- raw CustomMsg parsing ------------------------------------------------
    # rclpy builds one Python CustomPoint object per point when it deserializes a
    # CustomMsg: 134 ms for a 20k-point message on a Pi 5, paid before the callback
    # is even entered, i.e. ~134% of a core at 10 Hz. A raw subscription hands over
    # the CDR bytes instead, and the points are a plain fixed-stride array numpy can
    # read in 0.23 ms. Layout (verified against rclpy's own deserialisation):
    #
    #   4 bytes encapsulation header, then all CDR alignment is relative to what
    #   follows it -- getting that wrong shifts everything by 4 bytes:
    #     int32 sec, uint32 nanosec, uint32 len + frame_id (len includes the NUL),
    #     align 8: uint64 timebase, uint32 point_num, uint8 lidar_id, uint8 rsvd[3],
    #     align 4: uint32 sequence length, then the points.
    #   CustomPoint is uint32 offset_time, float32 x/y/z, uint8 reflectivity/tag/line
    #   = 19 bytes padded to a 20-byte stride, EXCEPT the last element, which CDR
    #   leaves unpadded -- hence the n*20 - 1 read below.
    POINT_STRIDE = 20

    def _parse_custom_raw(self, buf):
        """CDR bytes of a CustomMsg -> (xyz float32 array, tag array). None if unsure."""
        if len(buf) < 24 or buf[1] != 1:
            return None  # big-endian CDR: vanishingly rare, let rclpy handle it
        o = 8                                    # 4 encapsulation + sec, at nanosec
        o += 4                                   # past nanosec
        slen = int(np.frombuffer(buf, np.uint32, 1, o)[0])
        o += 4 + slen
        o = 4 + ((o - 4 + 7) & ~7)               # align 8 (relative to the body)
        o += 8 + 4 + 1 + 3                       # timebase, point_num, lidar_id, rsvd
        o = 4 + ((o - 4 + 3) & ~3)               # align 4
        n = int(np.frombuffer(buf, np.uint32, 1, o)[0])
        o += 4

        stride = self.POINT_STRIDE
        body = len(buf) - o
        # Whether CDR pads the FINAL element is RMW-dependent: FastDDS leaves the last
        # 19-byte point unpadded (n*20 - 1), CycloneDDS pads it to the full stride
        # (n*20). Accept both. Insisting on one silently costs the entire point of
        # this path -- measured on /livox/lidar under rmw_cyclonedds_cpp, the check
        # rejected every message and fell back to 134 ms of rclpy deserialisation
        # per scan, i.e. ~134% of a core at 10 Hz, in place of 1.6 ms.
        if n <= 0 or body not in (n * stride, n * stride - 1):
            return None  # not the layout we expect -- fall back rather than guess
        a = np.frombuffer(buf, np.uint8, body, o)
        if body < n * stride:
            a = np.append(a, np.uint8(0))
        a = a.reshape(n, stride)
        xyz = a[:, 4:16].copy().view(np.float32).reshape(n, 3)
        return xyz, a[:, 17]

    def add_custom_raw(self, buf):
        """Raw-subscription callback: read the CDR directly when we trust the layout."""
        if self.raw_ok:
            parsed = self._parse_custom_raw(buf)
            if parsed is not None and (self._raw_verified or self._verify_raw(parsed)):
                self._store_custom(*parsed)
                return
            if self._raw_verified:
                # One odd message after the layout was already confirmed (a truncated
                # publish, say). Drop it rather than demote a path known to work.
                self.dropped += 1
                return
            self.raw_ok = False
            print("  raw CustomMsg parse rejected on the first message -- falling back "
                  "to rclpy deserialisation (slower, but correct)")
        self._deserialize_fallback(buf)

    def _deserialize_fallback(self, buf):
        from rclpy.serialization import deserialize_message
        try:
            self.add_custom_msg(deserialize_message(buf, self.custom_msg_type))
        except Exception:
            # A message rclpy itself cannot parse: count it and carry on. Raising here
            # would take down the executor thread and with it every later scan.
            self.dropped += 1

    def _store_custom(self, xyz, tag):
        if self.stride > 1:
            xyz, tag = xyz[::self.stride], tag[::self.stride]
        keep = np.any(xyz != 0.0, axis=1) & np.isfinite(xyz).all(axis=1)
        if self.filter_tags:
            keep &= (tag & 0x30) <= 0x10        # same rejection as the C++ colorizer
        self.add(np.ascontiguousarray(xyz[keep]))

    def _verify_raw(self, parsed):
        """One-time sanity check that the byte layout really is what we assumed."""
        xyz, _ = parsed
        finite = np.isfinite(xyz)
        if not (finite.all() and np.abs(xyz).max() < 1000.0):
            return False
        self._raw_verified = True
        print(f"  raw CustomMsg parse verified on the first message "
              f"({len(xyz)} points, no rclpy deserialisation)")
        return True

    def add_custom_msg(self, msg):
        st = self.stride
        if self.filter_tags:
            # Same noisy/low-confidence rejection the C++ colorizer applies.
            pts = [(p.x, p.y, p.z) for p in msg.points[::st]
                   if (p.tag & 0x30) in (0x00, 0x10) and (p.x or p.y or p.z)]
        else:
            pts = [(p.x, p.y, p.z) for p in msg.points[::st] if (p.x or p.y or p.z)]
        self.add(np.asarray(pts, dtype=np.float32).reshape(-1, 3))

    def add_pointcloud2(self, msg):
        offs = {f.name: f.offset for f in msg.fields}
        if not {'x', 'y', 'z'} <= offs.keys():
            return
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        raw = raw[::self.stride]
        xyz = np.stack([raw[:, offs[c]:offs[c] + 4].copy().view(np.float32).ravel()
                        for c in ('x', 'y', 'z')], axis=1)
        xyz = xyz[np.isfinite(xyz).all(axis=1) & np.any(xyz != 0.0, axis=1)]
        self.add(np.ascontiguousarray(xyz))


def start_lidar(args, buf):
    """Spin a ROS 2 node feeding `buf` on a background thread. Returns a stopper."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    except ImportError:
        raise SystemExit(
            "no rclpy -- source the ROS 2 workspace (source install/setup.bash) first, "
            "or pass --no-lidar to just look at the cut views.")

    rclpy.init(args=None)
    node = Node('insta360_live_view_overlay')
    qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.BEST_EFFORT)

    fmt = args.lidar_format
    if fmt == 'auto':
        # Ask the graph rather than guessing: msg_MID360_launch.py publishes CustomMsg,
        # rviz_MID360_launch.py publishes PointCloud2, on the same topic name.
        fmt = 'custom'
        deadline = time.time() + 3.0
        while time.time() < deadline:
            types = dict(node.get_topic_names_and_types()).get(args.lidar_topic)
            if types:
                fmt = 'pointcloud2' if any('PointCloud2' in t for t in types) else 'custom'
                break
            time.sleep(0.2)

    if fmt == 'pointcloud2':
        from sensor_msgs.msg import PointCloud2
        node.create_subscription(PointCloud2, args.lidar_topic, buf.add_pointcloud2, qos)
    else:
        try:
            from livox_ros_driver2.msg import CustomMsg
        except ImportError:
            raise SystemExit(
                "cannot import livox_ros_driver2.msg.CustomMsg -- source the workspace "
                "(source install/setup.bash) first, or pass --lidar-format pointcloud2.")
        buf.custom_msg_type = CustomMsg
        if args.lidar_raw:
            # raw=True delivers the CDR bytes and skips building 20k Python objects.
            node.create_subscription(CustomMsg, args.lidar_topic, buf.add_custom_raw,
                                     qos, raw=True)
        else:
            node.create_subscription(CustomMsg, args.lidar_topic, buf.add_custom_msg, qos)

    raw_note = "" if fmt != 'custom' else (", raw parse" if args.lidar_raw
                                           else ", rclpy deserialisation")
    print(f"lidar      : {args.lidar_topic} ({fmt}{raw_note}), "
          f"{args.lidar_window:.2f}s window, every {buf.stride} point(s)")

    stop = threading.Event()

    def spin():
        while not stop.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()

    def shutdown():
        stop.set()
        thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return shutdown


# -------------------------------------------------------------------------- camera

class PanoramaSource:
    """Newest-frame-wins reader for the UVC panorama stream."""

    def __init__(self, args):
        # Decoding is the single most expensive thing in the loop (~23 ms of CPU for a
        # 2880x1440 MJPEG frame on a Pi 5) and it is not parallelised, so the grab loop
        # decodes at most `fps` frames a second: cap.grab() drains the USB queue without
        # touching the JPEG, and only a frame we are going to draw is retrieved. Every
        # cap call stays on this one thread -- VideoCapture is not thread-safe.
        self.min_period = 1.0 / args.fps if args.fps > 0 else 0.0
        self.cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(args.device)
        if not self.cap.isOpened():
            raise SystemExit(f"cannot open capture device {args.device}")

        # FOURCC before resolution: this camera ignores the size otherwise.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"camera     : device {args.device}, {w:.0f}x{h:.0f}")
        if abs(w / max(h, 1) - 2.0) > 1e-3:
            print(f"  WARNING: {w:.0f}x{h:.0f} is not 2:1. The views assume a full "
                  f"360x180 equirectangular panorama; anything else projects wrong.")

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.dropped = 0
        self.skipped = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        """Drain the USB stream continuously so the display always gets the newest frame."""
        last_decode = 0.0
        while self.running:
            if not self.cap.grab():
                self.dropped += 1
                time.sleep(0.01)
                continue
            now = time.monotonic()
            if now - last_decode < self.min_period:
                self.skipped += 1  # dropped before the JPEG was ever decoded
                continue
            ok, frame = self.cap.retrieve()
            if not ok or frame is None:
                self.dropped += 1
                continue
            last_decode = now
            with self.lock:
                self.frame = frame

    def latest(self):
        with self.lock:
            frame, self.frame = self.frame, None
        return frame

    def close(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


# -------------------------------------------------------------------------- drawing

# The ramp only ever takes 256 values, so build it once and index it. Calling
# applyColorMap on the frame's points instead costs 3x as much for the same output.
_DEPTH_LUT = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1),
                               cv2.COLORMAP_TURBO).reshape(256, 3)


def depth_colors(rng, dmin, dmax):
    t = np.clip((rng - dmin) * (255.0 / max(1e-6, dmax - dmin)), 0.0, 255.0)
    return _DEPTH_LUT[t.astype(np.uint8)]


def paint(img, u, v, rng, radius, dmin, dmax, alpha=1.0):
    """Stamp depth-coloured points into img. Returns how many landed."""
    h, w = img.shape[:2]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (rng >= dmin)
    u, v, rng = u[inside], v[inside], rng[inside]
    if len(u) == 0:
        return 0

    # Far to near, so the nearest point wins where two overlap (numpy fancy-index
    # assignment keeps the last write) -- the cheap stand-in for a depth buffer.
    order = np.argsort(-rng)
    ui = u[order].astype(np.int32)
    vi = v[order].astype(np.int32)
    colors = depth_colors(rng[order], dmin, dmax)

    r = int(radius)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            yy = np.clip(vi + dy, 0, h - 1)
            xx = np.clip(ui + dx, 0, w - 1)
            if alpha >= 1.0:
                img[yy, xx] = colors
            else:
                # Blending keeps the underlying image readable when the cloud is
                # dense -- which is exactly when you need to see what it covers.
                img[yy, xx] = (img[yy, xx] * (1.0 - alpha) + colors * alpha).astype(img.dtype)
    return len(u)


MATCH_COLORS = {'fast': (255, 255, 255), 'cluster': (0, 220, 255)}


def draw_anchors(img, anchor, color=(255, 255, 255)):
    """Mark where the top / central / bottom samples actually landed."""
    tx, ty, cx_, cy_, bx, by = anchor['pixels']
    for (x, y), r in (((tx, ty), 4), ((cx_, cy_), 5), ((bx, by), 4)):
        cv2.drawMarker(img, (int(x), int(y)), color, cv2.MARKER_CROSS, r * 2, 1,
                       cv2.LINE_AA)
    cv2.line(img, (int(tx), int(ty)), (int(bx), int(by)), color, 1, cv2.LINE_AA)


def label_view(img, text, sub=None):
    cv2.rectangle(img, (0, 0), (img.shape[1], 46 if sub else 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 220, 255), 1,
                    cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (60, 60, 60), 1)


def panorama_overlay(panorama, p_pano, args, radius, color_max, scale_w):
    # p_pano must already be range-filtered; ranges are recomputed here because this
    # path is off by default and not worth threading an extra array through.
    """The same points drawn straight onto the panorama -- a global sanity check."""
    ph, pw = panorama.shape[:2]
    scale = scale_w / pw
    # INTER_NEAREST, not INTER_AREA: this is a coarse whole-sphere sanity check and
    # area-averaging a 4 MP frame costs ~19 ms of CPU against ~3 ms for nearest.
    small = cv2.resize(panorama, (int(pw * scale), int(ph * scale)),
                       interpolation=cv2.INTER_NEAREST)
    if len(p_pano):
        rng = np.linalg.norm(p_pano, axis=1)
        px = iv.bearing_to_pixel(p_pano, pw, ph) * scale
        paint(small, px[:, 0], px[:, 1], rng, radius, args.min_depth, color_max,
              args.alpha)
    label_view(small, "panorama (equirectangular) with the same cloud")
    return small


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    v = p.add_argument_group('views')
    v.add_argument('--view', type=float, nargs='+', action='append', metavar='N',
                   help='YAW [PITCH [ROLL [FOV]]] in degrees; repeat once per view')
    v.add_argument('--ring', type=int, default=0,
                   help='additionally cut N evenly spaced views (the calibration ring)')
    v.add_argument('--yaw-start', type=float, default=0.0, help='first yaw of --ring')
    v.add_argument('--pitch', type=float, default=0.0, help='default pitch for views')
    v.add_argument('--roll', type=float, default=0.0, help='default roll for views')
    v.add_argument('--fov', type=float, default=60.0, help='default view fov (default 60)')
    v.add_argument('--size', type=int, default=480, help='view side length in px')
    v.add_argument('--max-cols', type=int, default=3, help='views per row in the window')
    v.add_argument('--every', type=int, default=1,
                   help='project the cloud only every Nth frame, reusing the previous '
                        'projection in between (the video stays at full rate)')

    e = p.add_argument_group('extrinsic (both frames x=fwd, y=left, z=up)')
    e.add_argument('--translation', type=float, nargs=3, default=[0.18, 0.0, -0.13],
                   metavar=('X', 'Y', 'Z'), help='camera origin in LiDAR coords, metres')
    e.add_argument('--rpy', type=float, nargs=3, default=[90.0, 0.0, -23.0],
                   metavar=('YAW', 'PITCH', 'ROLL'),
                   help='camera mounting: yaw>0 front swings LEFT, pitch>0 tilts UP, '
                        'roll>0 lifts its LEFT side')
    e.add_argument('--extrinsic', help='instead, load a 4x4/3x4 panorama-frame '
                                       'T_cam_lidar (e.g. the solved result)')

    c = p.add_argument_group('camera')
    c.add_argument('--device', type=int, default=0, help='V4L2 index of the Insta360')
    c.add_argument('--width', type=int, default=2880)
    c.add_argument('--height', type=int, default=1440)
    c.add_argument('--fps', type=float, default=10.0,
                   help='cap how many frames a second are decoded and drawn (0 = every '
                        'frame the camera sends). Each decode costs ~23 ms of CPU at '
                        '2880x1440, so this is the biggest single cost knob')
    c.add_argument('--cubic', action='store_true',
                   help='bicubic remap instead of bilinear: sharper cut-outs for about '
                        'twice the CPU. Bilinear is plenty for judging alignment')

    l = p.add_argument_group('lidar')
    l.add_argument('--lidar-topic', default='/livox/lidar')
    l.add_argument('--lidar-format', choices=('auto', 'custom', 'pointcloud2'),
                   default='auto')
    l.add_argument('--lidar-window', type=float, default=0.4,
                   help='seconds of scans to accumulate; longer is denser but smears '
                        'if the rig moves (no motion compensation)')
    l.add_argument('--point-stride', type=int, default=1,
                   help='keep every Nth point (raise it if the loop is CPU-bound)')
    l.add_argument('--no-filter-tags', dest='filter_tags', action='store_false',
                   help='keep points Livox flags noisy/low-confidence')
    l.add_argument('--no-lidar-raw', dest='lidar_raw', action='store_false',
                   help='let rclpy deserialise CustomMsg instead of reading the CDR '
                        'bytes directly. ~500x more CPU per message; only for '
                        'comparing against the fast path')
    l.add_argument('--no-lidar', action='store_true',
                   help='just show the cut views, no ROS, no projection')

    d = p.add_argument_group('display')
    d.add_argument('--min-depth', type=float, default=0.3,
                   help='drop points nearer than this -- the drone body self-occludes')
    d.add_argument('--max-depth', type=float, default=30.0,
                   help='colour ramp saturation range; points beyond it are dropped')
    d.add_argument('--point-radius', type=int, default=1)
    d.add_argument('--stats', action='store_true',
                   help='print a per-frame status line: points, fps, and the frames '
                        'and LiDAR messages that were dropped')
    y = p.add_argument_group('yolo (Hailo)')
    y.add_argument('--yolo', metavar='HEF',
                   help='run a Hailo YOLO HEF on the selected views and draw the boxes')
    y.add_argument('--yolo-view', type=int, action='append', metavar='N',
                   help='which view index to detect on; repeat for several '
                        '(default: view 0 only -- each extra view is another inference)')
    y.add_argument('--yolo-all-views', action='store_true',
                   help='detect on every view')
    y.add_argument('--yolo-conf', type=float, default=0.25,
                   help='score threshold (the HEF also has one baked into its NMS)')
    y.add_argument('--yolo-every', type=int, default=1,
                   help='infer every Nth frame, reusing the last boxes in between')
    y.add_argument('--yolo-box-shrink', type=float, default=0.6,
                   help='fraction of the box sampled for depth, about its centre. '
                        'Lower rejects more background, at the cost of points')
    y.add_argument('--yolo-cluster-gap', type=float, default=0.5,
                   help='metres of range separating the object from what is behind it')
    y.add_argument('--yolo-min-points', type=int, default=5,
                   help='fewest LiDAR returns a box needs before a position is claimed')
    y.add_argument('--yolo-print', action='store_true',
                   help='echo each detection\'s top/central/bottom LiDAR coordinates')
    y.add_argument('--match', choices=('fast', 'cluster', 'both'), default='fast',
                   help="how to separate the object from its background inside the box: "
                        "'fast' = nearest range run (cheap), 'cluster' = depth-mode gate "
                        "+ 3D voxel components (robust when the background sits at the "
                        "same range), 'both' = run each on the same box and draw both, "
                        "which is the only way to compare them on identical input")
    y.add_argument('--match-voxel', type=float, default=0.08,
                   help="voxel size for 'cluster' connectivity, metres")
    y.add_argument('--anchor-axis', choices=('row', 'z'), default='row',
                   help="define top/bottom by image row (default) or by world-up z "
                        "in the LiDAR frame")
    d.add_argument('--color-max', type=float, default=0.0,
                   help='range where the colour ramp saturates (0 = use --max-depth); '
                        'drop it to ~10 indoors, where everything is otherwise one blue')
    d.add_argument('--alpha', type=float, default=1.0,
                   help='point opacity 0-1; below 1 blends so the image stays visible '
                        'under a dense cloud')
    d.add_argument('--show-panorama', action='store_true',
                   help='also show the whole panorama with the cloud on it')
    d.add_argument('--no-display', action='store_true',
                   help='headless: write snapshots instead of opening a window')
    d.add_argument('--save-dir', default=None,
                   help='where "s" and --snapshot-interval write')
    d.add_argument('--snapshot-interval', type=float, default=0.0,
                   help='seconds between automatic snapshots (0 = only on "s")')
    args = p.parse_args()

    views = parse_views(args)
    color_max = args.color_max or args.max_depth
    r_pano, t_pano = build_extrinsic(args)
    print("views      : " + "; ".join(f"[{i}] {vw.label}" for i, vw in enumerate(views)))

    save_dir = args.save_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'live_snapshots')
    # if args.no_display and not args.snapshot_interval:
    #     args.snapshot_interval = 2.0

    buf = None
    shutdown = None
    if not args.no_lidar:
        buf = LidarBuffer(args.lidar_window, args.point_stride, args.filter_tags)
        shutdown = start_lidar(args, buf)

    # Opened after the camera for the same reason live_cluster_match does it: an
    # activated Hailo network group still open at interpreter teardown segfaults in
    # libhailort, so nothing that can fail may run between claiming it and the
    # try/finally that releases it.
    cam = PanoramaSource(args)

    detector = None
    yolo_views = set()
    if args.yolo:
        from hailo_yolo import HailoYolo, draw_detections
        try:
            detector = HailoYolo(args.yolo, conf=args.yolo_conf)
        except Exception as e:
            cam.close()
            raise SystemExit(
                f"cannot load {args.yolo}: {e}\n"
                f"If this is an architecture mismatch, compare\n"
                f"  hailortcli parse-hef {args.yolo}\n"
                f"  hailortcli fw-control identify\n"
                f"-- a HEF only runs on the architecture it was compiled for.")
        if args.yolo_all_views:
            yolo_views = set(range(len(views)))
        else:
            yolo_views = set(args.yolo_view or [0])
        bad = [i for i in yolo_views if i >= len(views)]
        if bad:
            raise SystemExit(f"--yolo-view {bad} out of range (only {len(views)} views)")
        print(f"yolo       : {args.yolo} {detector.input_size[0]}x{detector.input_size[1]} "
              f"on view(s) {sorted(yolo_views)}, conf {args.yolo_conf}"
              + (f", every {args.yolo_every} frames" if args.yolo_every > 1 else ""))
        print(f"match      : {args.match}"
              + (" (white = fast, cyan = cluster)" if args.match == 'both' else ""))

    canvas_maker = ViewCanvas(views, args.max_cols,
                              cv2.INTER_CUBIC if args.cubic else cv2.INTER_LINEAR)

    show_pano = args.show_panorama
    radius = args.point_radius
    fps, n_frames, t_fps = 0.0, 0, time.monotonic()
    last_snapshot = 0.0
    warned_empty = False
    frame_no = 0
    detections = [[] for _ in views]
    yolo_ms = 0.0
    methods = ('fast', 'cluster') if args.match == 'both' else (args.match,)
    match_ms = {m: 0.0 for m in methods}
    projections = [(np.empty(0), np.empty(0), np.empty(0))] * len(views)
    d_cam = np.empty((0, 3), dtype=np.float32)
    rng = np.empty(0, dtype=np.float32)
    near = np.empty(0, dtype=bool)
    r32 = r_pano.astype(np.float32)
    # The camera origin in LiDAR coords -- p_pano = R(p_lidar - origin), so working
    # camera-relative from the start lets the cull run in the LiDAR's own axes.
    cam_origin32 = (-r_pano.T @ t_pano).astype(np.float32)
    for vw in views:
        vw.set_extrinsic(r32)

    print("\nrunning -- q quit, s snapshot, p panorama overlay, [ / ] point size\n")
    try:
        while True:
            frame = cam.latest()
            if frame is None:
                time.sleep(0.005)
                continue

            frame_no += 1
            if buf is not None and frame_no % args.every == 0:
                # float32 throughout: the cloud is metres at centimetre accuracy, and
                # halving the width halves the memory traffic this loop is bound by.
                pts = buf.snapshot()
                if len(pts):
                    # Nothing is rotated yet: shift to camera-relative, measure range
                    # (unchanged by the rotation still to come), and let each view
                    # cull its own cone before paying for the transform.
                    d_cam = pts - cam_origin32
                    rng = np.linalg.norm(d_cam, axis=1)
                    near = (rng >= args.min_depth) & (rng <= args.max_depth)
                    projections = [vw.project(d_cam, rng, near) for vw in views]
                else:
                    d_cam = np.empty((0, 3), dtype=np.float32)
                    rng, near = np.empty(0), np.empty(0, dtype=bool)
                    projections = [(np.empty(0), np.empty(0), np.empty(0))] * len(views)
                    if not warned_empty:
                        print(f"  (no points yet on {args.lidar_topic} -- is the driver up?)")
                        warned_empty = True

            canvas = canvas_maker.render(frame)
            for i, vw in enumerate(views):
                tile = canvas_maker.tile(canvas, vw)

                # Detect BEFORE the cloud is painted on: the detector should see the
                # photograph, not a tile speckled with LiDAR returns.
                if detector is not None and i in yolo_views:
                    if frame_no % args.yolo_every == 0:
                        t_yolo = time.monotonic()
                        detections[i] = detector.infer(tile)
                        yolo_ms = (time.monotonic() - t_yolo) * 1000.0

                u, vv, rng = projections[i]
                n = paint(tile, u, vv, rng, radius, args.min_depth, color_max, args.alpha)
                if detector is not None and i in yolo_views:
                    draw_detections(tile, detections[i], detector)
                    located = 0
                    for det in detections[i]:
                        found = {}
                        for method in methods:
                            t_m = time.perf_counter()
                            a = bm.match(vw, det, u, vv, rng, cam_origin32, method,
                                         args.yolo_box_shrink, args.yolo_cluster_gap,
                                         args.yolo_min_points, args.match_voxel,
                                         args.anchor_axis)
                            match_ms[method] = (time.perf_counter() - t_m) * 1000.0
                            if a is not None:
                                found[method] = a
                                draw_anchors(tile, a, MATCH_COLORS[method])
                        if not found:
                            continue
                        located += 1
                        shown = found.get('cluster', found.get('fast'))
                        c = shown['central']
                        cv2.putText(tile, f"{shown['range']:.1f}m "
                                          f"({c[0]:+.1f},{c[1]:+.1f},{c[2]:+.1f})",
                                    (int(det[0]) + 2, min(int(det[3]) - 4, vw.size - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                                    cv2.LINE_AA)
                        if args.yolo_print:
                            for method, a in found.items():
                                print(f"  [{i}] {detector.label(det[5])} {det[4]:.2f} "
                                      f"{method:>7s}: {a['n']:4d} pts  "
                                      f"range {a['range']:5.2f} m  "
                                      f"top {np.round(a['top'], 2)}  "
                                      f"central {np.round(a['central'], 2)}  "
                                      f"bottom {np.round(a['bottom'], 2)}")
                            if len(found) == 2:
                                # The number that matters when comparing: how far apart
                                # the two methods place the same object.
                                gap3 = np.linalg.norm(found['fast']['central']
                                                      - found['cluster']['central'])
                                print(f"        fast vs cluster centre disagreement: "
                                      f"{gap3:.2f} m")
                    timing = "  ".join(f"{m}{match_ms[m]:.1f}ms" for m in methods)
                    label_view(tile, f"[{i}] {vw.label}",
                               f"{n} pts  {len(detections[i])} det  {located} located  "
                               f"yolo {yolo_ms:.0f} ms  {timing}  {fps:.1f} fps")
                else:
                    label_view(tile, f"[{i}] {vw.label}", f"{n} pts  {fps:.1f} fps")
            if show_pano:
                # The only consumer that needs every point in the panorama frame, so
                # it pays for that rotation itself -- and only while it is on screen.
                p_pano = d_cam[near] @ r32.T
                over = panorama_overlay(frame, p_pano, args, radius, color_max,
                                        scale_w=max(canvas.shape[1], 640))
                if over.shape[1] != canvas.shape[1]:
                    h = int(over.shape[0] * canvas.shape[1] / over.shape[1])
                    over = cv2.resize(over, (canvas.shape[1], h))
                canvas = np.vstack([canvas, over])

            n_frames += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                fps = n_frames / (now - t_fps)
                n_frames, t_fps = 0, now

            if args.stats:
                # One line per frame, rewritten in place. Printing inside the view loop
                # instead makes the views overwrite each other, and costs a syscall per
                # view rather than per frame.
                print(f"\rframe {frame_no}  {len(d_cam)} pts  {fps:4.1f} fps  "
                      f"camera dropped {cam.dropped} skipped {cam.skipped}"
                      + (f"  lidar msgs {buf.n_msgs} dropped {buf.dropped}"
                         if buf is not None else "") + "   ", end='', flush=True)

            snap = bool(args.snapshot_interval) and now - last_snapshot >= args.snapshot_interval

            if not args.no_display:
                cv2.imshow('insta360 views + livox', canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('s'):
                    snap = True
                elif key == ord('p'):
                    show_pano = not show_pano
                elif key == ord(']'):
                    radius = min(radius + 1, 8)
                elif key == ord('['):
                    radius = max(radius - 1, 0)

            if snap:
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, time.strftime('live_%Y%m%d_%H%M%S.png'))
                if cv2.imwrite(path, canvas):
                    print(f"  wrote {path}")
                else:
                    print(f"  FAILED to write {path}")
                last_snapshot = now
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        if detector is not None:
            detector.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        if shutdown is not None:
            shutdown()


if __name__ == '__main__':
    main()
