#!/usr/bin/env python3
"""YOLO bbox -> object keypoints in the GLOBAL (world) frame.

The rig is rigid but its extrinsic is not. The Insta360 X5's webcam stream is
gravity-locked -- the horizon stays level however the drone is banked -- while the
LiDAR is bolted to the airframe and tilts with it. A single T_cam_lidar is therefore
only true at the attitude it was solved at, and `live_cluster_match.py` inherits that
error the moment the drone stops being level.

The calibration data says exactly this. data/T_cam_lidar_indoor_level.txt was solved
with the LiDAR 22.75 deg off level (data/calib_indoor_level/gravity.txt), yet its
camera UP axis sits 0.57 deg from gravity up. Rotate that extrinsic into the levelled
frame and it collapses to a pure yaw:

    camera axes in the levelled frame        yaw offset  = +86.79 deg
    [ 0.0560  -0.9984   0.0097 ]             residual tilt = 0.57 deg
    [ 0.9984   0.0559  -0.0019 ]
    [ 0.0014   0.0098   1.0000 ]

The extrinsic did not fail; it measured a hinge and reported it as a fixed angle. The
--rpy 90 0 -23 seed is that hinge -- yaw 90, and a roll of -23 that is simply the
LiDAR's own tilt at capture time.

So this tool does not use a camera<->LiDAR extrinsic at all. It builds camera<->world
directly, where the pose is known analytically because the stabiliser holds roll and
pitch at zero:

    R_world_pano = Rz(yaw(R_wb) + dpsi)          a PURE yaw, dpsi from the calibration
    p_world_cam  = R_wi @ (t_lidar_cam + t_il) + p_wi

Two things survive the recalibration: one scalar (dpsi) and one lever arm. Everything
else arrives from /Odometry every frame. Tilt cannot be expressed in this chain.

WHY THE MATCH STILL RUNS ON A LOCAL WINDOW
------------------------------------------
Points are carried into world coordinates, but only a short window of them -- this is
NOT matching against /Laser_map, and that is deliberate.

  drift    The points and the camera pose come from the same instant, so WHICH points
           fall inside a box depends only on relative geometry and SLAM drift cancels
           exactly. It enters once, at the end, on three points. Matching against an
           accumulated map re-introduces it into the association: points placed
           minutes ago carry that moment's drift while the pose carries now's, and
           the failure is discrete (the wrong object) rather than graceful. FAST-LIO2
           has no loop closure, so drift is monotone -- 0.1-0.5% of path is 10-50 cm
           after 100 m, which at 5 m range is 9-45 px in a 480 px / 60 deg view
           against a ~50 px person.
  occlusion A LiDAR return IS a visible surface. An accumulated map holds everything
           ever seen, including what is now hidden behind a wall, and no depth gate
           recovers a surface that was never mapped.
  motion   A world-frame accumulation smears anything that moves: v * T. For a walker
           at 1.4 m/s to stay inside their own 0.5 m width needs T < 0.36 s -- which
           is almost exactly the 0.4 s window the LiDAR tools already use.
  cost     Measured on a Pi 5, per frame with 3 detections: 29k points (0.4 s) costs
           48% of a core; a 60 s world buffer 87%; a 300 s one 118%, i.e. over budget
           at 10 fps. /Laser_map itself is worse than unusable -- laserMapping.cpp
           never clears pcl_wait_pub, so the topic republishes the WHOLE accumulation
           at 1 Hz: ~64 MB/message after ten minutes, 99.85% of it unchanged.

What the world frame buys is DESKEW, not persistence. `LidarBuffer` accumulates in
the raw LiDAR frame with no motion compensation, which is why its window is pinned at
0.4 s; here each message is placed by its own odometry pose, so the ego-motion smear
is gone and --window can be raised safely. Object motion is then the only limit left.

MATCHING
--------
'fast' only (bbox_match.select_nearest_run): sort the in-box ranges, split at gaps
bigger than --gap, keep the nearest run with real support. 0.79 ms per detection
against 2.34 ms for 'cluster', and it skips _voxel_labels entirely -- whose
interpreted union-find is 76% of the cluster cost once the candidate set grows
(24.6 ms of 32.4 ms at 56k candidates). Use live_cluster_match.py when a box's object
does NOT stand clear of its background in depth; that is the one case 'fast' cannot
handle and 'cluster' can.

Because the points are already world-referenced, `anchors(anchor_axis='z')` splits
top from bottom along TRUE gravity up -- in the LiDAR-frame tools that axis tilts with
the airframe -- and top / central / bottom come out as global XYZ with no transform.

    python3 global_bbox_match.py --view 240 --yolo <hef> --translation 0.18 0.0 -0.13
    python3 global_bbox_match.py --view 240 --yolo <hef> --publish-markers   # RViz

Rendering, as in live_cluster_match.py -- seeing what was thrown away is the point:

    grey    inside the box, before anything was rejected
    green   the nearest range run -- the object, and the only points the keypoints
            are computed from
    C/B/T   centre, bottom, top, drawn where they reproject

Keys: q / Esc quit, s snapshot, l toggle the faint full cloud, p pause.
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
import live_view_overlay as lvo
import live_cluster_match as lcm      # stamp(), draw_keypoints(): shared so the two
                                      # tools' rendering cannot drift apart

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXTRINSIC = os.path.join(HERE, 'data', 'T_cam_lidar_indoor_level.txt')
DEFAULT_GRAVITY = os.path.join(HERE, 'data', 'calib_indoor_level', 'gravity.txt')

STAGE_COLORS = {'candidate': (150, 150, 150), 'object': (90, 230, 50)}


# ------------------------------------------------------------------ pose algebra

def quat_to_matrix(q):
    """xyzw quaternion -> rotation matrix. Normalised first: odometry quaternions
    arrive slightly off unit length and the error squares into the matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_of(r):
    """Heading of a rotation about world up -- the ONLY part of the body attitude the
    gravity-locked camera follows."""
    return math.atan2(r[1, 0], r[0, 0])


def rz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def load_gravity_rotation(path):
    """The `rotation` block of a gravity.txt: R such that p_level = R @ p_raw."""
    vals, grab = [], False
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            if line.lower().startswith('rotation'):
                grab = True
                continue
            if grab:
                vals.extend(float(v) for v in line.replace(',', ' ').split())
                if len(vals) >= 9:
                    break
    if len(vals) < 9:
        raise SystemExit(f"{path}: no 3x3 `rotation` block found")
    return np.array(vals[:9], dtype=float).reshape(3, 3)


def derive_yaw_offset(extrinsic_path, gravity_path):
    """dpsi: the camera's heading in the LEVELLED body frame, from the calibration.

    The solved extrinsic is p_pano = R @ p_lidar, so R.T holds the camera's axes in
    LiDAR coordinates. Levelling those with the gravity rotation measured at the same
    capture gives the camera's axes in a frame whose z is gravity up and whose yaw is
    the body's -- exactly the frame /Odometry reports its heading in. What is left is
    one angle. The residual tilt is returned too: it is the error floor of the whole
    gravity-lock assumption, so it is worth printing rather than trusting silently.
    """
    r_pano_lidar, _ = lvo.load_extrinsic_file(extrinsic_path)
    r_level = load_gravity_rotation(gravity_path)
    r_lev_cam = r_level @ r_pano_lidar.T
    dpsi = math.degrees(math.atan2(r_lev_cam[1, 0], r_lev_cam[0, 0]))
    tilt = math.degrees(math.acos(np.clip(r_lev_cam[2, 2], -1.0, 1.0)))
    return dpsi, tilt


class OdomBuffer:
    """Recent /Odometry poses, interpolated to an arbitrary stamp.

    FAST-LIO publishes at 10 Hz, the same rate the LiDAR and the camera run at, so
    almost every query lands between two samples rather than on one. Interpolating
    rather than snapping to the nearest matters most exactly when it is hardest to
    notice: during a fast yaw, half a sample is several degrees.
    """

    def __init__(self, window_s=5.0):
        self.window_s = window_s
        self.stamps = deque()
        self.poses = deque()
        self.lock = threading.Lock()
        self.n = 0
        # Message stamps are NOT wall clock. The Livox timebase is not disciplined to
        # the host (common.time_sync_en is false), and FAST-LIO stamps its odometry
        # with lidar_end_time, so the whole ROS timeline runs at a fixed offset from
        # time.time() -- measured ~1.04 s behind on this rig. The camera stream
        # carries no stamp at all, so a frame can only be placed by wall-clock
        # arrival; without this correction every lookup lands a second past the
        # newest pose and the tool silently matches nothing. The offset also absorbs
        # odometry's own transport-plus-processing latency, which is what makes it
        # the right quantity to subtract: it puts camera arrival and odometry arrival
        # on one timeline, leaving --cam-latency to trim only the difference between
        # the two pipelines.
        self.offsets = deque(maxlen=64)
        self.offset = 0.0

    def add(self, msg):
        st = msg.header.stamp
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        stamp = st.sec + st.nanosec * 1e-9
        arrived = time.time()
        with self.lock:
            self.stamps.append(stamp)
            self.poses.append((np.array([p.x, p.y, p.z]),
                               np.array([o.x, o.y, o.z, o.w])))
            self.offsets.append(arrived - stamp)
            # Median, not mean: a single late message under load would drag a mean
            # by its whole delay and mis-place every frame until it aged out.
            self.offset = float(np.median(self.offsets))
            self.n += 1
            while len(self.stamps) > 1 and self.stamps[-1] - self.stamps[0] > self.window_s:
                self.stamps.popleft()
                self.poses.popleft()

    def pose_at_wall(self, t_wall, max_extrap=0.15):
        """(R_wb, p_wb) for a wall-clock instant, via the measured stamp offset."""
        if not self.offsets:
            return None
        return self.pose_at(t_wall - self.offset, max_extrap)

    def pose_at(self, t, max_extrap=0.15):
        """(R_wb, p_wb) at ROS time `t`, or None if the buffer cannot cover it.

        Returning None rather than the nearest pose is deliberate: a frame placed with
        a pose 300 ms stale is not a slightly worse measurement, it is a wrong one, and
        silently emitting it would put objects at coordinates nothing flags as bad.
        """
        with self.lock:
            if not self.stamps:
                return None
            ts = np.fromiter(self.stamps, dtype=float, count=len(self.stamps))
            poses = list(self.poses)

        i = int(np.searchsorted(ts, t))
        if i == 0:
            return self._pose(poses[0]) if ts[0] - t <= max_extrap else None
        if i >= len(ts):
            return self._pose(poses[-1]) if t - ts[-1] <= max_extrap else None

        t0, t1 = ts[i - 1], ts[i]
        (p0, q0), (p1, q1) = poses[i - 1], poses[i]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        # nlerp with a hemisphere fix. Over a 100 ms gap the angle between the two
        # quaternions is small enough that nlerp and slerp differ far below the
        # 0.57 deg residual tilt of the gravity lock itself.
        if q0 @ q1 < 0.0:
            q1 = -q1
        return self._pose((p0 + a * (p1 - p0), q0 + a * (q1 - q0)))

    @staticmethod
    def _pose(pq):
        p, q = pq
        return quat_to_matrix(q), p


# ------------------------------------------------------------------- world cloud

class WorldLidarBuffer(lvo.LidarBuffer):
    """LidarBuffer whose window is carried into world coordinates, per message.

    Two changes to the base class, both forced by the same fact -- that a scan can
    only be placed once its pose is known:

    1. Every scan keeps the ROS stamp of the message it came from, not just arrival
       time, so each is placed by the pose at ITS instant. That is the deskew: the
       base class accumulates in the raw LiDAR frame with no compensation, which is
       why its docstring pins the window at 0.4 s.
    2. The transform is LAZY. FAST-LIO publishes /Odometry *after* it has processed
       the scan, so at callback time the pose for that stamp does not exist yet.
       Transforming eagerly would drop the newest -- and only fresh -- message every
       time. Each scan is instead converted the first time it is asked for and cached,
       so a scan is paid for once (0.15 ms per 6.7k points) no matter how many frames
       it survives.

    Points are placed the way laserMapping.cpp places them:
        p_world = R_wi @ (R_il @ p_lidar + t_il) + p_wi
    with R_il / t_il the mapping.extrinsic_R / extrinsic_T of the FAST-LIO config, so
    the LiDAR-to-IMU offset is not quietly folded into the camera lever arm.
    """

    def __init__(self, window_s, stride, filter_tags, odom, t_il):
        super().__init__(window_s, stride, filter_tags)
        self.odom = odom
        self.t_il = np.asarray(t_il, dtype=np.float32)
        self._stamp = None
        self.unplaced = 0

    # -- stamp capture, one override per ingest path ------------------------------
    def add_custom_raw(self, buf):
        # sec/nanosec sit immediately after the 4-byte encapsulation header; see the
        # layout note in LidarBuffer. Read before super(), which consumes the rest.
        if len(buf) >= 12:
            sec = int(np.frombuffer(buf, np.int32, 1, 4)[0])
            nsec = int(np.frombuffer(buf, np.uint32, 1, 8)[0])
            self._stamp = sec + nsec * 1e-9
        super().add_custom_raw(buf)

    def add_custom_msg(self, msg):
        self._stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        super().add_custom_msg(msg)

    def add_pointcloud2(self, msg):
        self._stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        super().add_pointcloud2(msg)

    # -- storage and placement ----------------------------------------------------
    def add(self, xyz):
        """Store body-frame points with their stamp; world comes later."""
        if self._stamp is None or not len(xyz):
            return
        now = time.monotonic()
        with self.lock:
            self.scans.append([now, self._stamp, xyz, None])
            self.n_msgs += 1
            while self.scans and now - self.scans[0][0] > self.window_s:
                self.scans.popleft()

    def snapshot(self):
        """The whole window in WORLD coordinates. Unplaceable scans are skipped."""
        with self.lock:
            cutoff = time.monotonic() - self.window_s
            pending = [s for s in self.scans if s[0] >= cutoff]

        out, unplaced = [], 0
        for scan in pending:
            if scan[3] is None:
                pose = self.odom.pose_at(scan[1])
                if pose is None:
                    unplaced += 1
                    continue
                r_wi, p_wi = pose
                scan[3] = ((scan[2] + self.t_il) @ r_wi.T.astype(np.float32)
                           + p_wi.astype(np.float32))
            out.append(scan[3])
        self.unplaced = unplaced
        if not out:
            return np.empty((0, 3), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(out, axis=0), dtype=np.float32)


# --------------------------------------------------------------------------- ros

def start_ros(args, buf, odom):
    """One node for all three jobs: LiDAR in, odometry in, markers out.

    Returns (shutdown, publish) where publish(objects) is a no-op unless
    --publish-markers was given.
    """
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from nav_msgs.msg import Odometry
    except ImportError:
        raise SystemExit(
            "no rclpy -- source the ROS 2 workspace (source install/setup.bash) first. "
            "This tool cannot run without odometry: the camera pose IS the odometry.")

    rclpy.init(args=None)
    node = Node('global_bbox_match')
    qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.BEST_EFFORT)

    # NOTE: node.get_clock() IS the system clock here, so comparing it to time.time()
    # proves nothing. What actually matters is the offset between MESSAGE STAMPS and
    # wall clock, which cannot be known until odometry arrives -- OdomBuffer measures
    # it per message and main() reports it once the first few are in.

    fmt = args.lidar_format
    if fmt == 'auto':
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
                "first, or pass --lidar-format pointcloud2.")
        buf.custom_msg_type = CustomMsg
        if args.lidar_raw:
            node.create_subscription(CustomMsg, args.lidar_topic, buf.add_custom_raw,
                                     qos, raw=True)
        else:
            node.create_subscription(CustomMsg, args.lidar_topic, buf.add_custom_msg, qos)

    node.create_subscription(Odometry, args.odom_topic, odom.add, qos)

    raw_note = "" if fmt != 'custom' else (", raw parse" if args.lidar_raw
                                           else ", rclpy deserialisation")
    print(f"lidar      : {args.lidar_topic} ({fmt}{raw_note}), "
          f"{args.lidar_window:.2f}s window, every {buf.stride} point(s)")
    print(f"odometry   : {args.odom_topic} -> world frame, interpolated per message")

    publish = _make_marker_publisher(node, args) if args.publish_markers else (lambda o: None)

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

    return shutdown, publish


def _make_marker_publisher(node, args):
    """MarkerArray of the keypoints, in `world`. Colours match the on-screen C/B/T."""
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from builtin_interfaces.msg import Duration

    pub = node.create_publisher(MarkerArray, args.marker_topic, 5)
    print(f"markers    : {args.marker_topic} (visualization_msgs/MarkerArray, "
          f"frame '{args.world_frame}')")
    rgb = {'T': (1.0, 0.0, 1.0), 'C': (1.0, 0.16, 0.16), 'B': (0.0, 0.63, 1.0)}
    life = Duration(sec=0, nanosec=int(args.marker_lifetime * 1e9))

    def base(kind, mid, stamp):
        m = Marker()
        m.header.frame_id = args.world_frame
        m.header.stamp = stamp
        m.ns = 'yolo_objects'
        m.id = mid
        m.type = kind
        m.action = Marker.ADD
        m.lifetime = life
        m.pose.orientation.w = 1.0
        return m

    def publish(objects):
        arr = MarkerArray()
        # DELETEALL first: ids are assigned by detection index, so a frame with fewer
        # detections than the last would otherwise leave the surplus on screen until
        # their lifetime expired.
        clear = Marker()
        clear.header.frame_id = args.world_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        stamp = node.get_clock().now().to_msg()
        for k, (label, anchor) in enumerate(objects):
            t_, c_, b_ = anchor['top'], anchor['central'], anchor['bottom']

            line = base(Marker.LINE_LIST, 4 * k, stamp)
            line.scale.x = 0.03
            line.color.r, line.color.g, line.color.b, line.color.a = 0.35, 0.9, 0.2, 0.9
            line.points = [Point(x=float(b_[0]), y=float(b_[1]), z=float(b_[2])),
                           Point(x=float(t_[0]), y=float(t_[1]), z=float(t_[2]))]
            arr.markers.append(line)

            for j, (tag, p) in enumerate((('T', t_), ('C', c_), ('B', b_))):
                s = base(Marker.SPHERE, 4 * k + 1 + j, stamp)
                s.scale.x = s.scale.y = s.scale.z = 0.12 if tag == 'C' else 0.08
                s.color.r, s.color.g, s.color.b = rgb[tag]
                s.color.a = 0.95
                s.pose.position.x, s.pose.position.y, s.pose.position.z = map(float, p)
                arr.markers.append(s)

            txt = base(Marker.TEXT_VIEW_FACING, 4 * (k + 1) + 1000, stamp)
            txt.scale.z = 0.22
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            txt.pose.position.x, txt.pose.position.y = float(t_[0]), float(t_[1])
            txt.pose.position.z = float(t_[2]) + 0.25
            txt.text = f"{label} {anchor['range']:.1f}m ({anchor['n']} pts)"
            arr.markers.append(txt)

        pub.publish(arr)

    return publish


# ---------------------------------------------------------------------- matching

def match_and_draw(tile, view, det, u, v, rng, cam_origin, args):
    """One detection, 'fast': draw both stages, return (anchor, counts) in WORLD xyz."""
    x1, y1, x2, y2 = det[:4]
    mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    # Shrink is per-axis on purpose. Narrowing the frustum is what keeps background
    # out of a method that has no depth gate -- but the box's VERTICAL extent is
    # precisely what carries top and bottom, and cropping it truncates the two
    # numbers this tool exists to report. Measured on a synthetic 1.8 m pole: a
    # uniform 0.6 shrink puts its top at 1.40 m and its base at 0.39 m. Sideways
    # neighbours (a wall beside the object) are what actually leak, so shrink there
    # and leave the vertical span alone.
    hw = (x2 - x1) * 0.5 * args.box_shrink
    hh = (y2 - y1) * 0.5 * args.box_shrink_v
    inside = (u >= mx - hw) & (u <= mx + hw) & (v >= my - hh) & (v <= my + hh)
    n_cand = int(inside.sum())
    if n_cand < args.min_points:
        return None, (n_cand, 0)

    ub, vb, rb = u[inside], v[inside], rng[inside]
    lcm.stamp(tile, ub, vb, STAGE_COLORS['candidate'], args.point_radius)

    sel = bm.select_nearest_run(rb, args.gap, args.min_points)
    if sel is None or len(sel) < args.min_points:
        return None, (n_cand, 0)

    us, vs, rs = ub[sel], vb[sel], rb[sel]
    lcm.stamp(tile, us, vs, STAGE_COLORS['object'], args.point_radius + 1)
    # Only the selected run is unprojected -- (u, v, range) is a complete description
    # of a projected point, so nothing extra rides through the per-frame path.
    pts3 = view.unproject(us, vs, rs, cam_origin)
    anchor = bm.anchors(pts3, us, vs, rs, np.arange(len(sel)), args.anchor_axis)
    lcm.draw_keypoints(tile, anchor)
    return anchor, (n_cand, len(sel))


def draw_legend(img, extra=()):
    lines = [("grey  = in box", STAGE_COLORS['candidate']),
             ("green = nearest range run (object)", STAGE_COLORS['object'])]
    y = img.shape[0] - 8 - 13 * (len(lines) + len(extra))
    for text, color in list(lines) + list(extra):
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
                    cv2.LINE_AA)
        y += 13


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    v = p.add_argument_group('views')
    v.add_argument('--view', type=float, nargs='+', action='append', metavar='N',
                   help='YAW [PITCH [ROLL [FOV]]] in degrees; repeat once per view')
    v.add_argument('--ring', type=int, default=0)
    v.add_argument('--yaw-start', type=float, default=0.0)
    v.add_argument('--pitch', type=float, default=0.0)
    v.add_argument('--roll', type=float, default=0.0)
    v.add_argument('--fov', type=float, default=60.0)
    v.add_argument('--size', type=int, default=480)
    v.add_argument('--max-cols', type=int, default=3)

    e = p.add_argument_group('mounting (all that survives the gravity lock)')
    e.add_argument('--translation', type=float, nargs=3, default=[0.18, 0.0, -0.13],
                   metavar=('X', 'Y', 'Z'),
                   help='camera ORIGIN in LiDAR coords, metres -- rigid, unaffected '
                        'by attitude (default: %(default)s)')
    e.add_argument('--yaw-offset', type=float, default=None, metavar='DEG',
                   help='dpsi: camera heading relative to the body, in the levelled '
                        'frame. Derived from --extrinsic + --gravity if omitted')
    e.add_argument('--extrinsic', default=DEFAULT_EXTRINSIC,
                   help='calibrated panorama-frame T_cam_lidar, to derive dpsi from')
    e.add_argument('--gravity', default=DEFAULT_GRAVITY,
                   help='gravity.txt from the SAME capture as --extrinsic')
    e.add_argument('--imu-lidar-t', type=float, nargs=3,
                   default=[-0.011, -0.02329, 0.04412], metavar=('X', 'Y', 'Z'),
                   help="FAST-LIO mapping.extrinsic_T (LiDAR origin in IMU frame)")

    c = p.add_argument_group('camera')
    c.add_argument('--device', type=int, default=0)
    c.add_argument('--width', type=int, default=2880)
    c.add_argument('--height', type=int, default=1440)
    c.add_argument('--fps', type=float, default=10.0)
    c.add_argument('--cubic', action='store_true')
    c.add_argument('--cam-latency', type=float, default=0.05, metavar='S',
                   help='exposure-to-latest() delay, subtracted before looking up the '
                        'pose. The dominant sync error during fast yaw: at 90 deg/s, '
                        '50 ms is 4.5 deg. Tune until a pole stops leading or lagging '
                        'its points as you rotate (default: %(default)s)')

    l = p.add_argument_group('lidar and odometry')
    l.add_argument('--lidar-topic', default='/livox/lidar')
    l.add_argument('--odom-topic', default='/Odometry')
    l.add_argument('--world-frame', default='world')
    l.add_argument('--lidar-format', choices=('auto', 'custom', 'pointcloud2'),
                   default='auto')
    l.add_argument('--lidar-window', type=float, default=0.4,
                   help='seconds of scans to accumulate. Kept at the LiDAR-frame tools '
                        'default so results stay comparable; deskew makes it safe to '
                        'raise, but object motion smears at v*T -- a 1.4 m/s walker '
                        'exceeds their own width past 0.36 s (default: %(default)s)')
    l.add_argument('--max-pose-age', type=float, default=0.15, metavar='S',
                   help='refuse to place a scan or a frame if odometry is this stale')
    l.add_argument('--point-stride', type=int, default=1)
    l.add_argument('--no-filter-tags', dest='filter_tags', action='store_false')
    l.add_argument('--no-lidar-raw', dest='lidar_raw', action='store_false')
    l.add_argument('--min-depth', type=float, default=0.3)
    l.add_argument('--max-depth', type=float, default=30.0)

    y = p.add_argument_group('detector and match')
    y.add_argument('--yolo', required=True, metavar='HEF')
    y.add_argument('--yolo-conf', type=float, default=0.25)
    y.add_argument('--yolo-every', type=int, default=1)
    y.add_argument('--yolo-view', type=int, action='append', metavar='N')
    y.add_argument('--yolo-all-views', action='store_true')
    y.add_argument('--box-shrink', type=float, default=0.6,
                   help="HORIZONTAL fraction of the box used as the frustum. 0.6 as "
                        "bbox_match's 'fast' default: with no depth gate and no "
                        "components to reject background, narrowing the frustum is "
                        "what keeps it out (default: %(default)s)")
    y.add_argument('--box-shrink-v', type=float, default=1.0,
                   help='VERTICAL fraction. Kept at 1.0 unlike the horizontal one: '
                        'the box height is what carries top and bottom, so cropping '
                        'it truncates them directly -- a 0.6 vertical shrink reports '
                        "a 1.8 m pole's top at 1.40 m (default: %(default)s)")
    y.add_argument('--gap', type=float, default=0.5, metavar='M',
                   help='range jump that splits one run from the next')
    y.add_argument('--min-points', type=int, default=5)
    y.add_argument('--anchor-axis', choices=('row', 'z'), default='z',
                   help='top/bottom along TRUE world up (z) -- correct here in a way '
                        'it is not in the LiDAR-frame tools -- or by image row')

    d = p.add_argument_group('output')
    d.add_argument('--publish-markers', action='store_true',
                   help='publish keypoints as a visualization_msgs/MarkerArray')
    d.add_argument('--marker-topic', default='/yolo_objects')
    d.add_argument('--marker-lifetime', type=float, default=0.5, metavar='S')
    d.add_argument('--point-radius', type=int, default=1)
    d.add_argument('--show-cloud', action='store_true')
    d.add_argument('--print', dest='do_print', action='store_true')
    d.add_argument('--no-display', action='store_true')
    d.add_argument('--save-dir')
    d.add_argument('--snapshot-interval', type=float, default=0.0)
    args = p.parse_args()

    views = lvo.parse_views(args)
    if args.yaw_offset is None:
        args.yaw_offset, tilt = derive_yaw_offset(args.extrinsic, args.gravity)
        print(f"yaw offset : {args.yaw_offset:+.2f} deg, derived from "
              f"{os.path.basename(args.extrinsic)} + {os.path.basename(args.gravity)}")
        print(f"             residual camera tilt {tilt:.2f} deg -- the error floor of "
              f"the gravity-lock assumption ({math.tan(math.radians(tilt)) * 10:.2f} m "
              f"at 10 m)")
    else:
        print(f"yaw offset : {args.yaw_offset:+.2f} deg (given)")
    dpsi = math.radians(args.yaw_offset)
    t_lidar_cam = np.asarray(args.translation, dtype=float)
    t_il = np.asarray(args.imu_lidar_t, dtype=float)
    print("views      : " + "; ".join(f"[{i}] {vw.label}" for i, vw in enumerate(views)))

    from hailo_yolo import HailoYolo, draw_detections

    # The accelerator is opened LAST and released in the finally below. An activated
    # Hailo network group still open at interpreter teardown segfaults inside
    # libhailort, so anything that can fail -- a missing ROS, a busy camera -- must
    # fail before the device is ever claimed.
    odom = OdomBuffer()
    buf = WorldLidarBuffer(args.lidar_window, args.point_stride, args.filter_tags,
                           odom, t_il)
    shutdown, publish = start_ros(args, buf, odom)
    cam = detector = None
    try:
        cam = lvo.PanoramaSource(args)
        detector = HailoYolo(args.yolo, conf=args.yolo_conf)
    except BaseException as exc:
        # BaseException, not Exception: PanoramaSource and HailoYolo both report a
        # busy device by raising SystemExit, which does NOT derive from Exception.
        # Missing it leaves the ROS executor thread spinning into interpreter
        # teardown, and the process aborts ("terminate called without an active
        # exception") instead of printing why the camera would not open -- burying
        # the one line that says what to fix under a core dump.
        if cam is not None:
            cam.close()
        shutdown()
        if isinstance(exc, SystemExit):
            raise
        raise SystemExit(f"startup failed: {exc}")

    yolo_views = (set(range(len(views))) if args.yolo_all_views
                  else set(args.yolo_view or [0]))
    print(f"yolo       : {args.yolo} {detector.input_size[0]}x{detector.input_size[1]} "
          f"on view(s) {sorted(yolo_views)}")
    print(f"match      : fast (nearest range run, gap {args.gap} m), frustum "
          f"{args.box_shrink:.2f}x horizontal / {args.box_shrink_v:.2f}x vertical, "
          f"anchors along {args.anchor_axis}, output in '{args.world_frame}'")

    canvas_maker = lvo.ViewCanvas(views, args.max_cols,
                                  cv2.INTER_CUBIC if args.cubic else cv2.INTER_LINEAR)
    save_dir = args.save_dir or os.path.join(HERE, 'live_snapshots')
    if args.no_display and not args.snapshot_interval:
        args.snapshot_interval = 2.0

    detections = [[] for _ in views]
    d_world = np.empty((0, 3), np.float32)
    rng = np.empty(0, np.float32)
    near = np.empty(0, bool)
    cam_origin = np.zeros(3, np.float32)
    fps, n_frames, t_fps = 0.0, 0, time.monotonic()
    frame_no, last_snapshot, n_nopose = 0, 0.0, 0
    show_cloud, paused = args.show_cloud, False
    yolo_ms = match_ms = 0.0
    have_pose = False

    # Wait for enough odometry to measure the stamp offset, so the first frames are
    # placed rather than thrown away -- and so a dead /Odometry is reported here
    # instead of as a silent absence of every match.
    deadline = time.time() + 5.0
    while odom.n < 5 and time.time() < deadline:
        time.sleep(0.05)
    if odom.n < 5:
        print(f"  WARNING: only {odom.n} odometry message(s) on {args.odom_topic} in "
              f"5 s. Without a pose nothing can be placed in world coordinates.")
    else:
        print(f"stamp offset: {odom.offset:+.3f} s (message stamps to wall clock, "
              f"median of {len(odom.offsets)}); camera frames are looked up at "
              f"wall - {args.cam_latency:.3f} s - offset")

    print("\nrunning -- q quit, s snapshot, l faint cloud, p pause\n")
    try:
        while True:
            frame = cam.latest()
            if frame is None or paused:
                time.sleep(0.005)
                if not paused:
                    continue
            frame_no += 1

            if not paused:
                # The camera stream carries no stamp, so the frame is placed at
                # wall-clock arrival less the pipeline delay. This is the weakest
                # link in the chain and the reason --cam-latency exists.
                pose = odom.pose_at_wall(time.time() - args.cam_latency,
                                         args.max_pose_age)
                have_pose = pose is not None
                if have_pose:
                    r_wi, p_wi = pose
                    # Camera attitude is a PURE yaw: the stabiliser holds roll and
                    # pitch at zero, so only the body's heading reaches the panorama.
                    r_pano_world = rz(yaw_of(r_wi) + dpsi).T
                    for vw in views:
                        vw.set_extrinsic(r_pano_world.astype(np.float32))
                    cam_origin = (r_wi @ (t_lidar_cam + t_il) + p_wi).astype(np.float32)

                    pts = buf.snapshot()
                    if len(pts):
                        d_world = pts - cam_origin
                        rng = np.linalg.norm(d_world, axis=1)
                        near = (rng >= args.min_depth) & (rng <= args.max_depth)
                    else:
                        d_world, rng, near = (np.empty((0, 3), np.float32),
                                              np.empty(0), np.empty(0, bool))
                else:
                    n_nopose += 1
                    d_world, rng, near = (np.empty((0, 3), np.float32),
                                          np.empty(0), np.empty(0, bool))

            canvas = canvas_maker.render(frame)
            objects = []
            for i, vw in enumerate(views):
                tile = canvas_maker.tile(canvas, vw)
                if i in yolo_views and frame_no % args.yolo_every == 0:
                    t0 = time.perf_counter()
                    detections[i] = detector.infer(tile)
                    yolo_ms = (time.perf_counter() - t0) * 1000.0

                u, v, r = vw.project(d_world, rng, near)
                if show_cloud:
                    lcm.stamp(tile, u, v, (70, 70, 70), 0)

                located = 0
                t0 = time.perf_counter()
                for det in detections[i]:
                    anchor, counts = match_and_draw(tile, vw, det, u, v, r,
                                                    cam_origin, args)
                    if anchor is None:
                        continue
                    located += 1
                    label = detector.label(det[5])
                    objects.append((label, anchor))
                    c_ = anchor['central']
                    cv2.putText(tile, f"{label} {anchor['range']:.1f}m "
                                      f"({c_[0]:+.1f},{c_[1]:+.1f},{c_[2]:+.1f})",
                                (int(det[0]) + 2, max(int(det[1]) - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                                cv2.LINE_AA)
                    if args.do_print:
                        t_, b_ = anchor['top'], anchor['bottom']
                        print(f"  [{i}] {label} {det[4]:.2f}  {counts[0]} in box -> "
                              f"{counts[1]} object  range {anchor['range']:.2f} m | "
                              f"world top {np.round(t_, 2)}  central {np.round(c_, 2)}"
                              f"  bottom {np.round(b_, 2)}")
                match_ms = (time.perf_counter() - t0) * 1000.0

                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
                              (60, 60, 60), 1)
                draw_detections(tile, detections[i], detector)
                lvo.label_view(tile, f"[{i}] {vw.label}",
                               f"{len(detections[i])} det  {located} matched  "
                               f"yolo {yolo_ms:.0f} ms  match {match_ms:.1f} ms  "
                               f"{fps:.1f} fps")
            publish(objects)

            note = [] if have_pose else [("NO POSE -- waiting for odometry", (0, 0, 255))]
            if buf.unplaced:
                note.append((f"{buf.unplaced} scan(s) unplaced", (0, 165, 255)))
            draw_legend(canvas, note)

            n_frames += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                fps, n_frames, t_fps = n_frames / (now - t_fps), 0, now

            snap = bool(args.snapshot_interval) and now - last_snapshot >= args.snapshot_interval
            if not args.no_display:
                cv2.imshow('yolo -> global match', canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('s'):
                    snap = True
                elif key == ord('l'):
                    show_cloud = not show_cloud
                elif key == ord('p'):
                    paused = not paused
            if snap:
                os.makedirs(save_dir, exist_ok=True)
                path = os.path.join(save_dir, time.strftime('global_%Y%m%d_%H%M%S.png'))
                print(f"  {'wrote' if cv2.imwrite(path, canvas) else 'FAILED to write'} {path}")
                last_snapshot = now
    except KeyboardInterrupt:
        pass
    finally:
        if n_nopose:
            print(f"\n{n_nopose} frame(s) had no usable pose within "
                  f"{args.max_pose_age}s -- is {args.odom_topic} publishing?")
        if cam is not None:
            cam.close()
        if detector is not None:
            detector.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        shutdown()


if __name__ == '__main__':
    main()
