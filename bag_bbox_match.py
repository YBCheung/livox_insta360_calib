#!/usr/bin/env python3
"""global_bbox_match.py for a laptop: a ros2 bag in, an Ultralytics .pt for the detector.

Same pose chain, same matcher, same output as global_bbox_match.py -- read that file
first; the reasoning behind the world-frame camera pose and the 'fast' match lives
there and is not repeated. What changes is only where the four inputs come from:

    detector    Ultralytics .pt on CPU or CUDA, instead of a HEF on a Hailo-8L
    panorama    /insta360/image_raw/compressed, instead of cv2.VideoCapture
    cloud       /livox/lidar out of the bag, parsed from its CDR bytes
    pose        /Odometry out of the bag

The bag is read DIRECTLY with rosbag2_py -- do not `ros2 bag play` alongside this.
Reading it rather than replaying it is what buys the three things below, none of
which the live tool can have:

  every pose is interpolated, never extrapolated
        All of /Odometry is loaded in a first pass before a single image is touched,
        so a frame's pose is always bracketed by two real samples. Live, odometry for
        an instant arrives after it, and the newest frame can only extrapolate.
  the LiDAR window is CENTRED on the frame
        Offline there is lookahead: an image is held back until scans past its stamp
        have been read, so it matches against [t - w/2, t + w/2] instead of the
        trailing [t - w, t] a live loop is stuck with. Same point budget, half the
        mean temporal offset. --trailing-window restores the live behaviour when the
        point is to reproduce it.
  stamps, not arrival times
        Every message here carries its own stamp and the panorama's is taken in the
        grab thread (insta360_ros_publisher.py stamps at capture, not at publish), so
        association is stamp-to-stamp throughout. The live tool has to place camera
        frames by wall-clock arrival because a UVC stream has no stamp at all, and
        needs a measured clock offset to do it; none of that machinery is used here.

WHICH CLOUD TOPIC
-----------------
/livox/lidar, not /cloud_registered -- even though the latter is already in the world
frame and would need no pose at all. Measured on kuusamo/manual_l2:

    /livox/lidar       19,968 points per message (median)
    /cloud_registered      675 points per message (median)

/cloud_registered is feats_down_body, voxel-filtered at mapping.filter_size_surf
(0.5 m), so a person at 5 m survives as a handful of points -- below --min-points
before the matcher even starts. It becomes usable only with dense_publish_en: true
in the FAST-LIO config, which would publish feats_undistort instead.

The CDR bytes rosbag2 hands back are byte-identical to what a raw subscription
delivers, so LidarBuffer's numpy parser reads them directly: 1.6 ms per message
against 134 ms for deserialize_message, i.e. ~3 s versus ~5 minutes over this bag.
It also means livox_ros_driver2 does NOT have to be built on the laptop -- the
parser never needs the message class, only the fallback would.

    python3 bag_bbox_match.py --bag ~/kuusamo/manual_l2 --view 240 --yolo yolov8s.pt
    python3 bag_bbox_match.py --bag ~/kuusamo/manual_l2 --view 240 --yolo yolov8s.pt \
        --device cuda --start 30 --duration 20 --csv objects.csv

Keys: q / Esc quit, space pause, s snapshot, l toggle the faint full cloud.
"""

import argparse
import bisect
import csv
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_view_overlay as lvo
import live_cluster_match as lcm
import global_bbox_match as gbm

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------- detector

class UltralyticsYolo:
    """A .pt through Ultralytics, wearing HailoYolo's interface.

    Same five members hailo_yolo.HailoYolo exposes -- input_size, label(), infer(),
    close() and the (x1, y1, x2, y2, score, cls) tuple shape -- so every drawing and
    matching call downstream is shared with the on-device tool rather than
    reimplemented. draw_detections() from hailo_yolo works on this unchanged.
    """

    def __init__(self, weights, conf=0.25, device='cpu', imgsz=640, classes=None):
        from ultralytics import YOLO
        self.conf = conf
        self.device = device
        self.imgsz = imgsz
        self.classes = classes
        self.model = YOLO(weights)
        # Ultralytics keeps names as {id: name}; HailoYolo.label() expects a .get().
        self.names = dict(self.model.names)

    @property
    def input_size(self):
        return (self.imgsz, self.imgsz)

    def label(self, cls_id):
        return self.names.get(int(cls_id), f"class {cls_id}")

    def infer(self, image):
        """BGR image -> [(x1, y1, x2, y2, score, cls), ...] in that image's pixels.

        Ultralytics takes a BGR ndarray directly and returns boxes already in the
        input image's coordinates, so unlike the HEF path there is no normalised
        0..1 box to scale back and no y-first ordering to get wrong.
        """
        res = self.model.predict(image, conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, classes=self.classes,
                                 verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        return [(float(x1), float(y1), float(x2), float(y2), float(s), int(c))
                for (x1, y1, x2, y2), s, c in zip(xyxy, scores, clses)]

    def close(self):
        pass


# ------------------------------------------------------------------- bag clouds

class BagCloudWindow(lvo.LidarBuffer):
    """Scans from the bag, placed in world coordinates and indexed by stamp.

    Subclasses LidarBuffer purely to inherit its CDR parser and the point hygiene
    that follows it (--point-stride, the tag rejection, the finite/non-zero mask):
    the bytes out of rosbag2 are exactly what add_custom_raw expects. Only storage
    changes -- the base class windows on time.monotonic(), which is meaningless when
    messages are read as fast as the disk allows rather than arriving at 10 Hz.

    Placement is eager here, not lazy as in WorldLidarBuffer, and can afford to be:
    every pose is already loaded, so a scan is never seen before its odometry.
    """

    def __init__(self, odom, t_il, stride, filter_tags):
        super().__init__(window_s=float('inf'), stride=stride, filter_tags=filter_tags)
        self.odom = odom
        self.t_il = np.asarray(t_il, dtype=np.float32)
        self._stamp = None
        self.stamps = []            # ascending: bag order is stamp order for one topic
        self.clouds = []
        self.unplaced = 0

    @staticmethod
    def stamp_of(cdr):
        """sec/nanosec sit right after the 4-byte encapsulation header."""
        if len(cdr) < 12:
            return None
        return (int(np.frombuffer(cdr, np.int32, 1, 4)[0])
                + int(np.frombuffer(cdr, np.uint32, 1, 8)[0]) * 1e-9)

    def feed(self, cdr):
        """One /livox/lidar message as stored. Returns its stamp, or None."""
        self._stamp = self.stamp_of(cdr)
        if self._stamp is None:
            return None
        self.add_custom_raw(cdr)
        return self._stamp

    def add(self, xyz):
        """Called by the inherited _store_custom once the points are clean."""
        if self._stamp is None or not len(xyz):
            return
        pose = self.odom.pose_at(self._stamp)
        if pose is None:
            self.unplaced += 1
            return
        r_wi, p_wi = pose
        world = (xyz + self.t_il) @ r_wi.T.astype(np.float32) + p_wi.astype(np.float32)
        self.stamps.append(self._stamp)
        self.clouds.append(np.ascontiguousarray(world, dtype=np.float32))

    def window(self, t0, t1):
        """Every world point stamped in [t0, t1]."""
        lo = bisect.bisect_left(self.stamps, t0)
        hi = bisect.bisect_right(self.stamps, t1)
        if hi <= lo:
            return np.empty((0, 3), np.float32)
        return np.concatenate(self.clouds[lo:hi], axis=0)

    def drop_before(self, t):
        """Release scans no future frame can ask for -- 20k points a scan at 10 Hz is
        ~500 MB over this bag if nothing is ever freed."""
        k = bisect.bisect_left(self.stamps, t)
        if k:
            del self.stamps[:k]
            del self.clouds[:k]

    @property
    def newest(self):
        return self.stamps[-1] if self.stamps else None


# ------------------------------------------------------------------- bag reading

def open_bag(path, storage_id=''):
    import rosbag2_py
    if not os.path.exists(path):
        raise SystemExit(f"no such bag: {path}")
    if not storage_id:
        storage_id = 'mcap' if any(f.endswith('.mcap') for f in os.listdir(path)
                                   if os.path.isfile(os.path.join(path, f))) else 'sqlite3'
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id=storage_id),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    return reader, {t.name: t.type for t in reader.get_all_topics_and_types()}


def load_odometry(path, storage_id, topic):
    """Pass one: every pose, before any image is looked at.

    This is the whole reason to read the bag rather than replay it -- with the full
    trajectory in hand, no frame ever has to extrapolate a pose.
    """
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Odometry
    reader, types = open_bag(path, storage_id)
    if topic not in types:
        raise SystemExit(f"{path}: no {topic}. Present: {', '.join(sorted(types))}")
    odom = gbm.OdomBuffer(window_s=float('inf'))
    import rosbag2_py
    f = rosbag2_py.StorageFilter()
    f.topics = [topic]
    reader.set_filter(f)
    while reader.has_next():
        _, data, _ = reader.read_next()
        odom.add(deserialize_message(data, Odometry))
    if odom.n < 2:
        raise SystemExit(f"{topic} holds {odom.n} message(s) -- nothing can be placed.")
    span = odom.stamps[-1] - odom.stamps[0]
    print(f"odometry   : {odom.n} poses over {span:.1f} s from {topic}")
    return odom


# ------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    b = p.add_argument_group('bag')
    b.add_argument('--bag', required=True, help='directory holding the .db3/.mcap + metadata.yaml')
    b.add_argument('--storage-id', default='', help="'sqlite3' or 'mcap'; guessed if omitted")
    b.add_argument('--image-topic', default='/insta360/image_raw/compressed')
    b.add_argument('--lidar-topic', default='/livox/lidar')
    b.add_argument('--odom-topic', default='/Odometry')
    b.add_argument('--start', type=float, default=0.0, metavar='S',
                   help='skip this many seconds from the start of the bag')
    b.add_argument('--duration', type=float, default=0.0, metavar='S',
                   help='stop after this many seconds of bag time (0 = to the end)')
    b.add_argument('--every', type=int, default=1, metavar='N',
                   help='process every Nth panorama frame')
    b.add_argument('--rate', type=float, default=0.0, metavar='X',
                   help='sleep to approximate X times real time (0 = as fast as possible)')

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
                   metavar=('X', 'Y', 'Z'), help='camera ORIGIN in LiDAR coords, metres')
    e.add_argument('--yaw-offset', type=float, default=None, metavar='DEG',
                   help='dpsi; derived from --extrinsic + --gravity if omitted. Those '
                        'files describe the rig they were captured on -- override this '
                        'if the bag came from a different mounting')
    e.add_argument('--extrinsic', default=gbm.DEFAULT_EXTRINSIC)
    e.add_argument('--gravity', default=gbm.DEFAULT_GRAVITY)
    e.add_argument('--imu-lidar-t', type=float, nargs=3,
                   default=[-0.011, -0.02329, 0.04412], metavar=('X', 'Y', 'Z'),
                   help='FAST-LIO mapping.extrinsic_T (LiDAR origin in IMU frame)')
    e.add_argument('--cam-latency', type=float, default=0.0, metavar='S',
                   help='subtracted from the panorama stamp before the pose lookup. '
                        'Near zero here: the publisher stamps in the grab thread, so '
                        'only the USB + decode delay before that remains (default: '
                        '%(default)s)')

    l = p.add_argument_group('cloud')
    l.add_argument('--window', type=float, default=0.4, metavar='S',
                   help='seconds of scans per frame, CENTRED on the frame stamp '
                        '(default: %(default)s)')
    l.add_argument('--trailing-window', action='store_true',
                   help='use [t-w, t] as the live tool must, instead of centring')
    l.add_argument('--point-stride', type=int, default=1)
    l.add_argument('--no-filter-tags', dest='filter_tags', action='store_false')
    l.add_argument('--min-depth', type=float, default=0.3)
    l.add_argument('--max-depth', type=float, default=30.0)

    y = p.add_argument_group('detector and match')
    y.add_argument('--yolo', default='yolov8s.pt', metavar='PT',
                   help='Ultralytics weights (default: %(default)s)')
    y.add_argument('--device', default='cpu', help="'cpu', 'cuda', 'cuda:0', '0' ...")
    y.add_argument('--imgsz', type=int, default=640)
    y.add_argument('--yolo-conf', type=float, default=0.25)
    y.add_argument('--classes', type=int, nargs='+', default=None,
                   help='restrict to these class ids (e.g. 0 for person)')
    y.add_argument('--yolo-view', type=int, action='append', metavar='N')
    y.add_argument('--yolo-all-views', action='store_true')
    y.add_argument('--box-shrink', type=float, default=0.6,
                   help='HORIZONTAL fraction of the box used as the frustum')
    y.add_argument('--box-shrink-v', type=float, default=1.0,
                   help='VERTICAL fraction; 1.0 because the box height is what '
                        'carries top and bottom')
    y.add_argument('--gap', type=float, default=0.5, metavar='M')
    y.add_argument('--min-points', type=int, default=5)
    y.add_argument('--anchor-axis', choices=('row', 'z'), default='z')

    d = p.add_argument_group('output')
    d.add_argument('--csv', help='write one row per matched detection')
    d.add_argument('--video', metavar='PATH',
                   help='write the annotated canvas to a video. .mp4 uses mp4v, .avi '
                        'uses MJPG (OpenCV here has no H.264 encoder)')
    d.add_argument('--video-fps', type=float, default=0.0, metavar='F',
                   help='0 = derive from the frame stamps so the video plays at 1x '
                        'real time (default: %(default)s)')
    d.add_argument('--point-radius', type=int, default=1)
    d.add_argument('--show-cloud', action='store_true')
    d.add_argument('--print', dest='do_print', action='store_true')
    d.add_argument('--no-display', action='store_true')
    d.add_argument('--save-dir')
    d.add_argument('--snapshot-every', type=int, default=0, metavar='N',
                   help='write a PNG every Nth processed frame')
    args = p.parse_args()

    views = lvo.parse_views(args)
    if args.yaw_offset is None:
        args.yaw_offset, tilt = gbm.derive_yaw_offset(args.extrinsic, args.gravity)
        print(f"yaw offset : {args.yaw_offset:+.2f} deg, derived from "
              f"{os.path.basename(args.extrinsic)} + {os.path.basename(args.gravity)}"
              f"  (residual camera tilt {tilt:.2f} deg)")
    else:
        print(f"yaw offset : {args.yaw_offset:+.2f} deg (given)")
    dpsi = math.radians(args.yaw_offset)
    t_lidar_cam = np.asarray(args.translation, dtype=float)
    t_il = np.asarray(args.imu_lidar_t, dtype=float)
    print("views      : " + "; ".join(f"[{i}] {vw.label}" for i, vw in enumerate(views)))

    odom = load_odometry(args.bag, args.storage_id, args.odom_topic)
    cloud = BagCloudWindow(odom, t_il, args.point_stride, args.filter_tags)

    reader, types = open_bag(args.bag, args.storage_id)
    for t in (args.image_topic, args.lidar_topic):
        if t not in types:
            raise SystemExit(f"{args.bag}: no {t}. Present: {', '.join(sorted(types))}")
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    f = rosbag2_py.StorageFilter()
    f.topics = [args.image_topic, args.lidar_topic]
    reader.set_filter(f)

    detector = UltralyticsYolo(args.yolo, conf=args.yolo_conf, device=args.device,
                               imgsz=args.imgsz, classes=args.classes)
    from hailo_yolo import draw_detections
    yolo_views = (set(range(len(views))) if args.yolo_all_views
                  else set(args.yolo_view or [0]))
    print(f"yolo       : {args.yolo} @{args.imgsz} on {args.device}, "
          f"view(s) {sorted(yolo_views)}")
    half = args.window if args.trailing_window else args.window * 0.5
    print(f"match      : fast (gap {args.gap} m), frustum {args.box_shrink:.2f}x h / "
          f"{args.box_shrink_v:.2f}x v, "
          f"{'trailing' if args.trailing_window else 'centred'} {args.window:.2f}s window")

    canvas_maker = lvo.ViewCanvas(views, args.max_cols, cv2.INTER_LINEAR)
    save_dir = args.save_dir or os.path.join(HERE, 'bag_snapshots')
    writer = fh = None
    if args.csv:
        fh = open(args.csv, 'w', newline='')
        writer = csv.writer(fh)
        writer.writerow(['stamp', 'view', 'cls', 'label', 'conf', 'n_box', 'n_obj',
                         'range_m', 'top_x', 'top_y', 'top_z', 'cx', 'cy', 'cz',
                         'bot_x', 'bot_y', 'bot_z', 'cam_x', 'cam_y', 'cam_z'])

    t_bag0 = odom.stamps[0]
    t_from = t_bag0 + args.start
    t_to = (t_from + args.duration) if args.duration > 0 else float('inf')

    # The writer cannot open until the canvas size is known, and --video-fps 0 also
    # needs a few stamps to work out the real frame interval, so the opening is
    # deferred and the first frames are buffered until both are settled.
    vid = {'w': None, 'buf': [], 'stamps': [], 'n': 0}

    def video_write(canvas, stamp):
        if vid['w'] is not None:
            vid['w'].write(canvas)
            vid['n'] += 1
            return
        vid['buf'].append(canvas.copy())
        vid['stamps'].append(stamp)
        if args.video_fps <= 0 and len(vid['buf']) < 5:
            return                       # still measuring the interval
        fps = args.video_fps
        if fps <= 0:
            d_ = np.diff(vid['stamps'])
            fps = 1.0 / float(np.median(d_)) if len(d_) and np.median(d_) > 0 else 10.0
        fps = float(np.clip(fps, 1.0, 60.0))
        h, w = canvas.shape[:2]
        fourcc = 'MJPG' if args.video.lower().endswith('.avi') else 'mp4v'
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if not writer.isOpened():
            raise SystemExit(f"cannot open {args.video} for writing "
                             f"(fourcc {fourcc}, {w}x{h} @ {fps:.2f} fps)")
        print(f"video      : {args.video} {w}x{h} @ {fps:.2f} fps ({fourcc})")
        vid['w'] = writer
        for c in vid['buf']:
            writer.write(c)
            vid['n'] += 1
        vid['buf'] = []

    pending = deque()          # images held back until the cloud has caught up
    n_img = n_done = n_obj = n_det = n_nopose = n_baddecode = 0
    show_cloud, paused, quit_now = args.show_cloud, False, False
    t_wall0 = time.monotonic()

    def process(stamp, jpeg):
        """One panorama: place the camera, project, detect, match, draw."""
        nonlocal n_done, n_obj, n_det, n_nopose, n_baddecode, show_cloud, paused, quit_now
        pose = odom.pose_at(stamp - args.cam_latency)
        if pose is None:
            n_nopose += 1
            return
        r_wi, p_wi = pose
        r_pano_world = gbm.rz(gbm.yaw_of(r_wi) + dpsi).T.astype(np.float32)
        for vw in views:
            vw.set_extrinsic(r_pano_world)
        cam_origin = (r_wi @ (t_lidar_cam + t_il) + p_wi).astype(np.float32)

        lo = stamp - args.window if args.trailing_window else stamp - half
        pts = cloud.window(lo, stamp if args.trailing_window else stamp + half)
        if len(pts):
            d_world = pts - cam_origin
            rng = np.linalg.norm(d_world, axis=1)
            near = (rng >= args.min_depth) & (rng <= args.max_depth)
        else:
            d_world, rng, near = (np.empty((0, 3), np.float32), np.empty(0),
                                  np.empty(0, bool))

        frame = cv2.imdecode(np.asarray(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            # Counted, never silent: a decode that fails on every frame otherwise
            # reports as "0 processed, 0 without a pose, 0 objects" -- which reads
            # like an empty bag rather than a broken image path.
            n_baddecode += 1
            return
        canvas = canvas_maker.render(frame)
        n_done += 1
        for i, vw in enumerate(views):
            tile = canvas_maker.tile(canvas, vw)
            dets = detector.infer(tile) if i in yolo_views else []
            n_det += len(dets)
            u, v_, r = vw.project(d_world, rng, near)
            if show_cloud:
                lcm.stamp(tile, u, v_, (70, 70, 70), 0)
            located = 0
            for det in dets:
                anchor, counts = gbm.match_and_draw(tile, vw, det, u, v_, r,
                                                    cam_origin, args)
                if anchor is None:
                    continue
                located += 1
                n_obj += 1
                label = detector.label(det[5])
                t_, c_, b_ = anchor['top'], anchor['central'], anchor['bottom']
                cv2.putText(tile, f"{label} {anchor['range']:.1f}m "
                                  f"({c_[0]:+.1f},{c_[1]:+.1f},{c_[2]:+.1f})",
                            (int(det[0]) + 2, max(int(det[1]) - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                            cv2.LINE_AA)
                if args.do_print:
                    print(f"  t{stamp - t_bag0:7.2f} [{i}] {label} {det[4]:.2f}  "
                          f"{counts[0]}->{counts[1]} pts  range {anchor['range']:.2f} m"
                          f" | top {np.round(t_, 2)} central {np.round(c_, 2)} "
                          f"bottom {np.round(b_, 2)}")
                if writer:
                    writer.writerow([f"{stamp:.6f}", i, det[5], label, f"{det[4]:.4f}",
                                     counts[0], counts[1], f"{anchor['range']:.4f}",
                                     *[f"{q:.4f}" for q in (*t_, *c_, *b_, *cam_origin)]])
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
                          (60, 60, 60), 1)
            draw_detections(tile, dets, detector)
            lvo.label_view(tile, f"[{i}] {vw.label}",
                           f"t{stamp - t_bag0:.2f}s  {len(dets)} det  "
                           f"{located} matched  {len(pts)} pts")
        gbm.draw_legend(canvas)

        if args.video:
            video_write(canvas, stamp)
        if args.snapshot_every and n_done % args.snapshot_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"bag_{stamp - t_bag0:08.2f}.png")
            print(f"  {'wrote' if cv2.imwrite(path, canvas) else 'FAILED'} {path}")
        if not args.no_display:
            cv2.imshow('bag -> global match', canvas)
            while True:
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key in (ord('q'), 27):
                    quit_now = True
                elif key == ord(' '):
                    paused = not paused
                    continue
                elif key == ord('l'):
                    show_cloud = not show_cloud
                elif key == ord('s'):
                    os.makedirs(save_dir, exist_ok=True)
                    path = os.path.join(save_dir, f"bag_{stamp - t_bag0:08.2f}.png")
                    print(f"  wrote {path}")
                    cv2.imwrite(path, canvas)
                break
        if args.rate > 0:
            target = (stamp - t_from) / args.rate
            lag = target - (time.monotonic() - t_wall0)
            if lag > 0:
                time.sleep(min(lag, 1.0))

    print(f"\nreading {args.bag} ...\n")
    try:
        while reader.has_next():
            topic, data, _ = reader.read_next()
            if topic == args.lidar_topic:
                cloud.feed(data)
            else:
                stamp = BagCloudWindow.stamp_of(data)
                if stamp is None or stamp < t_from or stamp > t_to:
                    if stamp is not None and stamp > t_to:
                        break
                    continue
                n_img += 1
                if (n_img - 1) % args.every:
                    continue
                # The bag stores the CDR-serialised CompressedImage, not the JPEG:
                # the payload sits past the header and two length-prefixed strings.
                # Handing the raw CDR to cv2.imdecode returns None, silently.
                pending.append((stamp, deserialize_message(data, CompressedImage).data))

            # An image is only ready once scans past its centred window have been
            # read; until then the cloud behind it is still arriving.
            newest = cloud.newest
            while pending and newest is not None and pending[0][0] + half <= newest:
                st, jpeg = pending.popleft()
                process(st, jpeg)
                cloud.drop_before(st - args.window)
                if quit_now:
                    raise KeyboardInterrupt
        for st, jpeg in pending:               # tail of the bag: no lookahead left
            process(st, jpeg)
            if quit_now:
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        detector.close()
        if args.video:
            if vid['w'] is None and vid['buf']:
                # Fewer frames than the interval probe needs: open at a fallback rate
                # rather than silently writing nothing.
                args.video_fps = args.video_fps or 10.0
                buf, stamps = vid['buf'], vid['stamps']
                vid['buf'], vid['stamps'] = [], []
                for c, st in zip(buf, stamps):
                    video_write(c, st)
            if vid['w'] is not None:
                vid['w'].release()
                print(f"wrote {args.video} ({vid['n']} frames)")
        if fh is not None:
            fh.close()
            print(f"wrote {args.csv}")
        if not args.no_display:
            cv2.destroyAllWindows()

    dt = time.monotonic() - t_wall0
    print(f"\n{n_done} frame(s) processed, {n_det} detection(s), "
          f"{n_obj} object(s) located, "
          f"{cloud.unplaced} scan(s) unplaced, {n_nopose} frame(s) without a pose"
          + (f", {n_baddecode} frame(s) FAILED TO DECODE" if n_baddecode else ""))
    print(f"{dt:.1f} s wall, {n_done / max(dt, 1e-6):.2f} frames/s")


if __name__ == '__main__':
    main()
