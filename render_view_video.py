#!/usr/bin/env python3
"""One virtual view out of a bag, with the LiDAR cloud painted on it, as a video.

bag_bbox_match.py without the detector. Same bag, same pose chain, same projection
-- what is dropped is YOLO, the matcher and the CSV, because none of them are needed
to answer the only question this tool asks: does the cloud sit where the image says
the world is, for a whole flight rather than one frame.

Two things differ from the tools next to it, both forced by where this runs:

  no ROS 2                 the .db3 is read straight through sqlite3 and the three
                           message types are parsed from their CDR bytes here, so
                           this works on a machine with no ROS install at all.
                           rosbag2's sqlite3 schema is two tables (topics, messages)
                           and rosbag2_py adds nothing on top of them for a plain
                           sequential read.
  H.264 through ffmpeg     cv2.VideoWriter's mp4v is not what a browser or Keynote
                           wants; frames are piped to ffmpeg as raw BGR instead, so
                           nothing is staged on disk (render_presentation.py writes
                           a PNG per frame -- 1651 of them here).

    python3 render_view_video.py --bag data/kuusamo/manual_l2 \
        --view 270 -20 --fov 60 --out manual_l2_yaw270.mp4

WHICH CLOUD
-----------
--cloud scan (default) is /livox/lidar: one 0.4 s window of raw scans, each placed by
its own odometry pose. NOT /cloud_registered, which is feats_down_body voxel-filtered
at 0.5 m -- 675 points a message against 19,968 (see bag_bbox_match.py's header).

--cloud map is /Laser_map, the accumulated world map, drawn only up to the frame's own
stamp so nothing appears before it was scanned. It fills the wedge no single scan can
reach: the MID-360's cone stops 7 deg below its own horizontal, which the airframe's
forward pitch carries down to about -30 deg of WORLD elevation in level flight -- the
measured floor of a 0.4 s window on this bag is -29 to -34 deg, while the map reaches
-87 deg, i.e. straight down, because the drone swept that ground from somewhere else
in the flight. Density follows the same story and takes time to arrive: in frame at
yaw 270 / pitch -20, the map holds 0.8x the scan's points at t=41 s, 4.6x at t=105 s
and 2.5x at t=170 s. Before roughly a minute of flight it is the thinner cloud, not
the denser one. What it costs is stated in
global_bbox_match.py's header and is real here too: the map holds every surface ever
seen, including ones now hidden behind a tree, and it carries FAST-LIO2's drift (no
loop closure) rather than cancelling it the way a same-instant scan does. Good for
looking at coverage; --cloud scan remains the honest one for judging the extrinsic.

--cloud both draws each view twice, side by side, which is the only way to compare
them on identical geometry.

DETECTIONS
----------
--yolo <weights.pt> runs the detector on every view and matches each box against the
cloud with bag_bbox_match.py's own matcher -- gbm.match_and_draw, imported rather
than reimplemented, so the boxes in this video and the rows in that tool's CSV come
from identical code. Rendering follows live_cluster_match.py:

    grey    inside the box, before anything was rejected
    green   the nearest range run -- the matched tree, and the only points the
            keypoints are computed from
    C/B/T   its centre, bottom and top, drawn where they reproject

The match runs against WHICHEVER cloud that tile draws, so what you see is what the
matcher saw. Against --cloud map that is a real difference, not a detail: the map
holds surfaces now hidden behind the tree in front, and they land inside the box too.
select_nearest_run takes the nearest run with support, which is the right instinct
here, but a box whose tree is thinner than its background is where it will fail.

The cloud is painted with the same turbo depth ramp live_view_overlay.py uses, so a
frame out of this video and a frame off the live tool are the same picture.
"""

import argparse
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import deque
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bag_bbox_match as bbm
import global_bbox_match as gbm
import live_view_overlay as lvo

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------- CDR reading
#
# Every offset below is absolute, but CDR alignment is relative to the 4-byte
# encapsulation header that starts the buffer -- hence the (o - 4) in _align. Getting
# that wrong shifts a string read by up to three bytes and yields a plausible-looking
# garbage length, so it is worth spelling out rather than inlining.

def _align(o, n):
    return 4 + ((o - 4 + n - 1) & ~(n - 1))


def _read_string(buf, o):
    o = _align(o, 4)
    n = int(np.frombuffer(buf, np.uint32, 1, o)[0])
    o += 4
    return buf[o:o + n - 1].decode('utf-8', 'replace'), o + n   # n counts the NUL


def _read_header(buf):
    """std_msgs/Header -> (stamp, frame_id, offset just past it)."""
    sec = int(np.frombuffer(buf, np.int32, 1, 4)[0])
    nsec = int(np.frombuffer(buf, np.uint32, 1, 8)[0])
    frame_id, o = _read_string(buf, 12)
    return sec + nsec * 1e-9, frame_id, o


def parse_compressed_image(buf):
    """CDR of a sensor_msgs/CompressedImage -> the JPEG bytes alone.

    The payload sits past the header and the `format` string; handing the whole CDR
    to cv2.imdecode returns None, silently.
    """
    _, _, o = _read_header(buf)
    _, o = _read_string(buf, o)                  # format ('jpeg')
    o = _align(o, 4)
    n = int(np.frombuffer(buf, np.uint32, 1, o)[0])
    return np.frombuffer(buf, np.uint8, n, o + 4)


def parse_odometry(buf):
    """CDR of a nav_msgs/Odometry -> the shim OdomBuffer.add expects.

    Only stamp, position and orientation are read; the covariances and the twist that
    follow them are never looked at, so there is nothing to parse past the quaternion.
    """
    stamp, _, o = _read_header(buf)
    _, o = _read_string(buf, o)                  # child_frame_id
    v = np.frombuffer(buf, np.float64, 7, _align(o, 8))   # align 8 for the doubles
    sec = int(stamp)
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(
            sec=sec, nanosec=int(round((stamp - sec) * 1e9)))),
        pose=SimpleNamespace(pose=SimpleNamespace(
            position=SimpleNamespace(x=v[0], y=v[1], z=v[2]),
            orientation=SimpleNamespace(x=v[3], y=v[4], z=v[5], w=v[6]))))


def parse_pointcloud2(buf):
    """CDR of a sensor_msgs/PointCloud2 -> its xyz as (N, 3) float32.

    The field offsets are read rather than assumed: /Laser_map carries FAST-LIO's
    PointXYZINormal, so x/y/z sit inside a 48-byte stride next to a normal and a
    curvature, and hard-coding a 16-byte XYZI stride would read normals as positions.
    """
    _, _, o = _read_header(buf)
    o = _align(o, 4) + 8                         # past height, width
    n_fields = int(np.frombuffer(buf, np.uint32, 1, o)[0])
    o += 4
    offsets = {}
    for _ in range(n_fields):
        name, o = _read_string(buf, o)
        o = _align(o, 4)
        offsets[name] = int(np.frombuffer(buf, np.uint32, 1, o)[0])
        o += 4 + 1                               # offset, datatype
        o = _align(o, 4) + 4                     # count
    o = _align(o + 1, 4) + 4                     # is_bigendian, point_step
    step = int(np.frombuffer(buf, np.uint32, 1, o - 4)[0])
    o += 4                                       # row_step
    n = int(np.frombuffer(buf, np.uint32, 1, o)[0])
    o += 4
    raw = np.frombuffer(buf, np.uint8, n, o).reshape(-1, step)
    return np.stack([raw[:, offsets[c]:offsets[c] + 4].copy().view(np.float32).ravel()
                     for c in ('x', 'y', 'z')], axis=1)


class LaserMap:
    """The accumulated /Laser_map, indexed by the stamp each point first appeared at.

    laserMapping.cpp's publish_map does `*pcl_wait_pub += *laserCloudWorld` and never
    clears it, so message k is a strict PREFIX of message k+1: the LAST message is the
    whole map, and the point at index i was first published by the earliest message
    wider than i. That needs only every message's WIDTH and stamp -- never a
    point-by-point comparison -- so exactly one message body is ever read, out of the
    212 that would otherwise total ~1 GB. extract_bag_layers.py documents the same
    property and checks it byte-wise; the monotone widths are re-checked here.

    Birth times come out non-decreasing in index by construction, so `upto` is a
    prefix slice and no per-frame mask over 177k points is needed.
    """

    def __init__(self, conn, tid, topic):
        widths, stamps = [], []
        for length, head in conn.execute(
                'SELECT length(data), substr(data, 1, 32) FROM messages '
                'WHERE topic_id = ? ORDER BY timestamp', (tid,)):
            head = bytes(head)
            if len(head) < 32:
                continue
            stamps.append(int(np.frombuffer(head, np.int32, 1, 4)[0])
                          + int(np.frombuffer(head, np.uint32, 1, 8)[0]) * 1e-9)
            # height, width sit right after the header's frame_id ('world', so the
            # offset is fixed); width is the point count for an unordered cloud.
            widths.append(int(np.frombuffer(head, np.uint32, 1, 28)[0]))
        if not widths:
            raise SystemExit(f"{topic} holds no messages -- nothing to accumulate.")
        widths = np.asarray(widths)
        if np.any(np.diff(widths) < 0):
            raise SystemExit(f"{topic} shrinks between messages -- it is not the "
                             "append-only accumulation this assumes.")

        (data,) = conn.execute('SELECT data FROM messages WHERE topic_id = ? '
                               'ORDER BY timestamp DESC LIMIT 1', (tid,)).fetchone()
        self.xyz = np.ascontiguousarray(parse_pointcloud2(bytes(data)), dtype=np.float32)
        # searchsorted on the widths: index i was born in the first message wider
        # than i, i.e. at widths.searchsorted(i, 'right') -- one vectorised lookup.
        born = np.searchsorted(widths, np.arange(len(self.xyz)), side='right')
        self.birth = np.asarray(stamps)[np.minimum(born, len(stamps) - 1)]
        print(f"laser map  : {len(self.xyz)} points from {len(widths)} {topic} "
              f"messages, born over {self.birth[-1] - self.birth[0]:.1f} s")

    def upto(self, stamp):
        """Every point already scanned at `stamp` -- no lookahead into the future."""
        return self.xyz[:int(np.searchsorted(self.birth, stamp, side='right'))]


def open_bag(path):
    """(sqlite3 connection, {topic name: type}) for a rosbag2 sqlite3 bag directory."""
    if os.path.isdir(path):
        dbs = sorted(f for f in os.listdir(path) if f.endswith('.db3'))
        if not dbs:
            raise SystemExit(f"{path}: no .db3 inside. Only sqlite3 bags are read here.")
        if len(dbs) > 1:
            raise SystemExit(f"{path}: {len(dbs)} .db3 files -- split bags are not read here.")
        path = os.path.join(path, dbs[0])
    elif not os.path.exists(path):
        raise SystemExit(f"no such bag: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    return conn, {n: t for n, t in conn.execute('SELECT name, type FROM topics')}


def topic_id(conn, types, topic, bag):
    if topic not in types:
        raise SystemExit(f"{bag}: no {topic}. Present: {', '.join(sorted(types))}")
    return conn.execute('SELECT id FROM topics WHERE name = ?', (topic,)).fetchone()[0]


def load_odometry(conn, tid, topic):
    """Pass one: every pose, before any image is looked at -- so no frame extrapolates."""
    odom = gbm.OdomBuffer(window_s=float('inf'))
    for (data,) in conn.execute('SELECT data FROM messages WHERE topic_id = ? '
                                'ORDER BY timestamp', (tid,)):
        odom.add(parse_odometry(bytes(data)))
    if odom.n < 2:
        raise SystemExit(f"{topic} holds {odom.n} message(s) -- nothing can be placed.")
    print(f"odometry   : {odom.n} poses over {odom.stamps[-1] - odom.stamps[0]:.1f} s "
          f"from {topic}")
    return odom


# ----------------------------------------------------------------------- drawing

def draw_colorbar(img, dmin, dmax, width=18, margin=12):
    """The depth ramp itself, down the right edge -- the picture is meaningless
    without it, since every colour in the cloud is a distance."""
    h = img.shape[0]
    top, bot = margin + 54, h - margin - 18
    if bot - top < 40:
        return
    ramp = lvo.depth_colors(np.linspace(dmax, dmin, bot - top), dmin, dmax)
    x = img.shape[1] - margin - width
    img[top:bot, x:x + width] = ramp[:, None, :]
    cv2.rectangle(img, (x - 1, top - 1), (x + width, bot), (40, 40, 40), 1)
    for y, text in ((top, f"{dmax:.0f} m"), (bot, f"{dmin:.1f} m")):
        cv2.putText(img, text, (x - 46, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x - 46, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (235, 235, 235), 1, cv2.LINE_AA)


def draw_progress(img, frac, margin=12):
    """A one-pixel-tall clock. A 3.5-minute flight of trees looks much the same at
    0:30 and at 2:30, so without this there is no way to tell the video is advancing."""
    w = img.shape[1]
    y = img.shape[0] - margin
    cv2.line(img, (margin, y), (w - margin, y), (55, 60, 68), 3, cv2.LINE_AA)
    cv2.line(img, (margin, y), (margin + int((w - 2 * margin) * frac), y),
             (235, 200, 120), 3, cv2.LINE_AA)


# -------------------------------------------------------------------------- video

class H264Writer:
    """Raw BGR frames straight into ffmpeg -- no intermediate files, H.264 out.

    Where there is no ffmpeg the frames go to a raw BGR file instead and the one
    command that finishes them is printed. That case is not hypothetical: the
    livox-360-yolo-ego image carries torch and ultralytics but no encoder, so the
    --yolo run happens inside it and the encode happens outside. Raw rather than
    falling back to cv2's mp4v, which does open there -- mp4v would be a generation
    of loss on a file that is about to be re-encoded anyway, whereas raw frames
    produce a video bit-identical to the direct path.
    """

    ENCODE = ['-c:v', 'libx264', '-preset', 'medium', '-pix_fmt', 'yuv420p',
              '-movflags', '+faststart']

    def __init__(self, path, size, fps, crf):
        w, h = size
        self.path, self.n, self.proc, self.raw = path, 0, None, None
        src = ['-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
               '-framerate', f'{fps:.4f}', '-i']
        if shutil.which('ffmpeg'):
            self.proc = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error'] + src + ['-',
                 '-crf', str(crf)] + self.ENCODE + [path], stdin=subprocess.PIPE)
            print(f"video      : {path} {w}x{h} @ {fps:.2f} fps (H.264, crf {crf})")
            return
        self.raw_path = path + '.rawvideo'
        self.raw = open(self.raw_path, 'wb')
        self.cmd = ' '.join(['ffmpeg', '-y'] + src + [self.raw_path,
                            '-crf', str(crf)] + self.ENCODE + [path])
        print(f"video      : no ffmpeg on PATH -- writing raw BGR to {self.raw_path}\n"
              f"             {w}x{h} @ {fps:.2f} fps; finish it with the command "
              f"printed at the end")

    def write(self, frame):
        (self.proc.stdin if self.proc else self.raw).write(
            np.ascontiguousarray(frame).tobytes())
        self.n += 1

    def close(self):
        if self.raw is not None:
            self.raw.close()
            print(f"\nencode with:\n  {self.cmd}\n  rm {self.raw_path}")
            return
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise SystemExit(f"ffmpeg failed writing {self.path}")


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    b = p.add_argument_group('bag')
    b.add_argument('--bag', required=True, help='bag directory (or the .db3 itself)')
    b.add_argument('--image-topic', default='/insta360/image_raw/compressed')
    b.add_argument('--lidar-topic', default='/livox/lidar')
    b.add_argument('--odom-topic', default='/Odometry')
    b.add_argument('--start', type=float, default=0.0, metavar='S',
                   help='seconds into the bag to begin')
    b.add_argument('--duration', type=float, default=0.0, metavar='S',
                   help='seconds to render (0 = to the end)')
    b.add_argument('--every', type=int, default=1, metavar='N',
                   help='render every Nth panorama')

    v = p.add_argument_group('views')
    v.add_argument('--view', type=float, nargs='+', action='append', metavar='N',
                   help='YAW [PITCH [ROLL [FOV]]] in degrees; repeat once per view')
    v.add_argument('--ring', type=int, default=0)
    v.add_argument('--yaw-start', type=float, default=0.0)
    v.add_argument('--pitch', type=float, default=0.0)
    v.add_argument('--roll', type=float, default=0.0)
    v.add_argument('--fov', type=float, default=60.0)
    v.add_argument('--size', type=int, default=720)
    v.add_argument('--max-cols', type=int, default=3)

    e = p.add_argument_group('mounting (all that survives the gravity lock)')
    e.add_argument('--translation', type=float, nargs=3, default=[0.18, 0.0, -0.13],
                   metavar=('X', 'Y', 'Z'), help='camera origin in LiDAR coords, metres')
    e.add_argument('--yaw-offset', type=float, default=None, metavar='DEG',
                   help='camera heading in the levelled body frame; derived from '
                        '--extrinsic + --gravity when omitted')
    e.add_argument('--extrinsic', default=gbm.DEFAULT_EXTRINSIC)
    e.add_argument('--gravity', default=gbm.DEFAULT_GRAVITY)
    e.add_argument('--imu-lidar-t', type=float, nargs=3,
                   default=[-0.011, -0.02329, 0.04412], metavar=('X', 'Y', 'Z'),
                   help="mapping.extrinsic_T of the FAST-LIO config")
    e.add_argument('--cam-latency', type=float, default=0.0, metavar='S')

    l = p.add_argument_group('cloud')
    l.add_argument('--window', type=float, default=0.4, metavar='S',
                   help='LiDAR seconds per frame, centred on its stamp')
    l.add_argument('--point-stride', type=int, default=1)
    l.add_argument('--no-filter-tags', dest='filter_tags', action='store_false')
    l.add_argument('--min-depth', type=float, default=0.3)
    l.add_argument('--max-depth', type=float, default=30.0)
    l.add_argument('--color-max', type=float, default=0.0, metavar='M',
                   help='range the ramp saturates at (0 = --max-depth)')
    l.add_argument('--point-radius', type=int, default=1)
    l.add_argument('--alpha', type=float, default=0.85,
                   help='point opacity -- below 1 keeps the image under a dense cloud')
    l.add_argument('--cloud', choices=('scan', 'map', 'both'), default='scan',
                   help="'scan' = a --window of /livox/lidar; 'map' = /Laser_map up to "
                        "the frame's stamp; 'both' = each view drawn twice, side by side")
    l.add_argument('--map-topic', default='/Laser_map')

    y = p.add_argument_group('detector and match (bag_bbox_match.py\'s, unchanged)')
    y.add_argument('--yolo', metavar='PT', help='Ultralytics .pt; omit for no detector')
    y.add_argument('--device', default='cpu', help="'cpu', 'cuda', 'cuda:0', '0' ...")
    y.add_argument('--imgsz', type=int, default=640)
    y.add_argument('--yolo-conf', type=float, default=0.25)
    y.add_argument('--classes', type=int, nargs='+', default=None)
    y.add_argument('--box-shrink', type=float, default=0.6,
                   help='horizontal frustum shrink -- keeps sideways background out')
    y.add_argument('--box-shrink-v', type=float, default=1.0,
                   help='vertical shrink; 1.0 because the box height IS the tree height')
    y.add_argument('--gap', type=float, default=0.5, metavar='M',
                   help='range gap that separates the tree from what is behind it')
    y.add_argument('--min-points', type=int, default=5)
    y.add_argument('--anchor-axis', choices=('row', 'z'), default='z')
    y.add_argument('--match-radius', type=int, default=0,
                   help='point radius INSIDE a box (the object draws one bigger). '
                        'Separate from --point-radius because these renderers were '
                        'tuned on a 0.4s scan: the accumulated map puts thousands of '
                        'points on one tree, and at the cloud\'s own radius the green '
                        'fills the whole box and hides what it is sitting on')

    d = p.add_argument_group('output')
    d.add_argument('--out', required=True, metavar='PATH', help='.mp4 to write')
    d.add_argument('--fps', type=float, default=0.0,
                   help='0 = the bag\'s own frame rate, so playback is real time')
    d.add_argument('--crf', type=int, default=20)
    d.add_argument('--stills', type=float, nargs='+', metavar='S',
                   help='seconds into the bag; writes a PNG per instant instead of a '
                        'video, to --out with the second appended')
    args = p.parse_args()

    # --cloud both draws every view twice. The duplicates must be distinct View
    # objects: ViewCanvas stamps each one's origin onto the object itself, so a
    # shared instance would put both tiles in the same place.
    views = lvo.parse_views(args)
    sources = ['map' if args.cloud == 'map' else 'scan'] * len(views)
    if args.cloud == 'both':
        # Interleaved, not appended: each view's two clouds must sit side by side on
        # one row to be comparable, and ViewCanvas fills row-major at --max-cols 2.
        views = [v for pair in zip(views, lvo.parse_views(args)) for v in pair]
        sources = ['scan', 'map'] * (len(views) // 2)
        args.max_cols = 2
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
    color_max = args.color_max or args.max_depth
    print("views      : " + "; ".join(f"[{i}] {vw.label} ({src})"
                                      for i, (vw, src) in enumerate(zip(views, sources))))

    conn, types = open_bag(args.bag)
    wanted = [args.odom_topic, args.image_topic]
    if 'scan' in sources:
        wanted.append(args.lidar_topic)
    if 'map' in sources:
        wanted.append(args.map_topic)
    tids = {t: topic_id(conn, types, t, args.bag) for t in wanted}
    odom = load_odometry(conn, tids[args.odom_topic], args.odom_topic)
    cloud = (bbm.BagCloudWindow(odom, t_il, args.point_stride, args.filter_tags)
             if 'scan' in sources else None)
    laser_map = (LaserMap(conn, tids[args.map_topic], args.map_topic)
                 if 'map' in sources else None)

    detector = draw_detections = None
    if args.yolo:
        from hailo_yolo import draw_detections
        detector = bbm.UltralyticsYolo(args.yolo, conf=args.yolo_conf,
                                       device=args.device, imgsz=args.imgsz,
                                       classes=args.classes)
        print(f"yolo       : {args.yolo} @{args.imgsz} on {args.device}, "
              f"conf {args.yolo_conf}, classes {sorted(detector.names.values())}")
        print(f"match      : fast (gap {args.gap} m), frustum {args.box_shrink:.2f}x h / "
              f"{args.box_shrink_v:.2f}x v, against the tile's own cloud")
    # match_and_draw stamps at args.point_radius (and the object one bigger); the base
    # cloud keeps its own, so the two are set independently.
    match_args = argparse.Namespace(**vars(args))
    match_args.point_radius = args.match_radius

    half = args.window * 0.5
    t_bag0 = odom.stamps[0]
    t_from = t_bag0 + args.start
    t_to = (t_from + args.duration) if args.duration > 0 else float('inf')
    stills = deque(sorted(args.stills or []))
    if stills:
        t_from = min(t_from, t_bag0 + stills[0])
        t_to = t_bag0 + stills[-1] + 1.0      # nothing past the last still is drawn

    canvas_maker = lvo.ViewCanvas(views, args.max_cols, cv2.INTER_LINEAR)
    writer = None
    pending = deque()
    n_img = n_done = n_nopose = n_baddecode = n_det = n_obj = 0
    t_wall0 = time.monotonic()

    # The bag's own rate, so a second of video is a second of flight. Measured from
    # the image stamps rather than assumed: the 360 stream free-runs at whatever the
    # USB pipeline manages, which is neither 10 Hz nor constant.
    img_stamps = [r[0] * 1e-9 for r in conn.execute(
        'SELECT timestamp FROM messages WHERE topic_id = ? ORDER BY timestamp',
        (tids[args.image_topic],))]
    dt = float(np.median(np.diff(img_stamps))) if len(img_stamps) > 1 else 0.1
    fps = args.fps or float(np.clip(1.0 / (max(dt, 1e-6) * max(args.every, 1)), 1.0, 60.0))

    def relative(pts, cam_origin):
        """Camera-relative points + their ranges + the shared range mask.

        Computed once per cloud SOURCE, not once per view: --cloud both has two views
        looking the same way, and the cull inside View.project is what differs between
        them, not this.
        """
        if not len(pts):
            return (np.empty((0, 3), np.float32), np.empty(0), np.empty(0, bool))
        d_world = pts - cam_origin
        rng = np.linalg.norm(d_world, axis=1)
        return d_world, rng, (rng >= args.min_depth) & (rng <= args.max_depth)

    def process(stamp, jpeg):
        nonlocal n_done, n_nopose, n_baddecode, n_det, n_obj, writer
        if args.stills and (not stills or stamp - t_bag0 < stills[0]):
            return          # not a requested instant: skip the decode and the detector
        pose = odom.pose_at(stamp - args.cam_latency)
        if pose is None:
            n_nopose += 1
            return
        r_wi, p_wi = pose
        # Pure yaw: the X5's webcam stream is gravity-locked, so roll and pitch of the
        # airframe never reach the panorama. See global_bbox_match.py's header.
        r_pano_world = gbm.rz(gbm.yaw_of(r_wi) + dpsi).T.astype(np.float32)
        for vw in views:
            vw.set_extrinsic(r_pano_world)
        cam_origin = (r_wi @ (t_lidar_cam + t_il) + p_wi).astype(np.float32)

        clouds = {}
        if cloud is not None:
            clouds['scan'] = cloud.window(stamp - half, stamp + half)
        if laser_map is not None:
            clouds['map'] = laser_map.upto(stamp)
        rel = {k: relative(v, cam_origin) for k, v in clouds.items()}

        frame = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        if frame is None:
            n_baddecode += 1
            return
        canvas = canvas_maker.render(frame)
        n_done += 1
        dets_by_view = {}
        for i, (vw, src) in enumerate(zip(views, sources)):
            tile = canvas_maker.tile(canvas, vw)
            # Inference BEFORE anything is painted: the detector must see the image,
            # not the cloud drawn over it. --cloud both gives two tiles the same
            # geometry and the same pixels, so the second reuses the first's boxes
            # rather than paying for an identical inference.
            dets = []
            if detector is not None:
                if vw.label not in dets_by_view:
                    dets_by_view[vw.label] = detector.infer(tile)
                dets = dets_by_view[vw.label]

            u, v_, r = vw.project(*rel[src])
            n = lvo.paint(tile, u, v_, r, args.point_radius, args.min_depth,
                          color_max, args.alpha)

            located = 0
            for det in dets:
                anchor, _ = gbm.match_and_draw(tile, vw, det, u, v_, r,
                                               cam_origin, match_args)
                if anchor is None:
                    continue
                located += 1
                c_ = anchor['central']
                cv2.putText(tile, f"{detector.label(det[5])} {anchor['range']:.1f}m "
                                  f"({c_[0]:+.1f},{c_[1]:+.1f},{c_[2]:+.1f})",
                            (int(det[0]) + 2, max(int(det[1]) - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                            cv2.LINE_AA)
            n_det += len(dets)
            n_obj += located
            if draw_detections is not None:
                draw_detections(tile, dets, detector)

            note = (f"{args.window:.1f}s scan window" if src == 'scan'
                    else "accumulated map")
            if detector is not None:
                note = f"{len(dets)} det  {located} matched   " + note
            lvo.label_view(tile, f"[{i}] {vw.label}  {src}",
                           f"t{stamp - t_bag0:6.1f}s   {n} of {len(clouds[src])} pts"
                           f"   {note}")
        if detector is not None:
            gbm.draw_legend(canvas)
        draw_colorbar(canvas, args.min_depth, color_max)
        draw_progress(canvas, (stamp - t_from) / max(img_stamps[-1] - t_from, 1e-6))

        if stills:
            # The first frame at or past each requested instant, so a still always
            # lands on real data rather than the nearest thing to an exact stamp.
            while stills and stamp - t_bag0 >= stills[0]:
                want = stills.popleft()
                path = f"{os.path.splitext(args.out)[0]}_t{want:g}s.png"
                cv2.imwrite(path, canvas)
                print(f"  wrote {path}  (t={stamp - t_bag0:.2f}s)")
            return
        if writer is None:
            writer = H264Writer(args.out, (canvas.shape[1], canvas.shape[0]), fps,
                                args.crf)
        writer.write(canvas)
        if n_done % 100 == 0:
            el = time.monotonic() - t_wall0
            print(f"  {n_done} frames  t={stamp - t_bag0:6.1f}s  "
                  f"{n_done / max(el, 1e-6):.1f} fps")

    print(f"\nreading {args.bag} ...\n")
    # Only /Laser_map wanted: no scan stream to wait for, so every image is ready
    # the moment it is read and `pending` never holds more than one frame.
    lidar_tid = tids[args.lidar_topic] if cloud is not None else None
    read_tids = [tids[args.image_topic]] + ([lidar_tid] if lidar_tid is not None else [])
    rows = conn.execute(
        'SELECT topic_id, data FROM messages WHERE topic_id IN '
        f"({','.join('?' * len(read_tids))}) ORDER BY timestamp", read_tids)
    try:
        for tid, data in rows:
            data = bytes(data)
            if tid == lidar_tid:
                cloud.feed(data)
            else:
                stamp = bbm.BagCloudWindow.stamp_of(data)
                if stamp is None or stamp < t_from:
                    continue
                if stamp > t_to:
                    break
                n_img += 1
                if (n_img - 1) % args.every:
                    continue
                pending.append((stamp, parse_compressed_image(data)))

            # An image is only ready once scans past its centred window have been
            # read; until then the cloud behind it is still arriving.
            newest = cloud.newest if cloud is not None else float('inf')
            while pending and newest is not None and pending[0][0] + half <= newest:
                st, jpeg = pending.popleft()
                process(st, jpeg)
                if cloud is not None:
                    cloud.drop_before(st - args.window)
            if args.stills and not stills:
                break                          # every requested instant is written
        for st, jpeg in pending:               # tail of the bag: no lookahead left
            process(st, jpeg)
    except KeyboardInterrupt:
        print("\ninterrupted -- closing the video at the frames written so far")
    finally:
        if writer is not None:
            written = writer.path if writer.raw is None else writer.raw_path
            writer.close()
            mb = os.path.getsize(written) / 1e6
            print(f"\nwrote {written} ({writer.n} frames, {mb:.1f} MB, "
                  f"{writer.n / fps:.1f} s)")
        conn.close()

    el = time.monotonic() - t_wall0
    unplaced = f"{cloud.unplaced} scan(s) unplaced, " if cloud is not None else ""
    found = f"{n_det} detection(s), {n_obj} matched, " if detector is not None else ""
    print(f"{n_done} frame(s) rendered, {found}{unplaced}"
          f"{n_nopose} frame(s) without a pose"
          + (f", {n_baddecode} frame(s) FAILED TO DECODE" if n_baddecode else ""))
    print(f"{el:.1f} s wall, {n_done / max(el, 1e-6):.2f} frames/s")


if __name__ == '__main__':
    main()
