#!/usr/bin/env python3
"""Real-time YOLO -> point-cloud object match, using interactive_bbox_match's method.

This is interactive_bbox_match.py with the mouse replaced by a detector and the static
capture replaced by the live camera and LiDAR. The matching is the same one, stage for
stage -- frustum candidates, depth-mode gate, 3D voxel components, size-vs-centrality
pick -- and so is the staged rendering that makes it readable:

    grey    inside the box, before anything was rejected
    orange  survived the depth-mode gate (the dominant depth peak)
    green   the chosen 3D component -- the object, and the only points the
            keypoints are computed from
    C/B/T   centre, bottom, top, drawn where they reproject

Seeing what each stage threw away is the point. A box whose "object" is really the wall
behind it is obvious the instant the gate keeps the wall, and no single 3D coordinate
printed on its own would have told you.

    python3 live_cluster_match.py --view 240 --yolo <hef> --translation 0.18 0.0 -0.13 \
        --rpy 90 0 -23

Against live_view_overlay.py --match cluster, which runs the same matcher: that one is
a calibration overlay that can also detect, so it paints the whole cloud by depth and
gives a box one line of text. This one shows a detection being taken apart. Same
matcher, different question -- "is my extrinsic right" versus "why did this object end
up there". Both read the same modules, so neither can drift from the other.

Keys: q / Esc quit, s snapshot, l toggle the faint full cloud, p pause.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bbox_match as bm
import live_view_overlay as lvo

STAGE_COLORS = {
    'candidate': (150, 150, 150),   # BGR: grey
    'gated': (0, 150, 255),         # orange
    'object': (90, 230, 50),        # green
}
KEYPOINT_COLORS = {'C': (40, 40, 255), 'B': (255, 160, 0), 'T': (255, 0, 255)}


_STAMP_OFFSETS = {}


def stamp(img, u, v, color, radius=1):
    """Draw points as small squares in ONE indexed assignment.

    A pass per offset (9 of them at radius 1, 25 at radius 2) costs more than the
    match itself at these point counts, so the offsets are broadcast into a single
    index array and written once.
    """
    if len(u) == 0:
        return
    h, w = img.shape[:2]
    if radius not in _STAMP_OFFSETS:
        span = np.arange(-radius, radius + 1)
        dy, dx = np.meshgrid(span, span, indexing='ij')
        _STAMP_OFFSETS[radius] = (dy.ravel(), dx.ravel())
    dys, dxs = _STAMP_OFFSETS[radius]
    vi = v.astype(np.int32)[:, None] + dys[None, :]
    ui = u.astype(np.int32)[:, None] + dxs[None, :]
    np.clip(vi, 0, h - 1, out=vi)
    np.clip(ui, 0, w - 1, out=ui)
    img[vi.ravel(), ui.ravel()] = color


def draw_keypoints(img, anchor):
    """Circle centre/bottom/top where they reproject, as the interactive tool does."""
    tx, ty, cu, cv_, bx, by = anchor['pixels']
    for tag, (x, y) in (('T', (tx, ty)), ('C', (cu, cv_)), ('B', (bx, by))):
        cv2.circle(img, (int(x), int(y)), 6, KEYPOINT_COLORS[tag], 2, cv2.LINE_AA)
        cv2.putText(img, tag, (int(x) + 8, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    KEYPOINT_COLORS[tag], 1, cv2.LINE_AA)


def draw_legend(img, extra=()):
    lines = [("grey  = in box", STAGE_COLORS['candidate']),
             ("orange = depth-gated", STAGE_COLORS['gated']),
             ("green  = object cluster", STAGE_COLORS['object'])]
    y = img.shape[0] - 8 - 13 * (len(lines) + len(extra))
    for text, color in list(lines) + list(extra):
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1,
                    cv2.LINE_AA)
        y += 13


def match_and_draw(tile, view, det, u, v, rng, cam_origin, args):
    """One detection: run the staged match, draw every stage, return the anchors."""
    x1, y1, x2, y2 = det[:4]
    mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    hw, hh = (x2 - x1) * 0.5 * args.box_shrink, (y2 - y1) * 0.5 * args.box_shrink
    inside = (u >= mx - hw) & (u <= mx + hw) & (v >= my - hh) & (v <= my + hh)
    n_cand = int(inside.sum())
    if n_cand < args.min_points:
        return None, (n_cand, 0, 0)

    ub, vb, rb = u[inside], v[inside], rng[inside]
    stamp(tile, ub, vb, STAGE_COLORS['candidate'], args.point_radius)

    pts3 = view.unproject(ub, vb, rb, cam_origin)
    idx_gate, idx_obj = bm.cluster_stages(pts3, ub, vb, rb, (x1, y1, x2, y2),
                                          voxel=args.match_voxel,
                                          min_size=args.min_size,
                                          min_points=args.min_points)
    if idx_gate is None:
        return None, (n_cand, 0, 0)
    stamp(tile, ub[idx_gate], vb[idx_gate], STAGE_COLORS['gated'], args.point_radius)
    if idx_obj is None or len(idx_obj) < args.min_points:
        return None, (n_cand, len(idx_gate), 0)

    stamp(tile, ub[idx_obj], vb[idx_obj], STAGE_COLORS['object'], args.point_radius + 1)
    anchor = bm.anchors(pts3, ub, vb, rb, idx_obj, args.anchor_axis)
    draw_keypoints(tile, anchor)
    return anchor, (n_cand, len(idx_gate), len(idx_obj))


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

    e = p.add_argument_group('extrinsic (both frames x=fwd, y=left, z=up)')
    e.add_argument('--translation', type=float, nargs=3, default=[0.18, 0.0, -0.13],
                   metavar=('X', 'Y', 'Z'))
    e.add_argument('--rpy', type=float, nargs=3, default=[90.0, 0.0, -23.0],
                   metavar=('YAW', 'PITCH', 'ROLL'))
    e.add_argument('--extrinsic', help='4x4/3x4 panorama-frame T_cam_lidar instead')

    c = p.add_argument_group('camera')
    c.add_argument('--device', type=int, default=0)
    c.add_argument('--width', type=int, default=2880)
    c.add_argument('--height', type=int, default=1440)
    c.add_argument('--fps', type=float, default=10.0)
    c.add_argument('--cubic', action='store_true')

    l = p.add_argument_group('lidar')
    l.add_argument('--lidar-topic', default='/livox/lidar')
    l.add_argument('--lidar-format', choices=('auto', 'custom', 'pointcloud2'), default='auto')
    l.add_argument('--lidar-window', type=float, default=0.4)
    l.add_argument('--point-stride', type=int, default=1)
    l.add_argument('--no-filter-tags', dest='filter_tags', action='store_false')
    l.add_argument('--no-lidar-raw', dest='lidar_raw', action='store_false')
    l.add_argument('--no-lidar', action='store_true')
    l.add_argument('--min-depth', type=float, default=0.3)
    l.add_argument('--max-depth', type=float, default=30.0)

    y = p.add_argument_group('detector and match')
    y.add_argument('--yolo', required=True, metavar='HEF')
    y.add_argument('--yolo-conf', type=float, default=0.25)
    y.add_argument('--yolo-every', type=int, default=1)
    y.add_argument('--yolo-view', type=int, action='append', metavar='N')
    y.add_argument('--yolo-all-views', action='store_true')
    y.add_argument('--box-shrink', type=float, default=1.0,
                   help='fraction of the box used as the frustum. 1.0 keeps the whole '
                        'box, as the interactive tool does -- the gate and the '
                        'components are what reject background here')
    y.add_argument('--match-voxel', type=float, default=0.08,
                   help='voxel size for 3D connectivity, metres')
    y.add_argument('--min-size', type=int, default=15,
                   help='smallest component considered a candidate object')
    y.add_argument('--min-points', type=int, default=5)
    y.add_argument('--anchor-axis', choices=('row', 'z'), default='z',
                   help="top/bottom along world-up z (this tool's default, matching "
                        "interactive_bbox_match's up-axis) or by image row")

    d = p.add_argument_group('display')
    d.add_argument('--point-radius', type=int, default=1)
    d.add_argument('--show-cloud', action='store_true',
                   help='also paint the rest of the in-view cloud, faintly')
    d.add_argument('--print', dest='do_print', action='store_true')
    d.add_argument('--no-display', action='store_true')
    d.add_argument('--save-dir')
    d.add_argument('--snapshot-interval', type=float, default=0.0)
    args = p.parse_args()

    views = lvo.parse_views(args)
    r_pano, t_pano = lvo.build_extrinsic(args)
    r32 = r_pano.astype(np.float32)
    cam_origin = (-r_pano.T @ t_pano).astype(np.float32)
    for vw in views:
        vw.set_extrinsic(r32)
    print("views      : " + "; ".join(f"[{i}] {vw.label}" for i, vw in enumerate(views)))

    from hailo_yolo import HailoYolo, draw_detections

    # The accelerator is opened LAST and released in the finally below. An activated
    # Hailo network group that is still open when the interpreter tears down segfaults
    # inside libhailort, so anything that can fail -- a missing ROS, a busy camera --
    # must fail before the device is ever claimed.
    buf = shutdown = cam = detector = None
    if not args.no_lidar:
        buf = lvo.LidarBuffer(args.lidar_window, args.point_stride, args.filter_tags)
        shutdown = lvo.start_lidar(args, buf)
    cam = lvo.PanoramaSource(args)

    try:
        detector = HailoYolo(args.yolo, conf=args.yolo_conf)
    except Exception as exc:
        cam.close()
        if shutdown is not None:
            shutdown()
        raise SystemExit(f"cannot load {args.yolo}: {exc}\n"
                         f"If the file exists, compare `hailortcli parse-hef {args.yolo}` "
                         f"with `hailortcli fw-control identify` -- a HEF only runs on the "
                         f"architecture it was compiled for, and only one process at a "
                         f"time may hold the device.")
    yolo_views = (set(range(len(views))) if args.yolo_all_views
                  else set(args.yolo_view or [0]))
    print(f"yolo       : {args.yolo} {detector.input_size[0]}x{detector.input_size[1]} "
          f"on view(s) {sorted(yolo_views)}")
    print(f"match      : cluster (depth-mode gate + {args.match_voxel} m voxel components), "
          f"anchors along {args.anchor_axis}")
    canvas_maker = lvo.ViewCanvas(views, args.max_cols,
                                  cv2.INTER_CUBIC if args.cubic else cv2.INTER_LINEAR)
    save_dir = args.save_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             'live_snapshots')
    if args.no_display and not args.snapshot_interval:
        args.snapshot_interval = 2.0

    detections = [[] for _ in views]
    d_cam = np.empty((0, 3), np.float32)
    rng = np.empty(0, np.float32)
    near = np.empty(0, bool)
    fps, n_frames, t_fps = 0.0, 0, time.monotonic()
    frame_no, last_snapshot = 0, 0.0
    show_cloud, paused = args.show_cloud, False
    yolo_ms = match_ms = 0.0

    print("\nrunning -- q quit, s snapshot, l faint cloud, p pause\n")
    try:
        while True:
            frame = cam.latest()
            if frame is None or paused:
                time.sleep(0.005)
                if not paused:
                    continue
            frame_no += 1

            if buf is not None and not paused:
                pts = buf.snapshot()
                if len(pts):
                    d_cam = pts - cam_origin
                    rng = np.linalg.norm(d_cam, axis=1)
                    near = (rng >= args.min_depth) & (rng <= args.max_depth)
                else:
                    d_cam, rng, near = (np.empty((0, 3), np.float32), np.empty(0),
                                        np.empty(0, bool))

            canvas = canvas_maker.render(frame)
            for i, vw in enumerate(views):
                tile = canvas_maker.tile(canvas, vw)
                if i in yolo_views and frame_no % args.yolo_every == 0:
                    t0 = time.perf_counter()
                    detections[i] = detector.infer(tile)
                    yolo_ms = (time.perf_counter() - t0) * 1000.0

                u, v, r = vw.project(d_cam, rng, near)
                if show_cloud:
                    stamp(tile, u, v, (70, 70, 70), 0)

                located = 0
                t0 = time.perf_counter()
                for det in detections[i]:
                    anchor, counts = match_and_draw(tile, vw, det, u, v, r,
                                                    cam_origin, args)
                    if anchor is None:
                        continue
                    located += 1
                    c_ = anchor['central']
                    cv2.putText(tile, f"{detector.label(det[5])} {anchor['range']:.1f}m "
                                      f"({c_[0]:+.1f},{c_[1]:+.1f},{c_[2]:+.1f})",
                                (int(det[0]) + 2, max(int(det[1]) - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                                cv2.LINE_AA)
                    if args.do_print:
                        t_, b_ = anchor['top'], anchor['bottom']
                        print(f"  [{i}] {detector.label(det[5])} {det[4]:.2f}  "
                              f"{counts[0]} in box -> {counts[1]} gated -> {counts[2]} object"
                              f"  range {anchor['range']:.2f} m | top {np.round(t_, 2)}  "
                              f"central {np.round(c_, 2)}  bottom {np.round(b_, 2)}")
                match_ms = (time.perf_counter() - t0) * 1000.0

                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1),
                              (60, 60, 60), 1)
                draw_detections(tile, detections[i], detector)
                lvo.label_view(tile, f"[{i}] {vw.label}",
                               f"{len(detections[i])} det  {located} matched  "
                               f"yolo {yolo_ms:.0f} ms  match {match_ms:.1f} ms  "
                               f"{fps:.1f} fps")
            draw_legend(canvas)

            n_frames += 1
            now = time.monotonic()
            if now - t_fps >= 1.0:
                fps, n_frames, t_fps = n_frames / (now - t_fps), 0, now

            snap = bool(args.snapshot_interval) and now - last_snapshot >= args.snapshot_interval
            if not args.no_display:
                cv2.imshow('yolo -> cluster match', canvas)
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
                path = os.path.join(save_dir, time.strftime('cluster_%Y%m%d_%H%M%S.png'))
                print(f"  {'wrote' if cv2.imwrite(path, canvas) else 'FAILED to write'} {path}")
                last_snapshot = now
    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.close()
        if detector is not None:
            detector.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        if shutdown is not None:
            shutdown()


if __name__ == '__main__':
    main()
