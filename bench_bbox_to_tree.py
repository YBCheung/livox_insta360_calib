#!/usr/bin/env python3
"""Benchmark: fixed image bbox -> matched point-cloud cluster -> center/top/bottom.

Standalone timing rig for the "YOLO bbox -> tree keypoints" pipeline discussed
alongside project_pointcloud_to_view.py, with YOLO removed (fixed bbox instead)
so we can measure the point-cloud-matching cost in isolation before worrying
about detector throughput.

    python3 bench_bbox_to_tree.py --data-dir data/calib_data_yard_1 --view 0 --iters 50

No scipy/sklearn dependency on purpose -- this environment's scipy is broken
(numpy2/scipy ABI mismatch) and a Pi deployment should avoid the extra
dependency weight anyway. Clustering is a cheap iterative 3D re-gate instead
of DBSCAN.
"""
import argparse
import glob
import math
import os
import time

import numpy as np

from project_pointcloud_to_view import (load_pcd_xyzi, load_extrinsic, load_intrinsics,
                                        apply_gravity_correction)


def project_all(xyz, R, t, fx, fy, cx, cy):
    cam = (R @ xyz.T).T + t
    X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]
    Zsafe = np.where(Z == 0, 1e-9, Z)
    u = fx * X / Zsafe + cx
    v = fy * Y / Zsafe + cy
    depth = np.sqrt(X * X + Y * Y + Z * Z)
    return u, v, Z, depth


def depth_mode_gate(depth, k_mad=3.0, n_bins=40):
    """Keep the dominant depth peak inside the bbox candidates (the object),
    dropping background/foreground at other ranges, via a MAD window around
    the tallest histogram bin's median -- robust to a handful of stray points."""
    if len(depth) == 0:
        return np.zeros(0, dtype=bool)
    hist, edges = np.histogram(depth, bins=n_bins)
    peak = np.argmax(hist)
    lo, hi = edges[peak], edges[peak + 1]
    in_peak = (depth >= lo) & (depth < hi)
    if not in_peak.any():
        in_peak = np.abs(depth - np.median(depth)) < 1e-6
    peak_med = np.median(depth[in_peak])
    mad = np.median(np.abs(depth[in_peak] - peak_med)) + 1e-6
    window = k_mad * 1.4826 * mad
    return np.abs(depth - peak_med) < max(window, 0.15)


def voxel_connected_components(points, voxel=0.08):
    """6-connected components over a voxel grid -- a from-scratch stand-in for
    DBSCAN/scipy.ndimage.label (both unavailable/broken in this environment,
    and worth avoiding as a Pi dependency anyway). Fine for the few-thousand-
    point candidate sets this runs on; returns {root_id: point_indices}."""
    if len(points) == 0:
        return {}
    keys = np.floor(points / voxel).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    n = len(uniq)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    key_to_idx = {tuple(k): i for i, k in enumerate(map(tuple, uniq))}
    offsets = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    for i, k in enumerate(map(tuple, uniq)):
        for dx, dy, dz in offsets:
            j = key_to_idx.get((k[0] + dx, k[1] + dy, k[2] + dz))
            if j is not None:
                union(i, j)

    roots = np.array([find(i) for i in range(n)])
    point_roots = roots[inverse]
    return {r: np.where(point_roots == r)[0] for r in np.unique(point_roots)}


def pick_object_cluster(xyz_sub, reproject, bbox_center, bbox_half_diag_px, voxel=0.08, min_size=15,
                         merge_radius_m=0.3, merge_size_ratio=3.0):
    """Split the depth-gated points into connected components and pick the object.

    Size and image-space centrality alone each fail on their own real case:
    picking purely by size latches onto a large background surface a loose
    bbox partially includes (e.g. a wall/partition behind a chair -- 2300
    points vs. the chair's 350); picking purely by "nearest centroid to bbox
    center" instead loses a large, correctly-shaped object whose centroid
    isn't dead-center (e.g. a tree trunk filling most of the box) to a
    handful of stray noise points that happen to be more central.

    So: default to the largest cluster (the majority case for a reasonably
    tight box). Only override it when the largest cluster's centroid is
    clearly off-center (> half the box's half-diagonal) AND a meaningfully
    sized alternative sits clearly near the center (< 0.3x the half-diagonal)
    -- that combination is what actually indicates "a background surface
    leaked in," not just "the object isn't perfectly centered." Fragments
    close to the winner (comparable in size, per `merge_size_ratio`) are
    merged back in -- e.g. a chair's separated armrest/leg returns."""
    clusters = voxel_connected_components(xyz_sub, voxel=voxel)
    scored = []
    for idx in clusters.values():
        if len(idx) < min_size:
            continue
        c = np.median(xyz_sub[idx], axis=0)
        pu, pv = reproject(c)
        d_px = math.hypot(pu - bbox_center[0], pv - bbox_center[1])
        scored.append((len(idx), d_px, idx, c))
    if not scored:
        return xyz_sub  # nothing passed the size filter -- fall back to everything gated

    scored.sort(key=lambda e: e[0], reverse=True)  # largest first
    best_size, best_d, best_idx, best_c = scored[0]
    if best_d > 0.5 * bbox_half_diag_px:
        central = [e for e in scored[1:] if e[1] < 0.3 * bbox_half_diag_px]
        if central:
            best_size, best_d, best_idx, best_c = min(central, key=lambda e: e[1])

    merged = [best_idx]
    for size, d_px, idx, c in scored:
        if idx is best_idx:
            continue
        if size <= merge_size_ratio * best_size and np.linalg.norm(c - best_c) < merge_radius_m:
            merged.append(idx)
    return xyz_sub[np.concatenate(merged)]


def bbox_to_keypoints(xyz, R, t, fx, fy, cx, cy, bbox, up_axis, timings):
    x1, y1, x2, y2 = bbox

    t0 = time.perf_counter()
    u, v, Z, depth = project_all(xyz, R, t, fx, fy, cx, cy)
    t1 = time.perf_counter()

    in_box = (u >= x1) & (u < x2) & (v >= y1) & (v < y2) & (Z > 0.1)
    candidates = xyz[in_box]
    cand_depth = depth[in_box]
    t2 = time.perf_counter()

    gate = depth_mode_gate(cand_depth)
    gated = candidates[gate]
    t3 = time.perf_counter()

    bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    bbox_half_diag = 0.5 * math.hypot(x2 - x1, y2 - y1)

    def reproject(p):
        cam_pt = R @ p + t
        return fx * cam_pt[0] / cam_pt[2] + cx, fy * cam_pt[1] / cam_pt[2] + cy

    cluster = pick_object_cluster(gated, reproject, bbox_center, bbox_half_diag)
    t4 = time.perf_counter()

    if len(cluster) == 0:
        result = None
    else:
        up = cluster[:, up_axis]
        lo_h, hi_h = np.percentile(up, [1, 99])
        center = np.median(cluster, axis=0)
        result = {
            'center': center,
            'bottom_h': lo_h,
            'top_h': hi_h,
            'n_candidates': len(candidates),
            'n_gated': len(gated),
            'n_cluster': len(cluster),
        }
    t5 = time.perf_counter()

    timings['project'].append(t1 - t0)
    timings['bbox_mask'].append(t2 - t1)
    timings['depth_gate'].append(t3 - t2)
    timings['cluster'].append(t4 - t3)
    timings['keypoints'].append(t5 - t4)
    timings['total'].append(t5 - t0)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--view', type=int, default=0)
    p.add_argument('--iters', type=int, default=50)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--up-axis', type=int, default=2, help='0=x,1=y,2=z world-up column in lidar frame')
    p.add_argument('--bbox', type=float, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                    help='fixed bbox in pixels; defaults to a centered 200x260 box')
    p.add_argument('--extrinsic', help='pose file to use -- a plain 4x4/3x4 number file '
                                        '(e.g. a solved configs/results/extrinsic_NN.txt) or a '
                                        'livox_camera_calib config_NN.yaml; defaults to that '
                                        "view's seed guess")
    p.add_argument('--gravity', help='gravity.txt with a FRESH up_measured reading for the '
                                      "rig's CURRENT tilt -- re-derives the extrinsic for it "
                                      'instead of assuming the rig is still level (see '
                                      'project_pointcloud_to_view.apply_gravity_correction)')
    args = p.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    views_dir = os.path.join(data_dir, 'views')
    intr, manifest = load_intrinsics(views_dir)
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    width, height = int(intr['width']), int(intr['height'])

    view_file, yaw, pitch = manifest[args.view]
    if args.extrinsic:
        extrinsic_path = args.extrinsic
    else:
        guess_dir = os.path.join(os.path.dirname(data_dir), 'guesses')
        matches = glob.glob(os.path.join(guess_dir, f'guess_{args.view:02d}_*.txt'))
        if not matches:
            matches = glob.glob(os.path.join(data_dir, 'guesses', f'guess_{args.view:02d}_*.txt'))
        if not matches:
            raise SystemExit(f"no seed guess found for view {args.view}")
        extrinsic_path = matches[0]
    R, t = load_extrinsic(extrinsic_path)
    R, t = apply_gravity_correction(R, t, args.gravity)

    print(f"loading cloud from {data_dir} ...")
    pts = load_pcd_xyzi(os.path.join(data_dir, 'cloud.pcd'))
    xyz = pts[:, :3].astype(np.float64)
    print(f"cloud: {len(xyz)} points, view {args.view} ({view_file}), extrinsic {extrinsic_path}")

    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        bw, bh = 200, 260
        bbox = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
    print(f"fixed bbox: {bbox} (image {width}x{height})")

    timings = {k: [] for k in ('project', 'bbox_mask', 'depth_gate', 'cluster', 'keypoints', 'total')}
    result = None
    for i in range(args.warmup + args.iters):
        r = bbox_to_keypoints(xyz, R, t, fx, fy, cx, cy, bbox, args.up_axis, timings)
        if i == args.warmup:
            for k in timings:
                timings[k] = []
            result = r
        elif i > args.warmup:
            result = r

    print()
    if result is None:
        print("WARNING: no points survived the gate/cluster for this bbox -- pick a bbox that "
              "actually covers an object, timing below still reflects the empty-result cost.")
    else:
        print(f"candidates in bbox frustum : {result['n_candidates']}")
        print(f"after depth-mode gate      : {result['n_gated']}")
        print(f"after centroid re-gate     : {result['n_cluster']}")
        print(f"center (xyz)               : {np.round(result['center'], 3)}")
        print(f"bottom/top (up-axis h)     : {result['bottom_h']:.3f} / {result['top_h']:.3f}")

    print()
    print(f"per-stage timing over {args.iters} iters (ms):")
    for k in ('project', 'bbox_mask', 'depth_gate', 'cluster', 'keypoints', 'total'):
        arr = np.array(timings[k]) * 1000
        print(f"  {k:12s}: mean {arr.mean():7.3f}  median {np.median(arr):7.3f}  p95 {np.percentile(arr, 95):7.3f}")

    total_mean = np.mean(timings['total']) * 1000
    print()
    print(f"-> {1000.0 / total_mean:.1f} matches/sec on this machine (single bbox, single-threaded call)")


if __name__ == '__main__':
    main()
