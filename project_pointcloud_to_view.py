#!/usr/bin/env python3
"""Project the raw LiDAR cloud onto a cut view using ONLY a seed extrinsic.

No livox_camera_calib involved -- this is a from-scratch sanity check of one
(image, cloud, extrinsic) triple, useful for eyeballing whether a seed/solved
guess is even in the right neighborhood before trusting the calibrator's own
debug output.

    python3 project_pointcloud_to_view.py --data-dir data/calib_data_yard_1 --view 5

Defaults to that view's seed guess (data/guesses/guess_NN_*.txt); pass
--extrinsic to check any other 4x4/3x4 pose file (e.g. a solved result)
against the same image instead.
"""
import argparse
import glob
import math
import os

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def load_pcd_xyzi(path):
    with open(path, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            if not line:
                raise SystemExit(f"{path}: no 'DATA binary' header found")
            header += line
            if line.strip() == b'DATA binary':
                break
        data = f.read()
    return np.frombuffer(data, dtype=np.float32).reshape(-1, 4)


def load_extrinsic(path):
    """Read a 4x4, 3x4, or 16/12-number whitespace/comma-separated pose file."""
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


def load_intrinsics(views_dir):
    vals = {}
    manifest = []
    with open(os.path.join(views_dir, 'intrinsics.txt')) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 3 and parts[0].endswith('.png'):
                manifest.append((parts[0], float(parts[1]), float(parts[2])))
            elif len(parts) == 2:
                vals[parts[0]] = float(parts[1])
    return vals, manifest


def depth_ramp(t):
    """5-stop dark-blue -> teal -> yellow-green -> amber -> warm-red ramp, t in [0,1]."""
    stops = np.array([
        [0.059, 0.090, 0.161],
        [0.102, 0.451, 0.522],
        [0.549, 0.851, 0.310],
        [0.976, 0.749, 0.153],
        [0.980, 0.349, 0.251],
    ])
    t = np.clip(t, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    frac = (t - i)[:, None]
    return stops[i] * (1 - frac) + stops[i + 1] * frac


def project(args):
    data_dir = os.path.abspath(args.data_dir)
    views_dir = os.path.join(data_dir, 'views')
    intr, manifest = load_intrinsics(views_dir)
    fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
    width, height = int(intr['width']), int(intr['height'])

    if args.view >= len(manifest):
        raise SystemExit(f"--view {args.view} out of range (only {len(manifest)} views)")
    view_file, yaw, pitch = manifest[args.view]
    image_path = os.path.join(views_dir, view_file)

    if args.extrinsic:
        extrinsic_path = args.extrinsic
    else:
        pattern = os.path.join(HERE, 'data', 'guesses', f'guess_{args.view:02d}_*.txt')
        matches = glob.glob(pattern)
        if not matches:
            raise SystemExit(f"no seed guess found matching {pattern}")
        extrinsic_path = matches[0]

    R, t = load_extrinsic(extrinsic_path)
    pts = load_pcd_xyzi(os.path.join(data_dir, 'cloud.pcd'))
    xyz = pts[:, :3].astype(np.float64)

    cam = (R @ xyz.T).T + t
    X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]
    depth = np.sqrt(X**2 + Y**2 + Z**2)

    front = Z > args.near_clip
    u = fx * X / np.where(Z == 0, 1e-9, Z) + cx
    v = fy * Y / np.where(Z == 0, 1e-9, Z) + cy
    in_bounds = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    mask = front & in_bounds & (depth >= args.min_depth)
    if args.max_depth:
        mask &= depth < args.max_depth

    n_hit = int(mask.sum())
    n_self = int((front & in_bounds & (depth < args.min_depth)).sum())
    print(f"view       : {args.view} (yaw {yaw:+.1f} deg, pitch {pitch:+.1f} deg)")
    print(f"dropped    : {n_self} points under {args.min_depth}m (drone self-occlusion)")
    print(f"image      : {image_path}")
    print(f"extrinsic  : {extrinsic_path}")
    print(f"cloud      : {len(xyz)} points -> {n_hit} land in frame")
    if n_hit:
        print(f"depth range: [{depth[mask].min():.2f}, {depth[mask].max():.2f}] m, "
              f"median {np.median(depth[mask]):.2f} m")

    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    idx = np.where(mask)[0]
    if len(idx) > args.max_points:
        idx = np.random.default_rng(0).choice(idx, args.max_points, replace=False)

    us, vs, ds = u[idx], v[idx], depth[idx]
    if len(ds):
        dmin, dmax = ds.min(), ds.max()
        colors = (depth_ramp((ds - dmin) / max(1e-6, dmax - dmin)) * 255).astype(int)
        r = args.point_radius
        for uu, vv, (cr, cg, cb) in zip(us, vs, colors):
            draw.ellipse([uu - r, vv - r, uu + r, vv + r], fill=(int(cr), int(cg), int(cb)))

    out_path = args.out or os.path.join(
        HERE, f"project_view{args.view:02d}_{os.path.splitext(os.path.basename(extrinsic_path))[0]}.png")
    img.save(out_path)
    print(f"wrote      : {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='dataset dir holding views/ and cloud.pcd, '
                                                        'e.g. data/calib_data_yard_1')
    p.add_argument('--view', type=int, default=5, help='view index (0-based)')
    p.add_argument('--extrinsic', help='pose file to project with; defaults to that view\'s seed guess')
    p.add_argument('--out', help='output image path')
    p.add_argument('--near-clip', type=float, default=0.1, help='behind-camera guard on camera-frame Z, not real range')
    p.add_argument('--min-depth', type=float, default=0.3, help='drop points closer than this range -- the drone body self-occludes here')
    p.add_argument('--max-depth', type=float, default=0.0, help='drop points farther than this (0 = no limit)')
    p.add_argument('--point-radius', type=float, default=2.0)
    p.add_argument('--max-points', type=int, default=60000, help='subsample cap for draw speed')
    args = p.parse_args()
    project(args)


if __name__ == '__main__':
    main()
