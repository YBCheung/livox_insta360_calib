#!/usr/bin/env python3
"""Visualize the bbox -> tree-cluster matching pipeline from bench_bbox_to_tree.py.

Draws, on top of the view image: the fixed bbox, all frustum candidates (dim),
the points kept after the depth-mode gate (mid), the final cluster used for the
keypoints (bright), and the computed center/top/bottom points -- so you can
eyeball how well the separation is rejecting background/edge clutter before
trusting the numbers.

    python3 visualize_bbox_match.py --data-dir data/calib_data_yard_2 --view 0 \
        --out /tmp/bbox_match.png

render_match_overlay() below is reused by interactive_bbox_match.py so the
mouse-driven tool and this static CLI stay in sync.
"""
import argparse
import glob
import math
import os

import numpy as np
from PIL import Image, ImageDraw

from project_pointcloud_to_view import (load_pcd_xyzi, load_extrinsic, load_intrinsics,
                                        apply_gravity_correction)
from bench_bbox_to_tree import project_all, depth_mode_gate, pick_object_cluster


def find_extrinsic(data_dir, view_idx, explicit=None):
    if explicit:
        return explicit
    guess_dir = os.path.join(os.path.dirname(data_dir), 'guesses')
    matches = glob.glob(os.path.join(guess_dir, f'guess_{view_idx:02d}_*.txt'))
    if not matches:
        matches = glob.glob(os.path.join(data_dir, 'guesses', f'guess_{view_idx:02d}_*.txt'))
    if not matches:
        raise SystemExit(f"no seed guess found for view {view_idx}")
    return matches[0]


def render_match_overlay(base_img, xyz, u, v, Z, depth, R, t, fx, fy, cx, cy, bbox, up_axis,
                          point_radius=1.6):
    """Run the bbox->cluster match against precomputed full-cloud projection
    (u, v, Z, depth) and draw the result over a copy of base_img. Returns
    (annotated_image, info) where info holds counts/coords or None if the
    bbox caught nothing."""
    x1, y1, x2, y2 = bbox
    in_box = (u >= x1) & (u < x2) & (v >= y1) & (v < y2) & (Z > 0.1)
    candidates = xyz[in_box]
    cu, cv = u[in_box], v[in_box]
    cdepth = depth[in_box]

    gate = depth_mode_gate(cdepth)
    gated = candidates[gate]
    gu, gv = cu[gate], cv[gate]

    bbox_center = ((x1 + x2) / 2, (y1 + y2) / 2)
    bbox_half_diag = 0.5 * math.hypot(x2 - x1, y2 - y1)

    def reproject(p):
        cam_pt = R @ p + t
        return fx * cam_pt[0] / cam_pt[2] + cx, fy * cam_pt[1] / cam_pt[2] + cy

    cluster = pick_object_cluster(gated, reproject, bbox_center, bbox_half_diag)
    if len(cluster):
        cam = (R @ cluster.T).T + t
        clu = fx * cam[:, 0] / cam[:, 2] + cx
        clv = fy * cam[:, 1] / cam[:, 2] + cy
    else:
        clu = clv = np.zeros(0)

    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 230, 0), width=2)

    r = point_radius
    for uu, vv in zip(cu, cv):
        draw.ellipse([uu - r, vv - r, uu + r, vv + r], fill=(90, 90, 90))
    for uu, vv in zip(gu, gv):
        draw.ellipse([uu - r, vv - r, uu + r, vv + r], fill=(255, 140, 0))
    for uu, vv in zip(clu, clv):
        draw.ellipse([uu - r * 1.3, vv - r * 1.3, uu + r * 1.3, vv + r * 1.3], fill=(50, 230, 90))

    info = None
    if len(cluster):
        up = cluster[:, up_axis]
        center_xyz = np.median(cluster, axis=0)
        bottom_pt = cluster[np.argmin(up)]
        top_pt = cluster[np.argmax(up)]
        for label, pt, color in (('C', center_xyz, (255, 40, 40)),
                                  ('B', bottom_pt, (0, 160, 255)),
                                  ('T', top_pt, (255, 0, 255))):
            cam_pt = R @ pt + t
            pu = fx * cam_pt[0] / cam_pt[2] + cx
            pv = fy * cam_pt[1] / cam_pt[2] + cy
            cr = 6
            draw.ellipse([pu - cr, pv - cr, pu + cr, pv + cr], outline=color, width=3)
            draw.text((pu + cr + 2, pv - cr), label, fill=color)
        info = {
            'n_candidates': len(candidates),
            'n_gated': len(gated),
            'n_cluster': len(cluster),
            'center': center_xyz,
            'bottom': bottom_pt,
            'top': top_pt,
        }

    return img, info


def draw_legend(img):
    draw = ImageDraw.Draw(img)
    legend = [
        ("gray = frustum candidate (pre-gate)", (90, 90, 90)),
        ("orange = kept after depth-mode gate", (255, 140, 0)),
        ("green = final cluster (keypoints computed from this)", (50, 230, 90)),
        ("red/blue/magenta ring = Center / Bottom / Top", (255, 40, 40)),
    ]
    for i, (text, color) in enumerate(legend):
        draw.text((6, 6 + i * 14), text, fill=color)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--view', type=int, default=0)
    p.add_argument('--up-axis', type=int, default=2)
    p.add_argument('--bbox', type=float, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'))
    p.add_argument('--point-radius', type=float, default=1.6)
    p.add_argument('--out', help='output image path')
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

    view_file, yaw, pitch = manifest[args.view]
    extrinsic_path = find_extrinsic(data_dir, args.view, args.extrinsic)
    R, t = load_extrinsic(extrinsic_path)
    R, t = apply_gravity_correction(R, t, args.gravity)

    pts = load_pcd_xyzi(os.path.join(data_dir, 'cloud.pcd'))
    xyz = pts[:, :3].astype(np.float64)

    if args.bbox:
        bbox = tuple(args.bbox)
    else:
        bw, bh = 200, 260
        bbox = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)

    u, v, Z, depth = project_all(xyz, R, t, fx, fy, cx, cy)

    image_path = os.path.join(views_dir, view_file)
    base_img = Image.open(image_path).convert('RGB')
    img, info = render_match_overlay(base_img, xyz, u, v, Z, depth, R, t, fx, fy, cx, cy,
                                      bbox, args.up_axis, args.point_radius)
    draw_legend(img)

    if info:
        print(f"candidates: {info['n_candidates']}  after depth-gate: {info['n_gated']}  "
              f"final cluster: {info['n_cluster']}")
        print(f"center: {np.round(info['center'], 3)}  bottom: {np.round(info['bottom'], 3)}  "
              f"top: {np.round(info['top'], 3)}")
    else:
        print("no cluster matched for this bbox")

    out_path = args.out or os.path.join(
        views_dir, f"bbox_match_view{args.view:02d}.png")
    img.save(out_path)
    print(f"wrote: {out_path}")


if __name__ == '__main__':
    main()
