#!/usr/bin/env python3
"""Two ways to turn an image bbox into the 3D points of the object inside it.

Both answer the same question -- which LiDAR returns inside this box belong to the
object, and where are its top / centre / bottom -- and they disagree in exactly one
place: how the object is separated from what surrounds it.

  'fast'     Separate along RANGE only. Sort the in-box ranges, split wherever there
             is a jump bigger than `gap`, keep the nearest run with real support.
             Cheap (one sort of a few hundred values) and right whenever the object
             stands clear of its background in depth.

  'cluster'  Separate in 3D. Narrow to the dominant depth peak (histogram mode + a
             MAD window), then split what is left into 6-connected voxel components
             and pick the object among them by size, overruled by image centrality
             when the biggest component is clearly off-centre -- that combination is
             what says "a wall leaked into the box" rather than "the object is not
             perfectly centred". This is the pipeline from bench_bbox_to_tree.py /
             interactive_bbox_match.py.

The two stages of 'cluster' are a pair, not a filter plus a refinement: without the
depth gate the voxel components chain through the floor and merge the whole box into
one blob (measured: 31547 of 33493 points). Do not use the components alone.

Keypoint extraction is deliberately SHARED, so a difference between the methods is a
difference in point selection and nothing else.
"""

import math

import numpy as np


class PinholeView:
    """Minimal stand-in for live_view_overlay.View, for tools that hold a raw pose.

    live_view_overlay.View already exposes fx/fy/cx/cy/size/lidar_to_cv/unproject, so
    it satisfies the same duck type and both work with everything below.
    """

    def __init__(self, fx, fy, cx, cy, size, r_cv_lidar, t_cv_lidar):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.size = size
        r = np.asarray(r_cv_lidar, dtype=np.float64)
        # p_cv = R p_lidar + t, and this code works camera-relative on LiDAR axes:
        # p_cv = R (p_lidar - origin), so lidar_to_cv (a right-multiply) is R.T.
        self.lidar_to_cv = np.ascontiguousarray(r.T, dtype=np.float32)
        self.axis_lidar = np.ascontiguousarray(self.lidar_to_cv[:, 2])
        self.camera_origin = (-r.T @ np.asarray(t_cv_lidar, dtype=np.float64)).astype(np.float32)

    def unproject(self, u, v, rng, cam_origin):
        rays = np.stack([(u - self.cx) / self.fx,
                         (v - self.cy) / self.fy,
                         np.ones_like(u)], axis=-1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        return (rays * np.asarray(rng)[:, None]) @ self.lidar_to_cv.T + cam_origin


# ------------------------------------------------------------------ separation

def select_nearest_run(rb, gap=0.5, min_points=5):
    """'fast': indices of the nearest range run with enough support."""
    order = np.argsort(rb)
    cuts = np.flatnonzero(np.diff(rb[order]) > gap) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [len(rb)]))
    need = max(min_points, int(0.15 * len(rb)))
    for k in range(len(starts)):
        if ends[k] - starts[k] >= need:
            return order[starts[k]:ends[k]]
    return None


def depth_mode_gate(depth, k_mad=3.0, n_bins=40):
    """Keep the dominant depth peak: tallest histogram bin, widened by a MAD window."""
    if len(depth) == 0:
        return np.zeros(0, dtype=bool)
    # np.bincount over scaled indices rather than np.histogram: identical uniform
    # binning, without histogram's edge-array machinery, which costs more than the
    # connected components that follow it.
    lo, hi = float(depth.min()), float(depth.max())
    if hi <= lo:
        return np.ones(len(depth), dtype=bool)
    scale = n_bins / (hi - lo)
    idx = ((depth - lo) * scale).astype(np.int32)
    np.clip(idx, 0, n_bins - 1, out=idx)
    peak = int(np.argmax(np.bincount(idx, minlength=n_bins)))
    in_peak = idx == peak
    if not in_peak.any():
        in_peak = np.abs(depth - np.median(depth)) < 1e-6
    peak_med = np.median(depth[in_peak])
    mad = np.median(np.abs(depth[in_peak] - peak_med)) + 1e-6
    return np.abs(depth - peak_med) < max(k_mad * 1.4826 * mad, 0.15)


def _voxel_labels(points, voxel):
    """6-connected components over a voxel grid -> per-point label.

    Neighbours are found with searchsorted over an integer encoding of the occupied
    voxels rather than a dict lookup per voxel per offset, which keeps the expensive
    part in numpy; only the union-find over the discovered pairs stays in Python, and
    there are far fewer pairs than lookups.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    keys -= keys.min(axis=0)                       # non-negative, so the encoding is monotone
    span = keys.max(axis=0) + 3                    # +2 for the -1/+1 neighbour reach
    if np.prod(span.astype(float)) > 4e18:         # pathological spread: give up cleanly
        return np.zeros(len(points), dtype=np.int64)
    code = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]

    uniq, inverse = np.unique(code, return_inverse=True)
    n = len(uniq)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    strides = (span[1] * span[2], span[2], 1)
    for s in strides:                              # +1 along each axis; -1 is its mirror
        pos = np.searchsorted(uniq, uniq + s)
        pos = np.minimum(pos, n - 1)
        hit = np.flatnonzero(uniq[pos] == uniq + s)
        for a, b in zip(hit, pos[hit]):
            ra, rb_ = find(int(a)), find(int(b))
            if ra != rb_:
                parent[ra] = rb_

    roots = np.array([find(i) for i in range(n)])
    return roots[inverse]


def cluster_stages(pts3, ub, vb, rb, box, voxel=0.08, min_size=15,
                   merge_radius_m=0.3, merge_size_ratio=3.0, min_points=5):
    """'cluster', with the intermediate sets kept: (gated indices, object indices).

    Split out so a viewer can show what each stage rejected -- which is the whole
    point of the staged rendering: a box whose "object" is really the wall behind it
    looks obviously wrong the moment you see the gate keep the wall.
    """
    if len(rb) < min_points:
        return None, None
    idx_gate = np.flatnonzero(depth_mode_gate(rb))
    if len(idx_gate) < min_points:
        return None, None

    labels = _voxel_labels(pts3[idx_gate], voxel)
    x1, y1, x2, y2 = box[:4]
    bcx, bcy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_diag = 0.5 * math.hypot(x2 - x1, y2 - y1)

    scored = []
    for lab in np.unique(labels):
        members = idx_gate[labels == lab]
        if len(members) < min_size:
            continue
        # Centrality is measured on the pixels we already have -- no reprojection.
        d_px = math.hypot(np.median(ub[members]) - bcx, np.median(vb[members]) - bcy)
        scored.append((len(members), d_px, members, np.median(pts3[members], axis=0)))
    if not scored:
        return idx_gate, idx_gate                  # nothing big enough: keep the gate's word

    scored.sort(key=lambda e: e[0], reverse=True)
    best_size, best_d, best_idx, best_c = scored[0]
    if best_d > 0.5 * half_diag:
        central = [e for e in scored[1:] if e[1] < 0.3 * half_diag]
        if central:
            best_size, best_d, best_idx, best_c = min(central, key=lambda e: e[1])

    merged = [best_idx]
    for size, _, members, c in scored:
        if members is best_idx:
            continue
        if size <= merge_size_ratio * best_size and np.linalg.norm(c - best_c) < merge_radius_m:
            merged.append(members)
    return idx_gate, np.concatenate(merged)


def select_cluster(pts3, ub, vb, rb, box, **kw):
    """'cluster': depth-mode gate, then 3D components, then pick the object."""
    _, obj = cluster_stages(pts3, ub, vb, rb, box, **kw)
    return obj


# -------------------------------------------------------------------- anchors

def anchors(pts3, ub, vb, rb, sel, anchor_axis='row'):
    """Top / central / bottom of a selected point set -- shared by both variants."""
    pts, uu, vv, rr = pts3[sel], ub[sel], vb[sel], rb[sel]
    if anchor_axis == 'z':
        # World-up in the LiDAR frame: the right choice once views are pitched.
        key = -pts[:, 2]
    else:
        # Image row: what actually ties top/bottom to the box, and equivalent while
        # the panorama is gravity-locked and views are level.
        key = vv
    lo, hi = np.percentile(key, [10.0, 90.0])
    top, bottom = key <= lo, key >= hi
    return {
        'top': pts[top].mean(axis=0),
        'central': np.median(pts, axis=0),
        'bottom': pts[bottom].mean(axis=0),
        'pixels': (float(uu[top].mean()), float(vv[top].mean()),
                   float(np.median(uu)), float(np.median(vv)),
                   float(uu[bottom].mean()), float(vv[bottom].mean())),
        'range': float(np.median(rr)),
        'n': int(len(rr)),
    }


def match(view, box, u, v, rng, cam_origin, method='fast', shrink=0.6, gap=0.5,
          min_points=5, voxel=0.08, anchor_axis='row'):
    """Bbox -> {top, central, bottom, pixels, range, n} in the LiDAR frame, or None.

    `u`, `v`, `rng` are the projection the caller already computed for drawing; only
    the points inside the box are ever unprojected back to 3D.
    """
    x1, y1, x2, y2 = box[:4]
    mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    hw, hh = (x2 - x1) * 0.5 * shrink, (y2 - y1) * 0.5 * shrink
    inside = (u >= mx - hw) & (u <= mx + hw) & (v >= my - hh) & (v <= my + hh)
    if int(inside.sum()) < min_points:
        return None

    ub, vb, rb = u[inside], v[inside], rng[inside]
    if method == 'fast':
        sel = select_nearest_run(rb, gap, min_points)
        if sel is None:
            return None
        pts3 = view.unproject(ub[sel], vb[sel], rb[sel], cam_origin)
        return anchors(pts3, ub[sel], vb[sel], rb[sel], np.arange(len(sel)), anchor_axis)

    # 'cluster' needs 3D for the components, so everything in the box is unprojected.
    pts3 = view.unproject(ub, vb, rb, cam_origin)
    sel = select_cluster(pts3, ub, vb, rb, (x1, y1, x2, y2), voxel=voxel,
                         min_points=min_points)
    if sel is None or len(sel) < min_points:
        return None
    return anchors(pts3, ub, vb, rb, sel, anchor_axis)
