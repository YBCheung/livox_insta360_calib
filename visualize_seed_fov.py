#!/usr/bin/env python3
"""Render an interactive 3D viewer: point cloud + each view's seed camera FOV.

Self-contained -- writes one .html file (point cloud, camera poses, and the
whole WebGL viewer app baked in) that opens directly in any browser, no
server needed. Re-run any time a dataset changes; it just overwrites the
output file.

    python3 visualize_seed_fov.py --dataset calib_data_yard_1
    python3 visualize_seed_fov.py --dataset calib_data_yard_2 --out yard2.html

Pick "All" in the viewer for a fanned-out overview of all 6 seed FOVs sharing
one rig origin, or select a single view to extend its FOV boundary + a bright
centre beam to real room/yard scale, so it visibly slices through the cloud.
"""
import argparse
import base64
import glob
import json
import os

import numpy as np

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
    vals = []
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if line:
                vals.extend(float(v) for v in line.replace(',', ' ').split())
    arr = np.array(vals, dtype=float).reshape(4, 4)
    return arr[:3, :3], arr[:3, 3]


def fit_ground_plane(xyz, dist_thresh=0.08, iters=2000, seed=0, max_tilt_deg=35.0):
    """RANSAC ground-plane fit, biased toward finding the ground rather than a wall.

    Candidate triplets are drawn only from the lowest ~35% of points (the ground is
    typically the most extensive low structure), and any candidate whose normal
    isn't already roughly vertical is rejected outright -- without this a large
    fence/wall can out-vote the actual ground on raw inlier count.
    """
    rng = np.random.default_rng(seed)
    z_lo = np.percentile(xyz[:, 2], 35)
    pool = xyz[xyz[:, 2] < z_lo]
    cos_min = np.cos(np.radians(max_tilt_deg))

    best_inliers, best_count = None, -1
    for _ in range(iters):
        p0, p1, p2 = pool[rng.choice(len(pool), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if abs(normal[2]) < cos_min:
            continue
        if normal[2] < 0:
            normal = -normal
        dist = np.abs(xyz @ normal - normal @ p0)
        count = int((dist < dist_thresh).sum())
        if count > best_count:
            best_count, best_inliers = count, dist < dist_thresh

    pts = xyz[best_inliers]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, centroid, best_count / len(xyz)


def rotation_aligning(a, b):
    """Shortest-arc rotation matrix taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else _rotation_180(a)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _rotation_180(axis_hint):
    """Rotation by 180deg about any axis perpendicular to axis_hint (a is antiparallel to b)."""
    perp = np.cross(axis_hint, [1.0, 0.0, 0.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(axis_hint, [0.0, 1.0, 0.0])
    perp /= np.linalg.norm(perp)
    return 2 * np.outer(perp, perp) - np.eye(3)


def camera_pose_in_lidar(R, t):
    """R,t map p_cam = R @ p_lidar + t (OpenCV convention). Returns the camera's
    origin and its right/down/forward axes, all expressed in LiDAR coordinates."""
    origin = -R.T @ t
    return origin, R.T[:, 0], R.T[:, 1], R.T[:, 2]


def load_intrinsics(views_dir):
    vals, manifest = {}, []
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


def build_payload(dataset, n_points):
    data_dir = os.path.join(HERE, 'data', dataset)
    views_dir = os.path.join(data_dir, 'views')
    if not os.path.isdir(views_dir):
        raise SystemExit(f"no such dataset: {views_dir}")
    intr, manifest = load_intrinsics(views_dir)

    pts = load_pcd_xyzi(os.path.join(data_dir, 'cloud.pcd'))
    xyz = pts[:, :3].astype(np.float32)
    intensity = pts[:, 3].astype(np.float32)

    n = len(xyz)
    if n > n_points:
        idx = np.random.default_rng(0).choice(n, n_points, replace=False)
        xyz = xyz[idx]
        intensity = intensity[idx]
    print(f"points: {len(xyz)} (from {n})")

    xyz_b64 = base64.b64encode(xyz.tobytes()).decode('ascii')
    i_clip = np.clip(intensity, 0, np.percentile(intensity, 99))
    i_norm = (i_clip / max(i_clip.max(), 1e-6) * 255).astype(np.uint8)
    i_b64 = base64.b64encode(i_norm.tobytes()).decode('ascii')

    # Outdoor captures have a long tail of far points (sky/distant trees/noise);
    # framing the camera on raw min/max would zoom out until the near structure
    # is a speck. Use a robust 1st-99th percentile box for framing/coloring
    # instead -- all points are still uploaded and rendered either way.
    robust_min = np.percentile(xyz, 1, axis=0).tolist()
    robust_max = np.percentile(xyz, 99, axis=0).tolist()
    print(f"robust bounds (p1-p99): min={np.round(robust_min,2)} max={np.round(robust_max,2)}")

    normal, centroid, inlier_frac = fit_ground_plane(xyz.astype(np.float64))
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal @ [0, 0, 1], -1, 1))))
    level_R = rotation_aligning(normal, np.array([0.0, 0.0, 1.0]))
    u = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    xyz_leveled = xyz.astype(np.float64) @ level_R.T
    print(f"ground plane: tilt={tilt_deg:.2f}deg off level, inliers={inlier_frac*100:.1f}%")

    views = []
    for i, (view_file, yaw, pitch) in enumerate(manifest):
        pattern = os.path.join(HERE, 'data', 'guesses', f'guess_{i:02d}_*.txt')
        matches = glob.glob(pattern)
        if not matches:
            raise SystemExit(f"no guess file for view {i}: {pattern}")
        R, t = load_extrinsic(matches[0])
        origin, right, down, forward = camera_pose_in_lidar(R, t)
        views.append({
            'idx': i, 'yaw': yaw, 'pitch': pitch, 'fov_deg': intr['fov_deg'],
            'view_file': view_file,
            'seed': {'origin': origin.tolist(), 'right': right.tolist(),
                      'down': down.tolist(), 'forward': forward.tolist()},
        })
        print(f"view {i:02d} yaw={yaw:6.1f}  origin={np.round(origin,3)}  forward={np.round(forward,3)}")

    return {
        'dataset': dataset,
        'point_count': len(xyz),
        'xyz_b64': xyz_b64,
        'intensity_b64': i_b64,
        'robust_min': robust_min,
        'robust_max': robust_max,
        'views': views,
        'level': {
            'rotation': level_R.tolist(),
            'tilt_deg': tilt_deg,
            'inlier_pct': inlier_frac * 100.0,
            'ground_centroid': centroid.tolist(),
            'ground_u': u.tolist(),
            'ground_v': v.tolist(),
            'z_min': float(xyz_leveled[:, 2].min()),
            'z_max': float(xyz_leveled[:, 2].max()),
            'robust_min': np.percentile(xyz_leveled, 1, axis=0).tolist(),
            'robust_max': np.percentile(xyz_leveled, 99, axis=0).tolist(),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='calib_data_yard_1',
                     help='folder under data/ holding views/ and cloud.pcd')
    ap.add_argument('--n-points', type=int, default=280000, help='point-cloud subsample cap')
    ap.add_argument('--out', help='output .html path (default: seed_fov_<dataset>.html next to this script)')
    args = ap.parse_args()

    payload = build_payload(args.dataset, args.n_points)
    html = TEMPLATE.replace('__DATA_JSON__', json.dumps(payload))

    out_path = args.out or os.path.join(HERE, f'seed_fov_{args.dataset}.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"\nwrote {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")
    print(f"open it directly in a browser -- fully self-contained, no server needed")


TEMPLATE = '<title>Seed FOV Inspector</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n<style>\n  :root {\n    --bg: #0a0d13;\n    --bg-1: #0e131b;\n    --panel: rgba(19, 25, 35, 0.82);\n    --panel-border: #26303f;\n    --text: #e7ebf1;\n    --text-dim: #8b95a7;\n    --text-faint: #566072;\n    --seed: #52e0c4;\n    --seed-dim: #2a5f56;\n    --seq: #b98cff;\n    --focus: #7fb7ff;\n  }\n  * { box-sizing: border-box; }\n  html, body {\n    margin: 0; padding: 0; width: 100%; height: 100%;\n    background: var(--bg); color: var(--text);\n    font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;\n    overflow: hidden;\n  }\n  #gl { position: fixed; inset: 0; display: block; touch-action: none; cursor: grab; }\n  #gl.dragging { cursor: grabbing; }\n\n  .hud {\n    position: fixed;\n    background: var(--panel);\n    border: 1px solid var(--panel-border);\n    border-radius: 10px;\n    backdrop-filter: blur(10px);\n    -webkit-backdrop-filter: blur(10px);\n    box-shadow: 0 8px 24px rgba(0,0,0,0.35);\n  }\n\n  .panel-left { top: 16px; left: 16px; width: 300px; padding: 16px 18px; max-height: calc(100vh - 32px); overflow-y: auto; }\n  .panel-left h1 { font-size: 15px; font-weight: 600; margin: 0 0 3px; letter-spacing: 0.01em; text-wrap: balance; }\n  .panel-left .sub { font-size: 12px; color: var(--text-dim); line-height: 1.5; margin: 0 0 14px; }\n  .panel-left .sub code { font-family: "IBM Plex Mono", ui-monospace, monospace; color: var(--seed); font-size: 11px; }\n\n  .section-label { font-size: 10px; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-faint); margin: 14px 0 8px; }\n  .section-label:first-of-type { margin-top: 0; }\n\n  .legend { display: flex; flex-direction: column; gap: 7px; }\n  .legend-row { display: flex; align-items: center; gap: 9px; font-size: 12.5px; color: var(--text-dim); }\n  .swatch { width: 11px; height: 11px; border-radius: 3px; flex: none; }\n  .swatch.seed { background: var(--seed); box-shadow: 0 0 8px rgba(82,224,196,0.5); }\n  .swatch.seq { background: var(--seq); box-shadow: 0 0 8px rgba(185,140,255,0.5); }\n  .ramp { width: 11px; height: 11px; border-radius: 3px; flex: none; background: linear-gradient(90deg, #0f1729, #1a7385, #8cd94f, #f9bf27, #fa5940); }\n\n  table.views { width: 100%; border-collapse: collapse; font-size: 12px; }\n  table.views th { text-align: right; font-weight: 500; color: var(--text-faint); font-size: 10px; letter-spacing: 0.04em; text-transform: uppercase; padding: 0 0 6px; border-bottom: 1px solid var(--panel-border); }\n  table.views th:first-child, table.views td:first-child { text-align: left; }\n  table.views td { padding: 5px 0; border-bottom: 1px solid rgba(38,48,63,0.5); font-variant-numeric: tabular-nums; color: var(--text-dim); }\n  table.views td.num { text-align: right; font-family: "IBM Plex Mono", ui-monospace, monospace; }\n  table.views tr { cursor: pointer; }\n  table.views tr:hover td { color: var(--focus); }\n  table.views tr.active td { color: var(--seed); }\n\n  .panel-controls { top: 16px; right: 16px; width: 226px; padding: 14px 16px; }\n  .ctrl-group { margin-bottom: 14px; }\n  .ctrl-group:last-child { margin-bottom: 0; }\n  .toggle-row { display: flex; align-items: center; justify-content: space-between; font-size: 12.5px; color: var(--text); padding: 5px 0; cursor: pointer; user-select: none; }\n  .toggle-row input { accent-color: var(--focus); width: 14px; height: 14px; cursor: pointer; }\n  .seg { display: flex; background: var(--bg-1); border: 1px solid var(--panel-border); border-radius: 7px; padding: 2px; gap: 2px; }\n  .seg button { flex: 1; background: transparent; border: none; color: var(--text-dim); font-family: inherit; font-size: 11.5px; padding: 6px 0; border-radius: 5px; cursor: pointer; transition: background 0.12s, color 0.12s; }\n  .seg button.active { background: var(--focus); color: #0a0d13; font-weight: 600; }\n  .seg.wrap { flex-wrap: wrap; }\n  .seg.wrap button { flex: 1 1 40px; min-width: 40px; }\n  input[type="range"] { width: 100%; accent-color: var(--focus); }\n  .range-label { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-faint); margin-bottom: 4px; }\n  button.reset { width: 100%; background: var(--bg-1); border: 1px solid var(--panel-border); color: var(--text); font-family: inherit; font-size: 12px; padding: 8px 0; border-radius: 7px; cursor: pointer; margin-top: 2px; }\n  button.reset:hover { border-color: var(--focus); color: var(--focus); }\n  button:focus-visible, input:focus-visible, .toggle-row:focus-within { outline: 2px solid var(--focus); outline-offset: 2px; }\n\n  .inspect-note { font-size: 11px; line-height: 1.5; color: var(--text-faint); margin-top: 8px; }\n\n  .hint { position: fixed; bottom: 16px; left: 16px; font-size: 11px; color: var(--text-faint); font-family: "IBM Plex Mono", ui-monospace, monospace; letter-spacing: 0.01em; padding: 8px 12px; }\n\n  .vlabel { position: fixed; pointer-events: none; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; font-weight: 500; color: var(--seed); background: rgba(10,13,19,0.7); padding: 1px 5px; border-radius: 4px; transform: translate(-50%, -140%); white-space: nowrap; border: 1px solid var(--seed-dim); }\n\n  ::-webkit-scrollbar { width: 8px; }\n  ::-webkit-scrollbar-thumb { background: var(--panel-border); border-radius: 4px; }\n  ::-webkit-scrollbar-track { background: transparent; }\n\n  @media (max-width: 720px) {\n    .panel-left { width: calc(100vw - 32px); max-height: 42vh; }\n    .panel-controls { top: auto; bottom: 16px; right: 16px; width: 190px; }\n  }\n</style>\n\n<canvas id="gl"></canvas>\n\n<div class="hud panel-left">\n  <h1>Seed FOV Inspector</h1>\n  <p class="sub">Dataset: <code id="datasetName"></code> &middot; pre-calibration sanity check &mdash; does the hand-measured seed even point at real structure?</p>\n\n  <p class="section-label">Legend</p>\n  <div class="legend">\n    <div class="legend-row"><span class="swatch seed"></span> Seed FOV &mdash; one rig origin, 6 look directions</div>\n    <div class="legend-row"><span class="swatch seq"></span> View order 0&rarr;5 (&amp; floor grid)</div>\n    <div class="legend-row"><span class="ramp"></span> Point height (low &rarr; high)</div>\n  </div>\n\n  <p class="section-label">Views <span style="text-transform:none;font-weight:400;color:var(--text-faint)">(click to inspect)</span></p>\n  <table class="views" id="viewTable"></table>\n</div>\n\n<div class="hud panel-controls">\n  <div class="ctrl-group">\n    <p class="section-label">Inspect view FOV</p>\n    <div class="seg wrap" id="viewSeg">\n      <button data-view="-1" class="active">All</button>\n      <button data-view="0">V0</button>\n      <button data-view="1">V1</button>\n      <button data-view="2">V2</button>\n      <button data-view="3">V3</button>\n      <button data-view="4">V4</button>\n      <button data-view="5">V5</button>\n    </div>\n    <p class="inspect-note" id="inspectNote">Showing all 6 as short markers. Pick one to extend its FOV boundary + centre beam through the cloud.</p>\n  </div>\n  <div class="ctrl-group">\n    <p class="section-label">Point color</p>\n    <div class="seg" id="colorSeg">\n      <button data-mode="height" class="active">Height</button>\n      <button data-mode="intensity">Intensity</button>\n    </div>\n  </div>\n  <div class="ctrl-group">\n    <div class="range-label"><span>Point size</span></div>\n    <input type="range" id="ptSize" min="1" max="6" step="0.1" value="2.0">\n  </div>\n  <div class="ctrl-group">\n    <label class="toggle-row"><span>View labels</span><input type="checkbox" id="showLabels" checked></label>\n    <label class="toggle-row"><span>Origin axes</span><input type="checkbox" id="showAxes" checked></label>\n    <label class="toggle-row"><span>Sequence ring</span><input type="checkbox" id="showRing" checked></label>\n    <label class="toggle-row"><span>Floor grid</span><input type="checkbox" id="showFloor" checked></label>\n  </div>\n  <div class="ctrl-group" id="levelGroup">\n    <label class="toggle-row"><span>Level to gravity</span><input type="checkbox" id="showLevel"></label>\n    <p class="inspect-note" id="levelNote"></p>\n  </div>\n  <button class="reset" id="resetView">Reset view</button>\n</div>\n\n<div class="hint">drag to orbit &nbsp;&middot;&nbsp; scroll to zoom &nbsp;&middot;&nbsp; shift+drag to pan</div>\n<div id="labelLayer"></div>\n\n<script id="payload" type="application/json">__DATA_JSON__</script>\n<script>\n(function () {\n  "use strict";\n  const payload = JSON.parse(document.getElementById("payload").textContent);\n\n  function b64ToFloat32(b64) {\n    const bin = atob(b64);\n    const buf = new ArrayBuffer(bin.length);\n    const view = new Uint8Array(buf);\n    for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);\n    return new Float32Array(buf);\n  }\n  function b64ToUint8(b64) {\n    const bin = atob(b64);\n    const arr = new Uint8Array(bin.length);\n    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);\n    return arr;\n  }\n\n  const positions = b64ToFloat32(payload.xyz_b64);\n  const intensities = b64ToUint8(payload.intensity_b64);\n  const pointCount = payload.point_count;\n  const views = payload.views;\n  document.getElementById("datasetName").textContent = payload.dataset;\n\n  // ---------------- true bounds (for height colormap) + robust bounds (for framing) ----------------\n  let minZ = Infinity, maxZ = -Infinity;\n  for (let i = 0; i < pointCount; i++) {\n    const z = positions[i * 3 + 2];\n    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;\n  }\n  const rMin = payload.robust_min, rMax = payload.robust_max;\n  const center = [(rMin[0]+rMax[0])/2, (rMin[1]+rMax[1])/2, (rMin[2]+rMax[2])/2];\n  const extent = Math.max(rMax[0]-rMin[0], rMax[1]-rMin[1], rMax[2]-rMin[2]);\n\n  // ---------------- mat4 / vec3 helpers ----------------\n  const M = {\n    multiply(a, b) {\n      const o = new Float32Array(16);\n      for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {\n        o[c*4+r] = a[0*4+r]*b[c*4+0] + a[1*4+r]*b[c*4+1] + a[2*4+r]*b[c*4+2] + a[3*4+r]*b[c*4+3];\n      }\n      return o;\n    },\n    perspective(fovy, aspect, near, far) {\n      const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);\n      const o = new Float32Array(16);\n      o[0]=f/aspect; o[5]=f; o[10]=(far+near)*nf; o[11]=-1; o[14]=2*far*near*nf;\n      return o;\n    },\n    lookAt(eye, target, up) {\n      const z = V.normalize(V.sub(eye, target));\n      const x = V.normalize(V.cross(up, z));\n      const y = V.cross(z, x);\n      const o = new Float32Array(16);\n      o[0]=x[0]; o[1]=y[0]; o[2]=z[0]; o[3]=0;\n      o[4]=x[1]; o[5]=y[1]; o[6]=z[1]; o[7]=0;\n      o[8]=x[2]; o[9]=y[2]; o[10]=z[2]; o[11]=0;\n      o[12]=-V.dot(x,eye); o[13]=-V.dot(y,eye); o[14]=-V.dot(z,eye); o[15]=1;\n      return o;\n    }\n  };\n  const V = {\n    sub(a,b){ return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; },\n    add(a,b){ return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]; },\n    scale(a,s){ return [a[0]*s,a[1]*s,a[2]*s]; },\n    dot(a,b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; },\n    cross(a,b){ return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; },\n    length(a){ return Math.sqrt(V.dot(a,a)); },\n    normalize(a){ const l = V.length(a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; }\n  };\n  function project(mvp, p) {\n    const x = mvp[0]*p[0]+mvp[4]*p[1]+mvp[8]*p[2]+mvp[12];\n    const y = mvp[1]*p[0]+mvp[5]*p[1]+mvp[9]*p[2]+mvp[13];\n    const w = mvp[3]*p[0]+mvp[7]*p[1]+mvp[11]*p[2]+mvp[15];\n    return [x / w, y / w, w];\n  }\n\n  const cam = { target: center.slice(), yaw: -2.4, pitch: 0.55, dist: extent * 1.15 };\n  const camInitial = JSON.parse(JSON.stringify(cam));\n  function eyeFromCam() {\n    const cy = Math.cos(cam.pitch), sy = Math.sin(cam.pitch);\n    const cx = Math.cos(cam.yaw), sx = Math.sin(cam.yaw);\n    return [cam.target[0]+cam.dist*cy*cx, cam.target[1]+cam.dist*cy*sx, cam.target[2]+cam.dist*sy];\n  }\n\n  const canvas = document.getElementById("gl");\n  const gl = canvas.getContext("webgl", { antialias: true });\n  if (!gl) { document.body.innerHTML = "<p style=\'color:#e7ebf1;padding:24px;font-family:sans-serif\'>WebGL is not available in this browser.</p>"; return; }\n\n  function compile(type, src) {\n    const s = gl.createShader(type);\n    gl.shaderSource(s, src); gl.compileShader(s);\n    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s));\n    return s;\n  }\n  function program(vsSrc, fsSrc) {\n    const p = gl.createProgram();\n    gl.attachShader(p, compile(gl.VERTEX_SHADER, vsSrc));\n    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));\n    gl.linkProgram(p);\n    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(p));\n    return p;\n  }\n\n  const pointVS = `\n    attribute vec3 aPos;\n    attribute float aIntensity;\n    uniform mat4 uMVP;\n    uniform mat3 uWorld;\n    uniform float uPointSize;\n    uniform float uMinH, uMaxH;\n    uniform float uColorMode;\n    varying vec3 vColor;\n    vec3 ramp(float t) {\n      t = clamp(t, 0.0, 1.0);\n      vec3 c0 = vec3(0.059, 0.090, 0.161);\n      vec3 c1 = vec3(0.102, 0.451, 0.522);\n      vec3 c2 = vec3(0.549, 0.851, 0.310);\n      vec3 c3 = vec3(0.976, 0.749, 0.153);\n      vec3 c4 = vec3(0.980, 0.349, 0.251);\n      float seg = t * 4.0;\n      if (seg < 1.0) return mix(c0, c1, seg);\n      else if (seg < 2.0) return mix(c1, c2, seg - 1.0);\n      else if (seg < 3.0) return mix(c2, c3, seg - 2.0);\n      return mix(c3, c4, min(seg - 3.0, 1.0));\n    }\n    void main() {\n      vec3 wp = uWorld * aPos;\n      gl_Position = uMVP * vec4(wp, 1.0);\n      float hT = (wp.z - uMinH) / max(uMaxH - uMinH, 0.0001);\n      float iT = aIntensity;\n      vColor = ramp(mix(hT, iT, uColorMode));\n      gl_PointSize = uPointSize;\n    }`;\n  const pointFS = `\n    precision mediump float;\n    varying vec3 vColor;\n    void main() {\n      vec2 d = gl_PointCoord - vec2(0.5);\n      float r2 = dot(d, d);\n      if (r2 > 0.25) discard;\n      float edge = smoothstep(0.25, 0.16, r2);\n      gl_FragColor = vec4(vColor, edge);\n    }`;\n  const lineVS = `\n    attribute vec3 aPos;\n    uniform mat4 uMVP;\n    uniform mat3 uWorld;\n    void main() { gl_Position = uMVP * vec4(uWorld * aPos, 1.0); }`;\n  const lineFS = `precision mediump float; uniform vec4 uColor; void main() { gl_FragColor = uColor; }`;\n\n  const pointProg = program(pointVS, pointFS);\n  const lineProg = program(lineVS, lineFS);\n\n  const posBuf = gl.createBuffer();\n  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);\n  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);\n\n  const intBuf = gl.createBuffer();\n  gl.bindBuffer(gl.ARRAY_BUFFER, intBuf);\n  const intF = new Float32Array(pointCount);\n  for (let i = 0; i < pointCount; i++) intF[i] = intensities[i] / 255;\n  gl.bufferData(gl.ARRAY_BUFFER, intF, gl.STATIC_DRAW);\n\n  function makeBuf(arr) {\n    const buf = gl.createBuffer();\n    gl.bindBuffer(gl.ARRAY_BUFFER, buf);\n    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(arr), gl.STATIC_DRAW);\n    return { buf, count: arr.length / 3 };\n  }\n\n  function fovLines(origin, right, down, forward, len, halfFovTan, withRoof, withBeam) {\n    const hw = len * halfFovTan, hh = len * halfFovTan;\n    const base = V.add(origin, V.scale(forward, len));\n    const rS = V.scale(right, hw), dS = V.scale(down, hh);\n    const c0 = V.sub(V.sub(base, rS), dS);\n    const c1 = V.add(V.sub(base, dS), rS);\n    const c2 = V.add(V.add(base, rS), dS);\n    const c3 = V.sub(V.add(base, dS), rS);\n    let pts = [\n      ...origin, ...c0, ...origin, ...c1, ...origin, ...c2, ...origin, ...c3,\n      ...c0, ...c1, ...c1, ...c2, ...c2, ...c3, ...c3, ...c0\n    ];\n    if (withRoof) {\n      const roof = V.sub(base, V.scale(down, hh * 1.55));\n      pts = pts.concat([...c0, ...roof, ...c1, ...roof]);\n    }\n    if (withBeam) pts = pts.concat([...origin, ...base]);\n    return pts;\n  }\n\n  function axisLines(origin, len) {\n    return [\n      origin[0], origin[1], origin[2], origin[0]+len, origin[1], origin[2],\n      origin[0], origin[1], origin[2], origin[0], origin[1]+len, origin[2],\n      origin[0], origin[1], origin[2], origin[0], origin[1], origin[2]+len\n    ];\n  }\n\n  const HALF_FOV_TAN = Math.tan((views[0].fov_deg / 2) * Math.PI / 180);\n  const SHORT_LEN = extent * 0.06;\n  const LONG_LEN = extent * 0.55;\n\n  function buildOverviewBuffer() {\n    let all = [];\n    for (const v of views) all = all.concat(fovLines(v.seed.origin, v.seed.right, v.seed.down, v.seed.forward, SHORT_LEN, HALF_FOV_TAN, true, false));\n    return makeBuf(all);\n  }\n  const seedOverview = buildOverviewBuffer();\n\n  function buildInspectBuffer(viewIdx) {\n    if (viewIdx < 0) return makeBuf([]);\n    const v = views[viewIdx].seed;\n    return makeBuf(fovLines(v.origin, v.right, v.down, v.forward, LONG_LEN, HALF_FOV_TAN, true, true));\n  }\n  let selectedView = -1;\n  let seedInspect = buildInspectBuffer(-1);\n\n  function buildSequenceRing() {\n    const radius = SHORT_LEN * 1.9;\n    const n = views.length;\n    const pts = [];\n    for (let i = 0; i < n; i++) {\n      const o = views[i].seed.origin;\n      const f0 = views[i].seed.forward;\n      const f1 = views[(i + 1) % n].seed.forward;\n      const p0 = V.add(o, V.scale(f0, radius));\n      const p1 = V.add(o, V.scale(f1, radius));\n      pts.push(...p0, ...p1);\n      const seg = V.sub(p1, p0);\n      const segLen = V.length(seg);\n      const dir = V.normalize(seg);\n      let perp = V.cross([0, 0, 1], dir);\n      if (V.length(perp) < 1e-6) perp = V.cross([1, 0, 0], dir);\n      perp = V.normalize(perp);\n      const tip = V.add(p0, V.scale(dir, segLen * 0.86));\n      const wing = SHORT_LEN * 0.32;\n      const back = V.sub(tip, V.scale(dir, wing));\n      const a1 = V.add(back, V.scale(perp, wing * 0.55));\n      const a2 = V.sub(back, V.scale(perp, wing * 0.55));\n      pts.push(...tip, ...a1, ...tip, ...a2);\n    }\n    return makeBuf(pts);\n  }\n  const sequenceRing = buildSequenceRing();\n\n  // Built directly on the RANSAC-fitted ground plane (centroid + in-plane basis\n  // u,v), not a raw-Z bounding-box slice -- so that applying the SAME leveling\n  // rotation as everything else below makes this grid come out exactly flat.\n  function buildFloorGrid() {\n    const lvl = payload.level;\n    const c = lvl.ground_centroid, u = lvl.ground_u, v = lvl.ground_v;\n    const half = extent * 0.55;\n    const n = 12;\n    const pts = [];\n    for (let i = 0; i <= n; i++) {\n      const s = -half + (2 * half) * i / n;\n      const p0 = V.add(c, V.add(V.scale(u, s), V.scale(v, -half)));\n      const p1 = V.add(c, V.add(V.scale(u, s), V.scale(v, half)));\n      pts.push(...p0, ...p1);\n    }\n    for (let i = 0; i <= n; i++) {\n      const t = -half + (2 * half) * i / n;\n      const p0 = V.add(c, V.add(V.scale(u, -half), V.scale(v, t)));\n      const p1 = V.add(c, V.add(V.scale(u, half), V.scale(v, t)));\n      pts.push(...p0, ...p1);\n    }\n    return makeBuf(pts);\n  }\n  const floorGrid = buildFloorGrid();\n  const axisBuf = makeBuf(axisLines([0, 0, 0], extent * 0.05));\n\n  const IDENTITY_MAT3 = new Float32Array([1,0,0, 0,1,0, 0,0,1]);\n  const R3 = payload.level.rotation;\n  const LEVEL_MAT3 = new Float32Array([R3[0][0],R3[1][0],R3[2][0], R3[0][1],R3[1][1],R3[2][1], R3[0][2],R3[1][2],R3[2][2]]);\n  function mat3xVec3(m, p) {\n    return [m[0]*p[0]+m[3]*p[1]+m[6]*p[2], m[1]*p[0]+m[4]*p[1]+m[7]*p[2], m[2]*p[0]+m[5]*p[1]+m[8]*p[2]];\n  }\n\n  const state = { colorMode: 0, ptSize: 2.0, showLabels: true, showAxes: true, showRing: true, showFloor: true, level: false };\n\n  const lvl = payload.level;\n  document.getElementById("levelNote").textContent =\n    `Ground-plane RANSAC: ${lvl.tilt_deg.toFixed(1)}° off level, ${lvl.inlier_pct.toFixed(0)}% inliers. ` +\n    (lvl.tilt_deg < 1.0 ? "Already near level." : "Rotates points + all FOVs together; the floor grid is built on the fitted plane, so it should read flat once this is on.");\n  document.getElementById("showLevel").addEventListener("change", (e) => {\n    state.level = e.target.checked;\n    const mat = state.level ? LEVEL_MAT3 : IDENTITY_MAT3;\n    cam.target = mat3xVec3(mat, center);\n  });\n\n  document.getElementById("colorSeg").addEventListener("click", (e) => {\n    const btn = e.target.closest("button"); if (!btn) return;\n    document.querySelectorAll("#colorSeg button").forEach(b => b.classList.remove("active"));\n    btn.classList.add("active");\n    state.colorMode = btn.dataset.mode === "intensity" ? 1 : 0;\n  });\n  document.getElementById("ptSize").addEventListener("input", (e) => { state.ptSize = parseFloat(e.target.value); });\n  document.getElementById("showLabels").addEventListener("change", (e) => { state.showLabels = e.target.checked; syncLabels(); });\n  document.getElementById("showAxes").addEventListener("change", (e) => { state.showAxes = e.target.checked; });\n  document.getElementById("showRing").addEventListener("change", (e) => { state.showRing = e.target.checked; });\n  document.getElementById("showFloor").addEventListener("change", (e) => { state.showFloor = e.target.checked; });\n  document.getElementById("resetView").addEventListener("click", () => {\n    Object.assign(cam, JSON.parse(JSON.stringify(camInitial)));\n    cam.target = mat3xVec3(state.level ? LEVEL_MAT3 : IDENTITY_MAT3, center);\n  });\n\n  const viewNotes = ["Showing all 6 as short markers. Pick one to extend its FOV boundary + centre beam through the cloud."];\n  for (const v of views) viewNotes.push(`View ${v.idx} (yaw ${v.yaw.toFixed(0)}°): FOV boundary + centre beam extended through the cloud (${v.view_file}).`);\n\n  function selectView(idx) {\n    selectedView = idx;\n    document.querySelectorAll("#viewSeg button").forEach(b => b.classList.toggle("active", parseInt(b.dataset.view, 10) === idx));\n    document.querySelectorAll("#viewTable tr[data-view]").forEach(tr => tr.classList.toggle("active", parseInt(tr.dataset.view, 10) === idx));\n    seedInspect = buildInspectBuffer(idx);\n    document.getElementById("inspectNote").textContent = viewNotes[idx + 1];\n    syncLabels();\n  }\n  document.getElementById("viewSeg").addEventListener("click", (e) => {\n    const btn = e.target.closest("button"); if (!btn) return;\n    selectView(parseInt(btn.dataset.view, 10));\n  });\n\n  let dragging = false, panning = false, lastX = 0, lastY = 0;\n  canvas.addEventListener("pointerdown", (e) => {\n    dragging = true; panning = e.shiftKey || e.button === 2;\n    lastX = e.clientX; lastY = e.clientY;\n    canvas.classList.add("dragging");\n    canvas.setPointerCapture(e.pointerId);\n  });\n  canvas.addEventListener("pointerup", () => { dragging = false; canvas.classList.remove("dragging"); });\n  canvas.addEventListener("pointercancel", () => { dragging = false; canvas.classList.remove("dragging"); });\n  canvas.addEventListener("contextmenu", (e) => e.preventDefault());\n  canvas.addEventListener("pointermove", (e) => {\n    if (!dragging) return;\n    const dx = e.clientX - lastX, dy = e.clientY - lastY;\n    lastX = e.clientX; lastY = e.clientY;\n    if (panning) {\n      const eye = eyeFromCam();\n      const fwd = V.normalize(V.sub(cam.target, eye));\n      const right = V.normalize(V.cross(fwd, [0, 0, 1]));\n      const up = V.cross(right, fwd);\n      const scale = cam.dist * 0.0016;\n      cam.target = V.add(cam.target, V.add(V.scale(right, -dx * scale), V.scale(up, dy * scale)));\n    } else {\n      cam.yaw -= dx * 0.0055;\n      cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy * 0.0055));\n    }\n  }, { passive: true });\n  canvas.addEventListener("wheel", (e) => {\n    e.preventDefault();\n    cam.dist *= Math.exp(e.deltaY * 0.0011);\n    cam.dist = Math.max(extent * 0.03, Math.min(extent * 8, cam.dist));\n  }, { passive: false });\n\n  function resize() {\n    const dpr = Math.min(window.devicePixelRatio || 1, 2);\n    canvas.width = window.innerWidth * dpr;\n    canvas.height = window.innerHeight * dpr;\n    canvas.style.width = window.innerWidth + "px";\n    canvas.style.height = window.innerHeight + "px";\n    gl.viewport(0, 0, canvas.width, canvas.height);\n  }\n  window.addEventListener("resize", resize);\n  resize();\n\n  const labelLayer = document.getElementById("labelLayer");\n  const labelEls = [];\n  views.forEach((v) => {\n    const s = document.createElement("div"); s.className = "vlabel"; s.textContent = "V" + v.idx;\n    labelLayer.appendChild(s);\n    labelEls.push({ el: s, view: v });\n  });\n  function syncLabels() {\n    labelEls.forEach((l, i) => {\n      const relevant = selectedView < 0 || selectedView === i;\n      l.el.style.display = state.showLabels && relevant ? "block" : "none";\n    });\n  }\n  syncLabels();\n\n  const tbl = document.getElementById("viewTable");\n  let rows = \'<tr><th>View</th><th>Yaw</th><th>Pitch</th></tr>\';\n  views.forEach(v => {\n    rows += `<tr data-view="${v.idx}"><td>V${v.idx}</td><td class="num">${v.yaw.toFixed(0)}&deg;</td><td class="num">${v.pitch.toFixed(0)}&deg;</td></tr>`;\n  });\n  tbl.innerHTML = rows;\n  tbl.addEventListener("click", (e) => {\n    const tr = e.target.closest("tr[data-view]"); if (!tr) return;\n    selectView(parseInt(tr.dataset.view, 10));\n  });\n\n  const aPosLoc = gl.getAttribLocation(pointProg, "aPos");\n  const aIntLoc = gl.getAttribLocation(pointProg, "aIntensity");\n  const uMVP = gl.getUniformLocation(pointProg, "uMVP");\n  const uWorld = gl.getUniformLocation(pointProg, "uWorld");\n  const uPointSize = gl.getUniformLocation(pointProg, "uPointSize");\n  const uMinH = gl.getUniformLocation(pointProg, "uMinH");\n  const uMaxH = gl.getUniformLocation(pointProg, "uMaxH");\n  const uColorMode = gl.getUniformLocation(pointProg, "uColorMode");\n\n  const lPosLoc = gl.getAttribLocation(lineProg, "aPos");\n  const lMVP = gl.getUniformLocation(lineProg, "uMVP");\n  const lWorld = gl.getUniformLocation(lineProg, "uWorld");\n  const lColor = gl.getUniformLocation(lineProg, "uColor");\n\n  gl.enable(gl.DEPTH_TEST);\n  gl.enable(gl.BLEND);\n  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);\n  gl.clearColor(0.039, 0.051, 0.075, 1);\n\n  function drawLines(buf, colorRGBA, mvp, width) {\n    if (!buf.count) return;\n    gl.useProgram(lineProg);\n    gl.bindBuffer(gl.ARRAY_BUFFER, buf.buf);\n    gl.enableVertexAttribArray(lPosLoc);\n    gl.vertexAttribPointer(lPosLoc, 3, gl.FLOAT, false, 0, 0);\n    gl.uniformMatrix4fv(lMVP, false, mvp);\n    gl.uniformMatrix3fv(lWorld, false, state.level ? LEVEL_MAT3 : IDENTITY_MAT3);\n    gl.uniform4fv(lColor, colorRGBA);\n    gl.lineWidth(width || 1.5);\n    gl.drawArrays(gl.LINES, 0, buf.count);\n  }\n\n  let lastW = 0, lastH = 0;\n  function resizeIfNeeded() {\n    if (window.innerWidth !== lastW || window.innerHeight !== lastH) {\n      lastW = window.innerWidth; lastH = window.innerHeight;\n      resize();\n    }\n  }\n\n  function positionLabel(el, origin, mvp) {\n    const p = project(mvp, origin);\n    if (p[2] <= 0) return;\n    const sx = (p[0] * 0.5 + 0.5) * window.innerWidth;\n    const sy = (1 - (p[1] * 0.5 + 0.5)) * window.innerHeight;\n    el.style.left = sx + "px";\n    el.style.top = sy + "px";\n  }\n\n  function frame() {\n    resizeIfNeeded();\n    const eye = eyeFromCam();\n    const aspect = canvas.width / canvas.height;\n    const proj = M.perspective(50 * Math.PI / 180, aspect, Math.max(extent * 0.005, 0.02), extent * 30);\n    const view = M.lookAt(eye, cam.target, [0, 0, 1]);\n    const mvp = M.multiply(proj, view);\n\n    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);\n\n    gl.useProgram(pointProg);\n    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);\n    gl.enableVertexAttribArray(aPosLoc);\n    gl.vertexAttribPointer(aPosLoc, 3, gl.FLOAT, false, 0, 0);\n    gl.bindBuffer(gl.ARRAY_BUFFER, intBuf);\n    gl.enableVertexAttribArray(aIntLoc);\n    gl.vertexAttribPointer(aIntLoc, 1, gl.FLOAT, false, 0, 0);\n    gl.uniformMatrix4fv(uMVP, false, mvp);\n    gl.uniformMatrix3fv(uWorld, false, state.level ? LEVEL_MAT3 : IDENTITY_MAT3);\n    gl.uniform1f(uPointSize, state.ptSize * (Math.min(window.devicePixelRatio || 1, 2)));\n    gl.uniform1f(uMinH, state.level ? lvl.z_min : minZ);\n    gl.uniform1f(uMaxH, state.level ? lvl.z_max : maxZ);\n    gl.uniform1f(uColorMode, state.colorMode);\n    gl.drawArrays(gl.POINTS, 0, pointCount);\n\n    if (state.showFloor) drawLines(floorGrid, [0.35, 0.42, 0.52, 0.18], mvp);\n    if (state.showAxes) drawLines(axisBuf, [0.5, 0.55, 0.62, 0.9], mvp);\n\n    if (selectedView < 0) {\n      drawLines(seedOverview, [82/255, 224/255, 196/255, 0.95], mvp);\n      if (state.showRing) drawLines(sequenceRing, [185/255, 140/255, 255/255, 0.85], mvp);\n    } else {\n      drawLines(seedInspect, [82/255, 224/255, 196/255, 0.95], mvp, 2.2);\n    }\n\n    if (state.showLabels) {\n      labelEls.forEach((l, i) => {\n        if (selectedView >= 0 && selectedView !== i) return;\n        positionLabel(l.el, l.view.seed.origin, mvp);\n      });\n    }\n\n    requestAnimationFrame(frame);\n  }\n  requestAnimationFrame(frame);\n})();\n</script>\n'


if __name__ == '__main__':
    main()
