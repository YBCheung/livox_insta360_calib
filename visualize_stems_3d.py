#!/usr/bin/env python3
"""Render the merged stem list against the global Laser_map, as one self-contained page.

The question this answers is not "where did the pipeline put the saplings" -- the CSV
already says that -- but "is there actually a tree there". A coordinate that is
self-consistent across frames can still be self-consistently wrong, and only the map,
which was built by a completely separate process (FAST-LIO's own registration), is
independent enough to check it against.

So every stem is scored against the map before it is drawn:

    support     map points within --radius of the stem in xy and above local ground
    offset      distance from the stem's xy to the median xy of that support
    ground      10th percentile of z within 3 m, the local ground under the stem

Offset is the number that matters and the one the colouring encodes. It is honest in
a way the per-frame spread is not: the map had no part in producing the stem
positions, so agreement is evidence rather than circularity.

Sibling of visualize_seed_fov.py and deliberately built the same way -- one .html with
the cloud, the stems and the whole WebGL viewer baked in, no server and no libraries,
so it can be mailed to someone who has neither ROS nor this repo.

    python3 visualize_stems_3d.py --map laser_map_xyz.npy --stems stems_manual_l2.csv
    python3 visualize_stems_3d.py --map cloud.pcd --stems stems.csv --out check.html
"""

import argparse
import base64
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_map(path):
    """.npy (N,3) written by the bag extractor, or a binary PCD with x y z intensity."""
    if path.endswith('.npy'):
        xyz = np.load(path).astype(np.float32)
        ipath = path.replace('_xyz.npy', '_i.npy')
        inten = (np.load(ipath).astype(np.float32) if os.path.exists(ipath)
                 else np.zeros(len(xyz), np.float32))
        return xyz, inten
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise SystemExit(f"{path}: no 'DATA binary' header")
            if line.strip() == b'DATA binary':
                break
        raw = np.frombuffer(f.read(), dtype=np.float32).reshape(-1, 4)
    return raw[:, :3].copy(), raw[:, 3].copy()


def load_traj(path):
    """(t, x, y, z) rows written by the bag extractor, sorted by stamp."""
    a = np.load(path)
    if a.ndim != 2 or a.shape[1] != 4:
        raise SystemExit(f"{path}: expected an (N,4) t,x,y,z array, got {a.shape}")
    return a[np.argsort(a[:, 0])]


def load_sightings(path, stems, snap):
    """Per-frame rows from bag_bbox_match.py --csv, each tied to a merged stem.

    The merged list says where things ended up; this says how the pipeline got there,
    one detection at a time. Each row keeps the camera position it was seen from, so
    the viewer can draw the actual sight line rather than just the endpoint -- which
    is what makes a weak baseline visible as a bundle of near-parallel rays.
    """
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path}: empty")
    sxy = np.array([[float(s['x']), float(s['y'])] for s in stems])
    out = []
    for r in rows:
        c = np.array([float(r['cx']), float(r['cy']), float(r['cz'])])
        d = np.linalg.norm(sxy - c[:2], axis=1)
        k = int(np.argmin(d))
        out.append((float(r['stamp']), *c,
                    float(r['cam_x']), float(r['cam_y']), float(r['cam_z']),
                    k if d[k] <= snap else -1))
    a = np.array(out, dtype=np.float64)
    return a[np.argsort(a[:, 0])]


def load_stems(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path}: empty")
    return rows


def score(stems, xyz, radius, ground_radius, above_h):
    """Attach map-support metrics to every stem. See the module docstring."""
    out = []
    xy = xyz[:, :2]
    for s in stems:
        p = np.array([float(s['x']), float(s['y'])])
        d = np.linalg.norm(xy - p, axis=1)
        near = xyz[d < ground_radius]
        if len(near) < 10:
            out.append({'support': 0, 'offset': None, 'nearest': None, 'ground': None})
            continue
        ground = float(np.percentile(near[:, 2], 10))
        can = xyz[(d < radius) & (xyz[:, 2] > ground + above_h)]
        if len(can) < 3:
            out.append({'support': int(len(can)), 'offset': None, 'nearest': None,
                        'ground': ground})
            continue
        # Two different questions, and they deserve separate numbers.
        #   offset  distance to the CENTROID of nearby structure. Depends on `radius`:
        #           widen it and neighbouring trees drag the median away (measured on
        #           manual_l2: 0.13 m at 0.40 m, 0.24 at 0.75, 0.49 at 2.0, 1.31 at
        #           5.0). Read it as "agreement within this radius", never as accuracy.
        #   nearest distance to the NEAREST above-ground point, which has no averaging
        #           radius at all. It answers only "is the stem on structure or in
        #           thin air" -- a weak test where the canopy is dense, but an
        #           unambiguous one, and its floor is the map's own point spacing.
        above = xyz[:, 2] > ground + above_h
        out.append({'support': int(len(can)),
                    'offset': float(np.linalg.norm(np.median(can[:, :2], axis=0) - p)),
                    'nearest': (float(d[above].min()) if above.any() else None),
                    'ground': ground,
                    'map_top': float(can[:, 2].max())})
    return out


def build(args):
    xyz, inten = load_map(args.map)
    stems = load_stems(args.stems)

    sx = np.array([float(s['x']) for s in stems])
    sy = np.array([float(s['y']) for s in stems])
    # Crop to the surveyed area: the raw map runs 150 m across because a handful of
    # far returns survive, and framing on those puts every stem in one pixel.
    lo = np.array([sx.min(), sy.min()]) - args.margin
    hi = np.array([sx.max(), sy.max()]) + args.margin
    keep = ((xyz[:, 0] > lo[0]) & (xyz[:, 0] < hi[0]) &
            (xyz[:, 1] > lo[1]) & (xyz[:, 1] < hi[1]))
    zlo, zhi = np.percentile(xyz[keep][:, 2], [0.2, 99.8])
    keep &= (xyz[:, 2] > zlo - 2) & (xyz[:, 2] < zhi + 2)
    xyz, inten = xyz[keep], inten[keep]
    print(f"map        : {len(xyz)} points after cropping to the surveyed area")

    # Per-point birth time, so the map can grow with the timeline instead of standing
    # there finished. Sibling of --map by default, since extract_bag_layers.py writes
    # the two together and they must describe the same cloud.
    mt = None if args.static_map else args.map_times
    if mt is None and not args.static_map:
        guess = args.map.replace('_xyz.npy', '_t.npy')
        mt = guess if guess != args.map and os.path.exists(guess) else None
    mtimes = None
    if mt:
        mtimes = np.load(mt)
        if len(mtimes) != len(keep):
            raise SystemExit(f"{mt} has {len(mtimes)} times but {args.map} has {len(keep)} "
                             f"points -- they must come from the same extraction run")
        mtimes = mtimes[keep]
        print(f"             growth from {os.path.basename(mt)}: "
              f"{mtimes.max()-mtimes.min():.1f} s of accumulation")

    scored = score(stems, xyz, args.radius, args.ground_radius, args.above)

    if args.max_points and len(xyz) > args.max_points:
        idx = np.random.default_rng(0).choice(len(xyz), args.max_points, replace=False)
        idx.sort()
        xyz, inten = xyz[idx], inten[idx]
        if mtimes is not None:
            mtimes = mtimes[idx]
        print(f"             thinned to {len(xyz)} for the viewer")

    rng_i = float(inten.max()) or 1.0
    i8 = np.clip(inten / rng_i * 255.0, 0, 255).astype(np.uint8)

    items = []
    for s, m in zip(stems, scored):
        items.append({
            'label': s['label'], 'n': int(s['n']),
            'x': round(float(s['x']), 3), 'y': round(float(s['y']), 3),
            'top': round(float(s['top_z']), 3), 'bot': round(float(s['bot_z']), 3),
            'h': round(float(s['height']), 3),
            'pts': int(float(s['pts'])), 'conf': round(float(s['conf']), 3),
            'support': m['support'],
            'offset': (None if m['offset'] is None else round(m['offset'], 3)),
            'nearest': (None if m.get('nearest') is None else round(m['nearest'], 3)),
            'ground': (None if m['ground'] is None else round(m['ground'], 3)),
            'map_top': (None if m.get('map_top') is None else round(m['map_top'], 3)),
            # Absolute stamps, so the viewer can light up the stretch of flight path
            # over which this stem was actually observed.
            'first': float(s['first']), 'last': float(s['last']),
            'baseline': round(float(s.get('baseline', 0.0) or 0.0), 3),
        })

    good = [i for i in items if i['offset'] is not None]
    offs = np.array([i['offset'] for i in good]) if good else np.array([])
    summary = {
        'n_stems': len(items),
        'n_supported': len(good),
        'median_offset': (round(float(np.median(offs)), 3) if len(offs) else None),
        'p90_offset': (round(float(np.percentile(offs, 90)), 3) if len(offs) else None),
        'within_25': int((offs <= 0.25).sum()) if len(offs) else 0,
        'within_50': int((offs <= 0.50).sum()) if len(offs) else 0,
        'radius': args.radius,
    }
    nrs = np.array([i['nearest'] for i in items if i['nearest'] is not None])
    summary['median_nearest'] = round(float(np.median(nrs)), 3) if len(nrs) else None
    summary['n_nearest'] = int(len(nrs))
    for cut in (4, 10, 30):
        sel = [i['offset'] for i in good if i['n'] >= cut]
        summary[f'offset_{cut}'] = (round(float(np.median(sel)), 3) if len(sel) > 3 else None)
        summary[f'count_{cut}'] = len(sel)

    traj = None
    if args.odom:
        a = load_traj(args.odom)
        t = a[:, 0]
        step = np.linalg.norm(np.diff(a[:, 1:4], axis=0), axis=1)
        traj = {
            'xyz_b64': base64.b64encode(a[:, 1:4].astype(np.float32).tobytes()).decode('ascii'),
            # Time is sent normalised with its own range beside it: the colour ramp
            # wants 0..1 and the per-stem segment lookup wants absolute seconds, and
            # one array plus two scalars serves both.
            't_b64': base64.b64encode(
                ((t - t[0]) / max(t[-1] - t[0], 1e-9)).astype(np.float32).tobytes()).decode('ascii'),
            'n': int(len(a)), 't0': float(t[0]), 't1': float(t[-1]),
            'length': float(step.sum()), 'duration': float(t[-1] - t[0]),
            'z_lo': float(a[:, 3].min()), 'z_hi': float(a[:, 3].max()),
            'v_med': float(np.median(step / np.maximum(np.diff(t), 1e-6))),
        }
        print(f"trajectory : {traj['n']} poses, {traj['length']:.1f} m over "
              f"{traj['duration']:.1f} s, altitude {traj['z_lo']:.1f}..{traj['z_hi']:.1f} m")

    sight = None
    if args.sightings:
        a = load_sightings(args.sightings, stems, args.snap)
        t = a[:, 0]
        # first_seen per stem drives the "not yet discovered" ghosting in the viewer.
        first = np.full(len(stems), np.inf)
        for row in a:
            k = int(row[7])
            if k >= 0:
                first[k] = min(first[k], row[0])
        sight = {
            'n': int(len(a)),
            'obj_b64': base64.b64encode(a[:, 1:4].astype(np.float32).tobytes()).decode('ascii'),
            'cam_b64': base64.b64encode(a[:, 4:7].astype(np.float32).tobytes()).decode('ascii'),
            # Seconds since t0, NOT absolute stamps: a ROS stamp is ~1.79e9 and a
            # float32 resolves that to about 128 s, which would collapse the whole
            # timeline onto two or three distinct instants.
            't_b64': base64.b64encode((t - t[0]).astype(np.float32).tobytes()).decode('ascii'),
            'idx_b64': base64.b64encode(a[:, 7].astype(np.int32).tobytes()).decode('ascii'),
            't0': float(t[0]), 't1': float(t[-1]),
            'assigned': int((a[:, 7] >= 0).sum()),
        }
        # One hue per merged stem, stepped by the golden angle so consecutive indices
        # land far apart on the wheel. Stems are ordered by sighting count, which is
        # unrelated to position, so neighbours in space almost always get separable
        # hues -- which is the whole point: a cluster that swallowed two real trees,
        # or split one, shows up as the wrong colours sitting together.
        idx = a[:, 7].astype(int)
        rgb = np.zeros((len(a), 3), np.uint8)
        for k in range(len(a)):
            if idx[k] < 0:
                rgb[k] = (120, 128, 140)          # unassigned: neutral grey
                continue
            h = (idx[k] * 137.508 % 360.0) / 60.0
            c, x = 0.78, 0.78 * (1 - abs(h % 2 - 1))
            r, g, b = [(c,x,0),(x,c,0),(0,c,x),(0,x,c),(x,0,c),(c,0,x)][int(h) % 6]
            rgb[k] = np.round((np.array([r, g, b]) + 0.22) * 255).clip(0, 255)
        sight['rgb_b64'] = base64.b64encode(rgb.tobytes()).decode('ascii')
        # Merge diagnostics, computed here so the page can state them rather than
        # leaving the eye to judge. Leader clustering keeps SEEDS a radius apart, but
        # absorbing members moves the median afterwards -- so final centres can end up
        # closer than the radius, and a sighting can sit further from its centre than
        # the radius. Both are worth counting.
        ok = idx >= 0
        cen = np.array([[float(x['x']), float(x['y'])] for x in items])
        own = np.linalg.norm(a[ok, 1:3] - cen[idx[ok]], axis=1)
        dall = np.linalg.norm(a[ok, None, 1:3] - cen[None, :, :], axis=2)
        srt = np.sort(dall, axis=1)
        nnp = np.sort(np.linalg.norm(cen[:, None] - cen[None, :], axis=2)
                      + np.eye(len(cen)) * 1e9, axis=1)[:, 0]
        sight['merge'] = {
            'assigned': int(ok.sum()), 'unassigned': int((~ok).sum()),
            'own_med': round(float(np.median(own)), 3),
            'own_max': round(float(own.max()), 3),
            'misassigned': int((np.argmin(dall, axis=1) != idx[ok]).sum()),
            'ambiguous': int(((srt[:, 1] - srt[:, 0]) < 0.10).sum()),
            # --merge-radius, NOT --radius: the latter is the map-support radius and
            # has nothing to do with how the stems were clustered. Conflating them
            # made this row count pairs against the wrong number.
            'close_pairs': int((nnp < args.merge_radius).sum()),
            'nn_med': round(float(np.median(nnp)), 3),
            'nn_min': round(float(nnp.min()), 3),
            'radius': args.merge_radius,
        }
        sight['stem_b64'] = base64.b64encode(
            np.array([[float(stems[i]['x']), float(stems[i]['y']), float(stems[i]['z'])]
                      if i >= 0 else [0.0, 0.0, 0.0] for i in idx],
                     dtype=np.float32).tobytes()).decode('ascii')
        for i, s_ in enumerate(items):
            s_['seen'] = (None if not np.isfinite(first[i]) else round(float(first[i]), 3))
        print(f"sightings  : {sight['n']} rows, {sight['assigned']} tied to a stem")

    centre = [(float(xyz[:, k].min()) + float(xyz[:, k].max())) / 2 for k in range(3)]
    extent = float(max(xyz[:, k].max() - xyz[:, k].min() for k in range(3)))
    payload = {
        'source': os.path.basename(args.stems),
        'map_name': os.path.basename(args.map),
        'xyz_b64': base64.b64encode(xyz.astype(np.float32).tobytes()).decode('ascii'),
        'i_b64': base64.b64encode(i8.tobytes()).decode('ascii'),
        'count': int(len(xyz)),
        # Offsets from the map's own first stamp, not absolute: a float32 cannot hold
        # a ~1.79e9 ROS stamp to better than ~128 s.
        'map_t_b64': (None if mtimes is None else base64.b64encode(
            (mtimes - mtimes.min()).astype(np.float32).tobytes()).decode('ascii')),
        'map_t0': (None if mtimes is None else float(mtimes.min())),
        'stems': items, 'summary': summary, 'traj': traj, 'sight': sight,
        'centre': centre, 'extent': extent,
        'z_lo': float(np.percentile(xyz[:, 2], 1)),
        'z_hi': float(np.percentile(xyz[:, 2], 99)),
    }
    html = (TEMPLATE.replace('<title>Stem Match Inspector</title>',
                             f'<title>{args.title}</title>')
                    .replace('<h1>Stem Match Inspector</h1>', f'<h1>{args.title}</h1>')
                    .replace('__DATA_JSON__', json.dumps(payload)))
    with open(args.out, 'w') as f:
        f.write(html)
    mb = os.path.getsize(args.out) / 1e6
    print(f"stems      : {len(items)}, {len(good)} with map support")
    if summary['median_offset'] is not None:
        print(f"offset     : median {summary['median_offset']:.2f} m, "
              f"p90 {summary['p90_offset']:.2f} m")
    print(f"wrote {args.out} ({mb:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--map', required=True, help='.npy from the bag extractor, or a binary .pcd')
    p.add_argument('--stems', required=True, help='merged CSV from merge_objects.py')
    p.add_argument('--out', default=os.path.join(HERE, 'stem_match.html'))
    p.add_argument('--title', default='Stem Match Inspector',
                   help='page title, and the name it takes in an artifact gallery. '
                        'Give distinct runs distinct titles or they are impossible to '
                        'tell apart later')
    p.add_argument('--map-times', help='(N,) .npy of per-point birth stamps, to grow the '
                                       'map along the timeline. Defaults to the '
                                       '*_t.npy sitting beside --map')
    p.add_argument('--static-map', action='store_true',
                   help='ignore birth times and draw the finished map')
    p.add_argument('--odom', help='(N,4) t,x,y,z .npy of /Odometry, to draw the flight path')
    p.add_argument('--sightings', help='per-frame CSV from bag_bbox_match.py, to enable '
                                       'the timeline')
    p.add_argument('--merge-radius', type=float, default=1.0, metavar='M',
                   help='the --radius merge_objects.py was run with. Used only to '
                        'report how many merged stems ended up closer than it')
    p.add_argument('--snap', type=float, default=1.5, metavar='M',
                   help='how close a sighting must be to a merged stem to count as it')
    p.add_argument('--radius', type=float, default=0.75, metavar='M',
                   help='xy radius counted as support for a stem')
    p.add_argument('--ground-radius', type=float, default=3.0, metavar='M',
                   help='radius over which local ground height is estimated')
    p.add_argument('--above', type=float, default=0.3, metavar='M',
                   help='height above local ground a point must clear to be canopy')
    p.add_argument('--margin', type=float, default=15.0, metavar='M',
                   help='map crop margin around the stems')
    p.add_argument('--max-points', type=int, default=250000)
    build(p.parse_args())


TEMPLATE = r'''<title>Stem Match Inspector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
  /* Committed single dark theme, like visualize_seed_fov.py: a point cloud reads
     against a dark ground and every tool in this domain agrees. Every colour is
     painted explicitly so the page holds on any host background. */
  :root {
    --bg: #0a0d13;
    --bg-1: #0e131b;
    --panel: rgba(19, 25, 35, 0.86);
    --panel-border: #26303f;
    --text: #e7ebf1;
    --text-dim: #8b95a7;
    --text-faint: #566072;
    --accent: #52e0c4;
    --focus: #7fb7ff;
    --good: #5ad67d;
    --fair: #f2c14e;
    --poor: #f2705b;
    /* Violet reads as temporal order across this toolset -- visualize_seed_fov.py
       uses it for view sequence -- and it is the one hue left free by the height
       ramp and the good/fair/poor scale. */
    --path: #b98cff;
    --path-dim: #4b3570;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; width: 100%; height: 100%;
    background: var(--bg); color: var(--text);
    font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
    overflow: hidden;
  }
  #gl { position: fixed; inset: 0; display: block; touch-action: none; cursor: grab; }
  #gl.dragging { cursor: grabbing; }
  .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }

  .hud {
    position: fixed; background: var(--panel);
    border: 1px solid var(--panel-border); border-radius: 10px;
    backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  }
  .left { top: 16px; left: 16px; width: 330px; padding: 16px 18px;
          max-height: calc(100vh - 32px); overflow-y: auto; }
  .left h1 { font-size: 15px; font-weight: 600; margin: 0 0 3px; text-wrap: balance; }
  .left .sub { font-size: 12px; color: var(--text-dim); line-height: 1.5; margin: 0 0 14px; }
  .left .sub code { font-family: "IBM Plex Mono", ui-monospace, monospace; color: var(--accent); font-size: 11px; }

  .label { font-size: 10px; font-weight: 600; letter-spacing: 0.08em;
           text-transform: uppercase; color: var(--text-faint); margin: 16px 0 8px; }
  .label:first-of-type { margin-top: 0; }

  /* .verdict-note is still used by the offset caveat and the merge note. */
  .verdict-note { font-size: 12px; color: var(--text-dim); line-height: 1.5; margin: 6px 0 0; }

  .bars { display: flex; flex-direction: column; gap: 7px; margin-top: 12px; }
  .bar-row { display: grid; grid-template-columns: 62px 1fr 46px; align-items: center; gap: 9px;
             font-size: 11.5px; color: var(--text-dim); }
  .track { height: 6px; background: var(--bg-1); border-radius: 3px; overflow: hidden; }
  .fill { height: 100%; border-radius: 3px; }
  .bar-row .num { text-align: right; color: var(--text);
                  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 11px; }

  .legend { display: flex; flex-direction: column; gap: 7px; }
  .lrow { display: flex; align-items: center; gap: 9px; font-size: 12px; color: var(--text-dim); }
  .sw { width: 11px; height: 11px; border-radius: 3px; flex: none; }
  .ramp { width: 11px; height: 11px; border-radius: 3px; flex: none;
          background: linear-gradient(90deg,#0f1729,#1a7385,#8cd94f,#f9bf27,#fa5940); }

  table.stems { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  table.stems th { text-align: right; font-weight: 500; color: var(--text-faint); font-size: 10px;
                   letter-spacing: 0.04em; text-transform: uppercase; padding: 0 0 6px;
                   border-bottom: 1px solid var(--panel-border); position: sticky; top: 0;
                   background: var(--panel); cursor: pointer; }
  table.stems th:first-child { text-align: left; }
  table.stems td { padding: 5px 0; border-bottom: 1px solid rgba(38,48,63,0.5);
                   color: var(--text-dim); }
  table.stems td.n { text-align: right; font-family: "IBM Plex Mono", ui-monospace, monospace;
                     font-variant-numeric: tabular-nums; }
  table.stems tr { cursor: pointer; }
  table.stems tr:hover td { color: var(--focus); }
  table.stems tr.on td { color: var(--accent); }
  .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }

  .right { top: 16px; right: 16px; width: 234px; padding: 14px 16px; }
  .grp { margin-bottom: 14px; }
  .grp:last-child { margin-bottom: 0; }
  .trow { display: flex; align-items: center; justify-content: space-between;
          font-size: 12.5px; color: var(--text); padding: 5px 0; cursor: pointer; user-select: none; }
  .trow input { accent-color: var(--focus); width: 14px; height: 14px; cursor: pointer; }
  .seg { display: flex; background: var(--bg-1); border: 1px solid var(--panel-border);
         border-radius: 7px; padding: 2px; gap: 2px; }
  .seg button { flex: 1; background: transparent; border: none; color: var(--text-dim);
                font-family: inherit; font-size: 11.5px; padding: 6px 0; border-radius: 5px;
                cursor: pointer; transition: background .12s, color .12s; }
  .seg button.on { background: var(--focus); color: #0a0d13; font-weight: 600; }
  input[type=range] { width: 100%; accent-color: var(--focus); }
  .rlab { display: flex; justify-content: space-between; font-size: 11px;
          color: var(--text-faint); margin-bottom: 4px; }
  .rlab b { color: var(--text); font-weight: 500;
            font-family: "IBM Plex Mono", ui-monospace, monospace; }
  button.reset { width: 100%; background: var(--bg-1); border: 1px solid var(--panel-border);
                 color: var(--text); font-family: inherit; font-size: 12px; padding: 8px 0;
                 border-radius: 7px; cursor: pointer; }
  button.reset:hover { border-color: var(--focus); color: var(--focus); }
  button:focus-visible, input:focus-visible, tr:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }

  .detail { position: fixed; right: 16px; bottom: 16px; width: 234px; padding: 13px 16px; }
  .detail h2 { font-size: 12px; margin: 0 0 9px; font-weight: 600; color: var(--accent); }
  .kv { display: grid; grid-template-columns: auto 1fr; gap: 3px 10px; font-size: 11.5px; }
  .kv dt { color: var(--text-faint); }
  .kv dd { margin: 0; text-align: right; color: var(--text);
           font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }

  #flightBlock .kv { margin-top: 2px; }
  .timeline { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
              width: min(660px, calc(100vw - 620px)); min-width: 320px; padding: 12px 16px; }
  .tl-top { display: flex; align-items: center; gap: 12px; }
  .tl-play { flex: none; width: 30px; height: 30px; border-radius: 50%; cursor: pointer;
             background: var(--accent); border: none; color: #0a0d13; font-size: 12px;
             display: grid; place-items: center; font-family: inherit; }
  .tl-play:hover { filter: brightness(1.12); }
  .tl-time { flex: none; font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 12px;
             color: var(--text); font-variant-numeric: tabular-nums; min-width: 86px; }
  .tl-time span { color: var(--text-faint); }
  #tlRange { flex: 1; accent-color: var(--accent); }
  .tl-speed { flex: none; display: flex; gap: 2px; background: var(--bg-1);
              border: 1px solid var(--panel-border); border-radius: 6px; padding: 2px; }
  .tl-speed button { background: transparent; border: none; color: var(--text-dim);
                     font-family: inherit; font-size: 10.5px; padding: 3px 6px;
                     border-radius: 4px; cursor: pointer; }
  .tl-speed button.on { background: var(--focus); color: #0a0d13; font-weight: 600; }
  .tl-foot { display: flex; justify-content: space-between; gap: 12px; margin-top: 7px;
             font-size: 10.5px; color: var(--text-faint);
             font-family: "IBM Plex Mono", ui-monospace, monospace; }
  .tl-foot b { color: var(--accent); font-weight: 500; }
  .hint { position: fixed; bottom: 16px; left: 16px; font-size: 11px; color: var(--text-faint);
          font-family: "IBM Plex Mono", ui-monospace, monospace; padding: 8px 12px; }
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-thumb { background: var(--panel-border); border-radius: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
  @media (max-width: 860px) {
    .left { width: calc(100vw - 32px); max-height: 40vh; }
    .right { top: auto; bottom: 16px; right: 16px; width: 200px; }
    .detail { display: none; }
  }
</style>

<canvas id="gl"></canvas>

<div class="hud left">
  <h1>Stem Match Inspector</h1>
  <p class="label" style="margin-top:14px">Offset to local centroid, by sighting count</p>
  <div class="bars" id="bars"></div>
  <p class="verdict-note" id="vCaveat" style="margin-top:9px"></p>

  <p class="label">Stem colour</p>
  <div class="legend">
    <div class="lrow"><span class="sw" style="background:#5ad67d"></span> within 0.25 m of map structure</div>
    <div class="lrow"><span class="sw" style="background:#f2c14e"></span> 0.25 – 0.60 m</div>
    <div class="lrow"><span class="sw" style="background:#f2705b"></span> beyond 0.60 m, or no support</div>
    <div class="lrow"><span class="ramp"></span> map points by height</div>
    <div class="lrow"><span class="sw" style="background:linear-gradient(90deg,#e05c5c,#5ce0a8,#5c8ce0)"></span> raw sightings, one hue per cluster</div>
    <div class="lrow"><span class="sw" style="background:linear-gradient(90deg,#4b3570,#b98cff)"></span> flight path, dim&rarr;bright over time</div>
  </div>

  <div id="mergeBlock" hidden>
    <p class="label">Merge quality</p>
    <dl class="kv" id="mergeKv"></dl>
    <p class="verdict-note" id="mergeNote" style="margin-top:8px"></p>
  </div>

  <div id="flightBlock" hidden>
    <p class="label">Flight</p>
    <dl class="kv" id="flightKv"></dl>
  </div>

  <p class="label">Stems <span style="text-transform:none;font-weight:400;color:var(--text-faint)">(click to inspect)</span></p>
  <table class="stems" id="tbl"></table>
</div>

<div class="hud right">
  <div class="grp">
    <p class="label" style="margin-top:0">Map colour</p>
    <div class="seg" id="cseg">
      <button data-m="0" class="on">Height</button>
      <button data-m="1">Intensity</button>
    </div>
  </div>
  <div class="grp">
    <div class="rlab"><span>Min sightings</span><b id="lMin">1</b></div>
    <input type="range" id="rMin" min="1" max="40" step="1" value="1">
  </div>
  <div class="grp">
    <div class="rlab"><span>Point size</span><b id="lPt">2.0</b></div>
    <input type="range" id="rPt" min="1" max="6" step="0.1" value="2.0">
  </div>
  <div class="grp">
    <div class="rlab"><span>Map opacity</span><b id="lOp">100%</b></div>
    <input type="range" id="rOp" min="5" max="100" step="5" value="100">
  </div>
  <div class="grp">
    <label class="trow"><span>Stem markers</span><input type="checkbox" id="cStem" checked></label>
    <label class="trow"><span>Ground ticks</span><input type="checkbox" id="cTick" checked></label>
    <label class="trow"><span>Only unsupported</span><input type="checkbox" id="cBad"></label>
    <label class="trow" id="trajRow" hidden><span>Flight path</span><input type="checkbox" id="cTraj" checked></label>
    <label class="trow" id="sightRow" hidden><span>Raw sightings</span><input type="checkbox" id="cSight"></label>
    <label class="trow" id="spokeRow" hidden><span>Spokes to merged</span><input type="checkbox" id="cSpoke"></label>
  </div>
  <button class="reset" id="reset">Reset view</button>
</div>

<div class="hud detail" id="detail" hidden>
  <h2 id="dTitle">Stem</h2>
  <dl class="kv" id="dBody"></dl>
</div>

<div class="hud timeline" id="timeline" hidden>
  <div class="tl-top">
    <button class="tl-play" id="tlPlay" aria-label="Play">&#9654;</button>
    <span class="tl-time" id="tlTime">0.0<span> / 0.0 s</span></span>
    <input type="range" id="tlRange" min="0" max="1000" step="1" value="1000" aria-label="Time">
    <span class="tl-speed" id="tlSpeed">
      <button data-x="0.5">.5&times;</button><button data-x="1" class="on">1&times;</button>
      <button data-x="4">4&times;</button><button data-x="16">16&times;</button>
    </span>
  </div>
  <div class="tl-foot">
    <span><b id="tlFound">0</b> of <span id="tlTotal">0</span> stems found</span>
    <span><b id="tlActive">0</b> sight lines this instant</span>
    <span>drag orbit &middot; scroll zoom &middot; shift+drag pan</span>
  </div>
</div>

<div class="hint" id="hint">drag orbit &middot; scroll zoom &middot; shift+drag pan</div>

<script id="payload" type="application/json">__DATA_JSON__</script>
<script>
(function () {
  "use strict";
  const P = JSON.parse(document.getElementById("payload").textContent);
  const b64f32 = (b) => { const s = atob(b), a = new ArrayBuffer(s.length), v = new Uint8Array(a);
    for (let i = 0; i < s.length; i++) v[i] = s.charCodeAt(i); return new Float32Array(a); };
  const b64u8 = (b) => { const s = atob(b), a = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; };

  const pos = b64f32(P.xyz_b64), inten = b64u8(P.i_b64), N = P.count;
  const stems = P.stems, S = P.summary;

  const GOOD = [0.353, 0.839, 0.490], FAIR = [0.949, 0.757, 0.306], POOR = [0.949, 0.439, 0.357];
  const grade = (s) => (s.offset === null) ? POOR : (s.offset <= 0.25 ? GOOD : (s.offset <= 0.60 ? FAIR : POOR));
  const css = (c) => `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;

  // ---- summary ----
  document.getElementById("vCaveat").textContent =
    `Distance to the centroid of structure within ${S.radius} m. This one scales with that `
    + `radius (0.13 m at 0.40, 1.31 m at 5.0), so read the trend, not the value: it falls as `
    + `a stem is seen more often, which a radius artefact would not do. Neither number is `
    + `ground truth — the map shares its LiDAR and its odometry with the detections.`;
  const bars = [
    ["all", S.median_offset, S.n_supported],
    ["4+ seen", S.offset_4, S.count_4],
    ["10+ seen", S.offset_10, S.count_10],
    ["30+ seen", S.offset_30, S.count_30]
  ].filter(r => r[1] !== null && r[1] !== undefined);
  const worst = Math.max(...bars.map(r => r[1]), 0.3);
  document.getElementById("bars").innerHTML = bars.map(([k, v, n]) => {
    const c = v <= 0.25 ? "var(--good)" : (v <= 0.6 ? "var(--fair)" : "var(--poor)");
    return `<div class="bar-row"><span>${k}</span>`
      + `<span class="track"><span class="fill" style="width:${Math.max(4, 100*v/worst)}%;background:${c}"></span></span>`
      + `<span class="num">${v.toFixed(2)} m</span></div>`;
  }).join("");

  // ---- table ----
  const tbl = document.getElementById("tbl");
  let sortKey = "n", sortDir = -1;
  function draw() {
    const rows = stems.map((s, i) => ({ s, i }))
      .filter(({ s }) => s.n >= state.minN && (!state.badOnly || s.offset === null || s.offset > 0.60))
      .sort((a, b) => sortDir * ((a.s[sortKey] ?? 1e9) - (b.s[sortKey] ?? 1e9)));
    tbl.innerHTML = `<tr><th data-k="n">Seen</th><th data-k="h">Height</th><th data-k="offset">Offset</th></tr>`
      + rows.map(({ s, i }) =>
        `<tr data-i="${i}" class="${i === sel ? "on" : ""}"><td><span class="dot" style="background:${css(grade(s))}"></span>${s.n}</td>`
        + `<td class="n">${s.h.toFixed(2)} m</td>`
        + `<td class="n">${s.offset === null ? "—" : s.offset.toFixed(2) + " m"}</td></tr>`).join("");
  }
  tbl.addEventListener("click", (e) => {
    const th = e.target.closest("th");
    if (th) { const k = th.dataset.k; sortDir = (k === sortKey) ? -sortDir : -1; sortKey = k; draw(); return; }
    const tr = e.target.closest("tr[data-i]");
    if (tr) select(parseInt(tr.dataset.i, 10));
  });

  let sel = -1;
  const detail = document.getElementById("detail");
  function select(i) {
    sel = (sel === i) ? -1 : i;
    selBuf = selBuffer();                 // selBuffer is hoisted; reads the new sel
    segBuf = segmentFor(sel < 0 ? null : stems[sel]);
    selRays = raysForStem(sel);
    if (sel < 0) { detail.hidden = true; draw(); return; }
    const s = stems[sel];
    document.getElementById("dTitle").textContent = `${s.label} · seen ${s.n}×`;
    const kv = [
      ["x, y", `${s.x.toFixed(2)}, ${s.y.toFixed(2)}`],
      ["top z", s.top.toFixed(2) + " m"], ["bottom z", s.bot.toFixed(2) + " m"],
      ["height", s.h.toFixed(2) + " m"],
      ["nearest structure", s.nearest === null ? "—" : s.nearest.toFixed(2) + " m"],
      ["offset to centroid", s.offset === null ? "no support" : s.offset.toFixed(2) + " m"],
      ["map points", s.support], ["lidar pts / frame", s.pts],
      ["observed over", (s.last - s.first).toFixed(1) + " s"],
      ["view baseline", s.baseline.toFixed(2) + " m"],
      ["local ground", s.ground === null ? "—" : s.ground.toFixed(2) + " m"],
      ["map top z", s.map_top === null || s.map_top === undefined ? "—" : s.map_top.toFixed(2) + " m"],
      ["best conf", s.conf.toFixed(2)]
    ];
    document.getElementById("dBody").innerHTML = kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    detail.hidden = false;
    draw();
  }

  // ---- gl ----
  const canvas = document.getElementById("gl");
  const gl = canvas.getContext("webgl", { antialias: true });
  if (!gl) { document.body.innerHTML = '<p style="color:#e7ebf1;padding:24px;font-family:sans-serif">WebGL is not available in this browser.</p>'; return; }
  const comp = (t, src) => { const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s)); return s; };
  const prog = (v, f) => { const p = gl.createProgram(); gl.attachShader(p, comp(gl.VERTEX_SHADER, v));
    gl.attachShader(p, comp(gl.FRAGMENT_SHADER, f)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(p)); return p; };

  const pProg = prog(`
    attribute vec3 aPos; attribute float aI; attribute float aBorn;
    uniform mat4 uMVP; uniform float uSize, uLo, uHi, uMode, uMapNow;
    varying vec3 vC; varying float vFresh;
    vec3 ramp(float t){ t=clamp(t,0.0,1.0);
      vec3 c0=vec3(0.059,0.090,0.161),c1=vec3(0.102,0.451,0.522),c2=vec3(0.549,0.851,0.310),
           c3=vec3(0.976,0.749,0.153),c4=vec3(0.980,0.349,0.251);
      float s=t*4.0;
      if(s<1.0) return mix(c0,c1,s); if(s<2.0) return mix(c1,c2,s-1.0);
      if(s<3.0) return mix(c2,c3,s-2.0); return mix(c3,c4,min(s-3.0,1.0)); }
    void main(){ gl_Position=uMVP*vec4(aPos,1.0);
      vC=ramp(mix((aPos.z-uLo)/max(uHi-uLo,0.0001), aI, uMode));
      // < 0 means not yet mapped; 0..1 fades out over the seconds after it appeared.
      vFresh = (aBorn > uMapNow) ? -1.0 : (1.0 - clamp((uMapNow-aBorn)/3.0, 0.0, 1.0));
      gl_PointSize=uSize; }`,
    `precision mediump float; varying vec3 vC; varying float vFresh; uniform float uAlpha;
     void main(){ vec2 d=gl_PointCoord-vec2(0.5); float r2=dot(d,d);
       if(r2>0.25 || vFresh < 0.0) discard;
       // Just-added points flash toward the accent, so the map visibly grows rather
       // than silently getting bigger.
       vec3 c = mix(vC, vec3(0.322,0.878,0.769), vFresh*0.85);
       gl_FragColor=vec4(c, uAlpha*smoothstep(0.25,0.16,r2)); }`);
  const lProg = prog(`attribute vec3 aPos; uniform mat4 uMVP; void main(){ gl_Position=uMVP*vec4(aPos,1.0); }`,
    `precision mediump float; uniform vec4 uC; void main(){ gl_FragColor=uC; }`);
  // Raw per-frame sightings: position, the moment it happened, and its cluster's
  // colour. The same program draws them as POINTS and the spokes as LINES --
  // gl_PointSize is simply ignored in the line case.
  const gProg = prog(
    `attribute vec3 aPos; attribute float aT; attribute vec3 aCol;
     uniform mat4 uMVP; uniform float uSize, uNow;
     varying vec3 vC; varying float vOn;
     void main(){ vOn = step(aT, uNow); vC = aCol;
       gl_Position = uMVP*vec4(aPos,1.0); gl_PointSize = uSize; }`,
    `precision mediump float; varying vec3 vC; varying float vOn;
     uniform float uGhost, uAlpha, uRound;
     void main(){ vec2 d = gl_PointCoord - vec2(0.5);
       if (uRound > 0.5 && dot(d,d) > 0.25) discard;
       gl_FragColor = vec4(vC, uAlpha * mix(uGhost, 1.0, vOn)); }`);

  // Stems carry the time they were first detected, so the timeline can ghost the
  // ones the drone has not reached yet without rebuilding a buffer per tick.
  const sProg = prog(
    `attribute vec3 aPos; attribute float aFirst; uniform mat4 uMVP; uniform float uNow;
     varying float vOn;
     void main(){ vOn = step(aFirst, uNow); gl_Position = uMVP*vec4(aPos,1.0); }`,
    `precision mediump float; varying float vOn; uniform vec4 uC; uniform float uGhost;
     void main(){ gl_FragColor = vec4(uC.rgb, uC.a * mix(uGhost, 1.0, vOn)); }`);
  // Its own program because the path carries a per-vertex time, and a uniform
  // colour cannot show where in the flight a stretch of path was flown.
  const tProg = prog(
    `attribute vec3 aPos; attribute float aT; uniform mat4 uMVP; varying float vT;
     void main(){ vT=aT; gl_Position=uMVP*vec4(aPos,1.0); }`,
    `precision mediump float; varying float vT; uniform float uA, uNowN;
     void main(){ gl_FragColor=vec4(mix(vec3(0.294,0.208,0.439), vec3(0.725,0.549,1.0),
       clamp(vT,0.0,1.0)), uA * mix(0.10, 1.0, step(vT, uNowN))); }`);

  const posBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  const iBuf = gl.createBuffer(); const iF = new Float32Array(N);
  for (let k = 0; k < N; k++) iF[k] = inten[k] / 255;
  gl.bindBuffer(gl.ARRAY_BUFFER, iBuf); gl.bufferData(gl.ARRAY_BUFFER, iF, gl.STATIC_DRAW);
  // The attribute exists even for a static map: a zero array plus a huge uMapNow
  // keeps one code path rather than relying on a disabled-attribute default.
  const MAPT = P.map_t_b64 ? b64f32(P.map_t_b64) : new Float32Array(N);
  const bornBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bornBuf);
  gl.bufferData(gl.ARRAY_BUFFER, MAPT, gl.STATIC_DRAW);
  const mkBuf = (a) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(a), gl.STATIC_DRAW); return { b, n: a.length / 3 }; };

  // One line buffer per grade so each draws in its own colour with no per-stem calls.
  const NEVER = 1e18;
  function mkBuf2(pos, first) {
    const b = mkBuf(pos), fb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, fb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(first), gl.STATIC_DRAW);
    return { b: b.b, n: b.n, f: fb };
  }
  function stemBuffers() {
    const out = { good: [], fair: [], poor: [], tick: [] };
    const ft = { good: [], fair: [], poor: [], tick: [] };
    for (const s of stems) {
      if (s.n < state.minN) continue;
      if (state.badOnly && !(s.offset === null || s.offset > 0.60)) continue;
      const g = grade(s), key = g === GOOD ? "good" : (g === FAIR ? "fair" : "poor");
      // Relative to the page's time origin so the shader compares small floats;
      // absolute ROS stamps are ~1.8e9 and lose all precision in a float32.
      const f = (s.seen === null || s.seen === undefined) ? NEVER : (s.seen - T0);
      out[key].push(s.x, s.y, s.bot, s.x, s.y, s.top);
      ft[key].push(f, f);
      const t = 0.28;
      out.tick.push(s.x - t, s.y, s.bot, s.x + t, s.y, s.bot,
                    s.x, s.y - t, s.bot, s.x, s.y + t, s.bot);
      ft.tick.push(f, f, f, f);
    }
    return { good: mkBuf2(out.good, ft.good), fair: mkBuf2(out.fair, ft.fair),
             poor: mkBuf2(out.poor, ft.poor), tick: mkBuf2(out.tick, ft.tick) };
  }
  let SB = null;
  function rebuild() { SB = stemBuffers(); draw(); }

  function selBuffer() {
    if (sel < 0) return mkBuf([]);
    const s = stems[sel], up = Math.max(s.top + P.extent * 0.06, s.top + 2);
    const r = 0.6, a = [];
    for (let k = 0; k <= 24; k++) {
      const t0 = k / 24 * Math.PI * 2, t1 = (k + 1) / 24 * Math.PI * 2;
      a.push(s.x + r * Math.cos(t0), s.y + r * Math.sin(t0), s.bot,
             s.x + r * Math.cos(t1), s.y + r * Math.sin(t1), s.bot);
    }
    a.push(s.x, s.y, s.bot, s.x, s.y, up);
    return mkBuf(a);
  }
  let selBuf = mkBuf([]);

  // ---- flight path ----
  const TR = P.traj;
  let trajBuf = null, trajT = null, trajXYZ = null, segBuf = mkBuf([]);
  if (TR) {
    trajXYZ = b64f32(TR.xyz_b64);
    trajT = b64f32(TR.t_b64);
    trajBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, trajBuf); gl.bufferData(gl.ARRAY_BUFFER, trajXYZ, gl.STATIC_DRAW);
    const tb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, tb); gl.bufferData(gl.ARRAY_BUFFER, trajT, gl.STATIC_DRAW);
    trajBuf = { pos: trajBuf, t: tb, n: TR.n };
    document.getElementById("trajRow").hidden = false;
    document.getElementById("flightBlock").hidden = false;
    document.getElementById("flightKv").innerHTML = [
      ["path", TR.length.toFixed(0) + " m"],
      ["duration", TR.duration.toFixed(0) + " s"],
      ["altitude", TR.z_lo.toFixed(1) + " – " + TR.z_hi.toFixed(1) + " m"],
      ["poses", TR.n]
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  }

  // The stretch of path flown while one stem was being observed. Its length IS the
  // observation baseline -- a stem seen only from one spot cannot be triangulated,
  // however many frames it appeared in, and this shows that at a glance.
  function segmentFor(s) {
    if (!TR || !s) return mkBuf([]);
    const span = Math.max(TR.t1 - TR.t0, 1e-9), a = [];
    for (let k = 0; k < TR.n; k++) {
      const abs = TR.t0 + trajT[k] * span;
      if (abs >= s.first && abs <= s.last) a.push(trajXYZ[k*3], trajXYZ[k*3+1], trajXYZ[k*3+2]);
    }
    return mkBuf(a);
  }

  // ---- timeline ----
  const SG = P.sight;
  const T0 = SG ? Math.min(SG.t0, TR ? TR.t0 : SG.t0) : (TR ? TR.t0 : 0);
  const T1 = SG ? Math.max(SG.t1, TR ? TR.t1 : SG.t1) : (TR ? TR.t1 : 1);
  const SPAN = Math.max(T1 - T0, 1e-6);
  let now = T1, playing = false, speed = 1, lastTick = 0;
  let sgObj = null, sgCam = null, sgT = null, sgIdx = null, rayBuf = mkBuf([]), droneBuf = mkBuf([]);
  if (SG) {
    sgObj = b64f32(SG.obj_b64); sgCam = b64f32(SG.cam_b64); sgT = b64f32(SG.t_b64);
    sgIdx = new Int32Array(b64u8(SG.idx_b64).buffer);
    document.getElementById("timeline").hidden = false;
    document.getElementById("hint").hidden = true;
    document.getElementById("tlTotal").textContent = stems.length;
  }

  // Sight lines live only for a moment on either side of `now`: drawn as camera to
  // detected point, so a stem observed from one spot shows as a tight fan and a
  // well-triangulated one as a wide one. That is the baseline, made visible.
  const RAY_WIN = 0.35;
  function raysAt(tAbs) {
    if (!SG) return { buf: mkBuf([]), n: 0 };
    const lo = tAbs - SG.t0 - RAY_WIN, hi = tAbs - SG.t0 + RAY_WIN, a = [];
    let i = 0, j = SG.n - 1;
    while (i < j) { const m = (i + j) >> 1; if (sgT[m] < lo) i = m + 1; else j = m; }
    for (let k = i; k < SG.n && sgT[k] <= hi; k++)
      a.push(sgCam[k*3], sgCam[k*3+1], sgCam[k*3+2], sgObj[k*3], sgObj[k*3+1], sgObj[k*3+2]);
    return { buf: mkBuf(a), n: a.length / 6 };
  }
  function droneAt(tAbs) {
    if (!TR) return mkBuf([]);
    const f = Math.min(Math.max((tAbs - TR.t0) / Math.max(TR.t1 - TR.t0, 1e-9), 0), 1);
    let i = 0, j = TR.n - 1;
    while (i < j) { const m = (i + j) >> 1; if (trajT[m] < f) i = m + 1; else j = m; }
    const x = trajXYZ[i*3], y = trajXYZ[i*3+1], z = trajXYZ[i*3+2], r = EXT * 0.012;
    return mkBuf([x-r,y,z, x+r,y,z, x,y-r,z, x,y+r,z, x,y,z-r, x,y,z+r]);
  }
  let sgPts = null, sgSpokes = null;
  if (SG) {
    const rgb = b64u8(SG.rgb_b64), stemXYZ = b64f32(SG.stem_b64);
    const col = new Float32Array(SG.n * 3);
    for (let i = 0; i < SG.n * 3; i++) col[i] = rgb[i] / 255;
    const mk = (arr) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW); return b; };
    sgPts = { pos: mk(sgObj), t: mk(sgT), col: mk(col), n: SG.n };
    // A spoke per sighting, to the merged position it was assigned to. Long spokes
    // crossing into a neighbouring tree are what a bad merge looks like.
    const sp = new Float32Array(SG.n * 6), st = new Float32Array(SG.n * 2),
          sc = new Float32Array(SG.n * 6);
    let m = 0;
    for (let i = 0; i < SG.n; i++) {
      if (sgIdx[i] < 0) continue;
      sp.set([sgObj[i*3], sgObj[i*3+1], sgObj[i*3+2],
              stemXYZ[i*3], stemXYZ[i*3+1], stemXYZ[i*3+2]], m*6);
      st.set([sgT[i], sgT[i]], m*2);
      sc.set([col[i*3], col[i*3+1], col[i*3+2], col[i*3], col[i*3+1], col[i*3+2]], m*6);
      m++;
    }
    sgSpokes = { pos: mk(sp.subarray(0, m*6)), t: mk(st.subarray(0, m*2)),
                 col: mk(sc.subarray(0, m*6)), n: m * 2 };
    // Lives here, not in the flight-path block: `const SG` is declared after
    // that one, so reading SG.merge there threw a temporal-dead-zone
    // ReferenceError that killed the whole viewer before anything drew.
    const MQ = SG.merge;
    if (MQ) {
      document.getElementById("mergeBlock").hidden = false;
      document.getElementById("mergeKv").innerHTML = [
        ["sightings", `${MQ.assigned} + ${MQ.unassigned} loose`],
        ["to own centre", `${MQ.own_med.toFixed(2)} m med, ${MQ.own_max.toFixed(2)} max`],
        ["closer to another", MQ.misassigned],
        ["within 0.10 m of a rival", MQ.ambiguous],
        ["stem spacing", `${MQ.nn_med.toFixed(2)} m med, ${MQ.nn_min.toFixed(2)} min`],
        [`pairs under ${MQ.radius} m`, MQ.close_pairs]
      ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
      document.getElementById("mergeNote").textContent =
        `${MQ.misassigned} sighting(s) sit closer to another stem's centre than their own, `
        + `so the assignment is self-consistent. But ${MQ.close_pairs} pairs of merged `
        + `stems ended up closer together than the ${MQ.radius} m they were clustered at — `
        + `leader clustering separates the seeds, then absorbing members moves the median. `
        + `Turn on raw sightings and spokes to judge whether those are two trees or one `
        + `split in two.`;
    }
    document.getElementById("sightRow").hidden = false;
    document.getElementById("spokeRow").hidden = false;
  }

  let nowDirty = true;
  function setNow(t) { now = Math.min(Math.max(t, T0), T1); nowDirty = true; }

  // Every sight line for one stem, across the whole flight. A stem triangulated from
  // a real arc shows a fan; one seen repeatedly from a standstill shows a pencil --
  // which is the difference `view baseline` reports as a number.
  function raysForStem(k) {
    if (!SG || k < 0) return mkBuf([]);
    const a = [];
    for (let i = 0; i < SG.n; i++) if (sgIdx[i] === k)
      a.push(sgCam[i*3], sgCam[i*3+1], sgCam[i*3+2], sgObj[i*3], sgObj[i*3+1], sgObj[i*3+2]);
    return mkBuf(a);
  }
  let selRays = mkBuf([]);

  const C = P.centre, EXT = P.extent;
  const cam = { t: C.slice(), yaw: -2.2, pitch: 0.42, d: EXT * 0.95 };
  const cam0 = JSON.parse(JSON.stringify(cam));
  const eye = () => { const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    return [cam.t[0] + cam.d * cp * Math.cos(cam.yaw), cam.t[1] + cam.d * cp * Math.sin(cam.yaw), cam.t[2] + cam.d * sp]; };
  const V = { sub: (a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], add:(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]],
    mul:(a,s)=>[a[0]*s,a[1]*s,a[2]*s], dot:(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],
    cross:(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],
    norm(a){ const l=Math.hypot(a[0],a[1],a[2])||1; return [a[0]/l,a[1]/l,a[2]/l]; } };
  function persp(f, asp, n, fa) { const t = 1/Math.tan(f/2), nf = 1/(n-fa), o = new Float32Array(16);
    o[0]=t/asp; o[5]=t; o[10]=(fa+n)*nf; o[11]=-1; o[14]=2*fa*n*nf; return o; }
  function look(e, c, up) { const z = V.norm(V.sub(e,c)), x = V.norm(V.cross(up,z)), y = V.cross(z,x);
    const o = new Float32Array(16);
    o[0]=x[0];o[1]=y[0];o[2]=z[0];o[4]=x[1];o[5]=y[1];o[6]=z[1];o[8]=x[2];o[9]=y[2];o[10]=z[2];
    o[12]=-V.dot(x,e);o[13]=-V.dot(y,e);o[14]=-V.dot(z,e);o[15]=1; return o; }
  function mul(a, b) { const o = new Float32Array(16);
    for (let c=0;c<4;c++) for (let r=0;r<4;r++)
      o[c*4+r]=a[r]*b[c*4]+a[4+r]*b[c*4+1]+a[8+r]*b[c*4+2]+a[12+r]*b[c*4+3];
    return o; }

  const state = { mode: 0, size: 2.0, alpha: 1.0, minN: 1, stem: true, tick: true,
                  badOnly: false, traj: true, sight: false, spoke: false };
  const on = (id, ev, fn) => document.getElementById(id).addEventListener(ev, fn);
  document.getElementById("cseg").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    document.querySelectorAll("#cseg button").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); state.mode = +b.dataset.m; });
  on("rPt", "input", (e) => { state.size = +e.target.value; document.getElementById("lPt").textContent = state.size.toFixed(1); });
  on("rOp", "input", (e) => { state.alpha = +e.target.value / 100; document.getElementById("lOp").textContent = e.target.value + "%"; });
  on("rMin", "input", (e) => { state.minN = +e.target.value; document.getElementById("lMin").textContent = e.target.value; rebuild(); });
  on("cStem", "change", (e) => { state.stem = e.target.checked; });
  on("cTick", "change", (e) => { state.tick = e.target.checked; });
  on("cBad", "change", (e) => { state.badOnly = e.target.checked; rebuild(); });
  if (TR) on("cTraj", "change", (e) => { state.traj = e.target.checked; });
  if (SG) {
    on("cSight", "change", (e) => { state.sight = e.target.checked; });
    on("cSpoke", "change", (e) => { state.spoke = e.target.checked;
      // Spokes are unreadable without the dots they start from.
      if (state.spoke && !state.sight) {
        state.sight = true; document.getElementById("cSight").checked = true;
      } });
  }
  if (SG) {
    const rng = document.getElementById("tlRange"), btn = document.getElementById("tlPlay");
    rng.addEventListener("input", (e) => { playing = false; btn.innerHTML = "&#9654;";
      setNow(T0 + SPAN * (+e.target.value / 1000)); });
    btn.addEventListener("click", () => {
      // Hitting play at the very end restarts, rather than doing nothing.
      if (!playing && now >= T1 - 1e-3) setNow(T0);
      playing = !playing; lastTick = performance.now();
      btn.innerHTML = playing ? "&#10073;&#10073;" : "&#9654;";
    });
    document.getElementById("tlSpeed").addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      document.querySelectorAll("#tlSpeed button").forEach(x => x.classList.remove("on"));
      b.classList.add("on"); speed = +b.dataset.x;
    });
    addEventListener("keydown", (e) => {
      if (e.code === "Space" && e.target === document.body) { e.preventDefault(); btn.click(); }
    });
  }
  on("reset", "click", () => { Object.assign(cam, JSON.parse(JSON.stringify(cam0))); });

  let drag = false, pan = false, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", (e) => { drag = true; pan = e.shiftKey || e.button === 2;
    lx = e.clientX; ly = e.clientY; canvas.classList.add("dragging"); canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointerup", () => { drag = false; canvas.classList.remove("dragging"); });
  canvas.addEventListener("pointercancel", () => { drag = false; canvas.classList.remove("dragging"); });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - lx, dy = e.clientY - ly; lx = e.clientX; ly = e.clientY;
    if (pan) { const ev = eye(), f = V.norm(V.sub(cam.t, ev)), r = V.norm(V.cross(f, [0,0,1])), u = V.cross(r, f);
      const s = cam.d * 0.0016;
      cam.t = V.add(cam.t, V.add(V.mul(r, -dx*s), V.mul(u, dy*s))); }
    else { cam.yaw -= dx*0.0055; cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch + dy*0.0055)); }
  }, { passive: true });
  canvas.addEventListener("wheel", (e) => { e.preventDefault();
    cam.d = Math.max(EXT*0.02, Math.min(EXT*6, cam.d * Math.exp(e.deltaY*0.0011))); }, { passive: false });

  let lw = 0, lh = 0;
  function resize() { const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = innerWidth*dpr; canvas.height = innerHeight*dpr;
    canvas.style.width = innerWidth+"px"; canvas.style.height = innerHeight+"px";
    gl.viewport(0,0,canvas.width,canvas.height); lw = innerWidth; lh = innerHeight; }
  resize(); addEventListener("resize", resize);

  const aPos = gl.getAttribLocation(pProg,"aPos"), aI = gl.getAttribLocation(pProg,"aI");
  const aBorn = gl.getAttribLocation(pProg,"aBorn");
  const uMapNow = gl.getUniformLocation(pProg,"uMapNow");
  const uMVP = gl.getUniformLocation(pProg,"uMVP"), uSize = gl.getUniformLocation(pProg,"uSize");
  const uLo = gl.getUniformLocation(pProg,"uLo"), uHi = gl.getUniformLocation(pProg,"uHi");
  const uMode = gl.getUniformLocation(pProg,"uMode"), uAlpha = gl.getUniformLocation(pProg,"uAlpha");
  const lPos = gl.getAttribLocation(lProg,"aPos"), lMVP = gl.getUniformLocation(lProg,"uMVP"),
        lC = gl.getUniformLocation(lProg,"uC");

  gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0.039, 0.051, 0.075, 1);

  function lines(buf, col, mvp, w) {
    if (!buf.n) return;
    gl.useProgram(lProg); gl.bindBuffer(gl.ARRAY_BUFFER, buf.b);
    gl.enableVertexAttribArray(lPos); gl.vertexAttribPointer(lPos, 3, gl.FLOAT, false, 0, 0);
    gl.uniformMatrix4fv(lMVP, false, mvp); gl.uniform4fv(lC, col);
    gl.lineWidth(w || 1.5); gl.drawArrays(gl.LINES, 0, buf.n);
  }

  const sPos = gl.getAttribLocation(sProg, "aPos"), sF = gl.getAttribLocation(sProg, "aFirst");
  const sMVP = gl.getUniformLocation(sProg, "uMVP"), sNow = gl.getUniformLocation(sProg, "uNow");
  const sC = gl.getUniformLocation(sProg, "uC"), sGhost = gl.getUniformLocation(sProg, "uGhost");
  function timedLines(buf, col, mvp, w) {
    if (!buf.n) return;
    gl.useProgram(sProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.b);
    gl.enableVertexAttribArray(sPos); gl.vertexAttribPointer(sPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, buf.f);
    gl.enableVertexAttribArray(sF); gl.vertexAttribPointer(sF, 1, gl.FLOAT, false, 0, 0);
    gl.uniformMatrix4fv(sMVP, false, mvp); gl.uniform4fv(sC, col);
    gl.uniform1f(sNow, SG ? (now - T0) : 1e19);
    gl.uniform1f(sGhost, SG ? 0.07 : 1.0);
    gl.lineWidth(w || 1.5); gl.drawArrays(gl.LINES, 0, buf.n);
    gl.disableVertexAttribArray(sF);
  }

  rebuild();
  function tick() {
    if (!SG) return;
    if (playing) {
      const t = performance.now();
      setNow(now + (t - lastTick) / 1000 * speed);
      lastTick = t;
      if (now >= T1) { playing = false; document.getElementById("tlPlay").innerHTML = "&#9654;"; }
    }
    if (!nowDirty) return;
    nowDirty = false;
    const r = raysAt(now); rayBuf = r.buf; droneBuf = droneAt(now);
    document.getElementById("tlRange").value = Math.round((now - T0) / SPAN * 1000);
    document.getElementById("tlTime").innerHTML =
      `${(now - T0).toFixed(1)}<span> / ${SPAN.toFixed(0)} s</span>`;
    let found = 0;
    for (const s of stems) if (s.seen !== null && s.seen !== undefined && s.seen <= now) found++;
    document.getElementById("tlFound").textContent = found;
    document.getElementById("tlActive").textContent = r.n;
  }

  function frame() {
    tick();
    if (innerWidth !== lw || innerHeight !== lh) resize();
    const mvp = mul(persp(50*Math.PI/180, canvas.width/canvas.height, Math.max(EXT*0.004,0.02), EXT*30),
                    look(eye(), cam.t, [0,0,1]));
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(pProg);
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, iBuf);
    gl.enableVertexAttribArray(aI); gl.vertexAttribPointer(aI, 1, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, bornBuf);
    gl.enableVertexAttribArray(aBorn); gl.vertexAttribPointer(aBorn, 1, gl.FLOAT, false, 0, 0);
    // No birth times, or no timeline to scrub: 1e19 puts every point in the past.
    gl.uniform1f(uMapNow, (P.map_t0 !== null && P.map_t0 !== undefined && SG)
                          ? (now - P.map_t0) : 1e19);
    gl.uniformMatrix4fv(uMVP, false, mvp);
    gl.uniform1f(uSize, state.size * Math.min(devicePixelRatio || 1, 2));
    gl.uniform1f(uLo, P.z_lo); gl.uniform1f(uHi, P.z_hi);
    gl.uniform1f(uMode, state.mode); gl.uniform1f(uAlpha, state.alpha);
    gl.drawArrays(gl.POINTS, 0, N);

    if (state.stem) {
      timedLines(SB.poor, [...POOR, 0.95], mvp, 2.0);
      timedLines(SB.fair, [...FAIR, 0.95], mvp, 2.0);
      timedLines(SB.good, [...GOOD, 0.95], mvp, 2.0);
    }
    if (state.tick) timedLines(SB.tick, [0.55, 0.62, 0.72, 0.5], mvp, 1.0);

    if (state.traj && trajBuf) {
      gl.useProgram(tProg);
      const tp = gl.getAttribLocation(tProg, "aPos"), tt = gl.getAttribLocation(tProg, "aT");
      gl.bindBuffer(gl.ARRAY_BUFFER, trajBuf.pos);
      gl.enableVertexAttribArray(tp); gl.vertexAttribPointer(tp, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, trajBuf.t);
      gl.enableVertexAttribArray(tt); gl.vertexAttribPointer(tt, 1, gl.FLOAT, false, 0, 0);
      gl.uniformMatrix4fv(gl.getUniformLocation(tProg, "uMVP"), false, mvp);
      gl.uniform1f(gl.getUniformLocation(tProg, "uA"), sel >= 0 ? 0.35 : 0.95);
      gl.uniform1f(gl.getUniformLocation(tProg, "uNowN"),
                   SG ? (now - TR.t0) / Math.max(TR.t1 - TR.t0, 1e-9) : 1.0);
      gl.lineWidth(1.8);
      gl.drawArrays(gl.LINE_STRIP, 0, trajBuf.n);
      gl.disableVertexAttribArray(tt);
    }
    if (sel >= 0 && segBuf.n) {
      gl.useProgram(lProg); gl.bindBuffer(gl.ARRAY_BUFFER, segBuf.b);
      gl.enableVertexAttribArray(lPos); gl.vertexAttribPointer(lPos, 3, gl.FLOAT, false, 0, 0);
      gl.uniformMatrix4fv(lMVP, false, mvp);
      gl.uniform4fv(lC, [0.725, 0.549, 1.0, 1.0]);
      gl.lineWidth(3.0); gl.drawArrays(gl.LINE_STRIP, 0, segBuf.n);
    }
    if (SG && (state.sight || state.spoke)) {
      gl.useProgram(gProg);
      const gp = gl.getAttribLocation(gProg, "aPos"), gt = gl.getAttribLocation(gProg, "aT"),
            gc = gl.getAttribLocation(gProg, "aCol");
      gl.uniformMatrix4fv(gl.getUniformLocation(gProg, "uMVP"), false, mvp);
      gl.uniform1f(gl.getUniformLocation(gProg, "uNow"), now - T0);
      gl.uniform1f(gl.getUniformLocation(gProg, "uGhost"), 0.0);
      const bind = (o) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, o.pos);
        gl.enableVertexAttribArray(gp); gl.vertexAttribPointer(gp, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, o.t);
        gl.enableVertexAttribArray(gt); gl.vertexAttribPointer(gt, 1, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, o.col);
        gl.enableVertexAttribArray(gc); gl.vertexAttribPointer(gc, 3, gl.FLOAT, false, 0, 0);
      };
      if (state.spoke) {
        bind(sgSpokes);
        gl.uniform1f(gl.getUniformLocation(gProg, "uAlpha"), 0.30);
        gl.uniform1f(gl.getUniformLocation(gProg, "uRound"), 0.0);
        gl.lineWidth(1.0); gl.drawArrays(gl.LINES, 0, sgSpokes.n);
      }
      if (state.sight) {
        bind(sgPts);
        gl.uniform1f(gl.getUniformLocation(gProg, "uSize"),
                     Math.max(3.0, state.size * 1.6) * Math.min(devicePixelRatio || 1, 2));
        gl.uniform1f(gl.getUniformLocation(gProg, "uAlpha"), 0.9);
        gl.uniform1f(gl.getUniformLocation(gProg, "uRound"), 1.0);
        gl.drawArrays(gl.POINTS, 0, sgPts.n);
      }
      gl.disableVertexAttribArray(gt); gl.disableVertexAttribArray(gc);
    }
    if (SG) {
      if (sel >= 0) lines(selRays, [0.322, 0.878, 0.769, 0.18], mvp, 1.0);
      lines(rayBuf, [0.322, 0.878, 0.769, 0.5], mvp, 1.2);
      lines(droneBuf, [1.0, 1.0, 1.0, 0.95], mvp, 2.4);
    }
    if (sel >= 0) lines(selBuf, [0.322, 0.878, 0.769, 1.0], mvp, 2.4);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
</script>
'''

if __name__ == '__main__':
    main()
