#!/usr/bin/env python3
"""Merge per-frame matches from bag_bbox_match.py into one object per physical thing.

`--csv` writes one row per detection per frame, so a sapling seen in forty frames is
forty rows. This collapses them and, on the way, answers the question that decides
whether collapsing is even meaningful: are those coordinates GLOBAL?

THE TEST
--------
There is no need to take the frame on trust; the data settles it. For every group of
sightings of one object, compare three numbers:

    world spread     how far the reported world position moves between sightings
    camera-relative  the same sightings expressed as (object - camera), i.e. what a
                     LiDAR-frame tool would have reported
    camera baseline  how far the camera itself travelled across those sightings

If the coordinates are global, world spread stays small while camera-relative spread
tracks the baseline -- the object stands still and the observer moves. If they were
camera-relative, the two would swap. The ratio of world spread to camera baseline is
the single number worth reading: well under 1 means global, near 1 means the position
is riding along with the camera.

That comparison also doubles as an accuracy estimate. Nothing else in this pipeline
measures repeatability -- one frame's answer looks equally confident whether it is
right or not, and only re-observation from a different vantage point exposes it.

MERGING
-------
A leader algorithm, not single linkage: take the most-sighted unassigned observation,
absorb everything within --radius in xy, repeat. Single linkage would chain across a
row of saplings and merge a whole stand into one object. Positions are combined with
a median, not a mean, so one bad frame cannot drag a stem.

Grouping is in xy only. These objects are vertical and their z extent is the thing
being measured, so including z would split one tall sapling into two.

    python3 merge_objects.py saplings.csv --radius 1.0 --out stems.csv
"""

import argparse
import csv
import math

import numpy as np


def load(path, min_conf, classes, min_points):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if float(r['conf']) < min_conf or int(r['n_obj']) < min_points:
                continue
            if classes and int(r['cls']) not in classes:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit(f"{path}: no rows survived the filters")
    g = lambda k: np.array([float(r[k]) for r in rows])
    return {
        'stamp': g('stamp'), 'cls': np.array([int(r['cls']) for r in rows]),
        'label': [r['label'] for r in rows], 'conf': g('conf'),
        'n_obj': g('n_obj'), 'range': g('range_m'),
        'c': np.column_stack([g('cx'), g('cy'), g('cz')]),
        'top': np.column_stack([g('top_x'), g('top_y'), g('top_z')]),
        'bot': np.column_stack([g('bot_x'), g('bot_y'), g('bot_z')]),
        'cam': np.column_stack([g('cam_x'), g('cam_y'), g('cam_z')]),
    }


def leader_cluster(xy, order, radius):
    """Leader clustering: seed with `order`, absorb neighbours, never chain."""
    used = np.zeros(len(xy), bool)
    groups = []
    for i in order:
        if used[i]:
            continue
        grp = (np.linalg.norm(xy - xy[i], axis=1) < radius) & ~used
        used |= grp
        groups.append(np.flatnonzero(grp))
    return groups


def frame_to_frame(d, gate):
    """Repeatability with no clustering in the loop.

    Between two consecutive frames a static sapling has not moved and the camera has
    barely moved, so nearest-neighbour association is unambiguous and the leftover
    displacement IS the measurement noise. This is the honest accuracy number: the
    per-object `spread` below cannot exceed --radius and therefore measures the
    clustering, not the pipeline.
    """
    order = np.argsort(d['stamp'])
    stamps = d['stamp'][order]
    c = d['c'][order]
    bounds = np.flatnonzero(np.diff(stamps) > 1e-6) + 1
    frames = np.split(np.arange(len(stamps)), bounds)
    deltas = []
    for a, b in zip(frames, frames[1:]):
        if not len(a) or not len(b) or stamps[b[0]] - stamps[a[0]] > 1.0:
            continue
        pa, pb = c[a], c[b]
        used = set()
        for i in range(len(pa)):
            dist = np.linalg.norm(pb - pa[i], axis=1)
            for j in np.argsort(dist):
                if dist[j] > gate:
                    break
                if j not in used:
                    used.add(j)
                    deltas.append(dist[j])
                    break
    return np.array(deltas)


def spread(p):
    """Robust radial spread about the median: the 90th percentile deviation."""
    if len(p) < 2:
        return 0.0
    return float(np.percentile(np.linalg.norm(p - np.median(p, axis=0), axis=1), 90))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv', help='the --csv written by bag_bbox_match.py')
    ap.add_argument('--radius', type=float, default=1.0, metavar='M',
                    help='xy distance within which two sightings are one object')
    ap.add_argument('--min-conf', type=float, default=0.0)
    ap.add_argument('--min-points', type=int, default=0,
                    help='drop sightings backed by fewer than this many LiDAR points')
    ap.add_argument('--min-sightings', type=int, default=1,
                    help='only report objects seen at least this many times')
    ap.add_argument('--classes', type=int, nargs='+')
    ap.add_argument('--out', help='write the merged objects here')
    args = ap.parse_args()

    d = load(args.csv, args.min_conf, args.classes, args.min_points)
    n = len(d['conf'])
    print(f"{n} sighting(s) over {d['stamp'].max() - d['stamp'].min():.1f} s, "
          f"camera moved {np.linalg.norm(d['cam'].max(0) - d['cam'].min(0)):.1f} m")

    # Seed by sighting density so the leader of a group is in its middle, not its edge.
    xy = d['c'][:, :2]
    density = np.array([np.count_nonzero(np.linalg.norm(xy - p, axis=1) < args.radius)
                        for p in xy])
    groups = leader_cluster(xy, np.argsort(-density), args.radius)

    stems, diag = [], []
    for g in groups:
        if len(g) < args.min_sightings:
            continue
        c = d['c'][g]
        cam = d['cam'][g]
        heights = d['top'][g][:, 2] - d['bot'][g][:, 2]
        s_world = spread(c)
        s_rel = spread(c - cam)                     # what a camera-frame tool reports
        baseline = (float(np.percentile(np.linalg.norm(cam - np.median(cam, axis=0),
                                                       axis=1), 90)) if len(g) > 1 else 0.0)
        labels, counts = np.unique([d['label'][i] for i in g], return_counts=True)
        stems.append({
            'label': labels[np.argmax(counts)], 'n': len(g),
            'x': np.median(c[:, 0]), 'y': np.median(c[:, 1]), 'z': np.median(c[:, 2]),
            'top_z': np.median(d['top'][g][:, 2]), 'bot_z': np.median(d['bot'][g][:, 2]),
            'height': float(np.median(heights)),
            'spread': s_world, 'baseline': baseline,
            'conf': float(np.max(d['conf'][g])),
            'pts': int(np.median(d['n_obj'][g])),
            'first': float(d['stamp'][g].min()), 'last': float(d['stamp'][g].max()),
        })
        if len(g) >= 4 and baseline > 1.0:
            diag.append((s_world, s_rel, baseline))

    stems.sort(key=lambda s: -s['n'])
    print(f"{len(stems)} object(s) after a {args.radius:.2f} m xy merge "
          f"(min {args.min_sightings} sighting(s))")

    # ---- global or camera-relative? --------------------------------------------
    if diag:
        w, r, b = (np.array([x[i] for x in diag]) for i in range(3))
        print(f"\nCOORDINATE FRAME, from {len(diag)} object(s) seen 4+ times with the "
              f"camera moving >1 m:")
        print(f"  world-frame spread       median {np.median(w):6.2f} m")
        print(f"  camera-relative spread   median {np.median(r):6.2f} m")
        print(f"  camera baseline          median {np.median(b):6.2f} m")
        print(f"  spread / baseline        median {np.median(w / b):6.3f}"
              f"   ({'GLOBAL' if np.median(w / b) < 0.35 else 'NOT global'})")
        print(f"  camera-relative / baseline      {np.median(r / b):6.3f}"
              f"   (should be ~1: the object moves in the camera's frame)")
    else:
        print("\nnot enough re-observation to test the frame "
              "(need objects seen 4+ times with >1 m of camera motion)")

    # ---- repeatability, independent of --radius --------------------------------
    print("\nREPEATABILITY between consecutive frames (no clustering involved):")
    for gate in (0.5, 1.0, 2.0):
        dd = frame_to_frame(d, gate)
        if len(dd) < 10:
            print(f"  gate {gate:.1f} m : too few pairs ({len(dd)})")
            continue
        print(f"  gate {gate:.1f} m : {len(dd):>5} pair(s)  median {np.median(dd):.2f} m"
              f"   p90 {np.percentile(dd, 90):.2f} m")
    print("  (stable across gates = a real noise floor, not an artefact of the gate)")

    seen = np.array([s['n'] for s in stems])
    good = [s for s in stems if s['n'] >= 4]
    print(f"\nsightings per object : median {int(np.median(seen))}, max {seen.max()}, "
          f"{len(good)} seen 4+ times")
    if good:
        h = np.array([s['height'] for s in good])
        sp = np.array([s['spread'] for s in good])
        print(f"height (4+ sightings): median {np.median(h):.2f} m, "
              f"p10 {np.percentile(h, 10):.2f}, p90 {np.percentile(h, 90):.2f}")
        print(f"per-object spread    : median {np.median(sp):.2f} m "
              f"(BOUNDED BY --radius {args.radius:.2f} m -- not an accuracy figure; "
              f"use the repeatability above)")

    print(f"\n{'#':>4} {'label':<10} {'n':>4} {'x':>8} {'y':>8} {'z':>7} "
          f"{'height':>7} {'spread':>7} {'pts':>5}")
    for i, s in enumerate(stems[:15]):
        print(f"{i:>4} {s['label']:<10} {s['n']:>4} {s['x']:>8.2f} {s['y']:>8.2f} "
              f"{s['z']:>7.2f} {s['height']:>7.2f} {s['spread']:>7.2f} {s['pts']:>5}")
    if len(stems) > 15:
        print(f"     ... {len(stems) - 15} more")

    if args.out:
        cols = ['label', 'n', 'x', 'y', 'z', 'top_z', 'bot_z', 'height', 'spread',
                'baseline', 'conf', 'pts', 'first', 'last']
        with open(args.out, 'w', newline='') as f:
            w_ = csv.DictWriter(f, fieldnames=cols)
            w_.writeheader()
            for s in stems:
                w_.writerow({k: (f"{s[k]:.4f}" if isinstance(s[k], float) else s[k])
                             for k in cols})
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
