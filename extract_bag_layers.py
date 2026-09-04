#!/usr/bin/env python3
"""Pull the two viewer layers out of a bag: the global map, and the flight path.

visualize_stems_3d.py needs a point cloud and a trajectory, and neither is a CSV the
matcher produces -- they come straight from the bag. This writes both, so the inputs
to the viewer are reproducible from the bag alone rather than from whatever ad-hoc
snippet happened to create them:

    laser_map_xyz.npy   (N,3) float32, world frame, from the LAST /Laser_map message
    laser_map_i.npy     (N,)  float32 intensity, alongside it
    laser_map_t.npy     (N,)  float64 -- the stamp at which each point FIRST appeared
    odom_txyz.npy       (M,4) float64 t,x,y,z from every /Odometry message

The birth times come free. laserMapping.cpp's publish_map does `*pcl_wait_pub +=
*laserCloudWorld` and then publishes the whole accumulation, so appending is a
push_back and message k is a strict prefix of message k+1. A point at index i was
therefore first published by the earliest message longer than i -- which needs only
the message SIZES and stamps, never a point-by-point comparison across 212 messages.
The prefix property is checked rather than assumed, on sampled consecutive pairs.

The last /Laser_map message is the whole map: laserMapping.cpp appends to pcl_wait_pub
once a second and never clears it, so each message is a superset of the one before.
That is a liability when subscribing live -- ~64 MB a second after ten minutes -- but
exactly what is wanted here, where only the final accumulation matters.

    python3 extract_bag_layers.py --bag ~/kuusamo/manual_l2 --out-dir ~/kuusamo
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bag_bbox_match as bbm


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bag', required=True)
    p.add_argument('--out-dir', default='.')
    p.add_argument('--storage-id', default='')
    p.add_argument('--map-topic', default='/Laser_map')
    p.add_argument('--odom-topic', default='/Odometry')
    p.add_argument('--prefix', default='', help='prepended to every output filename')
    args = p.parse_args()

    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    import rosbag2_py

    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, args.prefix + name)

    reader, types = bbm.open_bag(args.bag, args.storage_id)
    for t in (args.map_topic, args.odom_topic):
        if t not in types:
            raise SystemExit(f"{args.bag}: no {t}. Present: {', '.join(sorted(types))}")
    f = rosbag2_py.StorageFilter()
    f.topics = [args.map_topic, args.odom_topic]
    reader.set_filter(f)

    last_map, poses, n_map = None, [], 0
    map_msgs = []              # (n_points, stamp, raw bytes) -- bytes kept for the check
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == args.map_topic:
            last_map = data
            n_map += 1
            mm = deserialize_message(data, PointCloud2)
            map_msgs.append((mm.width * mm.height,
                             mm.header.stamp.sec + mm.header.stamp.nanosec * 1e-9,
                             mm.data))
        else:
            m = deserialize_message(data, Odometry)
            q = m.pose.pose.position
            poses.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, q.x, q.y, q.z))

    if last_map is None:
        raise SystemExit(f"{args.bag}: {args.map_topic} carried no messages")
    m = deserialize_message(last_map, PointCloud2)
    off = {fl.name: fl.offset for fl in m.fields}
    if not {'x', 'y', 'z'} <= off.keys():
        raise SystemExit(f"{args.map_topic}: no x/y/z fields")
    raw = np.frombuffer(m.data, np.uint8).reshape(-1, m.point_step)
    rd = lambda c: raw[:, off[c]:off[c] + 4].copy().view(np.float32).ravel()
    xyz = np.stack([rd(c) for c in 'xyz'], axis=1)
    inten = rd('intensity') if 'intensity' in off else np.zeros(len(xyz), np.float32)
    keep = np.isfinite(xyz).all(axis=1)
    xyz, inten = xyz[keep], inten[keep]
    np.save(out('laser_map_xyz.npy'), xyz.astype(np.float32))
    np.save(out('laser_map_i.npy'), inten.astype(np.float32))

    # --- per-point birth time -------------------------------------------------
    sizes = np.array([m[0] for m in map_msgs], dtype=np.int64)
    stamps = np.array([m[1] for m in map_msgs], dtype=np.float64)
    bad = np.flatnonzero(np.diff(sizes) < 0)
    if len(bad):
        raise SystemExit(f"{args.map_topic} shrank at message {int(bad[0])+1} "
                         f"({sizes[bad[0]]} -> {sizes[bad[0]+1]} points). This topic is "
                         f"not append-only here, so per-point birth times cannot be "
                         f"derived from sizes. Re-run without --with-times.")
    checks, bad_prefix = 0, []
    for k in np.linspace(0, len(map_msgs) - 2, min(6, max(len(map_msgs) - 1, 1))).astype(int):
        a, b = map_msgs[k][2], map_msgs[k + 1][2]
        checks += 1
        if not (len(b) >= len(a) and bytes(a) == bytes(b[:len(a)])):
            bad_prefix.append(int(k))
    if bad_prefix:
        raise SystemExit(f"{args.map_topic} message {bad_prefix[0]} is not a prefix of the "
                         f"next: the cloud is being reordered or rewritten, so birth times "
                         f"from sizes would be wrong.")
    born = stamps[np.searchsorted(sizes, np.arange(int(sizes[-1])), side='right')]
    born = born[:len(keep)][keep[:len(born)]] if len(born) >= len(keep) else born
    if len(born) != len(xyz):
        raise SystemExit(f"birth-time length {len(born)} != {len(xyz)} points kept")
    np.save(out('laser_map_t.npy'), born)
    print(f"growth     : {n_map} snapshots, {sizes[0]} -> {sizes[-1]} points over "
          f"{stamps[-1]-stamps[0]:.1f} s, prefix verified on {checks} pair(s) "
          f"-> {out('laser_map_t.npy')}")
    print(f"map        : {len(xyz)} points (last of {n_map} messages), frame "
          f"'{m.header.frame_id}' -> {out('laser_map_xyz.npy')}")

    if not poses:
        raise SystemExit(f"{args.bag}: {args.odom_topic} carried no messages")
    a = np.array(poses, dtype=np.float64)
    a = a[np.argsort(a[:, 0])]
    np.save(out('odom_txyz.npy'), a)
    step = np.linalg.norm(np.diff(a[:, 1:4], axis=0), axis=1)
    print(f"trajectory : {len(a)} poses, {step.sum():.1f} m over {a[-1,0]-a[0,0]:.1f} s, "
          f"altitude {a[:,3].min():.1f}..{a[:,3].max():.1f} m -> {out('odom_txyz.npy')}")


if __name__ == '__main__':
    main()
