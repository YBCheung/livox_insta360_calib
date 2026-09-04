#!/usr/bin/env python3
"""Capture one calibration pair: a 360 panorama + a dense LiDAR cloud, in the
RAW LiDAR frame the real-time colorizer actually consumes.

Hold the rig COMPLETELY STILL while this runs. The MID-360 uses a non-repetitive
rosette scan, so a single frame is far too sparse to calibrate against; integrating
several seconds while stationary is what builds a dense cloud. Because nothing moves,
raw scans can simply be concatenated in the LiDAR frame -- no SLAM, no odometry, and
therefore no dependence on FAST-LIO's world-frame convention.

Why the saved cloud stays in the raw, tilted-with-the-mount frame by default:
lidar_camera_colorizer_node.cpp projects raw /livox/lidar points straight through
T_cam_lidar (p_c = r_cl_ * p_l + t_cl_) with NO real-time attitude correction --
odometry is only used for a separate feature. So if the LiDAR has a fixed mechanical
mounting tilt relative to the camera (e.g. pitched forward while the camera stays
level because the Insta360 stitches gravity-locked, FlowState/horizon-lock ON), that
tilt MUST stay baked into the calibrated T_cam_lidar, which means it must ALSO stay
baked into cloud.pcd here -- leveling it away would make the calibration wrong by
exactly the mount tilt at every frame, not more correct.

Gravity measurement (always on unless --no-record-imu): the MID-360's built-in IMU
(/livox/imu) has no onboard orientation fusion, but a STATIONARY accelerometer reads
pure +g pointing "up" in its own body frame -- exactly the LiDAR's frame, since the
IMU is rigidly co-mounted. Averaging that reading over the same still window this
script already requires gives a clean, empirical tilt measurement, written to
gravity.txt regardless of whether it's applied -- useful to confirm a known
mechanical mounting angle was actually reproduced during THIS capture (e.g. "rig was
held level, so the measured tilt should read close to the LiDAR's nominal mount
angle"). Pass --level to additionally rotate the saved cloud by that measurement --
only do this if your downstream consumer applies its own real-time attitude
correction and therefore expects a gravity-referenced cloud instead.

Outputs:
    <out>/panorama.png   equirectangular frame, for calib_prepare_views.py cut
    <out>/cloud.pcd      accumulated XYZI cloud, RAW (mount tilt included) unless --level
    <out>/gravity.txt    measured up vector, tilt angle, and whether it was applied

Run (with the Livox driver and insta360_ros_publisher.py already up):

    python3 calib_capture.py --seconds 10 --out calib_data

Then:
    python3 calib_prepare_views.py cut calib_data/panorama.png --out-dir views
"""

import argparse
import os
import struct
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Imu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from insta360_views import rotation_aligning


def write_pcd_binary(path, xyzi):
    """Write an XYZI cloud as a binary PCD (PCL-readable, no PCL dependency)."""
    n = len(xyzi)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.asarray(xyzi, dtype="<f4").tobytes())


class CalibCapture(Node):
    def __init__(self, args):
        super().__init__("calib_capture")
        self.args = args

        self.points = []
        self.n_scans = 0
        self.panorama = None
        self.accel_samples = []
        self.lock = threading.Lock()

        sensor_qos = QoSProfile(depth=20, history=HistoryPolicy.KEEP_LAST,
                                reliability=ReliabilityPolicy.BEST_EFFORT)

        # Image: try best_effort first (matches insta360_ros_publisher's default).
        self.create_subscription(CompressedImage, args.image_topic,
                                 self._image_cb, sensor_qos)

        if args.record_imu:
            self.create_subscription(Imu, args.imu_topic, self._imu_cb, sensor_qos)

        try:
            from livox_ros_driver2.msg import CustomMsg
        except ImportError:
            self.get_logger().fatal(
                "cannot import livox_ros_driver2.msg.CustomMsg -- source your workspace "
                "(source install/setup.bash) before running this.")
            raise
        self.create_subscription(CustomMsg, args.lidar_topic, self._lidar_cb, sensor_qos)

        self.get_logger().info(
            f"HOLD STILL. Accumulating {args.seconds}s from {args.lidar_topic} "
            f"and grabbing {args.image_topic} ...")
        self.start = self.get_clock().now()
        self.timer = self.create_timer(0.5, self._tick)
        self.done = False

    def _image_cb(self, msg):
        import cv2
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        with self.lock:
            self.panorama = img

    def _imu_cb(self, msg):
        with self.lock:
            if self.done:
                return
            a = msg.linear_acceleration
            self.accel_samples.append((a.x, a.y, a.z))

    def _lidar_cb(self, msg):
        with self.lock:
            if self.done:
                return
            for p in msg.points:
                # Drop the Livox "noisy/low-confidence" tag bits, as the colorizer does.
                if self.args.filter_tags and (p.tag & 0x30) not in (0x00, 0x10):
                    continue
                if p.x == 0.0 and p.y == 0.0 and p.z == 0.0:
                    continue
                self.points.append((p.x, p.y, p.z, float(p.reflectivity)))
            self.n_scans += 1

    def _tick(self):
        elapsed = (self.get_clock().now() - self.start).nanoseconds / 1e9
        with self.lock:
            npts, nscans = len(self.points), self.n_scans
            have_img, n_imu = self.panorama is not None, len(self.accel_samples)
        imu_str = f"imu={n_imu}  " if self.args.record_imu else ""
        self.get_logger().info(
            f"  {elapsed:5.1f}s  scans={nscans}  points={npts}  {imu_str}"
            f"panorama={'yes' if have_img else 'WAITING'}")

        if elapsed < self.args.seconds:
            return

        with self.lock:
            self.done = True
            pts = np.asarray(self.points, dtype=np.float32)
            pano = self.panorama
            accel = np.asarray(self.accel_samples, dtype=np.float64)

        os.makedirs(self.args.out, exist_ok=True)
        ok = True

        if len(pts) == 0:
            self.get_logger().error(f"no LiDAR points received on {self.args.lidar_topic}")
            ok = False
        else:
            # Measuring the tilt and *applying* it to the saved cloud are separate
            # decisions. The real-time colorizer projects raw /livox/lidar points
            # straight through T_cam_lidar with no attitude correction -- so if the
            # mount has a fixed tilt (e.g. LiDAR pitched forward relative to a level
            # camera), that tilt MUST stay baked into cloud.pcd for the calibrated
            # matrix to match real-time data. --level is therefore opt-in, for
            # capture setups that feed a *different* consumer that does its own
            # real-time attitude correction. Either way we still measure and report
            # the tilt (gravity.txt) as a diagnostic -- e.g. to confirm a known
            # mechanical mounting angle was actually reproduced during capture.
            if self.args.record_imu:
                if len(accel) < self.args.min_imu_samples:
                    self.get_logger().warn(
                        f"only {len(accel)} samples on {self.args.imu_topic} (need "
                        f"{self.args.min_imu_samples}) -- skipping gravity measurement. "
                        f"Is the topic name right / is the driver publishing IMU?")
                else:
                    mean_a = accel.mean(axis=0)
                    std_a = accel.std(axis=0)
                    up = mean_a / np.linalg.norm(mean_a)
                    tilt_deg = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
                    R = rotation_aligning(up, np.array([0.0, 0.0, 1.0]))
                    applied = self.args.level
                    if applied:
                        pts[:, :3] = (pts[:, :3].astype(np.float64) @ R.T).astype(np.float32)
                    self.get_logger().info(
                        f"measured tilt: {len(accel)} IMU samples, {tilt_deg:.2f}deg off gravity, "
                        f"accel std={np.round(std_a, 3)} m/s^2 (high std = rig wasn't still) -- "
                        f"{'APPLIED to cloud.pcd' if applied else 'saved cloud.pcd stays RAW (pass --level to apply)'}")
                    grav_path = os.path.join(self.args.out, "gravity.txt")
                    with open(grav_path, "w") as f:
                        f.write(f"# measured from {len(accel)} samples on {self.args.imu_topic}\n")
                        f.write(f"# up_measured: unit vector in the RAW LiDAR frame\n")
                        f.write(f"up_measured {up[0]:.9f} {up[1]:.9f} {up[2]:.9f}\n")
                        f.write(f"tilt_deg {tilt_deg:.6f}\n")
                        f.write(f"accel_std {std_a[0]:.6f} {std_a[1]:.6f} {std_a[2]:.6f}\n")
                        f.write(f"applied_to_cloud {'true' if applied else 'false'}\n")
                        f.write("# rotation that WOULD level cloud.pcd (p_level = R @ p_raw);\n")
                        f.write("# only actually applied above if applied_to_cloud is true\n")
                        f.write("rotation\n")
                        for row in R:
                            f.write(f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f}\n")
                    self.get_logger().info(f"wrote {grav_path}")

            if self.args.max_points and len(pts) > self.args.max_points:
                idx = np.random.default_rng(0).choice(len(pts), self.args.max_points,
                                                      replace=False)
                pts = pts[np.sort(idx)]
                self.get_logger().info(f"subsampled to {len(pts)} points")
            pcd_path = os.path.join(self.args.out, "cloud.pcd")
            write_pcd_binary(pcd_path, pts)
            rng = pts[:, :3]
            self.get_logger().info(
                f"wrote {pcd_path}  ({len(pts)} pts, "
                f"x[{rng[:,0].min():.1f},{rng[:,0].max():.1f}] "
                f"y[{rng[:,1].min():.1f},{rng[:,1].max():.1f}] "
                f"z[{rng[:,2].min():.1f},{rng[:,2].max():.1f}] m)")

        if pano is None:
            self.get_logger().error(f"no image received on {self.args.image_topic}")
            ok = False
        else:
            import cv2
            h, w = pano.shape[:2]
            png_path = os.path.join(self.args.out, "panorama.png")
            cv2.imwrite(png_path, pano)
            self.get_logger().info(f"wrote {png_path}  ({w}x{h})")
            if abs(w / h - 2.0) > 1e-3:
                self.get_logger().warn(
                    f"{w}x{h} is not 2:1 -- the equirectangular model assumes a full "
                    f"360x180 panorama. Calibration will be wrong.")

        self.get_logger().info("DONE" if ok else "FAILED -- see errors above")
        self.exit_code = 0 if ok else 1
        rclpy.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seconds", type=float, default=10.0,
                   help="how long to accumulate (default 10; longer = denser)")
    p.add_argument("--out", default="calib_data")
    p.add_argument("--lidar-topic", default="/livox/lidar")
    p.add_argument("--image-topic", default="/insta360/image_raw/compressed")
    p.add_argument("--imu-topic", default="/livox/imu")
    p.add_argument("--no-record-imu", dest="record_imu", action="store_false",
                   help="don't subscribe to IMU at all (skips the gravity.txt diagnostic)")
    p.add_argument("--level", action="store_true",
                   help="ALSO rotate the saved cloud.pcd so measured gravity becomes +Z. "
                        "Leave this off for the default real-time colorizer, which applies "
                        "T_cam_lidar to RAW /livox/lidar points with no attitude correction -- "
                        "any fixed LiDAR mounting tilt must stay baked into cloud.pcd for the "
                        "calibrated matrix to match real-time data. Only pass this if your "
                        "downstream consumer does its own real-time gravity correction.")
    p.add_argument("--min-imu-samples", type=int, default=20,
                   help="minimum /livox/imu samples required to trust the gravity measurement")
    p.add_argument("--max-points", type=int, default=3000000,
                   help="subsample above this many points (0 = keep all)")
    p.add_argument("--no-filter-tags", dest="filter_tags", action="store_false",
                   help="keep Livox points flagged noisy/low-confidence")
    args, _ = p.parse_known_args()

    rclpy.init()
    node = CalibCapture(args)
    node.exit_code = 1
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    code = getattr(node, "exit_code", 1)
    try:
        node.destroy_node()
    except Exception:
        pass
    sys.exit(code)


if __name__ == "__main__":
    main()
