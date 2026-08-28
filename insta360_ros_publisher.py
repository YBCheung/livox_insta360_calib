#!/usr/bin/env python3
"""Publish the Insta360 ONE X5 equirectangular stream as a ROS 2 CompressedImage.

The camera enumerates as a UVC device and delivers a pre-stitched 360x180 panorama
(2880x1440 = 2:1). This node only captures and republishes -- the equirectangular
projection math lives in insta360_views.py and the C++ colorizer.

Deliberately a standalone process on its own topic namespace:

  * It defaults to /insta360/image_raw/compressed, NOT /camera/color/image_raw/...,
    so starting it can never collide with the Orbbec pipeline or with FAST-LIO2.
  * Capture runs on a separate thread from the ROS timer so a slow or stalled USB
    read cannot back up the executor.
  * Nothing here subscribes to the LiDAR topic.

Run:
    python3 insta360_ros_publisher.py --ros-args -p device:=0 -p jpeg_quality:=80

Check it is alive:
    ros2 topic hz /insta360/image_raw/compressed
"""

import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage


class Insta360Publisher(Node):
    def __init__(self):
        super().__init__('insta360_publisher')

        self.declare_parameter('device', 0)
        self.declare_parameter('width', 2880)
        self.declare_parameter('height', 1440)
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('jpeg_quality', 80)
        self.declare_parameter('topic', '/insta360/image_raw/compressed')
        self.declare_parameter('frame_id', 'insta360')
        # The panorama is large; best_effort avoids head-of-line blocking a slow consumer
        # would otherwise inflict on the publisher.
        self.declare_parameter('reliable_qos', False)
        self.declare_parameter('flip_180', False)

        self.device = self.get_parameter('device').value
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = float(self.get_parameter('fps').value)
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
        topic = self.get_parameter('topic').value
        self.frame_id = self.get_parameter('frame_id').value
        reliable = bool(self.get_parameter('reliable_qos').value)
        self.flip_180 = bool(self.get_parameter('flip_180').value)

        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        )
        self.pub = self.create_publisher(CompressedImage, topic, qos)

        self.cap = self._open_capture()

        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = True
        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._grab_thread.start()

        self.timer = self.create_timer(1.0 / max(self.fps, 1e-3), self._publish_latest)

        self._published = 0
        self._dropped = 0
        self._last_report = self.get_clock().now()

        self.get_logger().info(
            f"Insta360 publisher up: device={self.device} {self.width}x{self.height} "
            f"@{self.fps}Hz -> {topic} (jpeg q={self.jpeg_quality}, "
            f"{'reliable' if reliable else 'best_effort'})")

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if abs(actual_w - self.width) > 1 or abs(actual_h - self.height) > 1:
            self.get_logger().warn(
                f"Camera returned {actual_w:.0f}x{actual_h:.0f}, requested "
                f"{self.width}x{self.height}. The equirectangular projection assumes a full "
                f"360x180 panorama (2:1 aspect); a different mode will not be 2:1 and the "
                f"LiDAR fusion will be wrong.")
        elif abs(actual_w / max(actual_h, 1) - 2.0) > 1e-3:
            self.get_logger().warn(
                f"Resolution {actual_w:.0f}x{actual_h:.0f} is not 2:1. The colorizer's "
                f"equirectangular mode assumes a full 360x180 panorama.")

    def _open_capture(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.get_logger().warn(f"V4L2 backend failed for device {self.device}; "
                                   f"falling back to default backend.")
            cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open Insta360 capture device {self.device}")

        # FOURCC must be set before the resolution for this camera.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low, drop stale frames
        return cap

    def _grab_loop(self):
        """Continuously drain the USB stream so only the newest frame is ever published."""
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._dropped += 1
                continue
            stamp = self.get_clock().now().to_msg()
            with self._frame_lock:
                self._frame = (frame, stamp)

    def _publish_latest(self):
        with self._frame_lock:
            item = self._frame
            self._frame = None
        if item is None:
            return
        frame, stamp = item

        if self.flip_180:
            frame = cv2.flip(frame, -1)

        ok, buf = cv2.imencode('.jpg', frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            self._dropped += 1
            return

        msg = CompressedImage()
        # Stamp is taken at grab time, not encode time, so downstream time-sync against
        # the LiDAR reflects when the photons arrived rather than when we got around to it.
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.format = 'jpeg'
        msg.data = np.asarray(buf).tobytes()
        self.pub.publish(msg)
        self._published += 1

        now = self.get_clock().now()
        if (now - self._last_report).nanoseconds > 5e9:
            self.get_logger().info(
                f"published={self._published} dropped={self._dropped} "
                f"payload={len(msg.data) / 1024:.0f}KiB")
            self._last_report = now

    def destroy_node(self):
        self._running = False
        if self._grab_thread.is_alive():
            self._grab_thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = Insta360Publisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
