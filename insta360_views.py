#!/usr/bin/env python3
"""Projection math shared by the Insta360 360-camera pipeline.

Two coordinate frames are used throughout, and mixing them up is the single easiest
way to get a silently wrong extrinsic:

  panorama / spherical frame   x = forward, y = left, z = up
      The frame of Insta360Mapper in pixel_to_orientation.py, and the frame the
      colorizer expects for T_cam_lidar when
      color_mapping.projection_model == "equirectangular".

  virtual camera frame         x = right, y = down, z = forward (OpenCV)
      The frame every off-the-shelf calibrator (livox_camera_calib, OpenCV solvePnP)
      reports its result in.

A virtual rectilinear ("pinhole") view cut out of the panorama is an exact gnomonic
projection, so a pinhole-model calibrator is genuinely valid on it -- no
approximation. Because we synthesize the view ourselves, its intrinsics are known in
closed form and never need calibrating; only the LiDAR->camera extrinsic does.

Views are extracted with our own cv2.remap rather than py360convert so the yaw/pitch
sign convention is defined here and stays consistent with the C++ colorizer.
"""

import math

import numpy as np

# Panorama spherical frame axes, for reference/readability.
FORWARD = np.array([1.0, 0.0, 0.0])
LEFT = np.array([0.0, 1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])

# "True up" as seen in a per-view OpenCV frame (x=right, y=down, z=forward): every
# extracted view is cut with its vertical image axis aligned to the panorama's UP
# (see view_rotation's `down = cross(forward, right)`), so +y (down) is real-world
# down and this is real-world up. Use this as `cam_up` in gravity_correct_extrinsic
# for any per-view extrinsic (a seed guess or a solved extrinsic_NN.txt); use UP
# above instead for a composed panorama-frame T_cam_lidar.
OPENCV_VIEW_UP = np.array([0.0, -1.0, 0.0])


def pixel_to_bearing(x, y, width, height):
    """Equirectangular pixel -> unit bearing vector in the panorama spherical frame.

    Inverse of bearing_to_pixel.

    x increasing (rightward across the panorama) is a CLOCKWISE turn as seen from
    above -- i.e. toward -y (right), not +y (left). That is the opposite of the
    naive atan2 sign (x/width - 0.5), which is why azimuth is built from
    (0.5 - x/width) below. Confirmed empirically against this rig's actual
    Insta360 X5 stitcher output: the naive sign produced left/right-mirrored virtual
    views with yaw off by negation (e.g. a view asked for at +90 deg showed the
    content actually at -90/270 deg).
    """
    azimuth = (0.5 - np.asarray(x, dtype=float) / width) * 2.0 * math.pi
    elevation = (0.5 - np.asarray(y, dtype=float) / height) * math.pi
    cos_e = np.cos(elevation)
    return np.stack([cos_e * np.cos(azimuth),
                     cos_e * np.sin(azimuth),
                     np.sin(elevation)], axis=-1)


def bearing_to_pixel(d, width, height):
    """Bearing vector(s) in the panorama spherical frame -> equirectangular pixel.

    Inverse of pixel_to_bearing -- see its docstring for the x/azimuth sign.
    Azimuth is periodic, so x wraps into [0, width).
    """
    d = np.asarray(d, dtype=float)
    norm = np.linalg.norm(d, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        azimuth = np.arctan2(d[..., 1], d[..., 0])
        elevation = np.arcsin(np.clip(d[..., 2] / norm, -1.0, 1.0))
    x = (0.5 - azimuth / (2.0 * math.pi)) * width
    y = (0.5 - elevation / math.pi) * height
    x = np.mod(x, width)
    y = np.clip(y, 0.0, height - 1.0)
    return np.stack([x, y], axis=-1)


def virtual_view_intrinsics(fov_deg, out_hw):
    """Exact pinhole intrinsics of a synthesized rectilinear view.

    fov_deg may be a scalar (applied horizontally and vertically) or (h_fov, v_fov).
    Returns (fx, fy, cx, cy). These are exact by construction -- feed them to the
    calibrator as known intrinsics rather than solving for them.
    """
    out_h, out_w = out_hw
    if np.isscalar(fov_deg):
        fov_h = fov_v = float(fov_deg)
    else:
        fov_h, fov_v = float(fov_deg[0]), float(fov_deg[1])

    fx = (out_w / 2.0) / math.tan(math.radians(fov_h) / 2.0)
    fy = (out_h / 2.0) / math.tan(math.radians(fov_v) / 2.0)
    cx = out_w / 2.0
    cy = out_h / 2.0
    return fx, fy, cx, cy


def view_rotation(yaw_deg, pitch_deg):
    """Rotation taking a virtual-camera (OpenCV) vector into the panorama frame.

    p_panorama = view_rotation(yaw, pitch) @ p_virtual_camera

    yaw rotates about the panorama's up axis (positive = toward +y = left);
    pitch tilts the view up (positive) or down (negative).

    At yaw=pitch=0 this is the pure OpenCV->spherical axis swap:
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    """
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    forward = np.array([math.cos(pitch) * math.cos(yaw),
                        math.cos(pitch) * math.sin(yaw),
                        math.sin(pitch)])

    # Right-handed basis around the optical axis, keeping the horizon level.
    right = np.cross(forward, UP)
    n = np.linalg.norm(right)
    if n < 1e-9:
        # Looking straight up/down: the horizon is undefined, pick a stable fallback.
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
        n = np.linalg.norm(right)
        if n < 1e-9:
            right = np.array([0.0, -1.0, 0.0])
            n = 1.0
    right = right / n
    down = np.cross(forward, right)

    # Columns are the virtual camera's x/y/z axes expressed in the panorama frame.
    return np.column_stack([right, down, forward])


# Pure OpenCV -> panorama-spherical axis swap (a forward-looking view).
OPENCV_TO_SPHERICAL = view_rotation(0.0, 0.0)


def view_pixel_to_bearing(px, py, fov_deg, yaw_deg, pitch_deg, out_hw):
    """Virtual-view pixel -> unit bearing in the panorama spherical frame."""
    fx, fy, cx, cy = virtual_view_intrinsics(fov_deg, out_hw)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    rays = np.stack([(px - cx) / fx,
                     (py - cy) / fy,
                     np.ones_like(px)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays @ view_rotation(yaw_deg, pitch_deg).T


def view_pixel_to_panorama_pixel(px, py, fov_deg, yaw_deg, pitch_deg, out_hw, pano_wh):
    """Virtual-view pixel -> equirectangular panorama pixel.

    Use this to lift detections made on a rectilinear view back into panorama
    coordinates, which is the frame the colorizer's YOLO fusion consumes.
    """
    bearing = view_pixel_to_bearing(px, py, fov_deg, yaw_deg, pitch_deg, out_hw)
    return bearing_to_pixel(bearing, pano_wh[0], pano_wh[1])


def view_bbox_to_panorama_bbox(bbox, fov_deg, yaw_deg, pitch_deg, out_hw, pano_wh):
    """Axis-aligned view bbox -> axis-aligned panorama bbox.

    Samples the box outline rather than just its corners, because a straight edge in
    the rectilinear view becomes a curve on the panorama and the corners alone would
    understate the extent.

    Returns (x1, y1, x2, y2) in panorama pixels, or None if the box straddles the
    azimuth seam (x wraps), which cannot be expressed as one axis-aligned box.
    """
    x1, y1, x2, y2 = bbox
    n = 32
    ts = np.linspace(0.0, 1.0, n)
    edge_x = np.concatenate([np.full(n, x1), np.full(n, x2), x1 + (x2 - x1) * ts, x1 + (x2 - x1) * ts])
    edge_y = np.concatenate([y1 + (y2 - y1) * ts, y1 + (y2 - y1) * ts, np.full(n, y1), np.full(n, y2)])

    pano = view_pixel_to_panorama_pixel(edge_x, edge_y, fov_deg, yaw_deg, pitch_deg,
                                        out_hw, pano_wh)
    xs = pano[..., 0]
    ys = pano[..., 1]

    # Detect a seam crossing: points clustered at both ends of the x range.
    width = pano_wh[0]
    if xs.max() - xs.min() > width / 2.0:
        return None

    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def build_view_maps(fov_deg, yaw_deg, pitch_deg, out_hw, pano_wh):
    """Precompute cv2.remap sampling maps for one virtual view.

    The maps depend only on geometry, so build them once and reuse them per frame.
    """
    out_h, out_w = out_hw
    grid_x, grid_y = np.meshgrid(np.arange(out_w, dtype=np.float32),
                                 np.arange(out_h, dtype=np.float32))
    pano = view_pixel_to_panorama_pixel(grid_x, grid_y, fov_deg, yaw_deg, pitch_deg,
                                        out_hw, pano_wh)
    return (np.ascontiguousarray(pano[..., 0], dtype=np.float32),
            np.ascontiguousarray(pano[..., 1], dtype=np.float32))


def extract_view(panorama, fov_deg, yaw_deg, pitch_deg, out_hw, maps=None,
                 interpolation=None):
    """Cut an exact rectilinear view out of an equirectangular panorama.

    Pass precomputed `maps` from build_view_maps to avoid rebuilding them per frame.
    Defaults to bicubic interpolation, which preserves the intensity edges that
    edge-alignment calibrators depend on better than bilinear.
    """
    import cv2

    if interpolation is None:
        interpolation = cv2.INTER_CUBIC
    if maps is None:
        pano_h, pano_w = panorama.shape[:2]
        maps = build_view_maps(fov_deg, yaw_deg, pitch_deg, out_hw, (pano_w, pano_h))
    return cv2.remap(panorama, maps[0], maps[1], interpolation,
                     borderMode=cv2.BORDER_WRAP)


def compose_lidar_extrinsic(r_cv_lidar, t_cv_lidar, yaw_deg, pitch_deg):
    """Lift a per-view calibration result into the panorama frame.

    A calibrator solving against the virtual view returns the OpenCV-convention pose
        p_virtual = r_cv_lidar @ p_lidar + t_cv_lidar
    This composes it with the known view rotation to give the 4x4 T_cam_lidar the
    colorizer wants in equirectangular mode:
        p_panorama = R @ p_lidar + t
    """
    r_pano_cv = view_rotation(yaw_deg, pitch_deg)
    r = r_pano_cv @ np.asarray(r_cv_lidar, dtype=float).reshape(3, 3)
    t = r_pano_cv @ np.asarray(t_cv_lidar, dtype=float).reshape(3)

    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


def average_extrinsics(transforms):
    """Average several per-view T_cam_lidar estimates into one.

    Rotations are averaged via the SVD-projected mean (nearest proper rotation to the
    elementwise mean), translations arithmetically. Use this to fuse the independent
    solutions from views at different yaws; large disagreement between them is a red
    flag that one view's calibration did not converge.
    """
    transforms = [np.asarray(T, dtype=float) for T in transforms]
    if not transforms:
        raise ValueError("no transforms to average")

    r_mean = sum(T[:3, :3] for T in transforms) / len(transforms)
    u, _, vt = np.linalg.svd(r_mean)
    r = u @ vt
    if np.linalg.det(r) < 0:  # reflect back to a proper rotation
        u[:, -1] *= -1.0
        r = u @ vt

    T = np.eye(4)
    T[:3, :3] = r
    T[:3, 3] = sum(T_i[:3, 3] for T_i in transforms) / len(transforms)
    return T


def rotation_aligning(a, b):
    """Shortest-arc rotation matrix taking unit vector a onto unit vector b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        perp /= np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def gravity_correct_extrinsic(R, t, up_lidar, cam_up):
    """Re-derive a fixed T_cam_lidar for the rig's CURRENT tilt.

    The Insta360 stitches gravity-locked (FlowState/horizon-lock ON): its output is
    continuously re-leveled to true vertical no matter how the rig is tilted. A
    calibrated (R, t) is only exempt from needing that same re-leveling because it
    was solved while the rig happened to be level -- at that one attitude, the
    stitcher had nothing to correct, so the rigid camera-lidar mount transform and
    the calibrated extrinsic coincide. At any other attitude they don't: the raw
    LiDAR cloud tilts with the rig while the image does not, and a fixed (R, t)
    silently drifts off by exactly that difference.

    This reproduces the same re-leveling rotation the stitcher applied, from a
    gravity reading in the RAW LiDAR frame, and folds it into (R, t):

        up_lidar    current "up" in the raw LiDAR frame -- a stationary
                    accelerometer average (see gravity.txt / calib_capture.py)
                    while hovering/still, or, for a moving rig where that
                    assumption breaks, the world "up" axis rotated into the
                    current LiDAR body frame by a LiDAR-inertial odometry
                    estimate (e.g. FAST-LIO's orientation, whose world frame is
                    itself gravity-aligned at init) instead of raw accel.
        cam_up      the direction that is "true up" in (R, t)'s output frame --
                    insta360_views.UP for a composed panorama-frame T_cam_lidar,
                    OPENCV_VIEW_UP for a per-view OpenCV extrinsic.

    Returns (R_eff, t_eff): use these in place of (R, t) for this instant.
    """
    up_in_cam_frame = R @ (np.asarray(up_lidar, dtype=float) / np.linalg.norm(up_lidar))
    R_delta = rotation_aligning(up_in_cam_frame, cam_up)
    return R_delta @ np.asarray(R, dtype=float), R_delta @ np.asarray(t, dtype=float)


def format_t_cam_lidar_yaml(T, indent=26):
    """Render a 4x4 transform as the T_cam_lidar block for mid360*.yaml."""
    pad = " " * indent
    rows = []
    for i in range(4):
        rows.append(", ".join(f"{v: .9g}" for v in T[i]))
    body = (",\n" + pad).join(rows)
    return f"T_cam_lidar: [ {body} ]"
