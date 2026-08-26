#!/usr/bin/env python3
"""Calibration helper for Insta360 (equirectangular) <-> Livox MID-360 extrinsics.

A pinhole calibrator such as livox_camera_calib cannot consume an equirectangular
panorama directly. This tool bridges that gap in two steps:

  cut     Slice the panorama into virtual rectilinear views and emit each view's
          EXACT intrinsics. A gnomonic cut is a mathematically perfect pinhole
          image, so the calibrator's model assumption is genuinely satisfied --
          and because we synthesize the view, its intrinsics never need solving.

  compose Take the per-view T_cam_lidar the calibrator produced (OpenCV
          convention: x=right, y=down, z=forward), rotate it into the panorama
          spherical frame (x=forward, y=left, z=up), average the views, and print
          the T_cam_lidar block ready to paste into mid360_insta360.yaml.

Why 60 degrees rather than 90: gnomonic magnification at a view corner is
1/cos^2(theta). A square 90-degree view puts its corners at theta=54.7deg -> 3.0x
upsampling (soft, interpolated, no real detail). A 60-degree view puts them at
39.2deg -> 1.67x. Since panorama and view both sample at ~8 px/deg near the
equator, a 60-degree view is close to 1:1 at its centre.

Typical session:

    # 1. Grab a panorama while the rig is stationary, and accumulate a dense cloud.
    #    MID-360 is a non-repetitive scanner: one frame is far too sparse, integrate
    #    several seconds.

    python3 calib_prepare_views.py cut panorama.png --out-dir views/ --fov 60 --views 6

    # 2. Run livox_camera_calib once per view, using the printed intrinsics and the
    #    accumulated cloud. Record each resulting 4x4 (or R|t) in OpenCV convention.

    python3 calib_prepare_views.py compose --view-yaw 0   --extrinsic v0.txt \\
                                           --view-yaw 60  --extrinsic v60.txt \\
                                           --view-yaw 120 --extrinsic v120.txt
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import insta360_views as iv


def cmd_cut(args):
    import cv2

    pano = cv2.imread(args.panorama, cv2.IMREAD_COLOR)
    if pano is None:
        raise SystemExit(f"cannot read panorama: {args.panorama}")
    pano_h, pano_w = pano.shape[:2]

    if abs(pano_w / pano_h - 2.0) > 1e-3:
        print(f"WARNING: {pano_w}x{pano_h} is not 2:1. An equirectangular panorama must "
              f"cover the full 360x180 sphere or every angle below is wrong.\n")

    os.makedirs(args.out_dir, exist_ok=True)

    yaws = [args.yaw_start + i * (360.0 / args.views) for i in range(args.views)]
    fx, fy, cx, cy = iv.virtual_view_intrinsics(args.fov, (args.size, args.size))

    pano_px_deg = pano_w / 360.0
    view_px_deg = args.size / args.fov
    native_size = int(round(args.fov * pano_px_deg))

    print(f"panorama : {pano_w}x{pano_h}  ({pano_px_deg:.2f} px/deg horizontally)")
    print(f"views    : {args.views} x {args.fov}deg at {args.size}x{args.size} "
          f"({view_px_deg:.2f} px/deg)")
    print(f"pitch    : {args.pitch}deg")
    ratio = view_px_deg / pano_px_deg
    if ratio > 1.05:
        print(f"           centre sampling {ratio:.2f}x the panorama -- upsampled, so no "
              f"detail is gained.\n           --size {native_size} would be ~1:1 and cheaper.")
    elif ratio < 0.95:
        print(f"           centre sampling {ratio:.2f}x the panorama -- detail is being "
              f"discarded.\n           --size {native_size} would be ~1:1.")
    print(f"           corner magnification "
          f"{1.0 / np.cos(np.arctan(np.sqrt(2) * np.tan(np.radians(args.fov / 2)))) ** 2:.2f}x "
          f"(gnomonic, unavoidable)\n")
    print("Intrinsics are EXACT for every view (identical, since all share fov/size):")
    print(f"    fx = {fx:.6f}")
    print(f"    fy = {fy:.6f}")
    print(f"    cx = {cx:.6f}")
    print(f"    cy = {cy:.6f}")
    print("    distortion = 0 0 0 0 0   (a gnomonic cut has no lens distortion)\n")

    manifest = []
    for i, yaw in enumerate(yaws):
        view = iv.extract_view(pano, args.fov, yaw, args.pitch, (args.size, args.size))
        name = f"view_{i:02d}_yaw{yaw:+07.2f}_pitch{args.pitch:+06.2f}.png"
        path = os.path.join(args.out_dir, name)
        cv2.imwrite(path, view)
        manifest.append((name, yaw, args.pitch))
        print(f"  wrote {path}")

    intr_path = os.path.join(args.out_dir, "intrinsics.txt")
    with open(intr_path, "w") as f:
        f.write("# Exact intrinsics of the synthesized rectilinear views.\n")
        f.write("# Feed these to the calibrator as KNOWN intrinsics; do not solve for them.\n")
        f.write(f"fov_deg {args.fov}\nwidth {args.size}\nheight {args.size}\n")
        f.write(f"fx {fx:.9f}\nfy {fy:.9f}\ncx {cx:.9f}\ncy {cy:.9f}\n")
        f.write("k1 0\nk2 0\np1 0\np2 0\nk3 0\n\n")
        f.write("# view_file yaw_deg pitch_deg\n")
        for name, yaw, pitch in manifest:
            f.write(f"{name} {yaw:.6f} {pitch:.6f}\n")
    print(f"\n  wrote {intr_path}")

    print("\nNext: run your pinhole calibrator per view, then feed the results to "
          "`compose` with the matching --view-yaw.")
    print("Keep calibration targets near each view's centre (corner pixels are "
          "upsampled) and several metres away (the ONE X5's dual-lens stitch has "
          "~2-3cm of parallax up close).")


def rpy_to_matrix(yaw_deg, pitch_deg, roll_deg):
    """Mounting rotation of the camera relative to the LiDAR, both x=fwd, y=left, z=up.

    Returns Q, whose COLUMNS are the camera's axes expressed in LiDAR coordinates:
        Q[:,0] = where the camera's forward points, in LiDAR coords
        Q[:,1] = the camera's left,    Q[:,2] = the camera's up

    Note this is NOT the lidar->camera coordinate transform; that is Q.T. Confusing
    the two silently inverts every angle, so convert explicitly at the point of use.

    Signs read the way you would describe the mounting out loud (intrinsic Z-Y'-X'',
    so roll is about the camera's own final forward axis):

        yaw   > 0  camera front swings LEFT of the LiDAR front (CCW seen from above)
        pitch > 0  camera front tilts UP
        roll  > 0  camera's LEFT side lifts (it rolls to the right)
    """
    yaw, pitch, roll = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    # +pitch must raise the nose: with y=left and z=up, that is a NEGATIVE
    # right-handed rotation about +y, hence the flipped sin signs versus the
    # textbook Ry. Getting this backwards silently mirrors the guess vertically.
    ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def matrix_to_rpy(r):
    """Inverse of rpy_to_matrix. Returns (yaw, pitch, roll) in degrees."""
    pitch = math.asin(float(np.clip(r[2, 0], -1.0, 1.0)))
    if abs(r[2, 0]) < 1.0 - 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: looking straight up or down, yaw and roll merge
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def rotation_from_bearing_pairs(lidar_dirs, pano_dirs):
    """Least-squares rotation taking LiDAR bearings onto panorama bearings (Wahba/Kabsch).

    Bearing-only, so the lever arm is ignored -- fine for an initial guess, since a few
    centimetres of offset subtends well under a degree on a landmark several metres away.
    """
    a = np.asarray(lidar_dirs, dtype=float)
    b = np.asarray(pano_dirs, dtype=float)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)

    h = b.T @ a
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def cmd_guess(args):
    """Emit the per-view initial extrinsic a pinhole calibrator needs to start from."""
    # Q's columns are the camera axes in LiDAR coords; the transform that takes LiDAR
    # coordinates into camera coordinates is its transpose.
    q_cam = rpy_to_matrix(*args.rpy) if args.rpy else np.eye(3)
    r_pano = q_cam.T
    # --translation is the camera's ORIGIN in LiDAR coords (what a ruler measures).
    # T_cam_lidar maps p_cam = R p_lidar + t, so t = -R * (camera origin), not the
    # offset itself.
    cam_origin = np.asarray(args.translation, dtype=float)
    t_pano = -r_pano @ cam_origin

    print("Assumed mounting (both frames x=fwd, y=left, z=up):")
    print(f"  camera origin in LiDAR coords    : {cam_origin}")
    print(f"  rotation rpy(yaw,pitch,roll) deg : {args.rpy or [0, 0, 0]}")
    print(f"  camera forward, in LiDAR coords  : {np.round(q_cam[:, 0], 5)}")
    print(f"  camera up,      in LiDAR coords  : {np.round(q_cam[:, 2], 5)}")
    print(f"  => T_cam_lidar translation       : {np.round(t_pano, 5)}")
    print("\nIf the 360 camera is mounted upright and facing the same way as the LiDAR,")
    print("identity rotation is already correct -- only the lever arm matters.\n")

    yaws = [args.yaw_start + i * (360.0 / args.views) for i in range(args.views)]
    print("Per-view initial guesses in OpenCV convention (x=right, y=down, z=forward),")
    print("which is what livox_camera_calib / solvePnP expect:\n")
    for i, yaw in enumerate(yaws):
        r_pano_cv = iv.view_rotation(yaw, args.pitch)
        r_cv = r_pano_cv.T @ r_pano
        t_cv = r_pano_cv.T @ t_pano
        print(f"  view {i:02d}  yaw={yaw:+7.2f} pitch={args.pitch:+6.2f}")
        for row in range(3):
            print(f"      [{r_cv[row,0]: .6f} {r_cv[row,1]: .6f} {r_cv[row,2]: .6f} "
                  f"{t_cv[row]: .6f}]")
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            T = np.eye(4)
            T[:3, :3] = r_cv
            T[:3, 3] = t_cv
            path = os.path.join(args.out_dir, f"guess_{i:02d}_yaw{yaw:+07.2f}.txt")
            np.savetxt(path, T, fmt="%.9f")
            print(f"      -> {path}")
        print()


def cmd_bearing(args):
    """Read the mounting angles straight off a panorama.

    A 360 camera has no intrinsic "front": its front is DEFINED as whatever direction
    the stitcher places at the horizontal centre of the equirectangular image. So
    rather than measuring angles with a protractor, put a landmark at a known bearing
    from the LiDAR, find its pixel here, and read the offset off.
    """
    import cv2

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read image: {args.image}")
    h, w = img.shape[:2]
    if abs(w / h - 2.0) > 1e-3:
        print(f"WARNING: {w}x{h} is not 2:1; angles below assume a full 360x180 panorama.\n")

    print(f"panorama {w}x{h}   centre column u={w/2:.0f} is azimuth 0 (the camera's FRONT)")
    print(f"                   centre row    v={h/2:.0f} is elevation 0 (the horizon)\n")

    # Preferred path: landmark correspondences between the cloud and the panorama.
    # This needs no landmark placed "dead ahead" and no reading of the LiDAR's axis
    # marking -- the cloud coordinates carry the LiDAR bearing directly.
    if args.pair:
        lidar_dirs, pano_dirs = [], []
        print("Landmark correspondences (LiDAR frame vs panorama):\n")
        for u, v, x, y, z in args.pair:
            p = np.array([x, y, z], dtype=float)
            if np.linalg.norm(p) < 1e-6:
                raise SystemExit("a landmark at the LiDAR origin has no bearing")
            lidar_dirs.append(p / np.linalg.norm(p))
            pano_dirs.append(iv.pixel_to_bearing(u, v, w, h))
            az_l = math.degrees(math.atan2(y, x))
            el_l = math.degrees(math.atan2(z, math.hypot(x, y)))
            az_p = (0.5 - u / w) * 360.0
            el_p = (0.5 - v / h) * 180.0
            print(f"  pixel({u:7.1f},{v:7.1f})  <->  xyz({x:+7.2f},{y:+7.2f},{z:+7.2f}) "
                  f"@ {np.linalg.norm(p):5.1f}m")
            print(f"      LiDAR az/el = {az_l:+8.2f} / {el_l:+7.2f}   "
                  f"panorama az/el = {az_p:+8.2f} / {el_p:+7.2f}")

        if len(args.pair) == 1:
            # One correspondence pins the front direction but leaves roll free.
            # A landmark sitting at LiDAR azimuth az_l that appears at panorama
            # azimuth az_p means the camera is rotated by -(az_p - az_l): if it shows
            # up further LEFT in the image, the camera is aimed further RIGHT.
            yaw = -(az_p - az_l)
            yaw = (yaw + 180.0) % 360.0 - 180.0
            pitch = -(el_p - el_l)
            print(f"\nOne landmark fixes the front direction but not roll:")
            print(f"    --rpy {yaw:.2f} {pitch:.2f} 0")
            print("Add a second landmark ~90deg away to solve roll as well.")
            return

        # Solves pano_dir ~= r @ lidar_dir, i.e. r is the lidar->camera transform.
        # The mounting rotation reported to the user is its transpose.
        r = rotation_from_bearing_pairs(lidar_dirs, pano_dirs)
        yaw, pitch, roll = matrix_to_rpy(r.T)
        resid = [math.degrees(math.acos(np.clip(np.dot(r @ a, b), -1, 1)))
                 for a, b in zip(lidar_dirs, pano_dirs)]
        print(f"\nSolved rotation from {len(args.pair)} correspondences "
              f"(least squares over all of them):")
        print(f"    --rpy {yaw:.2f} {pitch:.2f} {roll:.2f}")
        print(f"\n  per-landmark residual: "
              f"{', '.join(f'{e:.2f}deg' for e in resid)}  (max {max(resid):.2f}deg)")
        if max(resid) > 5.0:
            print("  WARNING: residuals above ~5deg usually mean a mis-picked "
                  "correspondence.\n  Re-pick the worst landmark rather than trusting this.")
        print("\n  This is only a seed; the calibrator refines it. Anything within "
              "~5-10deg is fine.")
        return

    picks = []
    if args.pixel:
        picks = [tuple(args.pixel)]
    else:
        clicked = []

        def on_click(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked.append((x, y))
                print(f"  picked ({x}, {y})")

        scale = min(1.0, 1600.0 / w)
        disp = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()
        cv2.line(disp, (int(w * scale / 2), 0), (int(w * scale / 2), disp.shape[0]),
                 (0, 255, 255), 1)
        cv2.line(disp, (0, int(h * scale / 2)), (disp.shape[1], int(h * scale / 2)),
                 (0, 255, 255), 1)
        try:
            cv2.namedWindow("panorama - click the landmark, then press q", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("panorama - click the landmark, then press q", on_click)
            print("Click the landmark that sits dead ahead of the LiDAR, then press q.")
            print("(Yellow crosshair marks the camera's front / horizon.)")
            while True:
                cv2.imshow("panorama - click the landmark, then press q", disp)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            cv2.destroyAllWindows()
        except cv2.error:
            raise SystemExit("no display available -- pass --pixel U V instead "
                             "(open the PNG in any image viewer and read the coordinates)")
        picks = [(x / scale, y / scale) for x, y in clicked]

    if not picks:
        raise SystemExit("nothing picked")

    for u, v in picks:
        azimuth = (0.5 - u / w) * 360.0
        elevation = (0.5 - v / h) * 180.0
        print(f"\npixel ({u:.0f}, {v:.0f})")
        print(f"  azimuth   = {azimuth:+8.3f} deg   (0 = camera front, + = toward image left)")
        print(f"  elevation = {elevation:+8.3f} deg   (0 = horizon, + = up)")
        print(f"  bearing   = {np.round(iv.pixel_to_bearing(u, v, w, h), 5)}  (x=fwd, y=left, z=up)")
        print(f"\n  If that landmark is DEAD AHEAD of the LiDAR (+x, elevation 0), then:")
        print(f"      --rpy {azimuth:.2f} {elevation:.2f} 0")
        print(f"  i.e. yaw={azimuth:.2f} pitch={elevation:.2f}; roll needs a second "
              f"landmark off to one side.")


def _load_extrinsic(path):
    """Read a 4x4, or a 3x4, or 12/16 whitespace-separated numbers."""
    vals = []
    with open(path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            vals.extend(float(v) for v in line.replace(',', ' ').split())

    arr = np.asarray(vals, dtype=float)
    if arr.size == 16:
        T = arr.reshape(4, 4)
    elif arr.size == 12:
        T = np.eye(4)
        T[:3, :4] = arr.reshape(3, 4)
    else:
        raise SystemExit(f"{path}: expected 12 or 16 numbers, got {arr.size}")

    r = T[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-4):
        raise SystemExit(f"{path}: rotation block is not orthonormal; is this really "
                         f"a rigid transform in OpenCV convention?")
    return T


def cmd_compose(args):
    if len(args.view_yaw) != len(args.extrinsic):
        raise SystemExit(f"got {len(args.view_yaw)} --view-yaw but "
                         f"{len(args.extrinsic)} --extrinsic; they must pair up")
    pitches = args.view_pitch or [0.0] * len(args.view_yaw)
    if len(pitches) != len(args.view_yaw):
        raise SystemExit("--view-pitch must be given once per view, or not at all")

    transforms = []
    print("Per-view results, lifted into the panorama spherical frame "
          "(x=forward, y=left, z=up):\n")
    for yaw, pitch, path in zip(args.view_yaw, pitches, args.extrinsic):
        T_cv = _load_extrinsic(path)
        T = iv.compose_lidar_extrinsic(T_cv[:3, :3], T_cv[:3, 3], yaw, pitch)
        transforms.append(T)
        t = T[:3, 3]
        print(f"  yaw={yaw:+7.2f} pitch={pitch:+6.2f}  {os.path.basename(path)}")
        print(f"      t = [{t[0]: .5f}, {t[1]: .5f}, {t[2]: .5f}]")

    if len(transforms) > 1:
        # Disagreement between independent views is the most useful diagnostic there is:
        # a view whose calibration failed to converge shows up here, not in the average.
        ts = np.array([T[:3, 3] for T in transforms])
        spread = ts.max(axis=0) - ts.min(axis=0)
        print(f"\n  translation spread across views: "
              f"[{spread[0]:.4f}, {spread[1]:.4f}, {spread[2]:.4f}] m")

        angs = []
        for i in range(len(transforms)):
            for j in range(i + 1, len(transforms)):
                dr = transforms[i][:3, :3].T @ transforms[j][:3, :3]
                angs.append(np.degrees(np.arccos(np.clip((np.trace(dr) - 1) / 2, -1, 1))))
        print(f"  max pairwise rotation disagreement: {max(angs):.3f} deg")

        if spread.max() > 0.05 or max(angs) > 2.0:
            print("\n  WARNING: views disagree noticeably. Expect ~1-3cm of translation\n"
                  "  spread from the ONE X5's dual-lens baseline, but more than that\n"
                  "  (or >2deg of rotation) usually means one view did not converge.\n"
                  "  Drop the outlier and re-run rather than averaging it in.")

    T = iv.average_extrinsics(transforms)

    print("\n" + "=" * 72)
    print("Averaged T_cam_lidar (panorama spherical frame). Paste into your config:")
    print("=" * 72 + "\n")
    print(" " * 12 + "projection_model: \"equirectangular\"")
    print(" " * 12 + iv.format_t_cam_lidar_yaml(T, indent=26))
    print()

    if args.out:
        np.savetxt(args.out, T, fmt="%.9f")
        print(f"wrote {args.out}")

    print("Validate by running the colorizer with this extrinsic and inspecting\n"
          "/cloud_registered_color: colour boundaries should sit crisply on depth\n"
          "edges (building corners, trunks against sky). Smearing across an edge\n"
          "tells you which axis is still off.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cut", help="slice a panorama into virtual pinhole views")
    c.add_argument("panorama")
    c.add_argument("--out-dir", default="calib_views")
    c.add_argument("--fov", type=float, default=60.0,
                   help="view FOV in degrees (default 60; 90 softens the corners 3x)")
    c.add_argument("--size", type=int, default=720, help="view size in pixels (square)")
    c.add_argument("--views", type=int, default=6, help="number of yaws around the circle")
    c.add_argument("--yaw-start", type=float, default=0.0)
    c.add_argument("--pitch", type=float, default=0.0,
                   help="view pitch; keep near 0 to match the MID-360's -7..+52deg band")
    c.set_defaults(func=cmd_cut)

    b = sub.add_parser("bearing", help="read mounting angles off a panorama")
    b.add_argument("image")
    b.add_argument("--pixel", type=float, nargs=2, metavar=("U", "V"),
                   help="landmark pixel; omit to click it interactively")
    b.add_argument("--pair", type=float, nargs=5, action="append",
                   metavar=("U", "V", "X", "Y", "Z"),
                   help="landmark seen at panorama pixel (U,V) and at LiDAR-frame "
                        "point (X,Y,Z) from cloud.pcd. Repeat; 2+ pairs solve full "
                        "yaw/pitch/roll and need no 'dead ahead' placement.")
    b.set_defaults(func=cmd_bearing)

    g = sub.add_parser("guess", help="per-view initial extrinsic for the calibrator")
    g.add_argument("--translation", type=float, nargs=3, required=True,
                   metavar=("X", "Y", "Z"),
                   help="camera origin in LiDAR coords (x=fwd, y=left, z=up), metres -- "
                        "i.e. what a ruler measures from the LiDAR to the camera")
    g.add_argument("--rpy", type=float, nargs=3, metavar=("YAW", "PITCH", "ROLL"),
                   help="camera orientation in the panorama frame, degrees (default identity)")
    g.add_argument("--fov", type=float, default=60.0)
    g.add_argument("--views", type=int, default=6)
    g.add_argument("--yaw-start", type=float, default=0.0)
    g.add_argument("--pitch", type=float, default=0.0)
    g.add_argument("--out-dir", help="also write each guess as a 4x4 text file")
    g.set_defaults(func=cmd_guess)

    m = sub.add_parser("compose", help="lift per-view extrinsics into the panorama frame")
    m.add_argument("--view-yaw", type=float, action="append", required=True,
                   help="yaw of a calibrated view (repeat, paired with --extrinsic)")
    m.add_argument("--view-pitch", type=float, action="append",
                   help="pitch of each view (defaults to 0)")
    m.add_argument("--extrinsic", action="append", required=True,
                   help="file with that view's 4x4 or 3x4 T_cam_lidar, OpenCV convention")
    m.add_argument("--out", help="also write the averaged 4x4 here")
    m.set_defaults(func=cmd_compose)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
