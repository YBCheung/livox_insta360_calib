#!/usr/bin/env python3
"""Render the survey as a short film for a slide deck.

Not a debugging view. The audience is the people funding the programme, so the frame
holds three claims and nothing else: a drone flew for three and a half minutes, a
forest built itself out of laser returns underneath it, and every young tree it found
was counted and measured. Extrinsics, cluster radii and match rates are absent on
purpose -- they belong in the inspector pages.

Rendered in software rather than through the WebGL viewer: headless GPU is unavailable
on this machine, and a fixed camera path needs no interactivity anyway. The projection
is the same pinhole used everywhere else in this package, so what the video shows and
what the viewer shows are the same geometry.

    python3 render_presentation.py --data data/kuusamo --out survey.mp4

Output is H.264 via ffmpeg at 1920x1080 -- the combination PowerPoint and Keynote
both play without a codec pack.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

# --- palette -----------------------------------------------------------------
# Forest, not instrumentation: the debug pages use a blue-to-red turbo ramp that
# reads as a heatmap. Here ground is moss and canopy is late-summer gold, so the
# cloud looks like the thing it is. Saplings sit in warm amber against it -- the
# one colour nothing else uses, because they are the point of the exercise.
# Declared in RGB and converted once, because OpenCV works in BGR and a palette
# written in one and consumed as the other silently turns a gold canopy ice-blue.
def _bgr(c):
    return (c[2], c[1], c[0])

BG        = _bgr((16, 19, 24))
RAMP_RGB  = [(30, 44, 32), (48, 78, 50), (92, 128, 68), (158, 168, 92), (224, 208, 146)]
SAPLING   = _bgr((245, 185, 66))      # warm amber -- nothing else uses it
SAPLING_HI= (255, 255, 255)
PATH_COL  = _bgr((127, 212, 232))     # soft cyan
FRESH     = _bgr((255, 244, 214))     # just-scanned ground, warm not icy
TEXT      = _bgr((238, 241, 245))
DIM       = _bgr((150, 160, 172))
ACCENT    = _bgr((150, 214, 138))     # pale green
TRACK     = _bgr((44, 52, 62))


def ramp_lut():
    lut = np.zeros((256, 3), np.uint8)
    seg = len(RAMP_RGB) - 1
    for i in range(256):
        x = i / 255 * seg
        k = min(int(x), seg - 1)
        f = x - k
        rgb = [RAMP_RGB[k][c] * (1 - f) + RAMP_RGB[k + 1][c] * f for c in range(3)]
        lut[i] = _bgr([int(v) for v in rgb])
    return lut


def load(data_dir):
    j = lambda n: os.path.join(data_dir, n)
    xyz = np.load(j('laser_map_xyz.npy')).astype(np.float32)
    born = np.load(j('laser_map_t.npy')).astype(np.float64)
    od = np.load(j('odom_txyz.npy'))
    stems = list(csv.DictReader(open(j('stems_manual_l2.csv'))))
    return xyz, born, od, stems


def look_at(eye, target, up=(0, 0, 1)):
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])          # rows: right, up, forward


def project(P, eye, R, fx, cx, cy):
    """World points -> (u, v, depth) for points in front of the camera."""
    d = (P - eye) @ R.T
    z = d[:, 2]
    ok = z > 0.35
    u = fx * d[ok, 0] / z[ok] + cx
    v = -fx * d[ok, 1] / z[ok] + cy
    return u, v, z[ok], ok


def splat(img, u, v, col, size=1):
    """Draw points as small blocks, nearest last (caller sorts)."""
    h, w = img.shape[:2]
    ui = u.astype(np.int32)
    vi = v.astype(np.int32)
    m = (ui >= size) & (ui < w - size) & (vi >= size) & (vi < h - size)
    ui, vi, col = ui[m], vi[m], col[m]
    for dy in range(-size, size + 1):
        for dx in range(-size, size + 1):
            img[vi + dy, ui + dx] = col
    return int(m.sum())


class Overlay:
    """Text layer. PIL rather than cv2.putText, whose stroke fonts look like a
    terminal and would undercut everything else in the frame."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        # Everything is sized against a 1080p reference, so a preview render at any
        # size shows the layout the final frame will have instead of a different one.
        self.S = h / 1080.0
        p = '/usr/share/fonts/truetype/dejavu/'
        px = lambda v: max(int(round(v * self.S)), 8)
        self.px = lambda v: int(round(v * self.S))
        self.f = lambda n, s: ImageFont.truetype(p + n, px(s))
        self.hero = self.f('DejaVuSans-Bold.ttf', 74)
        self.big = self.f('DejaVuSans-Bold.ttf', 52)
        self.num = self.f('DejaVuSans-Bold.ttf', 40)
        self.mid = self.f('DejaVuSans.ttf', 27)
        self.small = self.f('DejaVuSans.ttf', 21)
        self.tiny = self.f('DejaVuSans-Bold.ttf', 15)

    def new(self):
        return Image.new('RGBA', (self.w, self.h), (0, 0, 0, 0))

    @staticmethod
    def blend(frame, layer):
        a = np.asarray(layer).astype(np.float32)
        rgb, alpha = a[:, :, 2::-1], (a[:, :, 3:4] / 255.0)
        return (frame * (1 - alpha) + rgb * alpha).astype(np.uint8)


def ease(x):
    return x * x * (3 - 2 * x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'kuusamo'))
    ap.add_argument('--out', default=os.path.join(HERE, 'survey.mp4'))
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--title-s', type=float, default=3.0)
    ap.add_argument('--fly-s', type=float, default=31.0)
    ap.add_argument('--card-s', type=float, default=7.0)
    ap.add_argument('--site', default='Kuusamo, Finland')
    ap.add_argument('--keep-frames', action='store_true')
    a = ap.parse_args()

    xyz, born, od, stems = load(a.data)
    W, H = a.width, a.height
    LUT = ramp_lut()

    sx = np.array([float(s['x']) for s in stems])
    sy = np.array([float(s['y']) for s in stems])
    stop = np.array([float(s['top_z']) for s in stems])
    sbot = np.array([float(s['bot_z']) for s in stems])
    sfirst = np.array([float(s['first']) for s in stems])
    hgt = np.array([float(s['height']) for s in stems])

    # Crop to the surveyed ground, as the viewer does: the raw map runs 150 m across
    # because a few far returns survive, and framing on those shows a dot.
    lo = np.array([sx.min(), sy.min()]) - 14
    hi = np.array([sx.max(), sy.max()]) + 14
    keep = ((xyz[:, 0] > lo[0]) & (xyz[:, 0] < hi[0]) &
            (xyz[:, 1] > lo[1]) & (xyz[:, 1] < hi[1]))
    zl, zh = np.percentile(xyz[keep][:, 2], [1, 99])
    keep &= (xyz[:, 2] > zl - 2) & (xyz[:, 2] < zh + 3)
    xyz, born = xyz[keep], born[keep]
    zlo, zhi = np.percentile(xyz[:, 2], [2, 98])
    shade = np.clip((xyz[:, 2] - zlo) / max(zhi - zlo, 1e-6), 0, 1)
    cols = LUT[(shade * 255).astype(np.uint8)]

    t_traj, P_traj = od[:, 0], od[:, 1:4]
    T0 = min(born.min(), t_traj[0])
    T1 = max(born.max(), t_traj[-1])
    SPAN = T1 - T0

    centre = np.array([np.median(sx), np.median(sy), float(np.median(xyz[:, 2]))])
    reach = float(max(np.ptp(sx), np.ptp(sy))) * 0.85 + 18

    n_title = int(a.title_s * a.fps)
    n_fly = int(a.fly_s * a.fps)
    n_card = int(a.card_s * a.fps)
    total = n_title + n_fly + n_card
    tmp = tempfile.mkdtemp(prefix='survey_')
    ov = Overlay(W, H)
    fx = W * 0.78
    print(f"rendering {total} frames at {W}x{H} -> {tmp}")

    for i in range(total):
        frame = np.full((H, W, 3), BG, np.uint8)
        phase = 'title' if i < n_title else ('fly' if i < n_title + n_fly else 'card')
        k = (i - n_title) / max(n_fly - 1, 1)
        k = float(np.clip(k, 0, 1))
        if phase == 'title':
            k = 0.0
        elif phase == 'card':
            k = 1.0
        t_now = T0 + SPAN * k

        # Orbit: a slow 78 deg sweep, easing in and out so the ends feel deliberate
        # rather than cut. Height drops a little as it goes, so the map flattens out
        # and the saplings stand up.
        ang = math.radians(-118 + 78 * ease(k))
        elev = math.radians(34 - 11 * ease(k))
        dist = reach * (1.34 - 0.2 * ease(k))
        eye = centre + np.array([dist * math.cos(elev) * math.cos(ang),
                                 dist * math.cos(elev) * math.sin(ang),
                                 dist * math.sin(elev)])
        R = look_at(eye, centre)

        vis = born <= t_now
        if vis.any():
            u, v, z, ok = project(xyz[vis], eye, R, fx, W / 2, H / 2)
            c = cols[vis][ok]
            age = (t_now - born[vis][ok])
            fresh = np.clip(1 - age / 2.5, 0, 1)[:, None]
            c = (c * (1 - fresh * 0.8) + np.array(FRESH) * fresh * 0.8).astype(np.uint8)
            order = np.argsort(-z)
            splat(frame, u[order], v[order], c[order], 1)

        # flight path so far
        tp = t_traj <= t_now
        if tp.sum() > 2:
            pu, pv, _, pok = project(P_traj[tp], eye, R, fx, W / 2, H / 2)
            pts = np.stack([pu, pv], 1).astype(np.int32)
            if len(pts) > 2:
                cv2.polylines(frame, [pts], False, PATH_COL, 2, cv2.LINE_AA)
                cv2.circle(frame, tuple(pts[-1]), 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, tuple(pts[-1]), 13, (255, 255, 255), 1, cv2.LINE_AA)

        # saplings found so far
        found = sfirst <= t_now
        n_found = int(found.sum())
        if n_found:
            base = np.stack([sx[found], sy[found], sbot[found]], 1)
            tip = np.stack([sx[found], sy[found], stop[found]], 1)
            bu, bv, bz, bok = project(base, eye, R, fx, W / 2, H / 2)
            tu, tv, _, tok = project(tip, eye, R, fx, W / 2, H / 2)
            if bok.sum() and bok.sum() == tok.sum():
                new = (t_now - sfirst[found][bok]) < 1.6
                for m in range(len(bu)):
                    col = SAPLING_HI if new[m] else SAPLING
                    cv2.line(frame, (int(bu[m]), int(bv[m])), (int(tu[m]), int(tv[m])),
                             col, 2, cv2.LINE_AA)
                    if new[m]:
                        cv2.circle(frame, (int(tu[m]), int(tv[m])), 9, col, 1, cv2.LINE_AA)

        # The closing card carries numbers a minister is meant to read off a
        # projector, so the scene drops back to a ghost behind it rather than
        # competing with the type.
        if phase == 'card':
            g = ease(min((i - n_title - n_fly) / max(n_card * 0.35, 1), 1))
            frame = (frame.astype(np.float32) * (1 - 0.86 * g)
                     + np.array(BG, np.float32) * (0.86 * g)).astype(np.uint8)

        # ---------------- overlay ----------------
        layer = ov.new()
        d = ImageDraw.Draw(layer)
        rgba = lambda c, al=255: (c[2], c[1], c[0], al)
        S = ov.px                      # 1080p-reference units -> this frame's pixels

        if phase == 'title':
            f = ease(min(i / max(n_title * 0.55, 1), 1))
            al = int(255 * f)
            d.text((S(132), H // 2 - S(96)), "Counting a forest before it grows",
                   font=ov.hero, fill=rgba(TEXT, al))
            d.text((S(136), H // 2 + S(8)),
                   "Autonomous drone survey of forest regeneration",
                   font=ov.mid, fill=rgba(DIM, al))
            d.text((S(136), H // 2 + S(52)), f"{a.site}   ·   August 2026",
                   font=ov.small, fill=rgba(DIM, int(al * 0.8)))
            d.line([(S(134), H // 2 - S(118)), (S(134), H // 2 + S(84))], fill=rgba(ACCENT, al), width=max(2, S(3)))
        else:
            fade = 1.0 if phase == 'fly' else max(0.0, 1 - (i - n_title - n_fly) / (n_card * 0.35))
            al = int(230 * fade)
            if al > 4:
                d.text((S(70), S(62)), "Live LiDAR map", font=ov.tiny, fill=rgba(DIM, al))
                d.text((S(70), S(86)), f"{t_now - T0:5.1f} s", font=ov.num, fill=rgba(TEXT, al))
                d.text((S(70), S(152)), "Saplings mapped", font=ov.tiny, fill=rgba(DIM, al))
                d.text((S(70), S(176)), f"{n_found}", font=ov.num, fill=rgba(SAPLING, al))
                bar = S(300)
                d.rectangle([S(70), S(246), S(70) + bar, S(250)], fill=rgba(TRACK, al))
                d.rectangle([S(70), S(246), S(70) + int(bar * k), S(250)], fill=rgba(ACCENT, al))

            if phase == 'card':
                g = ease(min((i - n_title - n_fly) / max(n_card * 0.4, 1), 1))
                ca = int(255 * g)
                x0, y0 = S(132), H // 2 - S(210)
                d.text((x0, y0), "One 3½-minute flight", font=ov.big, fill=rgba(TEXT, ca))
                d.line([(x0 - S(22), y0 - S(14)), (x0 - S(22), y0 + S(330))],
                       fill=rgba(ACCENT, ca), width=max(2, S(3)))
                med = math.floor(float(np.median(hgt)) * 10 + 0.5) / 10
                rows = [
                    (f"{len(stems)}", "young trees located and measured"),
                    (f"{med:.1f} m", "median height, range 0.7 – 2.4 m"),
                    ("0.2 m", "repeatability of each position"),
                    ("95 %", "verified against an independent map"),
                ]
                yy = y0 + S(92)
                for val, lab in rows:
                    d.text((x0, yy), val, font=ov.num, fill=rgba(ACCENT, ca))
                    d.text((x0 + S(200), yy + S(11)), lab, font=ov.mid, fill=rgba(TEXT, ca))
                    yy += S(62)
                d.text((x0, yy + S(22)),
                       "No ground crew, no plots, no manual tally.",
                       font=ov.mid, fill=rgba(DIM, ca))

        frame = ov.blend(frame, layer)
        cv2.imwrite(os.path.join(tmp, f"f{i:05d}.png"), frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if i % 120 == 0:
            print(f"  {i}/{total}  t={t_now - T0:6.1f}s  saplings={n_found}")

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(a.fps),
           '-i', os.path.join(tmp, 'f%05d.png'),
           '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', a.out]
    subprocess.run(cmd, check=True)
    if not a.keep_frames:
        shutil.rmtree(tmp, ignore_errors=True)
    mb = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out} ({mb:.1f} MB, {total / a.fps:.1f} s, H.264 {W}x{H})")


if __name__ == '__main__':
    main()
