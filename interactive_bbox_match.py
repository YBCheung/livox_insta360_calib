#!/usr/bin/env python3
"""Interactive mouse-driven UI for the bbox -> point-cloud object match.

Click-drag a box over the view image; on release it's matched against the
point cloud and the result (center/bottom/top keypoints) is drawn and
reported -- the same pipeline as bench_bbox_to_tree.py / visualize_bbox_match.py,
here driven by a real mouse-drawn box instead of a hardcoded one, standing in
for a YOLO detection.

    python3 interactive_bbox_match.py --data-dir data/calib_indoor_level --view 5 \
        --extrinsic configs/results/extrinsic_05.txt

Built on Tkinter + PIL (not matplotlib -- this environment's system matplotlib
is broken against numpy2's ABI). The full-cloud projection is computed once at
startup and cached; each drag only reruns the cheap mask/gate/cluster/keypoint
stages, so matches feel instant -- this is also a working demo of the
"cache the projection, don't redo it per box" optimization from the real-time/
Pi discussion.

Controls: drag to draw a box and match. 'c' clears the overlay. 'q' or Esc quits.
"""
import argparse
import os
import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

import bbox_match as bm
from project_pointcloud_to_view import load_pcd_xyzi, load_extrinsic, load_intrinsics
from bench_bbox_to_tree import project_all
from visualize_bbox_match import find_extrinsic, render_match_overlay, draw_legend


class BBoxMatchApp:
    def __init__(self, root, data_dir, view_idx, extrinsic_path, up_axis, scale, point_radius,
                 match='cluster'):
        self.match = match
        self.up_axis = up_axis
        self.scale = scale
        self.point_radius = point_radius

        data_dir = os.path.abspath(data_dir)
        views_dir = os.path.join(data_dir, 'views')
        intr, manifest = load_intrinsics(views_dir)
        self.fx, self.fy, self.cx, self.cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
        view_file, yaw, pitch = manifest[view_idx]

        extrinsic_path = find_extrinsic(data_dir, view_idx, extrinsic_path)
        self.R, self.t = load_extrinsic(extrinsic_path)

        root.title(f"bbox match -- {os.path.basename(data_dir)} view {view_idx} "
                   f"({os.path.basename(extrinsic_path)})")

        status0 = "loading cloud..."
        self.status = tk.Label(root, text=status0, anchor='w', justify='left',
                                font=('monospace', 10), padx=6, pady=4)
        self.status.pack(fill='x', side='bottom')
        root.update()

        pts = load_pcd_xyzi(os.path.join(data_dir, 'cloud.pcd'))
        self.xyz = pts[:, :3].astype(np.float64)

        t0 = time.perf_counter()
        self.u, self.v, self.Z, self.depth = project_all(
            self.xyz, self.R, self.t, self.fx, self.fy, self.cx, self.cy)
        project_ms = (time.perf_counter() - t0) * 1000

        # The shared matcher works on (u, v, range) of points IN FRONT of the camera --
        # the same arrays the live tool already has per frame. Filtering Z here is what
        # keeps points behind the camera from landing inside a box by coincidence.
        front = self.Z > 0.1
        self.fu = self.u[front].astype(np.float32)
        self.fv = self.v[front].astype(np.float32)
        self.fr = self.depth[front].astype(np.float32)
        self.match_view = bm.PinholeView(self.fx, self.fy, self.cx, self.cy,
                                         480, self.R, self.t)

        image_path = os.path.join(views_dir, view_file)
        self.base_img = Image.open(image_path).convert('RGB')
        self.W, self.H = self.base_img.size
        disp_img = self.base_img.resize((int(self.W * scale), int(self.H * scale)))

        self.canvas = tk.Canvas(root, width=disp_img.width, height=disp_img.height, cursor='cross')
        self.canvas.pack()
        self.photo = ImageTk.PhotoImage(disp_img)
        self.canvas_img_id = self.canvas.create_image(0, 0, anchor='nw', image=self.photo)

        self.drag_rect_id = None
        self.start_xy = None
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        root.bind('c', self.on_clear)
        root.bind('q', lambda e: root.destroy())
        root.bind('<Escape>', lambda e: root.destroy())

        self.status.config(text=(
            f"cloud: {len(self.xyz)} pts  |  projected once in {project_ms:.0f} ms  |  "
            f"drag a box on the image to match  |  'c' clear, 'q' quit"))

    def on_press(self, event):
        self.start_xy = (event.x, event.y)
        if self.drag_rect_id is not None:
            self.canvas.delete(self.drag_rect_id)
        self.drag_rect_id = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline='#ffe600', width=2)

    def on_drag(self, event):
        if self.start_xy is None:
            return
        x0, y0 = self.start_xy
        self.canvas.coords(self.drag_rect_id, x0, y0, event.x, event.y)

    def on_release(self, event):
        if self.start_xy is None:
            return
        x0, y0 = self.start_xy
        x1, y1 = event.x, event.y
        self.start_xy = None
        if self.drag_rect_id is not None:
            self.canvas.delete(self.drag_rect_id)
            self.drag_rect_id = None

        bx1, bx2 = sorted((x0 / self.scale, x1 / self.scale))
        by1, by2 = sorted((y0 / self.scale, y1 / self.scale))
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(self.W, bx2), min(self.H, by2)
        if bx2 - bx1 < 6 or by2 - by1 < 6:
            self.status.config(text="box too small -- drag a bigger one")
            return

        # 'fast' / 'both' run the live tool's matcher on this same box, so the two
        # approaches can be compared on identical input rather than across two
        # programs -- which is impossible live anyway, since the camera and the Hailo
        # device are both exclusive to one process.
        alt = None
        alt_ms = 0.0
        if self.match in ('fast', 'both'):
            t_alt = time.perf_counter()
            alt = bm.match(self.match_view, (bx1, by1, bx2, by2), self.fu, self.fv, self.fr,
                           self.match_view.camera_origin, 'fast',
                           anchor_axis='z' if self.up_axis == 2 else 'row')
            alt_ms = (time.perf_counter() - t_alt) * 1000

        t0 = time.perf_counter()
        img, info = render_match_overlay(
            self.base_img, self.xyz, self.u, self.v, self.Z, self.depth,
            self.R, self.t, self.fx, self.fy, self.cx, self.cy,
            (bx1, by1, bx2, by2), self.up_axis, self.point_radius)
        draw_legend(img)
        match_ms = (time.perf_counter() - t0) * 1000

        disp_img = img.resize((int(self.W * self.scale), int(self.H * self.scale)))
        self.photo = ImageTk.PhotoImage(disp_img)
        self.canvas.itemconfig(self.canvas_img_id, image=self.photo)

        if alt is not None:
            self.draw_alt(img, alt)

        if info is None:
            extra = ""
            if alt is not None:
                a = alt['central']
                extra = (f"  |  fast: {alt_ms:.1f} ms  {alt['n']} pts  "
                         f"central=({a[0]:.2f},{a[1]:.2f},{a[2]:.2f})")
            self.status.config(text=f"match took {match_ms:.1f} ms  |  no cluster survived"
                                    f" -- try a bigger/different box{extra}")
            return

        c, b, t_ = info['center'], info['bottom'], info['top']
        text = (f"cluster: {match_ms:5.1f} ms  |  candidates {info['n_candidates']:5d} -> "
                f"gated {info['n_gated']:5d} -> cluster {info['n_cluster']:5d}  |  "
                f"center=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  "
                f"bottom=({b[0]:.2f},{b[1]:.2f},{b[2]:.2f})  "
                f"top=({t_[0]:.2f},{t_[1]:.2f},{t_[2]:.2f})")
        if alt is not None:
            a = alt['central']
            disagree = float(np.linalg.norm(np.asarray(a) - np.asarray(c)))
            text += (f"\nfast   : {alt_ms:5.1f} ms  |  {alt['n']} pts  "
                     f"central=({a[0]:.2f},{a[1]:.2f},{a[2]:.2f})  "
                     f"|  the two disagree by {disagree:.2f} m")
        self.status.config(text=text)

    def draw_alt(self, img, anchor):
        """Mark where the live tool's cheap matcher put the same object (white)."""
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        tx, ty, cu, cv_, bx, by = anchor['pixels']
        for (x, y), r in (((tx, ty), 5), ((cu, cv_), 7), ((bx, by), 5)):
            draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 255, 255), width=2)
        draw.line([tx, ty, bx, by], fill=(255, 255, 255), width=1)

    def on_clear(self, event=None):
        disp_img = self.base_img.resize((int(self.W * self.scale), int(self.H * self.scale)))
        self.photo = ImageTk.PhotoImage(disp_img)
        self.canvas.itemconfig(self.canvas_img_id, image=self.photo)
        self.status.config(text="cleared -- drag a box on the image to match")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--view', type=int, default=0)
    p.add_argument('--extrinsic', help='pose file to use; defaults to that view\'s seed guess '
                                        '(pass a solved configs/results/extrinsic_NN.txt for a real check)')
    p.add_argument('--up-axis', type=int, default=2, help='0=x,1=y,2=z world-up column')
    p.add_argument('--scale', type=float, default=1.6, help='display zoom factor for the (small) view images')
    p.add_argument('--point-radius', type=float, default=1.6)
    p.add_argument('--match', choices=('cluster', 'fast', 'both'), default='cluster',
                   help="which matcher to run: 'cluster' (this tool's original "
                        "depth-gate + voxel components), 'fast' (the live tool's "
                        "nearest-range run), or 'both' to see them on one box")
    args = p.parse_args()

    root = tk.Tk()
    BBoxMatchApp(root, args.data_dir, args.view, args.extrinsic, args.up_axis, args.scale,
                 args.point_radius, args.match)
    root.mainloop()


if __name__ == '__main__':
    main()
