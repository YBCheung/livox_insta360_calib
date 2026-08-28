#!/usr/bin/env python3
"""YOLO inference on a Hailo accelerator, for the virtual views cut from the panorama.

Keeps one HEF configured and its inference pipeline activated for the lifetime of the
object, because configuring a network group costs far more than a frame does -- doing
it per call would dominate the measurement and the runtime.

The HEF is expected to carry its NMS post-process on-chip (the YOLOv8/YOLO11 Hailo
export does by default), so the output vstream is HAILO_NMS_BY_CLASS: one array of
[y_min, x_min, y_max, x_max, score] per class, already thresholded and de-duplicated,
in normalised 0..1 coordinates. Note the y-first ordering -- reading it as x-first
puts every box on its side, which looks plausible enough on a square view to go
unnoticed.

An HEF only runs on the architecture it was compiled for: HailoRT rejects a HAILO8
HEF on a HAILO8L device outright. Check with `hailortcli parse-hef <file>` against
`hailortcli fw-control identify`.
"""

import os

import cv2
import numpy as np


def load_class_names(hef_path):
    """Read `names:` out of the metadata.yaml an Ultralytics Hailo export leaves beside the HEF."""
    meta = os.path.join(os.path.dirname(os.path.abspath(hef_path)), 'metadata.yaml')
    if not os.path.exists(meta):
        return {}
    names, in_names = {}, False
    with open(meta) as f:
        for line in f:
            if line.startswith('names:'):
                in_names = True
                continue
            if in_names:
                if not line.startswith((' ', '\t')):
                    break
                key, _, value = line.strip().partition(':')
                if key.strip().isdigit():
                    names[int(key)] = value.strip()
    return names


class HailoYolo:
    """One configured HEF, ready to run frames through."""

    def __init__(self, hef_path, conf=0.25, class_names=None):
        from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                                    InputVStreamParams, OutputVStreamParams, InferVStreams,
                                    FormatType)
        self.conf = conf
        self.hef_path = hef_path
        self.names = class_names if class_names is not None else load_class_names(hef_path)

        hef = HEF(hef_path)
        self.input_info = hef.get_input_vstream_infos()[0]
        self.output_info = hef.get_output_vstream_infos()[0]
        self.height, self.width = self.input_info.shape[:2]

        self._vdevice = VDevice(VDevice.create_params())
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        self._network_group = self._vdevice.configure(hef, cfg)[0]

        in_params = InputVStreamParams.make(self._network_group, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(self._network_group, format_type=FormatType.FLOAT32)

        # Hold both context managers open: activation is the expensive part.
        self._pipeline_cm = InferVStreams(self._network_group, in_params, out_params)
        self._pipeline = self._pipeline_cm.__enter__()
        self._activation_cm = self._network_group.activate(self._network_group.create_params())
        self._activation_cm.__enter__()

    @property
    def input_size(self):
        return (self.width, self.height)

    def label(self, cls_id):
        return self.names.get(cls_id, f"class {cls_id}")

    def infer(self, image):
        """BGR image -> [(x1, y1, x2, y2, score, cls), ...] in that image's pixel coords.

        The view tiles are square and so are these models' inputs, so a plain resize is
        already aspect-correct; the scale back to pixels is per-axis anyway, which keeps
        this honest if either ever stops being square.
        """
        h, w = image.shape[:2]
        if (w, h) != (self.width, self.height):
            resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image
        # The HEF wants RGB; the whole pipeline upstream of here is OpenCV BGR.
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        results = self._pipeline.infer({self.input_info.name: rgb[None]})
        per_class = results[self.output_info.name][0]

        out = []
        for cls_id, dets in enumerate(per_class):
            if len(dets) == 0:
                continue
            for y1, x1, y2, x2, score in dets:
                if score < self.conf:
                    continue
                out.append((x1 * w, y1 * h, x2 * w, y2 * h, float(score), cls_id))
        return out

    def close(self):
        for cm in (getattr(self, '_activation_cm', None), getattr(self, '_pipeline_cm', None)):
            if cm is not None:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass
        if getattr(self, '_vdevice', None) is not None:
            self._vdevice.release()
            self._vdevice = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw_detections(img, dets, model, color=(60, 220, 60)):
    for x1, y1, x2, y2, score, cls_id in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        text = f"{model.label(cls_id)} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(p1[1], th + 4)
        cv2.rectangle(img, (p1[0], ty - th - 4), (p1[0] + tw + 2, ty), color, -1)
        cv2.putText(img, text, (p1[0] + 1, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 1, cv2.LINE_AA)
