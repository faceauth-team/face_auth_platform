"""
Real anti-spoofing inference — Silent-Face MiniFASNet ensemble.

This is the trained spoof classifier the spec (Section 6, "Liveness
Detection" row; FR-6) calls for: two MiniFASNet models at different crop
scales (2.7 and 4.0), softmax-summed, argmax over 3 classes where class 1
== "real". Weights live in models/anti_spoof/ (downloaded from the
Silent-Face-Anti-Spoofing release, Apache-2.0). Architecture is vendored
in app/ml/_minifasnet.py so the published state_dict loads exactly.

Crop geometry (CropImage) is ported verbatim from the original repo: the
face bbox is expanded by the model's scale factor and resized to 80x80,
which is the distribution the models were trained on (the surrounding
context — screen bezels, paper edges, moire — is part of what the model
keys on, which is why it needs the original frame, not the tight aligned
crop the embedder uses).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.core import config

_MODELS_DIR = Path(os.getenv("ANTISPOOF_MODELS_DIR", str(config.BASE_DIR / "models" / "anti_spoof")))
# The two released models — different receptive scales, used as an ensemble.
_MODEL_FILES = ["2.7_80x80_MiniFASNetV2.pth", "4_0_0_80x80_MiniFASNetV1SE.pth"]


@dataclass
class SpoofScore:
    is_real: bool
    real_score: float       # softmax prob of the "real" class (0..1)
    label: int              # argmax class (0/1/2); 1 == real


def _get_new_box(src_w, src_h, bbox, scale):
    x, y, box_w, box_h = bbox
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1
    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def _crop(org_img, bbox, scale, out_w, out_h):
    src_h, src_w = org_img.shape[:2]
    x1, y1, x2, y2 = _get_new_box(src_w, src_h, bbox, scale)
    patch = org_img[y1:y2 + 1, x1:x2 + 1]
    return cv2.resize(patch, (out_w, out_h))


class SilentFaceAntiSpoof:
    """Loads both MiniFASNet models once and scores BGR frames + bbox."""

    def __init__(self, models_dir: Path = _MODELS_DIR):
        import torch  # local import: optional heavy dep, only when liveness model is active
        from app.ml._minifasnet import MODEL_MAPPING, get_kernel, parse_model_name

        self._torch = torch
        self.device = torch.device("cpu")
        self._models = []  # list of (model, scale, h, w)

        for fname in _MODEL_FILES:
            path = models_dir / fname
            h_input, w_input, model_type, scale = parse_model_name(fname)
            kernel = get_kernel(h_input, w_input)
            model = MODEL_MAPPING[model_type](conv6_kernel=kernel).to(self.device)

            state_dict = torch.load(str(path), map_location=self.device)
            first_key = next(iter(state_dict))
            if first_key.startswith("module."):
                from collections import OrderedDict
                state_dict = OrderedDict((k[7:], v) for k, v in state_dict.items())
            model.load_state_dict(state_dict)
            model.eval()
            self._models.append((model, scale, h_input, w_input))

    def score(self, frame_bgr: np.ndarray, bbox_xywh_px) -> SpoofScore:
        """Ensemble softmax over both scales. bbox is absolute [x,y,w,h]."""
        torch = self._torch
        prediction = np.zeros((1, 3))
        for model, scale, h_in, w_in in self._models:
            patch = _crop(frame_bgr, bbox_xywh_px, scale, w_in, h_in)
            # NOTE: these MiniFASNet weights expect raw BGR pixels in the
            # [0,255] range (NOT normalized to [0,1] and NOT BGR->RGB
            # swapped). Verified empirically against the repo's own labeled
            # samples (T1=real, F1/F2=fake): only this variant discriminates.
            tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = model.forward(tensor)
                prob = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
            prediction += prob
        label = int(np.argmax(prediction))
        real_score = float(prediction[0][1] / len(self._models))
        return SpoofScore(is_real=(label == 1), real_score=real_score, label=label)


_singleton: Optional[SilentFaceAntiSpoof] = None
_load_failed = False


def get_antispoof() -> Optional[SilentFaceAntiSpoof]:
    """Returns the loaded Silent-Face ensemble, or None if torch/weights
    are unavailable (so liveness can fall back to the heuristic). Cached."""
    global _singleton, _load_failed
    if _singleton is not None or _load_failed:
        return _singleton
    try:
        if not all((_MODELS_DIR / f).exists() for f in _MODEL_FILES):
            _load_failed = True
            return None
        _singleton = SilentFaceAntiSpoof()
    except Exception:
        _load_failed = True
        _singleton = None
    return _singleton
