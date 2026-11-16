"""Finishing touches for a VR180 image (one eye, or left | right side by side), float H x W x C 0..1:
a soft blur that brings MiniMax-H3's over-sharp detail down to what real VR180 photos have, and a
vignette that fades the very rim of the 180-degree view to black."""
from __future__ import annotations

import cv2
import numpy as np

REFERENCE_EYE_PX = 2048  # blur_px is given for an eye this wide and scales with the real size
# Measured: a Gaussian of 0.6 px at 2048 per eye brings H3's finest detail (1 px band, +35-45% over
# the l2r right eye) to the right eye's level at every view angle; real VR180 eyes sit at that level.
DEFAULT_BLUR_PX = 0.6
DEFAULT_VIGNETTE_START_DEG = 85.0  # the fade starts this far from straight ahead (90 = the rim)


def eye_slices(width: int, height: int) -> list[slice]:
    """Column ranges of the eyes: two halves for a side-by-side image (width = 2 x height), else one."""
    if width == 2 * height:
        return [slice(0, height), slice(height, width)]
    return [slice(0, width)]


def view_angle_deg(height: int, width: int) -> np.ndarray:
    """Degrees from straight ahead for each pixel of one half-equirectangular eye (90 = the rim)."""
    lon = ((np.arange(width, dtype=np.float64) + 0.5) / width - 0.5) * np.pi
    lat = (0.5 - (np.arange(height, dtype=np.float64) + 0.5) / height) * np.pi
    cos_t = np.cos(lat)[:, None] * np.cos(lon)[None, :]
    return np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))


def vignette_gain(height: int, width: int, start_deg: float, strength: float) -> np.ndarray:
    """Per-pixel gain for one eye: 1 up to start_deg, easing (smoothstep) down to 1 - strength at the rim."""
    if strength <= 0 or start_deg >= 90:
        return np.ones((height, width), dtype=np.float32)
    t = np.clip((view_angle_deg(height, width) - start_deg) / (90.0 - start_deg), 0.0, 1.0)
    ease = t * t * (3.0 - 2.0 * t)
    return (1.0 - min(strength, 1.0) * ease).astype(np.float32)


def vignette(image: np.ndarray, start_deg: float = DEFAULT_VIGNETTE_START_DEG, strength: float = 1.0) -> np.ndarray:
    """The image with each eye's rim faded towards black (strength 1 = pure black at the very edge)."""
    h, w = image.shape[:2]
    out = image.astype(np.float32, copy=True)
    for cols in eye_slices(w, h):
        gain = vignette_gain(h, cols.stop - cols.start, start_deg, strength)
        out[:, cols] *= gain[..., None] if out.ndim == 3 else gain
    return out


def soften(image: np.ndarray, blur_px: float = DEFAULT_BLUR_PX) -> np.ndarray:
    """The image with a Gaussian blur of blur_px (for a 2048-wide eye; scaled to the real eye size),
    each eye on its own so nothing bleeds across the middle of a side-by-side image."""
    h, w = image.shape[:2]
    out = image.astype(np.float32, copy=True)
    if blur_px <= 0:
        return out
    for cols in eye_slices(w, h):
        eye = np.ascontiguousarray(out[:, cols])
        sigma = blur_px * (cols.stop - cols.start) / REFERENCE_EYE_PX
        out[:, cols] = cv2.GaussianBlur(eye, (0, 0), sigma, borderType=cv2.BORDER_REFLECT).reshape(eye.shape)
    return out
