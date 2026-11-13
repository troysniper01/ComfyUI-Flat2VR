"""Flat2VR: VR180 flat2vr placement for ComfyUI: a vendored copy of vr180.project / vr180.views
(the code the LoRA was trained with). Must stay identical to training: the test
tests/test_comfy_nodes.py in the training project renders the same photos with both
and requires byte-identical canvases and identical captions.

numpy + OpenCV only (no ComfyUI imports), so it can be tested on its own.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

CANVAS_GREY = 128
EQ_PX = 2048
# The photo shapes (width:height) the LoRA was trained on: vr180.project.ASPECTS, the shapes
# the picker offered (16:9, 3:2, 4:3, 5:4, 1:1, landscape and portrait).
TRAINED_ASPECTS = ((16, 9), (3, 2), (4, 3), (5, 4), (1, 1), (4, 5), (3, 4), (2, 3), (9, 16))
FULL_FRAME_DIAG_MM = 43.27  # 35 mm-equivalent focal lengths refer to a 36x24 mm frame's diagonal

# The training caption: the fixed instruction (the trigger), then optional parts in this order.
INSTRUCTION = ("Fill in the grey area around the photo to complete this VR180 image: one seamless "
               "180-degree half-equirectangular scene. Keep the photo itself exactly as it is.")
# The l2r LoRA's caption (fixed: it was trained with this one text only).
L2R_INSTRUCTION = ("Create the right-eye image for this VR180 left-eye image: the same 180-degree "
                   "half-equirectangular scene, seen from the right eye's position.")
POV_PHRASE = "POV body: visible."
POV_NONE_PHRASE = "POV body: none."  # the v5 data (the MiniMax-H3 LoRA) labels no-body frames too
WATERMARK_PHRASE = "Watermark: visible."
VIGNETTE_PHRASE = "Vignette: visible."


@dataclass(frozen=True)
class View:
    width: int
    height: int
    focal: float  # pixels
    yaw: float  # radians
    pitch: float
    roll: float


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Camera-to-world rotation: roll about the view axis, then pitch, then yaw."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    r_pitch = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    r_roll = np.array([[cr, sr, 0], [-sr, cr, 0], [0, 0, 1]])
    return r_yaw @ r_pitch @ r_roll


def _even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def view_for(diag_fov_deg: float, aspect: float, yaw_deg: float, pitch_deg: float,
             roll_deg: float, eq_width: int) -> View:
    """A view at the canvas's pixel density: focal = equirect pixels per radian."""
    focal = eq_width / math.pi
    half_diag = focal * math.tan(math.radians(diag_fov_deg) / 2)
    height = 2 * half_diag / math.sqrt(1 + aspect * aspect)
    return View(_even(height * aspect), _even(height), focal,
                math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg))


def view_for_photo(width: int, height: int, yaw: float, pitch: float, diag_fov: float, eq_px: int = EQ_PX) -> View:
    """The view of a photo of this shape (level: roll 0)."""
    return view_for(diag_fov, width / height, yaw, pitch, 0.0, eq_px)


def resize_photo(photo: np.ndarray, view: View) -> np.ndarray:
    """The photo at its view's size: area averaging when shrinking, cubic when enlarging."""
    if photo.shape[:2] == (view.height, view.width):
        return photo
    shrink = view.width * view.height < photo.shape[0] * photo.shape[1]
    return cv2.resize(photo, (view.width, view.height), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_CUBIC)


def to_equirect(snapshot: np.ndarray, view: View, eq_w: int, eq_h: int,
                interpolation: int = cv2.INTER_CUBIC) -> tuple[np.ndarray, np.ndarray]:
    """Project a snapshot back onto a half-equirect canvas; returns (image, valid mask)."""
    v, u = np.mgrid[0:eq_h, 0:eq_w].astype(np.float64)
    lon = (u + 0.5) / eq_w * math.pi - math.pi / 2
    lat = math.pi / 2 - (v + 0.5) / eq_h * math.pi
    world = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], axis=-1)
    cam = world @ rotation(view.yaw, view.pitch, view.roll)  # world -> camera (R is orthonormal)
    in_front = cam[..., 2] > 1e-6
    z = np.where(in_front, cam[..., 2], 1.0)
    x = view.focal * cam[..., 0] / z + view.width / 2 - 0.5
    y = -view.focal * cam[..., 1] / z + view.height / 2 - 0.5
    # keep a 1 px margin so interpolation never reads outside the snapshot
    valid = in_front & (x >= 1) & (x <= view.width - 2) & (y >= 1) & (y <= view.height - 2)
    image = cv2.remap(snapshot, x.astype(np.float32), y.astype(np.float32), interpolation,
                      borderMode=cv2.BORDER_CONSTANT)
    return image, valid


def fits_in_180(view: View, per_edge: int = 64) -> bool:
    """Whether the whole photo lies inside the 180-degree view (every border ray in front)."""
    t = np.linspace(0, 1, per_edge)
    w, h = view.width - 0.5, view.height - 0.5
    xs = np.concatenate([t * w, t * w, np.zeros(per_edge) - 0.5, np.full(per_edge, w)])
    ys = np.concatenate([np.zeros(per_edge) - 0.5, np.full(per_edge, h), t * h, t * h])
    cam = np.stack([(xs + 0.5 - view.width / 2) / view.focal, -(ys + 0.5 - view.height / 2) / view.focal,
                    np.ones_like(xs)], axis=-1)
    cam /= np.linalg.norm(cam, axis=-1, keepdims=True)
    rays = cam @ rotation(view.yaw, view.pitch, view.roll).T
    return bool((rays[:, 2] > 0).all())


def place_photo(photo: np.ndarray, yaw: float, pitch: float, diag_fov: float,
                eq_px: int = EQ_PX) -> tuple[np.ndarray, np.ndarray]:
    """An RGB uint8 photo placed on the grey 180-degree canvas, exactly as in training.
    Returns (canvas, photo mask)."""
    view = view_for_photo(photo.shape[1], photo.shape[0], yaw, pitch, diag_fov, eq_px)
    canvas, valid = to_equirect(resize_photo(photo, view), view, eq_px, eq_px)
    canvas[~valid] = CANVAS_GREY
    return canvas, valid


def trained_crop(width: int, height: int) -> tuple[int, int, int, int, str]:
    """The centred crop (x, y, w, h) that gives a width x height photo the nearest trained shape
    while keeping as many of its pixels as possible, and that shape's name ("4:3", "9:16", ...)."""
    best = None
    for rw, rh in TRAINED_ASPECTS:
        if width * rh > height * rw:  # wider than this shape: trim the sides
            w, h = min(width, round(height * rw / rh)), height
        else:  # taller: trim top and bottom
            w, h = width, min(height, round(width * rh / rw))
        if best is None or w * h > best[2] * best[3]:
            best = ((width - w) // 2, (height - h) // 2, w, h, f"{rw}:{rh}")
    return best


def _half_up(v: float) -> int:
    return math.floor(v + 0.5)  # as JavaScript's Math.round, so the placer computes the same crop


def crop_rect(width: int, height: int, spec: str = "auto") -> tuple[int, int, int, int, str]:
    """The crop (x, y, w, h) and shape a crop spec asks for. "auto" is `trained_crop`; otherwise
    "<shape>,<zoom>,<cx>,<cy>": a trained shape, the size as a fraction of the largest crop of that
    shape, and the centre as fractions of the photo (the placer's Shift-crop writes these)."""
    if not spec or spec.strip().lower() == "auto":
        return trained_crop(width, height)
    try:
        shape, zoom, cx, cy = (p.strip() for p in spec.split(","))
        rw, rh = (int(v) for v in shape.split(":"))
        zoom, cx, cy = float(zoom), float(cx), float(cy)
    except ValueError as exc:
        raise ValueError(f"crop must be 'auto' or '<shape>,<zoom>,<cx>,<cy>', got {spec!r}") from exc
    if (rw, rh) not in TRAINED_ASPECTS:
        raise ValueError(f"crop shape {shape} isn't one the LoRA was trained on")
    largest = min(width, height * rw / rh)
    w = max(1, min(width, _half_up(largest * min(max(zoom, 0.0), 1.0))))
    h = max(1, min(height, _half_up(w * rh / rw)))
    x = min(max(_half_up(cx * width - w / 2), 0), width - w)
    y = min(max(_half_up(cy * height - h / 2), 0), height - h)
    return x, y, w, h, shape


def crop_photo(photo: np.ndarray, spec: str = "auto") -> np.ndarray:
    """The photo cut as `spec` says: a plain slice of its pixels, nothing resampled."""
    x, y, w, h, _ = crop_rect(photo.shape[1], photo.shape[0], spec)
    return photo[y:y + h, x:x + w]


def crop_to_trained_shape(photo: np.ndarray) -> np.ndarray:
    """The photo cut to its nearest trained shape (crop "auto")."""
    return crop_photo(photo, "auto")


def fov_after_crop(diag_fov_full: float, full_wh: tuple[int, int], crop_wh: tuple[int, int]) -> float:
    """The diagonal FOV of a crop of a photo whose whole frame spans diag_fov_full (same lens)."""
    ratio = math.hypot(*crop_wh) / math.hypot(*full_wh)
    return math.degrees(2 * math.atan(math.tan(math.radians(diag_fov_full) / 2) * ratio))


def fov_from_35mm(focal_35mm: float) -> float:
    """Diagonal field of view of a photo from its 35 mm-equivalent focal length."""
    return math.degrees(2 * math.atan(FULL_FRAME_DIAG_MM / (2 * focal_35mm)))


def normalize_setting(text: str) -> str:
    """A setting description written the way the dataset captioner wrote them (vr180.settings.
    clean_setting, minus its rejection of texts that mention people): one line, no quotes or
    final full stop, a leading "Setting:" dropped, the first letter lower case unless it starts
    an acronym."""
    text = " ".join(str(text).split()).strip().strip("'\"").rstrip(".").strip()
    if text.lower().startswith(("setting:", "the setting is")):
        text = text.split(":", 1)[-1] if ":" in text else text[len("the setting is"):]
        text = text.strip()
    if not text:
        return ""
    return text[0].lower() + text[1:] if text[:2].isalpha() and not text[:2].isupper() else text


def caption(setting: str = "", position: str = "", pov: bool = False, watermark: bool = False,
            vignette: bool = False, label_no_body: bool = False) -> str:
    """The training caption: instruction, then Setting / Position / POV / Watermark / Vignette.
    label_no_body: say "POV body: none." when there is no body (v5 / H3 captions); off for the Qwen v4 LoRA."""
    text = INSTRUCTION
    if setting and setting.strip():
        text += f" Setting: {setting.strip().rstrip('.')}."
    if position and position.strip():
        text += f" Position: {position.strip()}."
    if pov:
        text += f" {POV_PHRASE}"
    elif label_no_body:
        text += f" {POV_NONE_PHRASE}"
    if watermark:
        text += f" {WATERMARK_PHRASE}"
    if vignette:
        text += f" {VIGNETTE_PHRASE}"
    return text


def refine_mask(photo_mask: np.ndarray, keep_px: int, ramp_px: int) -> np.ndarray:
    """Second-pass strength per pixel: 0 on the photo and for keep_px around it, rising
    smoothly to 1 over the next ramp_px (no hard edge for the sampler to show)."""
    outside = (~photo_mask).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 5)  # distance to the nearest photo pixel
    t = np.clip((dist - keep_px) / max(1, ramp_px), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)  # smoothstep


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The VR180 image: left eye | right eye (float 0..1). A right eye of another size is resized
    to the left eye's."""
    if right.shape[:2] != left.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_CUBIC).clip(0, 1)
    return np.concatenate([left, right], axis=1)


def paste_photo(image: np.ndarray, canvas: np.ndarray, photo_mask: np.ndarray, feather_px: int) -> np.ndarray:
    """The original photo pixels back over a generated image (float 0..1, same size), fading
    in over feather_px inside the photo's edge so the VAE round trip leaves no seam."""
    inside = photo_mask.astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)  # distance to the nearest non-photo pixel
    alpha = np.clip(dist / max(1, feather_px), 0.0, 1.0)[..., None].astype(np.float32)
    return canvas * alpha + image * (1.0 - alpha)
