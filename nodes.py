"""ComfyUI nodes for the Flat2VR LoRA (Qwen-Image 2.1 edit): load a 2D photo, cut it to its nearest
trained shape and place it on the 180-degree canvas exactly as in training (a live, draggable
placer in the node), build the prompt exactly as the dataset was captioned, paste the original
photo back, join the left eye with the l2r LoRA's right eye into the side-by-side VR180 image,
and look around the result in the node."""
from __future__ import annotations

import hashlib
import os
import uuid
from functools import lru_cache

import numpy as np
import torch

from . import vr180_geometry as geo

TRAINED_FOV = (38.5, 130.0)  # diagonal FOV range of the training photos
PREVIEW_PX = 512  # /flat2vr/preview: the same canvas, rendered smaller (a check; the node draws its own)
PREVIEW_JPEG_QUALITY = 95  # the 180 preview's temp file (the saved result stays lossless PNG)
EXIF_ORIENTATION = 0x0112
EXIF_SIDEWAYS = (5, 6, 7, 8)  # orientations that swap width and height
# The position tags the LoRA was trained with (spelled exactly as in training).
POSITIONS = ["none", "close-up", "blowjob", "missionary", "cowgirl", "reverse cowgirl", "doggystyle",
             "from below", "clothed", "handjob"]


def _input_images() -> list[str]:
    import folder_paths  # ComfyUI

    folder = folder_paths.get_input_directory()
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return sorted(folder_paths.filter_files_content_types(files, ["image"]))


def _image_path(name: str) -> str:
    import folder_paths  # ComfyUI

    return folder_paths.get_annotated_filepath(name)


@lru_cache(maxsize=8)
def _load_rgb(path: str, mtime: float) -> np.ndarray:
    """The photo as RGB uint8, turned upright by its EXIF orientation (as Load Image does)."""
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB")).copy()


def load_photo(name: str, crop: str = "auto") -> tuple[np.ndarray, tuple[int, int]]:
    """The photo, upright and cut as `crop` says (a slice: no resampling), and its full (w, h)."""
    path = _image_path(name)
    full = _load_rgb(path, os.path.getmtime(path))
    return geo.crop_photo(full, crop), (full.shape[1], full.shape[0])


def photo_info(name: str) -> dict:
    """What the placer in the browser needs: the upright size, the trained-shape crop, and where
    ComfyUI serves the file. Reads only the header, so it is instant even for huge photos."""
    import folder_paths  # ComfyUI
    from PIL import Image

    path = _image_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(name)
    with Image.open(path) as img:
        width, height = img.size
        if img.getexif().get(EXIF_ORIENTATION) in EXIF_SIDEWAYS:
            width, height = height, width
    x, y, w, h, shape = geo.trained_crop(width, height)
    clean, _ = folder_paths.annotated_filepath(name)
    kind = next((k for k in ("output", "temp", "input") if name.endswith(f"[{k}]")), "input")
    subfolder, _, filename = clean.replace("\\", "/").rpartition("/")
    return {"width": width, "height": height, "crop": [x, y, w, h], "shape": shape,
            "kept": round(w * h / (width * height), 4),
            "view": {"filename": filename, "subfolder": subfolder, "type": kind}}


def _fov(diag_fov: float, focal_35mm: float, full_wh: tuple[int, int], crop_wh: tuple[int, int]) -> float:
    """The placed photo's diagonal FOV: diag_fov as set, or the lens's (focal_35mm describes the
    whole frame, so a crop of it spans less)."""
    if focal_35mm and focal_35mm > 0:
        return geo.fov_after_crop(geo.fov_from_35mm(focal_35mm), full_wh, crop_wh)
    return diag_fov


class Flat2VRPlacePhoto:
    """Load a photo (drop it anywhere on the node, paste it, or upload) and place it on the grey
    2048x2048 half-equirect canvas: the LoRA's input. The photo is first cut to the nearest shape
    the LoRA was trained on (16:9, 3:2, 4:3, 5:4, 1:1 or a portrait one), trimming as little as
    possible. Drag in the placer to move it, scroll to change the field of view; hold Shift over
    it to crop by hand."""

    CATEGORY = "Flat2VR"
    FUNCTION = "place"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("canvas", "photo_mask")
    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": (_input_images(), {"image_upload": True}),
            "yaw": ("FLOAT", {"default": 0.0, "min": -89.0, "max": 89.0, "step": 0.5,
                              "tooltip": "Left/right of the photo's centre in the 180 view (degrees). Or drag in the preview."}),
            "pitch": ("FLOAT", {"default": -5.0, "min": -89.0, "max": 89.0, "step": 0.5,
                                "tooltip": "Up/down. Level or slightly down (-15..+5) works best. Or drag in the preview."}),
            "diag_fov": ("FLOAT", {"default": 80.0, "min": 10.0, "max": 170.0, "step": 0.5,
                                   "tooltip": "The photo's diagonal field of view. Trained range 38.5-130; a phone's main camera is about 75-85. Or scroll over the preview."}),
            "focal_35mm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.5,
                                     "tooltip": "If > 0: the photo's 35 mm-equivalent focal length (from its EXIF); overrides diag_fov."}),
            "canvas_px": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 32,
                                  "tooltip": "Canvas size; the LoRA was trained at 2048. Leave it."}),
            "crop": ("STRING", {"default": "auto",
                                "tooltip": "auto: the nearest trained shape, trimming as little as possible. Hold Shift over the placer to crop by hand (written here as shape,zoom,centre x,centre y)."}),
        }}

    def place(self, image, yaw, pitch, diag_fov, focal_35mm, canvas_px, crop="auto"):
        rgb, full_wh = load_photo(image, crop)
        fov = _fov(diag_fov, focal_35mm, full_wh, (rgb.shape[1], rgb.shape[0]))
        view = geo.view_for_photo(rgb.shape[1], rgb.shape[0], yaw, pitch, fov, canvas_px)
        if not geo.fits_in_180(view):
            raise ValueError(f"Flat2VR Place Photo: at yaw {yaw}, pitch {pitch}, FOV {fov:.1f} the photo leaves the "
                             "180-degree view; move it towards the centre or lower the FOV.")
        if not TRAINED_FOV[0] <= fov <= TRAINED_FOV[1]:
            print(f"[Flat2VR] FOV {fov:.1f} is outside the trained range {TRAINED_FOV}; results may be weaker.")
        canvas, valid = geo.place_photo(rgb.copy(), yaw, pitch, fov, canvas_px)
        return (torch.from_numpy(canvas.astype(np.float32) / 255.0)[None],
                torch.from_numpy(valid.astype(np.float32))[None])

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        with open(_image_path(image), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        import folder_paths  # ComfyUI

        return True if folder_paths.exists_annotated_filepath(image) else f"Invalid image file: {image}"


def render_preview(image: str, yaw: float, pitch: float, diag_fov: float, focal_35mm: float,
                   size: int = PREVIEW_PX, crop: str = "auto") -> tuple[bytes, bool]:
    """The canvas the node will output, rendered at `size` (PNG), and whether it fits in 180
    (a check for the browser's own drawing; the placer doesn't call it)."""
    import cv2

    rgb, full_wh = load_photo(image, crop)
    fov = _fov(diag_fov, focal_35mm, full_wh, (rgb.shape[1], rgb.shape[0]))
    fits = geo.fits_in_180(geo.view_for_photo(rgb.shape[1], rgb.shape[0], yaw, pitch, fov, size))
    canvas, _ = geo.place_photo(rgb.copy(), yaw, pitch, fov, size)
    ok, png = cv2.imencode(".png", canvas[:, :, ::-1])
    if not ok:
        raise ValueError("could not encode the preview")
    return png.tobytes(), fits


class Flat2VRRefineMask:
    """Strength map for the optional second pass: 0 on and near the photo, rising smoothly
    to 1 further out, so the refinement never shows a seam."""

    CATEGORY = "Flat2VR"
    FUNCTION = "build"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("refine_mask",)
    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "photo_mask": ("MASK",),
            "keep_px": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1,
                                "tooltip": "Band around the photo that the second pass leaves alone (pixels)."}),
            "ramp_px": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1,
                                "tooltip": "Distance over which the second pass fades in (pixels)."}),
        }}

    def build(self, photo_mask, keep_px, ramp_px):
        mask = photo_mask[0].detach().cpu().numpy() > 0.5
        return (torch.from_numpy(geo.refine_mask(mask, keep_px, ramp_px))[None],)


class Flat2VRPrompt:
    """The prompt exactly as the dataset was captioned: the instruction, then an optional
    setting (written the way the captioner wrote them), a position tag, and the POV body."""

    CATEGORY = "Flat2VR"
    FUNCTION = "build"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "setting": ("STRING", {"default": "", "multiline": True,
                                   "tooltip": "Optional: the place only, no people, e.g. 'bright bedroom with white walls and a large window'."}),
            "position": (POSITIONS, {"default": "none", "tooltip": "Optional: one of the trained position tags."}),
            "pov_body": ("BOOLEAN", {"default": False, "tooltip": "The camera person's body in shot."}),
        }}

    def build(self, setting, position, pov_body):
        return (geo.caption(geo.normalize_setting(setting), "" if position == "none" else position, pov_body),)


class Flat2VRPastePhoto:
    """The original photo pixels back over the generated image, with a soft inner edge."""

    CATEGORY = "Flat2VR"
    FUNCTION = "paste"
    RETURN_TYPES = ("IMAGE",)
    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "canvas": ("IMAGE",),
            "photo_mask": ("MASK",),
            "feather_px": ("INT", {"default": 12, "min": 0, "max": 256, "step": 1,
                                   "tooltip": "Fade-in width inside the photo's edge (pixels)."}),
        }}

    def paste(self, image, canvas, photo_mask, feather_px):
        cv = canvas[0, :, :, :3].detach().cpu().numpy()
        mask = photo_mask[0].detach().cpu().numpy() > 0.5
        out = []
        for img in image:
            gen = img[:, :, :3]
            if tuple(gen.shape[:2]) != cv.shape[:2]:
                gen = torch.nn.functional.interpolate(gen.movedim(-1, 0)[None], size=cv.shape[:2],
                                                      mode="bicubic", align_corners=False)[0].movedim(0, -1).clamp(0, 1)
            out.append(torch.from_numpy(geo.paste_photo(gen.detach().cpu().numpy(), cv, mask, feather_px)))
        return (torch.stack(out),)


class Flat2VRJoinEyes:
    """The final VR180 image: left eye | right eye side by side (4096x2048), the layout VR180
    players expect (save it with "_180_LR" in the name so they detect it)."""

    CATEGORY = "Flat2VR"
    FUNCTION = "join"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("vr180",)
    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"left_eye": ("IMAGE",), "right_eye": ("IMAGE",)}}

    def join(self, left_eye, right_eye):
        n = max(len(left_eye), len(right_eye))  # a single left eye pairs with every right eye
        out = [geo.side_by_side(left_eye[i % len(left_eye), :, :, :3].detach().cpu().numpy(),
                                right_eye[i % len(right_eye), :, :, :3].detach().cpu().numpy()) for i in range(n)]
        return (torch.from_numpy(np.stack(out)),)


def save_preview_jpeg(rgb: np.ndarray, folder: str) -> str:
    """The VR180 image as a high-quality JPEG in `folder` (full colour resolution); its name."""
    from PIL import Image

    name = f"flat2vr_vr180_{uuid.uuid4().hex[:12]}.jpg"
    Image.fromarray(rgb).save(os.path.join(folder, name), quality=PREVIEW_JPEG_QUALITY, subsampling=0)
    return name


class Flat2VRPreview180:
    """Look around the finished VR180 image inside the node: drag to look, scroll to zoom, and
    switch between the left and right eye. A preview only; Save Image keeps the result."""

    CATEGORY = "Flat2VR"
    FUNCTION = "preview"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vr180": ("IMAGE", {"tooltip": "The side-by-side VR180 image (or a single 180 eye)."})}}

    def preview(self, vr180):
        import folder_paths  # ComfyUI

        rgb = (vr180[0, :, :, :3].clamp(0, 1).detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        name = save_preview_jpeg(rgb, folder_paths.get_temp_directory())
        return {"ui": {"flat2vr_vr180": [{"filename": name, "subfolder": "", "type": "temp",
                                          "width": rgb.shape[1], "height": rgb.shape[0]}]}}


def _register_routes() -> None:
    """GET /flat2vr/photo?image= -> photo_info JSON (404 while the file isn't there yet).
    GET /flat2vr/preview?image=&yaw=&pitch=&diag_fov=&focal_35mm=&size= -> PNG (X-Fits: 1/0)."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:  # not running inside ComfyUI (tests)
        return

    @PromptServer.instance.routes.get("/flat2vr/photo")
    async def photo(request):
        name = request.rel_url.query.get("image", "")
        try:
            return web.json_response(photo_info(name), headers={"Cache-Control": "no-store"})
        except FileNotFoundError:
            return web.json_response({"error": f"{name} isn't in ComfyUI's input folder (yet)", "missing": True},
                                     status=404)
        except (OSError, ValueError) as exc:  # not an image PIL can read (a video, say)
            return web.json_response({"error": f"{name} isn't a photo ComfyUI can read ({exc})"}, status=415)

    @PromptServer.instance.routes.get("/flat2vr/preview")
    async def preview(request):
        q = request.rel_url.query
        try:
            png, fits = render_preview(q["image"], float(q.get("yaw", 0)), float(q.get("pitch", 0)),
                                       float(q.get("diag_fov", 80)), float(q.get("focal_35mm", 0)),
                                       max(128, min(1024, int(q.get("size", PREVIEW_PX)))), q.get("crop", "auto"))
        except (KeyError, ValueError, OSError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.Response(body=png, content_type="image/png",
                            headers={"X-Fits": "1" if fits else "0", "Cache-Control": "no-store"})


_register_routes()

NODE_CLASS_MAPPINGS = {
    "Flat2VRPlacePhoto": Flat2VRPlacePhoto,
    "Flat2VRRefineMask": Flat2VRRefineMask,
    "Flat2VRPrompt": Flat2VRPrompt,
    "Flat2VRPastePhoto": Flat2VRPastePhoto,
    "Flat2VRJoinEyes": Flat2VRJoinEyes,
    "Flat2VRPreview180": Flat2VRPreview180,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Flat2VRPlacePhoto": "Flat2VR Place Photo",
    "Flat2VRRefineMask": "Flat2VR Refine Mask",
    "Flat2VRPrompt": "Flat2VR Prompt",
    "Flat2VRPastePhoto": "Flat2VR Paste Photo",
    "Flat2VRJoinEyes": "Flat2VR Join Eyes (VR180)",
    "Flat2VRPreview180": "Flat2VR 180 Preview",
}
