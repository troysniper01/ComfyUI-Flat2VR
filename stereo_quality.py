"""Stereo quality of a generated right eye, from the two eyes alone (numbers, no viewing).

Shared by the ComfyUI nodes (Join Eyes corrections, Right Eye best-of) and scripts/l2r_eval.py.
From dense optical flow left -> right over the middle of the view (|lon|, |lat| <= 60 deg), textured
pixels only:
  near_shift   horizontal shift of the closest content (90th percentile of |dx|, px at 2048)
  relief       inside 64 px blocks of close content: spread of the shift (std of dx, px). A flat
               "cardboard" person has almost none; a real face or body has some
  pose_drift   share of points whose vertical offset is > 6 px off the image's overall fit:
               real stereo barely moves anything vertically, a changed pose or limb does
  scale        vertical stretch between the eyes (%), from that fit; real stereo is ~0
  far_shift    shift of the background: median of the 64 px blocks in the image's lowest quarter of
               shifts (px). Less = the far scene gets less depth
  far_bumps    unevenness of the background's shift: median spread (p90 - p10, px) inside its blocks.
               A wall or distant room is smooth in real stereo; bumps read as a lumpy background
  far_spikes   share of background pixels that shift > 24 px more than the blocks around them: bits of
               distant scenery that pop out as if close (an artifact)
  diverge      share of the view placed beyond infinity (the right eye shifted the wrong way by > 4 px):
               the eyes would have to splay outward to fuse it
  halo         around real depth edges (a person in front of a room): share of pixels whose shift is
               stuck halfway between the near and the far side. Smeared edges glue the background to
               the subject like a rubber sheet; real edges are a clean step
  bright_shift right eye minus left eye brightness on matched points (Lab L, 0-255); real eyes ~0
  sharp_ratio  fine-detail energy of the right eye / of the left eye seen through the match. The match
               itself blurs a little, so real pairs read ~1.2; lower = a softer right eye
The far numbers, diverge and halo come from stereo block matching (SGBM) of each right-eye pixel
along its row: the flow above smooths over small patches that jump in shift (a pop-out).

quality_score() turns them into one number (<= 1: within real stereo on every check), and
match_brightness() / fix_divergence() correct the two faults that can be fixed after the fact.
Images are BGR uint8, as OpenCV reads them.
"""
from __future__ import annotations

import cv2
import numpy as np

EYE = 2048
CENTRE = (30 / 180, 150 / 180)  # the middle +-60 deg of a 180 eye
BLOCK = 64
NEAR_Q = 0.75  # blocks whose shift is in the image's top quarter count as close content
FAR_Q = 0.25  # ... and those in its bottom quarter as the background
SPIKE_PX = 24.0  # a background pixel shifting this much more than the blocks around it pops out
SGBM_MIN, SGBM_NUM, SGBM_BLOCK = -16, 192, 7  # shift range searched at WORK_PX: -16 .. +175 px
# Tuned on the 30 real eval pairs: speckle filter + stricter uniqueness take far_spikes there from a
# 1.6% median (32% max) of matching noise to 0% (max 7.8%); synthetic pop-outs are still all found.
SGBM_SPECKLE, SGBM_UNIQUE = 200, 15
DIVERGE_PX = 4.0  # shifted the wrong way by more than this (px at 2048): beyond infinity
EDGE_PX = 40.0  # 3 x 3 blocks whose shifts span this much (px at 2048) hold a real depth edge
DRIFT_PX = 6.0
TEXTURE = 4.0
WORK_PX = 1024  # flow resolution per eye
FB_PX = 1.0  # left->right->left round trip must land within this (px at WORK_PX)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each pixel of a: its offset (dx, dy) to the matching pixel of b."""
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(_gray(a), _gray(b), None)


def stereo_metrics(left: np.ndarray, right: np.ndarray, centre: tuple[float, float] = CENTRE,
                   equirect: bool = True) -> dict:
    """The numbers for one stereo pair, in pixels at a 2048 eye. Flow is computed at WORK_PX (big
    shifts are where flow goes wrong; halving them helps) and only points that match the same way
    left -> right and right -> left count."""
    k = min(1.0, WORK_PX / left.shape[1])
    if k < 1:
        left = cv2.resize(left, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    to_2048 = EYE / left.shape[1]
    h, w = left.shape[:2]
    f_lr, f_rl = flow(left, right), flow(right, left)
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    back = cv2.remap(f_rl, X + f_lr[..., 0], Y + f_lr[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb = np.hypot(f_lr[..., 0] + back[..., 0], f_lr[..., 1] + back[..., 1])
    y0, y1, x0, x1 = int(h * centre[0]), int(h * centre[1]), int(w * centre[0]), int(w * centre[1])
    crop = (slice(y0, y1), slice(x0, x1))
    lap = np.abs(cv2.GaussianBlur(cv2.Laplacian(_gray(left), cv2.CV_32F, ksize=3), (0, 0), 2))[crop]
    tex = lap > TEXTURE
    ok = tex & (fb[crop] < FB_PX)
    dx, dy = f_lr[crop][..., 0] * to_2048, f_lr[crop][..., 1] * to_2048
    out = {"match": float(ok.sum() / max(1, tex.sum()))}
    if ok.sum() < 200:
        return {**out, "near_shift": 0.0, "relief": 0.0, "pose_drift": 1.0, "scale": 0.0,
                "far_shift": 0.0, "far_bumps": 0.0, "far_spikes": 0.0, "diverge": 0.0, "halo": 0.0,
                "bright_shift": 0.0, "sharp_ratio": 0.0}
    out["near_shift"] = float(np.percentile(np.abs(dx[ok]), 90))

    # On a half-equirect eye a sideways eye shift moves points along great circles through the left
    # and right poles (tan(lat) / cos(lon) stays constant), so close content above or below the
    # horizon also moves vertically. Take that expected move out first; what's left is drift.
    if equirect:
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        lon = (gx + 0.5) / w * np.pi - np.pi / 2
        lat = np.pi / 2 - (gy + 0.5) / h * np.pi
        lon2 = np.clip(lon + f_lr[crop][..., 0] * np.pi / w, -np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
        lat2 = np.arctan(np.tan(lat) * np.cos(lon2) / np.cos(lon))
        dy = dy - ((lat - lat2) * h / np.pi * to_2048).astype(np.float32)

    # vertical fit dy = a + s * (y - centre): s = vertical stretch; the residual = drift
    ys = (np.mgrid[0:dy.shape[0], 0:dy.shape[1]][0].astype(np.float32) - dy.shape[0] / 2) * to_2048
    A = np.stack([np.ones(ok.sum()), ys[ok]], 1)
    coef, *_ = np.linalg.lstsq(A, dy[ok], rcond=None)
    for _ in range(2):  # re-fit without the outliers the fit is meant to expose
        keep = np.abs(dy[ok] - A @ coef) < DRIFT_PX
        if keep.sum() > 100:
            coef, *_ = np.linalg.lstsq(A[keep], dy[ok][keep], rcond=None)
    resid = np.abs(dy - (coef[0] + coef[1] * ys))
    out["pose_drift"] = float((resid[ok] > DRIFT_PX).mean())
    out["scale"] = float(coef[1] * 100)

    b = max(8, round(BLOCK / to_2048))
    out["relief"] = _blocks(np.abs(dx), dx, ok, b)[0]

    disp = right_disparity(left, right)[crop] * to_2048
    lap_r = np.abs(cv2.GaussianBlur(cv2.Laplacian(_gray(right), cv2.CV_32F, ksize=3), (0, 0), 2))[crop]
    tex_r = lap_r > TEXTURE
    valid = tex_r & ~np.isnan(disp)
    out["diverge"] = float((disp[valid] < -DIVERGE_PX).mean()) if valid.any() else 0.0
    out["halo"] = _halo(np.nan_to_num(disp), valid, b)
    back_r = cv2.remap(f_lr, X + f_rl[..., 0], Y + f_rl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ok_r = tex_r & (np.hypot(f_rl[..., 0] + back_r[..., 0], f_rl[..., 1] + back_r[..., 1])[crop] < FB_PX)
    seen = cv2.remap(left, X + f_rl[..., 0], Y + f_rl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return {**out, **_far(np.nan_to_num(disp), valid, b), **_eye_match(right, seen, crop, ok_r)}


def _bgr(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _eye_match(right: np.ndarray, seen: np.ndarray, crop: tuple, ok: np.ndarray) -> dict:
    """bright_shift and sharp_ratio: the right eye against the left eye warped onto it (`seen`)."""
    if ok.sum() < 200:
        return {"bright_shift": 0.0, "sharp_ratio": 0.0}
    light = lambda im: cv2.GaussianBlur(cv2.cvtColor(_bgr(im), cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), (0, 0), 6)  # noqa: E731
    detail = lambda im: np.abs(cv2.Laplacian(_gray(im).astype(np.float32), cv2.CV_32F, ksize=3))[crop][ok]  # noqa: E731
    return {"bright_shift": float(np.mean((light(right) - light(seen))[crop][ok])),
            "sharp_ratio": float(detail(right).mean() / max(1e-6, detail(seen).mean()))}


def _halo(disp: np.ndarray, valid: np.ndarray, b: int) -> float:
    """Share of pixels around real depth edges whose shift sits in the middle 40% between the near
    and far side (block medians of the 3 x 3 blocks around)."""
    _, med, _ = _blocks(disp, disp, valid, b)
    middle = counted = 0
    for by in range(1, med.shape[0] - 1):
        for bx in range(1, med.shape[1] - 1):
            ring = med[by - 1:by + 2, bx - 1:bx + 2]
            if np.isnan(ring).sum() > 2 or np.nanmax(ring) - np.nanmin(ring) < EDGE_PX:
                continue
            lo, hi = np.nanmin(ring), np.nanmax(ring)
            sl = (slice(by * b, (by + 1) * b), slice(bx * b, (bx + 1) * b))
            frac = (disp[sl][valid[sl]] - lo) / (hi - lo)
            middle += int(((frac > 0.3) & (frac < 0.7)).sum())
            counted += int(valid[sl].sum())
    return middle / counted if counted else 0.0


def right_disparity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """For each right-eye pixel: how far left of its match in the left eye it sits (px; NaN = no
    reliable match). SGBM matches the left image's pixels, so both eyes are mirrored and swapped."""
    sgbm = cv2.StereoSGBM_create(minDisparity=SGBM_MIN, numDisparities=SGBM_NUM, blockSize=SGBM_BLOCK,
                                 P1=8 * SGBM_BLOCK ** 2, P2=32 * SGBM_BLOCK ** 2, uniquenessRatio=SGBM_UNIQUE,
                                 speckleWindowSize=SGBM_SPECKLE, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    mirror = lambda im: np.ascontiguousarray(_gray(im)[:, ::-1])  # noqa: E731
    d = sgbm.compute(mirror(right), mirror(left)).astype(np.float32)[:, ::-1] / 16
    d[d < SGBM_MIN] = np.nan
    return d


def _blocks(shift: np.ndarray, dx: np.ndarray, ok: np.ndarray, b: int, robust: bool = False
            ) -> tuple[float, np.ndarray, np.ndarray]:
    """(relief, block medians, block spreads) over b x b blocks (BLOCK px at 2048) of a shift map;
    NaN = too few points. Spread is the std of dx, or p90 - p10 when robust."""
    rows, cols = shift.shape[0] // b, shift.shape[1] // b
    med, spread = np.full((rows, cols), np.nan), np.full((rows, cols), np.nan)
    for by in range(rows):
        for bx in range(cols):
            sl = (slice(by * b, (by + 1) * b), slice(bx * b, (bx + 1) * b))
            m = ok[sl]
            if m.mean() > 0.3:
                v = dx[sl][m]
                med[by, bx] = np.median(shift[sl][m])
                spread[by, bx] = np.subtract(*np.percentile(v, [90, 10])) if robust else np.std(v)
    valid = ~np.isnan(med)
    if not valid.any():
        return 0.0, med, spread
    near = valid & (med >= np.nanquantile(med, NEAR_Q))
    return float(np.median(spread[near])), med, spread


def _far(shift: np.ndarray, ok: np.ndarray, b: int) -> dict:
    """far_shift, far_bumps and far_spikes of a shift map."""
    _, med, spread = _blocks(shift, shift, ok, b, robust=True)
    rows, cols = med.shape
    valid = ~np.isnan(med)
    if not valid.any():
        return {"far_shift": 0.0, "far_bumps": 0.0, "far_spikes": 0.0}
    far = valid & (med <= np.nanquantile(med, FAR_Q))
    out = {"far_shift": float(np.median(med[far])), "far_bumps": float(np.median(spread[far]))}
    # Background = blocks whose 8 neighbours are all in the far half (so no real depth edge, like the
    # outline of a person, runs through them); within them, pixels shifting much more than those
    # neighbours pop out of the distance.
    far_half = np.nanquantile(med, 0.5)
    around = np.full_like(med, np.nan)
    for by in range(rows):
        for bx in range(cols):
            ring = med[max(0, by - 1):by + 2, max(0, bx - 1):bx + 2].copy()
            ring[min(by, 1), min(bx, 1)] = np.nan  # the block itself may be the one popping out
            ring = ring[~np.isnan(ring)]
            if ring.size >= 5 and ring.max() <= far_half:
                around[by, bx] = np.median(ring)
    background = ~np.isnan(around)
    popped = counted = 0
    for by, bx in zip(*np.nonzero(background)):
        sl = (slice(by * b, (by + 1) * b), slice(bx * b, (bx + 1) * b))
        m = ok[sl]
        popped += int((shift[sl][m] > around[by, bx] + SPIKE_PX).sum())
        counted += int(m.sum())
    out["far_spikes"] = popped / max(1, counted)
    return out


# Real stereo (30 eval pairs): the value 90% of real pairs stay within, per check (relief: 90% above)
REAL_LIMITS = {"pose_drift": 0.19, "scale": 0.7, "diverge": 0.08, "halo": 0.29, "far_spikes": 0.02, "relief": 1.47}


def quality_score(m: dict) -> tuple[float, str]:
    """(score, worst check): each check divided by its real-stereo limit, the worst one counts.
    <= 1 means the pair is within real stereo on every check. Relief counts the other way round
    (too little = cardboard), so selecting by score never rewards a flatter right eye."""
    parts = {
        "pose_drift": m["pose_drift"] / REAL_LIMITS["pose_drift"],
        "scale": abs(m["scale"]) / REAL_LIMITS["scale"],
        "diverge": m["diverge"] / REAL_LIMITS["diverge"],
        "halo": m["halo"] / REAL_LIMITS["halo"],
        "far_spikes": m["far_spikes"] / REAL_LIMITS["far_spikes"],
        "relief": REAL_LIMITS["relief"] / max(1e-3, m["relief"]),
    }
    worst = max(parts, key=parts.get)
    return float(parts[worst]), worst


LIGHT_SIGMA_PX = 64.0  # brightness match: smoothness of the correction (px at 2048; lighting, not detail)
MAX_OFFSET = 20.0  # brightness match: at most +-20 levels per channel anywhere
MAX_SHIFT_PX = 32.0  # divergence fix: at most this horizontal shift (px at 2048)
# Real stereo has a little view slightly beyond infinity too (median 2.3%, 90% of pairs under 8% past
# -4 px): only act when over 5% of the view is past -4 px, then bring that 5% to -2 px.
DIVERGE_TARGET_PX = 2.0


def _left_seen_from_right(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The left eye warped onto the right eye's pixels, and where that match is trustworthy (both at
    the images' size; the match is found at WORK_PX)."""
    h, w = right.shape[:2]
    k = min(1.0, WORK_PX / w)
    small = lambda im: cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else im  # noqa: E731
    ls, rs = small(left), small(right)
    f_lr, f_rl = flow(ls, rs), flow(rs, ls)
    sh, sw = rs.shape[:2]
    X, Y = np.meshgrid(np.arange(sw, dtype=np.float32), np.arange(sh, dtype=np.float32))
    back = cv2.remap(f_lr, X + f_rl[..., 0], Y + f_rl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ok = np.hypot(f_rl[..., 0] + back[..., 0], f_rl[..., 1] + back[..., 1]) < FB_PX
    big = cv2.resize(f_rl, (w, h), interpolation=cv2.INTER_LINEAR) / k
    Xb, Yb = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    seen = cv2.remap(left, Xb + big[..., 0], Yb + big[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return seen, cv2.resize(ok.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def match_brightness(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """The right eye with a smooth per-channel brightness correction (a field, so uneven differences such
    as vignetting are matched too), fitted on the points both eyes see. Detail is left alone. Returns
    it and the mean correction per channel (levels)."""
    seen, ok = _left_seen_from_right(left, right)
    if ok.sum() < 1000:
        return right, [0.0] * right.shape[2]
    sigma = LIGHT_SIGMA_PX * right.shape[1] / EYE
    smooth = lambda im: cv2.GaussianBlur(im, (0, 0), sigma)  # noqa: E731
    weight = smooth(ok.astype(np.float32)) + 1e-3
    diff = (seen.astype(np.float32) - right.astype(np.float32)) * ok[..., None]
    field = np.clip(smooth(diff) / weight[..., None], -MAX_OFFSET, MAX_OFFSET)
    out = np.clip(right.astype(np.float32) + field + 0.5, 0, 255).astype(np.uint8)
    return out, [float(field[..., c][ok].mean()) for c in range(right.shape[2])]


def fix_divergence(left: np.ndarray, right: np.ndarray, allowed: float = 0.05) -> tuple[np.ndarray, float]:
    """If more than `allowed` of the view sits beyond infinity by over DIVERGE_PX (more than real stereo
    does), shift the whole right eye sideways until that part sits at -DIVERGE_TARGET_PX: on a 180 eye a
    sideways shift is a turn of the head, so nothing distorts and everything moves the same few px
    closer. Returns the right eye and the shift (px at 2048). Measure a right eye before this, not
    after: pose_drift expects close points to move vertically with depth, which a turn doesn't do."""
    h, w = right.shape[:2]
    k = min(1.0, WORK_PX / w)
    small = lambda im: cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else im  # noqa: E731
    d = right_disparity(small(left), small(right))
    sh, sw = d.shape
    crop = (slice(int(sh * CENTRE[0]), int(sh * CENTRE[1])), slice(int(sw * CENTRE[0]), int(sw * CENTRE[1])))
    lap = np.abs(cv2.GaussianBlur(cv2.Laplacian(_gray(small(right)), cv2.CV_32F, ksize=3), (0, 0), 2))
    v = d[crop][(lap[crop] > TEXTURE) & ~np.isnan(d[crop])]
    if v.size < 1000:
        return right, 0.0
    lowest = float(np.percentile(v, 100 * allowed)) / k * EYE / w  # px at 2048: where the lowest `allowed` begins
    if lowest >= -DIVERGE_PX:
        return right, 0.0
    shift = min(-lowest - DIVERGE_TARGET_PX, MAX_SHIFT_PX) * w / EYE  # px at the image's size
    moved = cv2.warpAffine(right, np.float32([[1, 0, -shift], [0, 1, 0]]), (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    return moved, shift * EYE / w
