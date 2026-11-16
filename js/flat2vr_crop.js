// Flat2VR Place Photo: the Shift-crop editor. Shows the whole photo with the crop box: drag a corner
// to change the shape (snapping to the shapes the LoRA was trained on), drag inside to move the box,
// scroll to zoom it, double-click for "auto". The crop is written as the node's `crop` text, which
// vr180_geometry.crop_rect turns into exactly the same pixels.
import { clamp } from "./flat2vr_gl.js";

export const SHAPES = [[16, 9], [3, 2], [4, 3], [5, 4], [1, 1], [4, 5], [3, 4], [2, 3], [9, 16]];
const HANDLE_PX = 16;  // corner grab radius, CSS px
const MIN_SIDE = 32;  // smallest crop side, photo px
const MARGIN = 0.06;  // of the square, around the photo
const ZOOM_PER_WHEEL = 0.0015;

const halfUp = (v) => Math.floor(v + 0.5);  // = vr180_geometry._half_up

/** vr180_geometry.crop_rect: (x, y, w, h, shape) for a spec; "auto" uses the server's trained crop. */
export function cropRect(W, H, spec, auto) {
  if (!spec || spec.trim().toLowerCase() === "auto") return { x: auto.crop[0], y: auto.crop[1], w: auto.crop[2], h: auto.crop[3], shape: auto.shape };
  const parts = spec.split(",").map((p) => p.trim());
  const [rw, rh] = (parts[0] || "").split(":").map(Number);
  const [zoom, cx, cy] = parts.slice(1).map(Number);
  if (parts.length !== 4 || !SHAPES.some(([a, b]) => a === rw && b === rh) || [zoom, cx, cy].some(Number.isNaN)) {
    return { ...cropRect(W, H, "auto", auto), invalid: true };
  }
  const largest = Math.min(W, (H * rw) / rh);
  const w = Math.max(1, Math.min(W, halfUp(largest * clamp(zoom, 0, 1))));
  const h = Math.max(1, Math.min(H, halfUp((w * rh) / rw)));
  const x = Math.min(Math.max(halfUp(cx * W - w / 2), 0), W - w);
  const y = Math.min(Math.max(halfUp(cy * H - h / 2), 0), H - h);
  return { x, y, w, h, shape: `${rw}:${rh}` };
}

/** The spec for a box of `shape` with width w centred at (cx, cy) in photo px. */
function specFor(W, H, shape, w, cx, cy) {
  const [rw, rh] = shape;
  const zoom = w / Math.min(W, (H * rw) / rh);
  return `${rw}:${rh},${zoom.toFixed(5)},${(cx / W).toFixed(5)},${(cy / H).toFixed(5)}`;
}

function nearestShape(aspect) {
  let best = SHAPES[0], d = Infinity;
  for (const s of SHAPES) {
    const e = Math.abs(Math.log(aspect) - Math.log(s[0] / s[1]));
    if (e < d) { d = e; best = s; }
  }
  return best;
}

export const CROP_CSS = `
.f2v-crop { position: absolute; inset: 0; width: 100%; height: 100%; display: block; opacity: 0; pointer-events: none;
  transition: opacity 120ms ease; background: #1a1b1e; touch-action: none; }
.f2v-crop.on { opacity: 1; pointer-events: auto; }
`;

export class CropEditor {
  /** frame: the square element; onChange(spec, rect) whenever the crop changes. */
  constructor(frame, onChange) {
    this.square = frame;
    this.onChange = onChange;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "f2v-crop";
    frame.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.photo = null;  // { image (small bitmap), W, H, auto }
    this.spec = "auto";
    this.isOpen = false;
    this.drag = null;
    this.dirty = true;
    this.#listen();
  }

  setPhoto(image, W, H, auto) {
    this.photo?.image?.close?.();
    this.photo = { image, W, H, auto };
    this.dirty = true;
  }

  setSpec(spec) {
    if (spec !== this.spec) { this.spec = spec; this.dirty = true; }
  }

  rect() {
    return this.photo ? cropRect(this.photo.W, this.photo.H, this.spec, this.photo.auto) : null;
  }

  open() {
    if (!this.photo || this.isOpen) return;
    this.isOpen = true;
    this.canvas.classList.add("on");
    this.dirty = true;
  }

  close() {
    if (!this.isOpen || this.drag) return;  // a drag in progress finishes first
    this.isOpen = false;
    this.canvas.classList.remove("on");
  }

  /** Photo px <-> canvas CSS px. */
  #layout() {
    const r = this.canvas.getBoundingClientRect();
    const S = r.width || 1, m = S * MARGIN;
    const k = Math.min((S - 2 * m) / this.photo.W, (S - 2 * m) / this.photo.H);
    return { r, S, k, ox: (S - this.photo.W * k) / 2, oy: (S - this.photo.H * k) / 2 };
  }

  #toPhoto(e, L) {
    const sx = (e.clientX - L.r.left) * (L.S / L.r.width), sy = (e.clientY - L.r.top) * (L.S / L.r.width);
    return { x: (sx - L.ox) / L.k, y: (sy - L.oy) / L.k, sx, sy };
  }

  #hit(p, L) {
    const c = this.rect();
    const corners = [[c.x, c.y], [c.x + c.w, c.y], [c.x, c.y + c.h], [c.x + c.w, c.y + c.h]];
    for (let i = 0; i < 4; i++) {
      const [x, y] = corners[i];
      if (Math.hypot(L.ox + x * L.k - p.sx, L.oy + y * L.k - p.sy) <= HANDLE_PX) {
        const [fx, fy] = corners[3 - i];  // the opposite corner stays put
        return { kind: "corner", fx, fy };
      }
    }
    if (p.x >= c.x && p.x <= c.x + c.w && p.y >= c.y && p.y <= c.y + c.h) return { kind: "move" };
    return null;
  }

  #emit(spec) {
    if (spec === this.spec) return;
    this.spec = spec;
    this.dirty = true;
    this.onChange(spec, this.rect());
  }

  #listen() {
    const cv = this.canvas;
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
    cv.addEventListener("pointerdown", (e) => {
      if (!this.isOpen || e.button !== 0) return;
      stop(e);
      const L = this.#layout(), p = this.#toPhoto(e, L), hit = this.#hit(p, L);
      if (!hit) return;
      cv.setPointerCapture(e.pointerId);
      const c = this.rect();
      this.drag = { ...hit, p0: p, c0: c };
    });
    cv.addEventListener("pointermove", (e) => {
      if (!this.isOpen) return;
      stop(e);
      const L = this.#layout(), p = this.#toPhoto(e, L);
      if (!this.drag) {
        const hit = this.#hit(p, L);
        const c = this.rect();
        const cornerCursor = hit?.kind === "corner" && ((hit.fx > c.x) === (hit.fy > c.y)) ? "nwse-resize" : "nesw-resize";
        cv.style.cursor = !hit ? "default" : hit.kind === "move" ? "move" : cornerCursor;
        return;
      }
      const { W, H } = this.photo;
      if (this.drag.kind === "move") {
        const c = this.drag.c0;
        const x = clamp(c.x + (p.x - this.drag.p0.x), 0, W - c.w), y = clamp(c.y + (p.y - this.drag.p0.y), 0, H - c.h);
        this.#emit(specFor(W, H, c.shape.split(":").map(Number), c.w, x + c.w / 2, y + c.h / 2));
        return;
      }
      const { fx, fy } = this.drag;
      const px = clamp(p.x, 0, W), py = clamp(p.y, 0, H);
      const dx = px - fx, dy = py - fy;
      const shape = nearestShape(Math.max(1, Math.abs(dx)) / Math.max(1, Math.abs(dy)));
      const r = shape[0] / shape[1];
      const roomW = dx >= 0 ? W - fx : fx, roomH = dy >= 0 ? H - fy : fy;
      const w = clamp((Math.abs(dx) + Math.abs(dy) * r) / 2, MIN_SIDE, Math.min(roomW, roomH * r));
      const h = w / r;
      const x = dx >= 0 ? fx : fx - w, y = dy >= 0 ? fy : fy - h;
      this.#emit(specFor(W, H, shape, w, x + w / 2, y + h / 2));
    });
    const end = (e) => {
      if (!this.drag) return;
      e.stopPropagation();
      this.drag = null;
    };
    cv.addEventListener("pointerup", end);
    cv.addEventListener("pointercancel", end);
    cv.addEventListener("dblclick", (e) => { if (this.isOpen) { stop(e); this.#emit("auto"); } });
  }

  /** Scroll: zoom the box about its centre. */
  wheel(e) {
    if (!this.isOpen) return false;
    const { W, H } = this.photo, c = this.rect();
    const shape = c.shape.split(":").map(Number), r = shape[0] / shape[1];
    const largest = Math.min(W, H * r);
    const w = clamp(c.w * Math.exp(e.deltaY * ZOOM_PER_WHEEL), Math.min(MIN_SIDE * Math.max(1, r), largest), largest);
    this.#emit(specFor(W, H, shape, w, c.x + c.w / 2, c.y + c.h / 2));
    return true;
  }

  /** Redraw if needed (called every animation frame by the placer). */
  redraw() {
    if (!this.isOpen || !this.photo) return;
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const px = Math.max(1, Math.round(r.width * dpr));
    if (this.canvas.width !== px) { this.canvas.width = this.canvas.height = px; this.dirty = true; }
    if (!this.dirty) return;
    this.dirty = false;
    const ctx = this.ctx, L = this.#layout(), s = px / L.S, c = this.rect();
    const { W, H, image } = this.photo;
    ctx.setTransform(s, 0, 0, s, 0, 0);
    ctx.clearRect(0, 0, L.S, L.S);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(image, L.ox, L.oy, W * L.k, H * L.k);
    const bx = L.ox + c.x * L.k, by = L.oy + c.y * L.k, bw = c.w * L.k, bh = c.h * L.k;
    ctx.fillStyle = "rgba(10,10,12,0.62)";  // dim everything outside the box
    ctx.beginPath();
    ctx.rect(L.ox, L.oy, W * L.k, H * L.k);
    ctx.rect(bx, by, bw, bh);
    ctx.fill("evenodd");
    ctx.strokeStyle = "rgba(255,255,255,0.35)";  // thirds
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const t of [1 / 3, 2 / 3]) {
      ctx.moveTo(bx + bw * t, by); ctx.lineTo(bx + bw * t, by + bh);
      ctx.moveTo(bx, by + bh * t); ctx.lineTo(bx + bw, by + bh * t);
    }
    ctx.stroke();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(bx, by, bw, bh);
    const arm = Math.min(18, bw / 3, bh / 3);  // L-shaped corner handles
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (const [x, y, sx, sy] of [[bx, by, 1, 1], [bx + bw, by, -1, 1], [bx, by + bh, 1, -1], [bx + bw, by + bh, -1, -1]]) {
      ctx.moveTo(x, y + sy * arm); ctx.lineTo(x, y); ctx.lineTo(x + sx * arm, y);
    }
    ctx.stroke();
    const auto = !this.spec || this.spec.trim().toLowerCase() === "auto";
    const label = `${c.shape}${auto ? " · auto" : ""} · ${c.w}×${c.h} · ${Math.round((100 * c.w * c.h) / (W * H))}% of the photo`;
    ctx.font = "600 12px system-ui, sans-serif";
    const tw = ctx.measureText(label).width + 16;
    const ly = by - 30 >= 4 ? by - 30 : by + 6;
    const lx = clamp(bx + bw / 2 - tw / 2, 4, L.S - tw - 4);
    ctx.fillStyle = "rgba(14,15,17,0.82)";
    ctx.beginPath();
    ctx.roundRect(lx, ly, tw, 24, 12);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textBaseline = "middle";
    ctx.fillText(label, lx + 8, ly + 12);
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = "rgba(235,235,240,0.75)";
    ctx.textAlign = "center";
    ctx.fillText("corners: shape · drag: move · scroll: zoom · double-click: auto · let go of Shift: done", L.S / 2, L.S - 10);
    ctx.textAlign = "start";
  }

  destroy() {
    this.photo?.image?.close?.();
    this.canvas.remove();
  }
}
