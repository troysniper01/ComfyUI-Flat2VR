// Flat2VR Place Photo: the placer. The square IS the canvas the node outputs (the whole 180-degree
// view), drawn in the browser with the node's own projection (vr180_geometry.to_equirect) at 60 fps.
// Drop a photo anywhere on the node (or paste / upload); drag to move it, scroll to resize it,
// hold Shift to crop it (flat2vr_crop.js).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
  ShaderView, SPINNER_CSS, captureWheel, clamp, deg, el, fetchBitmap, frameLoop, injectStyle,
  rotationColumns, sleep,
} from "./flat2vr_gl.js";
import { CROP_CSS, CropEditor, cropRect } from "./flat2vr_crop.js";

const NODE = "Flat2VRPlacePhoto";
const MAX_CANVAS_PX = 1400;
const WHEEL_DEG = 2;
const FINE_DRAG = 0.25;  // Ctrl-drag: a quarter of the speed
const CROP_VIEW_PX = 1200;  // the crop editor's copy of the photo (display only)
const STEP = 0.5;  // the widgets' step
const LOAD_TRIES = 8;  // a just-uploaded file can land a moment after the name does
const FULL_FRAME_DIAG_MM = 43.27;
const RAW_PREVIEW_WIDGET = "$$canvas-image-preview";
const START = { yaw: 0, pitch: -5, diag_fov: 80, focal_35mm: 0 };  // the node's defaults (nodes.py)
const START_CROP = "auto";

const FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uSize;
uniform vec3 uC0, uC1, uC2;  // rotation columns: camera = (c0.w, c1.w, c2.w)
uniform vec2 uHalf;          // the photo's half width / height, in focal lengths
uniform vec4 uCrop;          // the trained-shape crop, in texture coordinates
uniform float uHas;
out vec4 outColor;
const float PI = 3.141592653589793;
void main() {
  vec2 uv = vec2(gl_FragCoord.x / uSize.x, 1.0 - gl_FragCoord.y / uSize.y);
  float lon = uv.x * PI - PI / 2.0, lat = PI / 2.0 - uv.y * PI;
  vec3 w = vec3(cos(lat) * sin(lon), sin(lat), cos(lat) * cos(lon));
  vec3 c = vec3(dot(uC0, w), dot(uC1, w), dot(uC2, w));
  float z = max(c.z, 1e-6);
  vec2 p = vec2(c.x / z, -c.y / z) / (2.0 * uHalf) + 0.5;
  vec3 photo = texture(uTex, uCrop.xy + clamp(p, 0.0, 1.0) * uCrop.zw).rgb;
  bool on = uHas > 0.5 && c.z > 1e-6 && all(greaterThanEqual(p, vec2(0.0))) && all(lessThanEqual(p, vec2(1.0)));
  outColor = vec4(on ? photo : vec3(128.0 / 255.0), 1.0);
}`;

const STYLE = `
.f2v-place { display: flex; flex-direction: column; gap: 6px; width: 100%; height: 100%; min-height: 0; box-sizing: border-box; }
.f2v-place .stage { position: relative; flex: 1 1 auto; min-height: 0; display: flex; align-items: center; justify-content: center; }
.f2v-place .frame { position: relative; aspect-ratio: 1 / 1; height: 100%; max-width: 100%; max-height: 100%;
  border-radius: 6px; overflow: hidden; background: #808080; cursor: grab; touch-action: none; user-select: none;
  box-shadow: 0 0 0 2px #3a3a3a; transition: box-shadow 120ms ease; }
.f2v-place .frame.dragging { cursor: grabbing; }
.f2v-place .frame.outside { box-shadow: 0 0 0 2px #e5484d; }
.f2v-place .frame canvas { width: 100%; height: 100%; display: block; }
.f2v-place .guide { position: absolute; pointer-events: none; background: rgba(255,255,255,0.16); }
.f2v-place .guide.h { left: 0; right: 0; top: 50%; height: 1px; }
.f2v-place .guide.v { top: 0; bottom: 0; left: 50%; width: 1px; }
.f2v-place .empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; text-align: center;
  padding: 18px; color: #f1f1f1; font: 13px/1.4 sans-serif; background: rgba(0,0,0,0.38); pointer-events: none; }
.f2v-place .empty.error { color: #ffb4b4; }
.f2v-place .status { font: 12px/1.35 sans-serif; color: #a9a9a9; text-align: center; flex: none; min-height: 16px; }
.f2v-place .status.outside { color: #ff7b7b; }
.f2v-drop-hint { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; pointer-events: none;
  border: 2px dashed #6aa7ff; border-radius: 8px; background: rgba(40,90,170,0.22); color: #fff; font: 600 14px sans-serif; z-index: 5; }
.f2v-dropping .f2v-drop-hint { display: flex; }
[data-node-id]:has(.f2v-place) .image-preview { display: none !important; }
${SPINNER_CSS}
${CROP_CSS}`;

const widget = (node, name) => node.widgets?.find((w) => w.name === name);
const value = (node, name) => widget(node, name)?.value;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
const isImageFile = (f) => !!f && (f.type?.startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif|tiff?|avif|heic|heif)$/i.test(f.name));

const lensFov = (focal) => (2 * Math.atan(FULL_FRAME_DIAG_MM / (2 * focal)) * 180) / Math.PI;

/** The placed photo's diagonal FOV (nodes._fov): diag_fov, or the lens's narrowed to the crop. */
function fovOf(s, full, rect) {
  if (!(s.focal_35mm > 0)) return s.diag_fov;
  const ratio = full && rect ? Math.hypot(rect.w, rect.h) / Math.hypot(full.width, full.height) : 1;
  return (2 * Math.atan(Math.tan(deg(lensFov(s.focal_35mm)) / 2) * ratio) * 180) / Math.PI;
}

/** The photo's half width/height in focal lengths (vr180_geometry.view_for, without its even-pixel rounding). */
function halfExtent(aspect, fovDeg) {
  const t = Math.tan(deg(fovDeg) / 2) / Math.sqrt(1 + aspect * aspect);
  return [t * aspect, t];
}

/** Whether the whole photo stays in front (vr180_geometry.fits_in_180): z is linear over the frame, so the corners decide. */
function fits(cols, half) {
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
    const x = sx * half[0], y = sy * half[1];
    if (cols[0][2] * x + cols[1][2] * y + cols[2][2] <= 0) return false;
  }
  return true;
}

function setWidget(node, name, v) {
  const w = widget(node, name);
  if (!w) return;
  const { min = -Infinity, max = Infinity } = w.options ?? {};
  const next = clamp(Math.round(v / STEP) * STEP, min, max);
  if (w.value === next) return;
  w.value = next;
  w.callback?.(next, app.canvas, node);
  node.setDirtyCanvas?.(true, true);
}

function setText(node, name, v) {
  const w = widget(node, name);
  if (!w || w.value === v) return;
  w.value = v;
  w.callback?.(v, app.canvas, node);
  node.setDirtyCanvas?.(true, true);
}

function setup(node) {
  injectStyle("f2v-place-style", STYLE);
  const box = el("div", "f2v-place");
  const stage = el("div", "stage", box);
  const frame = el("div", "frame", stage);
  const canvas = el("canvas", null, frame);
  el("div", "guide h", frame);
  el("div", "guide v", frame);
  const empty = el("div", "empty", frame, "Drop a photo anywhere on this node");
  const spinner = el("div", "f2v-spinner", frame);
  const status = el("div", "status", box);
  el("div", "f2v-drop-hint", box, "Drop to use this photo");

  node.addDOMWidget("placer", "flat2vr_placer", box, { serialize: false, hideOnZoom: false, getMinHeight: () => 320 });

  // ---- Shift-crop: while Shift is held over the placer, edit the crop on the whole photo -----------
  let cropSession = null;  // the FOV and crop diagonal when cropping began
  const cropper = new CropEditor(frame, (spec, rect) => {
    setText(node, "crop", spec);
    // a manual FOV keeps the photo's angular scale (the crop trims, it doesn't zoom); a lens FOV follows by itself
    if (cropSession && !(value(node, "focal_35mm") > 0)) {
      const t = Math.tan(deg(cropSession.fov) / 2) * (Math.hypot(rect.w, rect.h) / cropSession.diag);
      setWidget(node, "diag_fov", (2 * Math.atan(t) * 180) / Math.PI);
    }
  });
  let hovering = false;
  const openCrop = () => {
    if (cropper.isOpen || !photo) return;
    const rect = cropper.rect();
    cropSession = rect ? { fov: value(node, "diag_fov"), diag: Math.hypot(rect.w, rect.h) } : null;
    cropper.open();
  };
  const closeCrop = () => { cropper.close(); if (!cropper.isOpen) cropSession = null; };
  frame.addEventListener("pointerenter", (e) => { hovering = true; if (e.shiftKey) openCrop(); });
  frame.addEventListener("pointerleave", () => { hovering = false; });
  frame.addEventListener("pointermove", (e) => {
    if (drag) return;
    if (e.shiftKey && !cropper.isOpen) openCrop();
    else if (!e.shiftKey && cropper.isOpen && !cropper.drag) closeCrop();
  }, true);
  const onKey = (e) => {
    if (e.key !== "Shift") return;
    if (e.type === "keydown" && hovering) openCrop();
    if (e.type === "keyup") closeCrop();
  };
  const onBlur = () => closeCrop();
  const afterCropDrag = () => { if (cropper.isOpen) setTimeout(() => { if (!heldShift) closeCrop(); }, 0); };
  let heldShift = false;
  const trackShift = (e) => { heldShift = e.shiftKey; };
  window.addEventListener("keydown", onKey);
  window.addEventListener("keyup", onKey);
  window.addEventListener("blur", onBlur);
  window.addEventListener("pointerup", afterCropDrag, true);
  window.addEventListener("pointermove", trackShift, true);
  window.addEventListener("pointerup", trackShift, true);
  const stopCropKeys = () => {
    window.removeEventListener("keydown", onKey);
    window.removeEventListener("keyup", onKey);
    window.removeEventListener("blur", onBlur);
    window.removeEventListener("pointerup", afterCropDrag, true);
    window.removeEventListener("pointermove", trackShift, true);
    window.removeEventListener("pointerup", trackShift, true);
  };

  let view;
  try {
    view = new ShaderView(canvas, FRAGMENT, MAX_CANVAS_PX);
  } catch (e) {
    empty.textContent = `The placer needs WebGL2 (${e.message}). The node still works with the numbers above.`;
    return;
  }

  // ---- loading: the selected file -> its trained-shape crop + an upright texture -------------
  let photo = null;  // { info, crop: [x, y, w, h] in texture coords, aspect }
  let wanted = null;  // the image name being shown or loaded
  let ticket = 0;

  const showMessage = (text, isError = false) => {
    empty.textContent = text;
    empty.classList.toggle("error", isError);
    empty.style.display = text ? "" : "none";
  };

  async function load(name) {
    const my = ++ticket;
    wanted = name;
    spinner.classList.add("on");
    if (!photo) showMessage("");
    let lastError = null;
    for (let i = 0; i < LOAD_TRIES; i++) {
      try {
        const res = await api.fetchApi(`/flat2vr/photo?image=${encodeURIComponent(name)}`, { cache: "no-store" });
        const info = await res.json();
        if (!res.ok) throw Object.assign(new Error(info.error || `HTTP ${res.status}`), { retry: !!info.missing });
        const bitmap = await fetchBitmap(info.view, view.gl);
        if (my !== ticket) { bitmap.close(); return; }
        view.setImage(bitmap);
        const k = Math.min(1, CROP_VIEW_PX / Math.max(bitmap.width, bitmap.height));
        const small = await createImageBitmap(bitmap, {
          resizeWidth: Math.max(1, Math.round(bitmap.width * k)), resizeHeight: Math.max(1, Math.round(bitmap.height * k)),
          resizeQuality: "high",
        });
        bitmap.close();
        if (my !== ticket) { small.close(); return; }
        cropper.setPhoto(small, info.width, info.height, info);
        photo = { info };
        spinner.classList.remove("on");
        showMessage("");
        dirty = true;
        return;
      } catch (e) {
        lastError = e;
        if (e.retry === false || my !== ticket) break;
        await sleep(250 * (i + 1));
      }
      if (my !== ticket) return;
    }
    if (my !== ticket) return;
    spinner.classList.remove("on");
    photo = null;
    dirty = true;
    showMessage(`Couldn't open ${name}: ${lastError?.message ?? "unknown error"}. Drop a photo (JPG, PNG, WebP) on the node.`, true);
  }

  // ---- uploading: drop anywhere on the node, paste, or the upload button -----------------------
  async function upload(file) {
    if (!isImageFile(file)) {
      showMessage(`${file?.name ?? "That"} isn't a photo. Drop a JPG, PNG or WebP.`, true);
      return false;
    }
    ++ticket;
    spinner.classList.add("on");
    try {
      const body = new FormData();
      body.append("image", file, file.name);
      const res = await api.fetchApi("/upload/image", { method: "POST", body });
      if (!res.ok) throw new Error(`upload failed (HTTP ${res.status})`);
      const d = await res.json();
      const name = d.subfolder ? `${d.subfolder}/${d.name}` : d.name;
      const w = widget(node, "image");
      if (w) {
        const values = w.options?.values;
        if (Array.isArray(values) && !values.includes(name)) values.push(name);
        w.value = name;
        w.callback?.(name);
      }
      node.setDirtyCanvas?.(true, true);
      return true;
    } catch (e) {
      spinner.classList.remove("on");
      showMessage(`Couldn't upload ${file.name}: ${e.message}`, true);
      return false;
    }
  }

  let dropTimer = 0;
  const hint = (on) => {
    clearTimeout(dropTimer);
    box.classList.toggle("f2v-dropping", on);
    if (on) dropTimer = setTimeout(() => box.classList.remove("f2v-dropping"), 400);
  };
  // the graph canvas (classic layout): LiteGraph asks the node under the pointer
  node.onDragOver = (e) => { const ok = hasFiles(e); if (ok) hint(true); return ok; };
  node.onDragDrop = (e) => { hint(false); const f = e.dataTransfer?.files?.[0]; if (!f) return false; upload(f); return true; };
  node.pasteFile = (f) => upload(f);
  node.pasteFiles = (files) => upload(files?.[0]);
  // DOM parts (the placer here, and the whole node in the Vue layout)
  const onDomDragOver = (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    e.dataTransfer.dropEffect = "copy";
    hint(true);
  };
  const onDomDrop = (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    hint(false);
    upload(e.dataTransfer.files[0]);
  };
  const dropTargets = new Set();
  const listenForDrops = (target) => {
    if (!target || dropTargets.has(target)) return;
    dropTargets.add(target);
    target.addEventListener("dragover", onDomDragOver, true);
    target.addEventListener("dragenter", onDomDragOver, true);
    target.addEventListener("drop", onDomDrop, true);
  };
  listenForDrops(box);

  // ---- the upload widget's raw-photo thumbnail: the placer replaces it, in both layouts --------
  Object.defineProperty(node, "imgs", { configurable: true, get: () => undefined, set: () => {} });
  const imageWidget = widget(node, "image");
  if (imageWidget) imageWidget.callback = () => node.setDirtyCanvas?.(true, true);
  const dropRawPreview = () => {
    const i = node.widgets?.findIndex((w) => w.name === RAW_PREVIEW_WIDGET) ?? -1;
    if (i >= 0) {
      node.widgets[i].onRemove?.();
      node.widgets.splice(i, 1);
      node.setDirtyCanvas?.(true, true);
    }
  };
  const clearStoredOutputs = () => {
    try {  // the Vue layout's output store (Pinia); the CSS rule covers the rest
      const host = document.getElementById("vue-app") || [...document.body.children].find((e) => e.__vue_app__);
      host?.__vue_app__?.config.globalProperties.$pinia?._s.get("nodeOutput")?.removeNodeOutputs?.(node.id);
    } catch (_) { /* another frontend layout */ }
  };
  requestAnimationFrame(() => requestAnimationFrame(clearStoredOutputs));

  // ---- drawing: every animation frame, only when something changed ------------------------------
  let dirty = true;
  let lastKey = "";
  let restored = false;  // the first image is the saved workflow's: keep its placement
  const state = () => ({
    image: value(node, "image"), yaw: value(node, "yaw") ?? 0, pitch: value(node, "pitch") ?? 0,
    diag_fov: value(node, "diag_fov") ?? 80, focal_35mm: value(node, "focal_35mm") ?? 0,
    crop: value(node, "crop") ?? START_CROP,
  });

  function render(s) {
    const cols = rotationColumns(s.yaw, s.pitch);
    const info = photo?.info;
    const rect = info ? cropRect(info.width, info.height, s.crop, info) : null;
    const fov = fovOf(s, info, rect);
    const half = halfExtent(rect ? rect.w / rect.h : 1, fov);
    const tex = rect ? [rect.x / info.width, rect.y / info.height, rect.w / info.width, rect.h / info.height] : [0, 0, 1, 1];
    view.draw({ uC0: cols[0], uC1: cols[1], uC2: cols[2], uHalf: half, uCrop: tex, uHas: photo ? 1 : 0 });
    const ok = !photo || fits(cols, half);
    frame.classList.toggle("outside", !ok);
    status.classList.toggle("outside", !ok);
    if (!photo) {
      status.textContent = "";
    } else if (!ok) {
      status.textContent = "The photo leaves the 180° view: drag it towards the centre or scroll to shrink it";
    } else if (rect.invalid) {
      status.classList.add("outside");
      status.textContent = `The crop text "${s.crop}" isn't valid: hold Shift to crop again, or type auto`;
    } else {
      const kept = (rect.w * rect.h) / (info.width * info.height);
      const how = s.crop.trim().toLowerCase() === "auto" ? "" : " by hand";
      const crop = kept >= 0.9995 ? rect.shape : `${rect.shape} (cropped ${(100 * (1 - kept)).toFixed(1)}%${how})`;
      const lens = s.focal_35mm > 0 ? `FOV ${fov.toFixed(1)}° (${s.focal_35mm} mm)` : `FOV ${fov.toFixed(1)}°`;
      status.textContent = `${crop} · yaw ${s.yaw}° · pitch ${s.pitch}° · ${lens}  —  drag: move (Ctrl: fine) · scroll: resize · Shift: crop`;
    }
  }

  const stopLoop = frameLoop(() => {
    dropRawPreview();
    if (!canvas.isConnected || canvas.offsetParent === null) return;  // off-screen or collapsed
    const s = state();
    if (s.image !== wanted) {
      if (s.image && restored) {  // a new photo starts from the start position
        for (const [name, v] of Object.entries(START)) setWidget(node, name, v);
        setText(node, "crop", START_CROP);
        Object.assign(s, START, { crop: START_CROP });
      }
      restored = true;
      if (s.image) load(s.image);
      else { wanted = s.image; photo = null; showMessage("Drop a photo anywhere on this node"); dirty = true; }
    }
    if (!dropTargets.has(box.closest("[data-node-id]"))) listenForDrops(box.closest("[data-node-id]"));
    cropper.setSpec(s.crop);
    cropper.frame();
    const resized = view.fit();
    const key = JSON.stringify(s);
    if (dirty || resized || key !== lastKey) {
      render(s);
      lastKey = key;
      dirty = false;
    }
  });

  // ---- drag to move: the square spans 180° x 180° -------------------------------------------------
  let drag = null;
  frame.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || !photo || cropper.isOpen) return;
    e.preventDefault();
    e.stopPropagation();
    frame.setPointerCapture(e.pointerId);
    frame.classList.add("dragging");
    drag = { x: e.clientX, y: e.clientY, yaw: value(node, "yaw"), pitch: value(node, "pitch"), size: frame.getBoundingClientRect().width, fine: e.ctrlKey };
  });
  frame.addEventListener("pointermove", (e) => {
    if (!drag) return;
    e.preventDefault();
    e.stopPropagation();
    const k = (180 / drag.size) * (e.ctrlKey ? FINE_DRAG : 1);
    if (e.ctrlKey !== drag.fine) {  // re-anchor when Ctrl changes, so the photo doesn't jump
      Object.assign(drag, { x: e.clientX, y: e.clientY, yaw: value(node, "yaw"), pitch: value(node, "pitch"), fine: e.ctrlKey });
      return;
    }
    setWidget(node, "yaw", drag.yaw + (e.clientX - drag.x) * k);
    setWidget(node, "pitch", drag.pitch - (e.clientY - drag.y) * k);
  });
  const endDrag = (e) => {
    if (!drag) return;
    e.stopPropagation();
    drag = null;
    frame.classList.remove("dragging");
  };
  frame.addEventListener("pointerup", endDrag);
  frame.addEventListener("pointercancel", endDrag);

  // ---- scroll to resize (the diagonal field of view) ------------------------------------------------
  const stopWheel = captureWheel(frame, (e) => {
    if (!photo) return false;
    if (cropper.isOpen) return cropper.wheel(e);
    const s = state();
    const step = (e.deltaY > 0 ? -WHEEL_DEG : WHEEL_DEG) * (e.ctrlKey ? FINE_DRAG : 1);
    if (s.focal_35mm > 0) {  // switch to a manual FOV, starting from the lens's
      const info = photo.info;
      setWidget(node, "diag_fov", fovOf(s, info, cropRect(info.width, info.height, s.crop, info)));
      setWidget(node, "focal_35mm", 0);
    }
    setWidget(node, "diag_fov", value(node, "diag_fov") + step);
    return true;
  });

  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    stopLoop();
    stopWheel();
    stopCropKeys();
    cropper.destroy();
    ++ticket;
    for (const t of dropTargets) {
      t.removeEventListener("dragover", onDomDragOver, true);
      t.removeEventListener("dragenter", onDomDragOver, true);
      t.removeEventListener("drop", onDomDrop, true);
    }
    view.destroy();
    return onRemoved?.apply(this, arguments);
  };

  // a workflow saved before the crop input existed shifts its values by one: crop then holds the
  // upload button's "image" (or ""). Anything that isn't a crop spec means "auto".
  const onConfigure = node.onConfigure;
  node.onConfigure = function () {
    const r = onConfigure?.apply(this, arguments);
    const w = widget(node, "crop");
    if (w && !/^\s*(auto|\d+:\d+\s*,[^,]*,[^,]*,[^,]*)\s*$/i.test(String(w.value ?? ""))) w.value = START_CROP;
    return r;
  };

  if (node.size[0] < 460 || node.size[1] < 700) node.setSize([Math.max(node.size[0], 460), Math.max(node.size[1], 700)]);
}

app.registerExtension({
  name: "Flat2VR.PlacePhoto",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      setup(this);
      return r;
    };
  },
});
