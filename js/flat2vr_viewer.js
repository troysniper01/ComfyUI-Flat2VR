// Flat2VR 180 Preview: look around the finished VR180 image in the node, as in a headset:
// drag to look, scroll to zoom, double-click to recentre, and switch between the eyes. 60 fps WebGL.
import { app } from "../../scripts/app.js";
import {
  ShaderView, SPINNER_CSS, captureWheel, clamp, deg, el, fetchBitmap, frameLoop, injectStyle, rotationColumns,
} from "./flat2vr_gl.js";

const NODE = "Flat2VRPreview180";
const MAX_CANVAS_PX = 2560;
const HOME = { yaw: 0, pitch: 0, fov: 90 };  // vertical FOV, degrees
const FOV_RANGE = [25, 125];
const YAW_LIMIT = 105;  // a little past the 180° edge, so the border can be inspected
const PITCH_LIMIT = 89;
const EASE = 0.28;  // per frame, towards the target view (wheel, recentre)
const FRICTION = 0.9;  // per frame, for the glide after a flick
const MIN_SPEED = 0.01;  // deg/frame below which the glide stops

const FRAGMENT = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uSize;
uniform vec3 uC0, uC1, uC2;  // view rotation columns: world = c0*x + c1*y + c2*z
uniform float uTanH;         // tan(vertical FOV / 2)
uniform vec2 uEye;           // the eye's horizontal offset and width in the texture
uniform float uEdge;         // half a texel of the eye, to keep mipmaps off the other eye
uniform float uHas;
out vec4 outColor;
const float PI = 3.141592653589793;
void main() {
  vec2 ndc = gl_FragCoord.xy / uSize * 2.0 - 1.0;
  vec3 cam = normalize(vec3(ndc.x * uTanH * uSize.x / uSize.y, ndc.y * uTanH, 1.0));
  vec3 w = uC0 * cam.x + uC1 * cam.y + uC2 * cam.z;
  float lon = atan(w.x, w.z), lat = asin(clamp(w.y, -1.0, 1.0));
  float u = clamp(lon / PI + 0.5, uEdge, 1.0 - uEdge);
  vec3 img = texture(uTex, vec2(uEye.x + u * uEye.y, 0.5 - lat / PI)).rgb;
  vec3 bg = vec3(0.055, 0.058, 0.066);
  float outside = smoothstep(PI / 2.0, PI / 2.0 + 0.012, abs(lon));
  outColor = vec4(mix(uHas > 0.5 ? img : bg, bg, outside), 1.0);
}`;

const STYLE = `
.f2v-view { position: relative; width: 100%; height: 100%; min-height: 0; border-radius: 8px; overflow: hidden;
  background: #0e0f11; box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06); user-select: none; touch-action: none;
  font: 12px/1.3 system-ui, -apple-system, "Segoe UI", sans-serif; color: #e8e8ea; }
.f2v-view canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; cursor: grab; }
.f2v-view.dragging canvas { cursor: grabbing; }
.f2v-view .empty { position: absolute; inset: 0; display: flex; flex-direction: column; gap: 8px; align-items: center;
  justify-content: center; text-align: center; padding: 24px; color: #9a9ca3; pointer-events: none; }
.f2v-view .empty b { color: #d8d9dd; font-size: 14px; font-weight: 600; }
.f2v-view .empty .globe { width: 42px; height: 42px; border-radius: 50%; border: 2px solid #3b3e46; position: relative; margin-bottom: 4px; }
.f2v-view .empty .globe::before, .f2v-view .empty .globe::after { content: ""; position: absolute; border: 2px solid #3b3e46; border-radius: 50%; }
.f2v-view .empty .globe::before { inset: -2px 11px; }
.f2v-view .empty .globe::after { inset: 11px -2px; }
.f2v-view .hint { position: absolute; top: 10px; left: 10px; padding: 5px 9px; border-radius: 999px; pointer-events: none;
  background: rgba(14,15,17,0.62); backdrop-filter: blur(6px); color: #c9cad0; transition: opacity 600ms ease; }
.f2v-view .hint.gone { opacity: 0; }
.f2v-view .bar { position: absolute; left: 10px; right: 10px; bottom: 10px; display: flex; align-items: center; gap: 8px;
  pointer-events: none; }
.f2v-view .bar > * { pointer-events: auto; }
.f2v-view .glass { background: rgba(14,15,17,0.66); backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.08);
  border-radius: 9px; }
.f2v-view .eyes { display: inline-flex; padding: 3px; gap: 2px; }
.f2v-view .eyes button { all: unset; cursor: pointer; padding: 5px 12px; border-radius: 6px; color: #b9bbc2; font-weight: 600;
  transition: background 120ms ease, color 120ms ease; }
.f2v-view .eyes button:hover { color: #fff; background: rgba(255,255,255,0.07); }
.f2v-view .eyes button.on { background: #e8e8ea; color: #111214; }
.f2v-view .eyes button:focus-visible, .f2v-view .tool:focus-visible { outline: 2px solid #6aa7ff; outline-offset: 1px; }
.f2v-view .mono { padding: 7px 11px; color: #b9bbc2; }
.f2v-view .spacer { flex: 1; pointer-events: none; }
.f2v-view .readout { padding: 7px 10px; color: #b9bbc2; font-variant-numeric: tabular-nums; }
.f2v-view .tool { all: unset; cursor: pointer; width: 30px; height: 30px; display: inline-flex; align-items: center; justify-content: center;
  color: #d8d9dd; transition: background 120ms ease; }
.f2v-view .tool:hover { background: rgba(255,255,255,0.1); }
.f2v-view .tool svg { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }
.f2v-view .bar.idle { opacity: 0.35; }
.f2v-view:fullscreen { border-radius: 0; }
${SPINNER_CSS}`;

const ICON_RECENTRE = `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4"/></svg>`;
const ICON_FULL = `<svg viewBox="0 0 24 24"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>`;
const ICON_EXIT = `<svg viewBox="0 0 24 24"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5"/></svg>`;

function setup(node) {
  injectStyle("f2v-view-style", STYLE);
  const root = el("div", "f2v-view");
  const canvas = el("canvas", null, root);
  const empty = el("div", "empty", root);
  el("div", "globe", empty);
  el("b", null, empty, "Run the workflow to look around the VR180 result here");
  el("span", null, empty, "Drag to look · scroll to zoom · double-click to recentre");
  const hint = el("div", "hint gone", root, "Drag to look · scroll to zoom · double-click to recentre");
  const spinner = el("div", "f2v-spinner", root);
  const bar = el("div", "bar idle", root);
  const eyes = el("div", "eyes glass", bar);
  const leftBtn = el("button", "on", eyes, "Left eye");
  const rightBtn = el("button", null, eyes, "Right eye");
  const mono = el("div", "mono glass", bar, "Single eye");
  mono.style.display = "none";
  el("div", "spacer", bar);
  const readout = el("div", "readout glass", bar, "");
  const recentre = el("button", "tool glass", bar);
  recentre.innerHTML = ICON_RECENTRE;
  recentre.title = "Recentre (double-click)";
  const full = el("button", "tool glass", bar);
  full.innerHTML = ICON_FULL;
  full.title = "Fullscreen";

  node.addDOMWidget("vr180", "flat2vr_viewer", root, { serialize: false, hideOnZoom: false, getMinHeight: () => 280 });

  let view;
  try {
    view = new ShaderView(canvas, FRAGMENT, MAX_CANVAS_PX);
  } catch (e) {
    empty.querySelector("b").textContent = `The 180 preview needs WebGL2 (${e.message})`;
    return;
  }

  // ---- state ---------------------------------------------------------------------------------------
  const cur = { ...HOME };
  const target = { ...HOME };
  const vel = { yaw: 0, pitch: 0 };
  let image = null;  // { stereo, eyePx }
  let eye = 0;  // 0 left, 1 right
  let dirty = true;
  let ticket = 0;

  const setEye = (e) => {
    if (!image?.stereo) e = 0;
    eye = e;
    leftBtn.classList.toggle("on", e === 0);
    rightBtn.classList.toggle("on", e === 1);
    leftBtn.setAttribute("aria-pressed", String(e === 0));
    rightBtn.setAttribute("aria-pressed", String(e === 1));
    dirty = true;
  };
  for (const [b, e] of [[leftBtn, 0], [rightBtn, 1]]) {
    b.addEventListener("pointerdown", (ev) => ev.stopPropagation());
    b.addEventListener("click", (ev) => { ev.stopPropagation(); setEye(e); });
  }

  const goHome = () => { Object.assign(target, HOME); vel.yaw = vel.pitch = 0; };
  recentre.addEventListener("pointerdown", (ev) => ev.stopPropagation());
  recentre.addEventListener("click", (ev) => { ev.stopPropagation(); goHome(); });
  full.addEventListener("pointerdown", (ev) => ev.stopPropagation());
  full.addEventListener("click", (ev) => {
    ev.stopPropagation();
    if (document.fullscreenElement === root) document.exitFullscreen();
    else root.requestFullscreen?.().catch(() => {});
  });
  const onFullscreen = () => {
    const on = document.fullscreenElement === root;
    full.innerHTML = on ? ICON_EXIT : ICON_FULL;
    full.title = on ? "Exit fullscreen (Esc)" : "Fullscreen";
    dirty = true;
  };
  document.addEventListener("fullscreenchange", onFullscreen);

  // ---- loading a result ------------------------------------------------------------------------------
  async function show(result) {
    if (!result) return;
    const my = ++ticket;
    spinner.classList.add("on");
    try {
      const bitmap = await fetchBitmap(result, view.gl);
      if (my !== ticket) { bitmap.close(); return; }
      view.setImage(bitmap);
      const stereo = bitmap.width >= 1.5 * bitmap.height;
      image = { stereo, eyePx: stereo ? bitmap.width / 2 : bitmap.width };
      bitmap.close();
      eyes.style.display = stereo ? "" : "none";
      mono.style.display = stereo ? "none" : "";
      setEye(stereo ? eye : 0);
      empty.style.display = "none";
      bar.classList.remove("idle");
      hint.classList.remove("gone");
      dirty = true;
    } catch (e) {
      if (my !== ticket) return;
      empty.style.display = "";
      empty.querySelector("b").textContent = `Couldn't load the preview (${e.message}). Run the workflow again.`;
    } finally {
      if (my === ticket) spinner.classList.remove("on");
    }
  }
  node.__flat2vrShow = show;

  // ---- interaction: drag to look (with a glide), scroll to zoom, double-click to recentre ------------
  let drag = null;
  const degPerPx = () => cur.fov / Math.max(1, canvas.getBoundingClientRect().height);
  const dismissHint = () => hint.classList.add("gone");
  canvas.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || !image) return;
    e.preventDefault();
    e.stopPropagation();
    canvas.setPointerCapture(e.pointerId);
    root.classList.add("dragging");
    drag = { x: e.clientX, y: e.clientY, t: performance.now() };
    vel.yaw = vel.pitch = 0;
    dismissHint();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    e.preventDefault();
    e.stopPropagation();
    const k = degPerPx();
    const dyaw = -(e.clientX - drag.x) * k, dpitch = (e.clientY - drag.y) * k;
    const now = performance.now(), frames = Math.max(1, (now - drag.t) / 16.7);
    vel.yaw = 0.6 * vel.yaw + 0.4 * (dyaw / frames);
    vel.pitch = 0.6 * vel.pitch + 0.4 * (dpitch / frames);
    target.yaw = cur.yaw = clamp(cur.yaw + dyaw, -YAW_LIMIT, YAW_LIMIT);
    target.pitch = cur.pitch = clamp(cur.pitch + dpitch, -PITCH_LIMIT, PITCH_LIMIT);
    Object.assign(drag, { x: e.clientX, y: e.clientY, t: now });
    dirty = true;
  });
  const endDrag = (e) => {
    if (!drag) return;
    e.stopPropagation();
    if (performance.now() - drag.t > 80) vel.yaw = vel.pitch = 0;  // held still before letting go: no glide
    drag = null;
    root.classList.remove("dragging");
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("dblclick", (e) => { e.stopPropagation(); goHome(); });
  const stopWheel = captureWheel(canvas, (e) => {
    if (!image) return false;
    target.fov = clamp(target.fov * Math.exp(e.deltaY * 0.0012), FOV_RANGE[0], FOV_RANGE[1]);
    dismissHint();
    return true;
  });

  // ---- the frame loop --------------------------------------------------------------------------------
  const stopLoop = frameLoop(() => {
    const r = canvas.getBoundingClientRect();
    if (!canvas.isConnected || r.width < 2 || r.height < 2) return;  // hidden or collapsed
    if (!drag && (Math.abs(vel.yaw) > MIN_SPEED || Math.abs(vel.pitch) > MIN_SPEED)) {
      target.yaw = cur.yaw = clamp(cur.yaw + vel.yaw, -YAW_LIMIT, YAW_LIMIT);
      target.pitch = cur.pitch = clamp(cur.pitch + vel.pitch, -PITCH_LIMIT, PITCH_LIMIT);
      vel.yaw *= FRICTION;
      vel.pitch *= FRICTION;
      dirty = true;
    }
    for (const k of ["yaw", "pitch", "fov"]) {
      const d = target[k] - cur[k];
      if (Math.abs(d) > 1e-3) { cur[k] += d * EASE; dirty = true; } else cur[k] = target[k];
    }
    if (view.fit()) dirty = true;
    if (!dirty) return;
    dirty = false;
    const cols = rotationColumns(cur.yaw, cur.pitch);
    const eyeW = image?.stereo ? 0.5 : 1;
    view.draw({
      uC0: cols[0], uC1: cols[1], uC2: cols[2], uTanH: Math.tan(deg(cur.fov) / 2),
      uEye: [image?.stereo && eye === 1 ? 0.5 : 0, eyeW], uEdge: image ? 0.5 / image.eyePx : 0, uHas: image ? 1 : 0,
    });
    readout.textContent = `${Math.round(cur.yaw)}° · ${Math.round(cur.pitch)}° · FOV ${Math.round(cur.fov)}°`;
  });

  const onRemoved = node.onRemoved;
  node.onRemoved = function () {
    ++ticket;
    stopLoop();
    stopWheel();
    document.removeEventListener("fullscreenchange", onFullscreen);
    view.destroy();
    return onRemoved?.apply(this, arguments);
  };

  if (node.size[0] < 520 || node.size[1] < 380) node.setSize([Math.max(node.size[0], 720), Math.max(node.size[1], 460)]);
}

app.registerExtension({
  name: "Flat2VR.Preview180",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      setup(this);
      return r;
    };
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      this.__flat2vrShow?.(message?.flat2vr_vr180?.[0]);
      return r;
    };
  },
});
