// Shared helpers for the Flat2VR placer and 180 preview: a WebGL2 full-screen shader view,
// photo textures, rAF-driven redraws and the wheel capture the graph canvas would otherwise eat.
// (ComfyUI loads every .js in this folder; this one only exports.)
import { api } from "../../scripts/api.js";

const VERTEX = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;
const MAX_TEXTURE_SIDE = 4096;  // previews only; the nodes themselves use the full image

export const PI = Math.PI;
export const deg = (d) => (d * PI) / 180;
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Camera-to-world rotation (yaw, then pitch; no roll), as vr180_geometry.rotation: returns its
 * three columns, so world = c0*x + c1*y + c2*z and camera = (c0.w, c1.w, c2.w). */
export function rotationColumns(yawDeg, pitchDeg) {
  const cy = Math.cos(deg(yawDeg)), sy = Math.sin(deg(yawDeg));
  const cp = Math.cos(deg(pitchDeg)), sp = Math.sin(deg(pitchDeg));
  return [[cy, 0, -sy], [-sy * sp, cp, -cy * sp], [sy * cp, sp, cy * cp]];
}

export function injectStyle(id, css) {
  if (document.getElementById(id)) return;
  const style = document.createElement("style");
  style.id = id;
  style.textContent = css;
  document.head.appendChild(style);
}

export function el(tag, cls, parent, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

/** A canvas drawn by one fragment shader over a full-screen triangle. */
export class ShaderView {
  constructor(canvas, fragment, maxPx) {
    this.canvas = canvas;
    this.maxPx = maxPx;
    const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 isn't available in this browser");
    this.gl = gl;
    this.program = this.#link(VERTEX, fragment);
    this.uniforms = new Map();
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.program, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.texture = null;
  }

  #link(vs, fs) {
    const gl = this.gl;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  #loc(name) {
    if (!this.uniforms.has(name)) this.uniforms.set(name, this.gl.getUniformLocation(this.program, name));
    return this.uniforms.get(name);
  }

  /** Match the drawing buffer to the canvas's on-screen size (zoom and HiDPI included). True if it changed. */
  fit() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const scale = Math.min(1, this.maxPx / Math.max(1, r.width * dpr, r.height * dpr));
    const w = Math.max(1, Math.round(r.width * dpr * scale));
    const h = Math.max(1, Math.round(r.height * dpr * scale));
    if (this.canvas.width === w && this.canvas.height === h) return false;
    this.canvas.width = w;
    this.canvas.height = h;
    return true;
  }

  /** Upload an ImageBitmap as the (mipmapped) texture. */
  setImage(bitmap) {
    const gl = this.gl;
    if (this.texture) gl.deleteTexture(this.texture);
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const aniso = gl.getExtension("EXT_texture_filter_anisotropic");
    if (aniso) gl.texParameterf(gl.TEXTURE_2D, aniso.TEXTURE_MAX_ANISOTROPY_EXT, 8);
  }

  draw(uniforms) {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.uniform1i(this.#loc("uTex"), 0);
    gl.uniform2f(this.#loc("uSize"), this.canvas.width, this.canvas.height);
    for (const [name, v] of Object.entries(uniforms)) {
      const loc = this.#loc(name);
      if (loc == null) continue;
      if (typeof v === "number") gl.uniform1f(loc, v);
      else if (v.length === 2) gl.uniform2fv(loc, v);
      else if (v.length === 3) gl.uniform3fv(loc, v);
      else gl.uniform4fv(loc, v);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  destroy() {
    const gl = this.gl;
    if (this.texture) gl.deleteTexture(this.texture);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}

/** Fetch a ComfyUI file (/view) as an upright ImageBitmap, shrunk if it's beyond the preview cap. */
export async function fetchBitmap(view, gl) {
  const q = new URLSearchParams({ filename: view.filename, subfolder: view.subfolder || "", type: view.type || "input" });
  const res = await api.fetchApi(`/view?${q}`, { cache: "force-cache" });
  if (!res.ok) throw Object.assign(new Error(`HTTP ${res.status}`), { status: res.status });
  const blob = await res.blob();
  const probe = await createImageBitmap(blob, { imageOrientation: "from-image" });
  const cap = Math.min(MAX_TEXTURE_SIDE, gl.getParameter(gl.MAX_TEXTURE_SIZE));
  if (Math.max(probe.width, probe.height) <= cap) return probe;
  const s = cap / Math.max(probe.width, probe.height);
  const sized = await createImageBitmap(blob, {
    imageOrientation: "from-image", resizeWidth: Math.round(probe.width * s),
    resizeHeight: Math.round(probe.height * s), resizeQuality: "high",
  });
  probe.close();
  return sized;
}

/** Route wheel events over `target` to `handler` before the graph canvas can zoom with them. */
export function captureWheel(target, handler) {
  const onWheel = (e) => {
    if (!target.isConnected || !target.contains(e.target)) return;
    if (handler(e) === false) return;
    e.preventDefault();
    e.stopImmediatePropagation();
  };
  window.addEventListener("wheel", onWheel, { capture: true, passive: false });
  return () => window.removeEventListener("wheel", onWheel, { capture: true });
}

/** Call `tick` every animation frame until `stop()`; cheap when tick has nothing to do. */
export function frameLoop(tick) {
  let alive = true;
  const step = (t) => {
    if (!alive) return;
    tick(t);
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  return () => { alive = false; };
}

export const SPINNER_CSS = `
.f2v-spinner { position: absolute; inset: 0; display: none; align-items: center; justify-content: center;
  background: rgba(20,20,22,0.35); backdrop-filter: blur(1px); pointer-events: none; }
.f2v-spinner.on { display: flex; }
.f2v-spinner::after { content: ""; width: 34px; height: 34px; border-radius: 50%;
  border: 3px solid rgba(255,255,255,0.25); border-top-color: #fff; animation: f2v-spin 0.8s linear infinite; }
@keyframes f2v-spin { to { transform: rotate(360deg); } }
`;
