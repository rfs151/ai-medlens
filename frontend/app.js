/* MedLens frontend - no build step, plain ES2020.
 * One menu per imaging type; each menu uses its own model (models/<model>.onnx) and its own demo samples. */
"use strict";

const $ = (id) => document.getElementById(id);
const COLORS = ["#ff3838", "#ff9d97", "#ff701f", "#ffb21d", "#cfd231", "#48f90a", "#3ddb86", "#1a9334"];
const FETCH_FLOOR_CONF = 0.05; // fetch once with a low threshold, filter on the client when the slider moves

// Each imaging menu = one independent model. A tooth model never sees blood cells and vice versa.
// mode "video" menus reuse an image menu's model and samples (samples: manifest key) but only show video detection.
const MODALITIES = [  // first entry is the default menu
  {
    key: "fracture", model: "fracture", icon: "🦴", title: "X 光骨折",
    intro: "在骨骼 X 光片上框出骨折部位。",
  },
  {
    key: "blood-cell", model: "blood-cell", icon: "🩸", title: "血细胞计数",
    intro: "显微镜血涂片中检测红细胞、血小板和 5 类白细胞并计数，可与传统 OpenCV 算法对比。",
    classic: true,
  },
  {
    key: "brain-tumor", model: "brain-tumor", icon: "🧠", title: "脑 MRI 肿瘤",
    intro: "在脑部 MRI 切片中检测胶质瘤、脑膜瘤和垂体瘤；示例里附了一张无肿瘤的对照图。",
  },
  {
    key: "dental", model: "dental", icon: "🦷", title: "口腔全景片",
    intro: "在口腔全景 X 光片上检测龋齿、根尖周病变和阻生齿。",
  },
  {
    key: "polyp", model: "polyp", icon: "🔬", title: "结肠镜息肉",
    intro: "在结肠镜截帧中检测息肉。临床上漏诊代价远高于误报，因此阈值偏低、追求高召回。",
  },
  {
    key: "blood-cell-video", model: "blood-cell", samples: "blood-cell", mode: "video", icon: "🎞", title: "血细胞视频检测",
    intro: "模拟显微镜下移动载玻片：逐帧检测并实时计数，显示 FPS 和延迟。也可以打开本地视频或摄像头。",
  },
  {
    key: "polyp-video", model: "polyp", samples: "polyp", mode: "video", icon: "🎥", title: "息肉视频检测",
    intro: "结肠镜检查视频逐帧实时检测息肉，显示 FPS 和延迟。也可以打开本地视频或摄像头。",
  },
];
const samplesOf = (mod = state.mod) => state.manifest[mod.samples || mod.key] || {};

const LABEL_ZH = {
  polyp: "息肉", lesion: "息肉", fracture: "骨折", RBC: "红细胞", WBC: "白细胞", Platelets: "血小板",
  Basophil: "嗜碱性粒细胞", Eosinophil: "嗜酸性粒细胞", Lymphocyte: "淋巴细胞", Monocyte: "单核细胞",
  Neutrophil: "中性粒细胞", glioma: "胶质瘤", meningioma: "脑膜瘤", pituitary: "垂体瘤", "pituitary tumor": "垂体瘤",
  caries: "龋齿", periapical_lesion: "根尖周病变", impacted_tooth: "阻生齿",
};
const zh = (label) => (LABEL_ZH[label] ? `${LABEL_ZH[label]} ${label}` : label);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = {
  models: [],
  backend: "python",
  manifest: {},
  mod: MODALITIES[0],
  image: null,        // HTMLImageElement being shown
  imageFile: null,
  imageCaption: "",
  result: null,       // last /api/detect response
  backendError: "",   // set when the selected backend cannot be reached
  cppPort: 8081,      // overridden by the Python service's /api/health (MEDLENS_CPP_PORT)
};

// ---------------------------------------------------------------- backends
function cppBaseUrl() {
  const q = new URLSearchParams(location.search).get("cpp");
  try { if (q) localStorage.setItem("medlens.cpp", q); } catch { /* ignore */ }
  let saved = null;
  try { saved = localStorage.getItem("medlens.cpp"); } catch { /* ignore */ }
  if (saved) return saved;
  // Behind the HTTPS gateway (https://medlens.edgebot.vip) the C++ service has no TLS port of its own:
  // the gateway proxies /cpp/* to it. Over plain http, go straight to its port on the same host.
  if (location.protocol === "https:") return `${location.origin}/cpp`;
  return `${location.protocol}//${location.hostname}:${state.cppPort}`;
}
function baseUrl(backend = state.backend) {
  return backend === "cpp" ? cppBaseUrl() : "";
}
function modelInfo(mod = state.mod) {
  return state.models.find((m) => m.name === mod.model);
}

async function api(path, { backend, method = "GET", body, params } = {}) {
  const url = new URL(baseUrl(backend) + path, location.href);
  Object.entries(params || {}).forEach(([k, v]) => v !== undefined && v !== null && v !== "" && url.searchParams.set(k, v));
  const res = await fetch(url, { method, body });
  if (!res.ok) {
    const text = await res.text();
    let msg = `${res.status}`;
    try { msg = JSON.parse(text).detail || text; } catch { msg = text || msg; }
    throw new Error(msg);
  }
  return res.json();
}

async function refreshBackend() {
  const status = $("backendStatus");
  status.textContent = "连接中…";
  status.className = "pill";
  try {
    const health = await api("/api/health");
    const { models } = await api("/api/models");
    state.models = models;
    state.backendError = "";
    const needed = [...new Set(MODALITIES.map((m) => m.model))]; // video menus share their image menu's model
    const installed = needed.filter((name) => models.some((m) => m.name === name)).length;
    status.textContent = `${health.backend} · ${installed}/${needed.length} 个模型`;
    status.className = installed === needed.length ? "pill ok" : installed ? "pill warn" : "pill err";
  } catch (e) {
    status.textContent = `后端不可用: ${e.message}`;
    status.className = "pill err";
    state.models = [];
    state.backendError = e.message;
  }
  renderMenu();
  renderModelCard();
}

async function loadManifest() {
  try {
    const res = await fetch("samples/manifest.json");
    state.manifest = res.ok ? await res.json() : {};
  } catch {
    state.manifest = {};
  }
}

// ---------------------------------------------------------------- drawing
function colorFor(cls) { return COLORS[cls % COLORS.length]; }

function drawDetections(canvas, img, boxes) {
  const ctx = canvas.getContext("2d");
  if (img) {
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    ctx.drawImage(img, 0, 0);
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  const lw = Math.max(2, Math.round((canvas.width + canvas.height) / 600));
  const font = Math.max(12, Math.round(canvas.width / 50));
  ctx.lineWidth = lw;
  ctx.font = `${font}px sans-serif`;
  boxes.forEach((b) => {
    const c = colorFor(b.cls);
    ctx.strokeStyle = c;
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    const text = b.conf < 1 ? `${b.label} ${b.conf.toFixed(2)}` : b.label;
    const tw = ctx.measureText(text).width + 6;
    const ty = Math.max(b.y1, font + 4);
    ctx.fillStyle = c;
    ctx.fillRect(b.x1, ty - font - 4, tw, font + 4);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, b.x1 + 3, ty - 4);
  });
}

function drawBarChart(canvas, counts) {
  const ctx = canvas.getContext("2d");
  const entries = Object.entries(counts);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!entries.length) return;
  const max = Math.max(...entries.map(([, v]) => v));
  const pad = 16, bw = (canvas.width - pad * 2) / entries.length;
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  entries.forEach(([label, v], i) => {
    const h = ((canvas.height - 48) * v) / max;
    const x = pad + i * bw + bw * 0.15, y = canvas.height - 24 - h;
    ctx.fillStyle = COLORS[i % COLORS.length];
    ctx.fillRect(x, y, bw * 0.7, h);
    ctx.fillStyle = getComputedStyle(document.body).color;
    ctx.fillText(String(v), x + bw * 0.35, y - 4);
    ctx.fillText(LABEL_ZH[label] || label, x + bw * 0.35, canvas.height - 8);
  });
}

function countBy(boxes) {
  return boxes.reduce((acc, b) => ((acc[b.label] = (acc[b.label] || 0) + 1), acc), {});
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("浏览器无法显示该图片"));
    img.src = src;
  });
}

async function sampleFile(entry) {
  const res = await fetch(`samples/${entry.file}`);
  if (!res.ok) throw new Error(`示例文件缺失：${entry.file}`);
  const blob = await res.blob();
  return new File([blob], entry.file.split("/").pop(), { type: blob.type });
}

// ---------------------------------------------------------------- menu + model card
function renderMenu() {
  $("modalityMenu").innerHTML = MODALITIES.map((m, i) => {
    const missing = !state.backendError && !modelInfo(m) ? `<span class="badge">未安装</span>` : "";
    const firstVideo = m.mode === "video" && MODALITIES[i - 1]?.mode !== "video";
    return `${firstVideo ? `<span class="menu-sep" aria-hidden="true"></span>` : ""}<button class="menu-item${m === state.mod ? " active" : ""}" data-key="${m.key}">
      <span class="icon">${m.icon}</span>${esc(m.title)}${missing}</button>`;
  }).join("");
  document.querySelectorAll(".menu-item").forEach((btn) =>
    btn.addEventListener("click", () => selectModality(btn.dataset.key)));
}

function renderModelCard() {
  const mod = state.mod;
  const m = modelInfo();
  $("modTitle").textContent = `${mod.icon} ${mod.title}`;
  $("modIntro").textContent = mod.intro;
  if (state.backendError) {
    const cpp = state.backend === "cpp";
    $("modelCard").innerHTML = `<div class="missing">
      <b>${cpp ? "C++ 后端" : "Python 后端"}未启动或连接不上</b>
      <p>${cpp ? `页面在找 <code>${esc(cppBaseUrl())}</code>。用 <code>docker compose --profile cpp up -d --build</code> 启动 C++ 服务，
      或者把右上角切回 Python 后端。` : "请确认 uvicorn 服务正在运行。"}</p></div>`;
    return;
  }
  if (!m) {
    $("modelCard").innerHTML = `<div class="missing">
      <b>模型 <code>${esc(mod.model)}</code> 未安装</b>
      <p>在项目根目录运行 <code>python scripts/fetch_models.py ${esc(mod.model)}</code>，然后重启后端。
      示例图片仍可浏览，但无法检测。</p></div>`;
    return;
  }
  const fmt = (v) => (typeof v === "number" ? v.toFixed(3) : "-");
  const mt = m.metrics || {};
  $("modelCard").innerHTML = `
    <div class="model-desc">${esc(m.description || m.name)}</div>
    <div class="kpis">
      <div><span>mAP@0.5</span><b>${fmt(mt.map50)}</b></div>
      <div><span>mAP@.5:.95</span><b>${fmt(mt.map50_95)}</b></div>
      <div><span>Precision</span><b>${fmt(mt.precision)}</b></div>
      <div><span>Recall</span><b>${fmt(mt.recall)}</b></div>
    </div>
    <div class="hint">类别：${m.classes.map((c) => esc(zh(c))).join("、")} · 输入 ${m.imgsz}px · ${esc(m.provider)}
      ${mt.source ? ` · 指标来自模型卡（作者验证集）` : ""}</div>
    <div class="hint">许可：${esc(m.license || "-")}${m.source ? ` · <a href="${esc(m.source)}" target="_blank" rel="noopener">模型来源</a>` : ""}</div>`;
}

function selectModality(key) {
  const mod = MODALITIES.find((m) => m.key === key) || MODALITIES[0];
  state.mod = mod;
  if (location.hash.slice(1) !== mod.key) history.replaceState(null, "", `#${mod.key}`);
  stopVideo();

  const m = modelInfo();
  const conf = m?.recommended_conf ?? 0.25;
  const iou = m?.recommended_iou ?? 0.45;
  $("confSlider").value = $("vConfSlider").value = conf;
  $("confValue").textContent = $("vConfValue").textContent = conf.toFixed(2);
  $("iouSlider").value = iou;
  $("iouValue").textContent = iou.toFixed(2);

  // sub tabs: video menus show only video detection; classic comparison only for blood cells
  const isVideo = mod.mode === "video";
  document.querySelector('.tab[data-tab="image"]').hidden = isVideo;
  document.querySelector('.tab[data-tab="video"]').hidden = !isVideo;
  document.querySelector('.tab[data-tab="classic"]').hidden = !mod.classic;
  $("subTabs").hidden = [...document.querySelectorAll(".tab")].filter((t) => !t.hidden).length < 2;
  showTab(isVideo ? "video" : "image");

  renderMenu();
  renderModelCard();
  renderGallery();
  renderVideoSamples();
  if (mod.classic) renderClassicGallery();
  resetImage();

  if (isVideo) {
    $("videoSamples").querySelector(".chip")?.click(); // play the first demo video and start detecting
  } else {
    const first = (samplesOf(mod).images || [])[0];
    if (first) onSample(first, $("gallery").querySelector(".thumb"));
  }
}

function showTab(name) {
  document.querySelectorAll(".tab, .panel").forEach((el) => el.classList.remove("active"));
  document.querySelector(`.tab[data-tab="${name}"]`).classList.add("active");
  $(`tab-${name}`).classList.add("active");
  if (name !== "video") stopVideo();
}

// ---------------------------------------------------------------- image tab
function renderGallery() {
  const images = samplesOf().images || [];
  $("gallery").innerHTML = images.map((e, i) => `
    <button class="thumb" data-i="${i}" title="${esc(e.caption)}">
      <img src="samples/${esc(e.file)}" alt="${esc(e.caption)}" loading="lazy"><span>${esc(e.caption)}</span>
    </button>`).join("") + `
    <label class="thumb local">
      <input type="file" id="imageInput" accept="image/*,.dcm,.dicom,application/dicom">
      <div class="plus">＋</div><span>本地图片 / DICOM</span>
    </label>`;
  $("gallery").querySelectorAll("button.thumb").forEach((btn) =>
    btn.addEventListener("click", () => onSample(images[+btn.dataset.i], btn)));
  $("imageInput").addEventListener("change", (e) => {
    markThumb(null);
    onImageFile(e.target.files[0], "本地文件");
  });
}

function markThumb(btn) {
  $("gallery").querySelectorAll(".thumb").forEach((t) => t.classList.toggle("active", t === btn));
}

async function onSample(entry, btn) {
  markThumb(btn);
  try {
    onImageFile(await sampleFile(entry), `${entry.caption}<br><small>来源：${esc(entry.source)} · ${esc(entry.license)}</small>`);
  } catch (e) {
    $("timing").innerHTML = `<span class="err-text">${esc(e.message)}</span>`;
  }
}

function resetImage() {
  state.image = state.imageFile = state.result = null;
  const c = $("imageCanvas");
  c.width = c.height = 0;
  $("imagePlaceholder").style.display = "";
  $("sampleCaption").innerHTML = "";
  renderImageResult();
}

// Some models occasionally emit one box covering the whole frame; models/<name>.json can cap the box area.
function plausibleBoxes(boxes, width, height) {
  const maxArea = modelInfo()?.max_box_area;
  if (!maxArea) return boxes;
  return boxes.filter((b) => (b.x2 - b.x1) * (b.y2 - b.y1) <= maxArea * width * height);
}

function visibleBoxes() {
  if (!state.result) return [];
  const conf = parseFloat($("confSlider").value);
  const { width, height } = state.result.image;
  return plausibleBoxes(state.result.boxes, width, height).filter((b) => b.conf >= conf);
}

function renderImageResult() {
  const boxes = visibleBoxes();
  if (state.image) drawDetections($("imageCanvas"), state.image, boxes);
  $("boxCount").textContent = state.result ? `${boxes.length} 个目标` : "";
  $("resultBody").innerHTML = boxes
    .map((b, i) => `<tr><td>${i + 1}</td><td><span class="dot" style="background:${colorFor(b.cls)}"></span>${esc(zh(b.label))}</td>
      <td>${(b.conf * 100).toFixed(1)}%</td><td>${[b.x1, b.y1, b.x2, b.y2].map(Math.round).join(", ")}</td></tr>`)
    .join("");
  drawBarChart($("countChart"), countBy(boxes));
  const r = state.result;
  const t = r?.timing_ms || {};
  const dicom = r?.dicom;
  if (r) {
    $("timing").innerHTML = `<div>模型 <b>${esc(r.model)}</b> · 后端 <b>${esc(r.backend)}</b> (${esc(r.provider || "-")})</div>
       <div>总耗时 <b>${t.total ?? "-"} ms</b>（预处理 ${t.preprocess ?? "-"} / 推理 ${t.inference ?? "-"} / 后处理 ${t.postprocess ?? "-"}）</div>
       <div>往返延迟 <b>${r.rtt_ms} ms</b></div>
       ${boxes.length ? "" : `<div>当前阈值下未检出目标</div>`}
       ${dicom ? `<div>DICOM：${esc(dicom.modality || "-")} · WC ${dicom.window_center.toFixed(0)} / WW ${dicom.window_width.toFixed(0)}</div>` : ""}`;
  }
  $("downloadBtn").disabled = $("reportBtn").disabled = !r;
}

async function runImageDetect() {
  if (!state.imageFile) return;
  const m = modelInfo();
  if (!m) {
    state.result = null;
    renderImageResult();
    $("timing").innerHTML = `<span class="err-text">${state.backendError ? "后端连接不上" : "该菜单的模型未安装"}，无法检测（见上方提示）。</span>`;
    return;
  }
  const file = state.imageFile;
  const fd = new FormData();
  fd.append("file", file);
  $("timing").textContent = "检测中…";
  const t0 = performance.now();
  try {
    const res = await api("/api/detect", {
      method: "POST",
      body: fd,
      params: {
        model: m.name, conf: FETCH_FLOOR_CONF, iou: $("iouSlider").value,
        wc: $("wcInput").value, ww: $("wwInput").value,
      },
    });
    if (file !== state.imageFile) return; // user picked another image meanwhile
    res.rtt_ms = Math.round(performance.now() - t0);
    if (res.image_png_base64) state.image = await loadImage(`data:image/png;base64,${res.image_png_base64}`);
    state.result = res;
    renderImageResult();
  } catch (e) {
    if (file !== state.imageFile) return;
    state.result = null;
    renderImageResult();
    $("timing").innerHTML = `<span class="err-text">检测失败：${esc(e.message)}</span>`;
  }
}

async function onImageFile(file, captionHtml = "") {
  if (!file) return;
  state.imageFile = file;
  state.result = null;
  state.imageCaption = file.name;
  $("sampleCaption").innerHTML = captionHtml;
  $("imagePlaceholder").style.display = "none";
  const isDicom = /\.(dcm|dicom)$/i.test(file.name) || file.type === "application/dicom";
  state.image = isDicom ? null : await loadImage(URL.createObjectURL(file)).catch(() => null);
  if (file !== state.imageFile) return;
  renderImageResult();
  await runImageDetect();
}

function exportReport() {
  const boxes = visibleBoxes();
  const imgUrl = $("imageCanvas").toDataURL("image/png");
  const r = state.result;
  const rows = boxes.map((b, i) => `<tr><td>${i + 1}</td><td>${esc(zh(b.label))}</td><td>${(b.conf * 100).toFixed(1)}%</td>
    <td>${[b.x1, b.y1, b.x2, b.y2].map(Math.round).join(", ")}</td></tr>`).join("");
  const counts = Object.entries(countBy(boxes)).map(([k, v]) => `${esc(zh(k))}: ${v}`).join("，") || "未检出目标";
  const w = window.open("", "_blank");
  if (!w) return alert("浏览器拦截了弹出窗口，请允许后重试");
  w.document.write(`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>MedLens 检测报告</title>
    <style>body{font-family:sans-serif;margin:32px;color:#222}img{max-width:100%;border:1px solid #ccc}
    table{border-collapse:collapse;width:100%;margin-top:12px}td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}
    .warn{color:#b00;font-weight:bold;border:2px solid #b00;padding:8px;margin:12px 0}</style></head><body>
    <h1>MedLens 影像检测报告 · ${esc(state.mod.title)}</h1>
    <div class="warn">本报告由 AI 演示系统自动生成，仅供技术演示，不用于临床诊断。</div>
    <p>文件：${esc(state.imageCaption || "-")}<br>模型：${esc(r.model)} · 后端：${esc(r.backend)}<br>
       置信度阈值：${$("confSlider").value} · IoU：${$("iouSlider").value}<br>
       生成时间：${new Date().toLocaleString()}<br>统计：${counts}</p>
    <img src="${imgUrl}">
    <table><thead><tr><th>#</th><th>类别</th><th>置信度</th><th>位置</th></tr></thead><tbody>${rows}</tbody></table>
    <script>window.onload=()=>window.print()<\/script></body></html>`);
  w.document.close();
}

// ---------------------------------------------------------------- video tab
const video = {
  running: false,
  inFlight: false,
  ws: null,
  frames: 0,
  lastFpsTime: 0,
  fps: 0,
  lastResult: null,
  latency: 0,
  grab: document.createElement("canvas"),
};

function renderVideoSamples() {
  const videos = samplesOf().videos || [];
  $("videoSamples").innerHTML = videos.map((e, i) =>
    `<button class="chip" data-i="${i}">▶ ${esc(e.caption)}</button>`).join("");
  $("videoSamples").querySelectorAll(".chip").forEach((btn) => btn.addEventListener("click", () => {
    const e = videos[+btn.dataset.i];
    $("videoSamples").querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c === btn));
    $("videoCaption").innerHTML = `${esc(e.caption)}<br><small>来源：${esc(e.source)} · ${esc(e.license)}</small>`;
    setVideoSource(`samples/${e.file}`, true);
  }));
  $("videoCaption").textContent = videos.length ? "选择上方示例视频，或打开本地视频 / 摄像头" : "打开本地视频或摄像头";
}

function openWs() {
  return new Promise((resolve, reject) => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const params = new URLSearchParams({ conf: $("vConfSlider").value, iou: $("iouSlider").value, model: state.mod.model });
    const ws = new WebSocket(`${proto}//${location.host}/ws/detect?${params}`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => resolve(ws);
    ws.onerror = () => reject(new Error("WebSocket 连接失败"));
    ws.onclose = () => { if (video.ws === ws) video.ws = null; };
  });
}

function grabFrame() {
  const v = $("video");
  const targetW = parseInt($("sendWidth").value, 10) || v.videoWidth;
  const scale = Math.min(1, targetW / v.videoWidth);
  video.grab.width = Math.round(v.videoWidth * scale);
  video.grab.height = Math.round(v.videoHeight * scale);
  video.grab.getContext("2d").drawImage(v, 0, 0, video.grab.width, video.grab.height);
  return new Promise((resolve) => video.grab.toBlob(resolve, "image/jpeg", 0.8)).then((blob) => ({ blob, scale }));
}

async function detectFrame(blob) {
  if (state.backend === "python") {
    if (!video.ws) video.ws = await openWs();
    const ws = video.ws;
    return new Promise((resolve, reject) => {
      ws.onmessage = (ev) => resolve(JSON.parse(ev.data));
      ws.onclose = () => { video.ws = null; reject(new Error("连接已断开")); };
      blob.arrayBuffer().then((buf) => ws.send(buf));
    });
  }
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  return api("/api/detect", { method: "POST", body: fd, params: { model: state.mod.model, conf: $("vConfSlider").value, iou: $("iouSlider").value } });
}

function videoLoop() {
  const v = $("video");
  if (!video.running) return;
  if (!video.inFlight && v.readyState >= 2 && v.videoWidth) {
    video.inFlight = true;
    const t0 = performance.now();
    grabFrame()
      .then(({ blob, scale }) => detectFrame(blob).then((res) => ({ res, scale })))
      .then(({ res, scale }) => {
        if (res.error) throw new Error(res.error);
        if (!video.running) return;
        // map boxes back from the down-scaled frame to the original video size
        res.boxes = plausibleBoxes(res.boxes, res.image.width, res.image.height);
        res.boxes.forEach((b) => { b.x1 /= scale; b.y1 /= scale; b.x2 /= scale; b.y2 /= scale; });
        res.counts = countBy(res.boxes);
        video.lastResult = res;
        video.latency = performance.now() - t0;
        video.frames++;
      })
      .catch((e) => { $("hud").textContent = `错误：${e.message}`; })
      .finally(() => { video.inFlight = false; });
  }
  const now = performance.now();
  if (now - video.lastFpsTime > 1000) {
    video.fps = (video.frames * 1000) / (now - video.lastFpsTime);
    video.frames = 0;
    video.lastFpsTime = now;
  }
  const overlay = $("videoOverlay");
  overlay.width = v.videoWidth;
  overlay.height = v.videoHeight;
  const res = video.lastResult;
  drawDetections(overlay, null, res ? res.boxes : []);
  const t = res?.timing_ms || {};
  $("hud").textContent = `检测 FPS ${video.fps.toFixed(1)} · 往返 ${video.latency.toFixed(0)} ms · 推理 ${t.inference ?? "-"} ms · 目标 ${res ? res.boxes.length : 0}`;
  $("videoStats").innerHTML = res
    ? Object.entries(res.counts || {}).map(([k, n]) => `<div>${esc(zh(k))}: <b>${n}</b></div>`).join("") || "<div>未检出目标</div>"
    : "";
  requestAnimationFrame(videoLoop);
}

function setVideoSource(src, autoStart = false) {
  const v = $("video");
  stopVideo();
  if (v.srcObject) v.srcObject.getTracks?.().forEach((t) => t.stop());
  if (src instanceof MediaStream) { v.srcObject = src; v.removeAttribute("src"); } else { v.srcObject = null; v.src = src; }
  v.play().catch(() => { /* autoplay may need a click */ });
  $("videoToggle").disabled = false;
  if (autoStart && modelInfo()) startVideo();
}

function startVideo() {
  if (!modelInfo()) {
    $("hud").textContent = "该菜单的模型未安装，无法检测";
    return;
  }
  video.running = true;
  video.lastFpsTime = performance.now();
  $("videoToggle").textContent = "停止检测";
  requestAnimationFrame(videoLoop);
}

function stopVideo() {
  video.running = false;
  video.lastResult = null;
  if (video.ws) { video.ws.close(); video.ws = null; }
  $("videoToggle").textContent = "开始检测";
  const o = $("videoOverlay");
  o.getContext("2d").clearRect(0, 0, o.width, o.height);
  $("hud").textContent = "";
}

// ---------------------------------------------------------------- classic comparison (blood cells)
function renderClassicGallery() {
  const images = samplesOf().images || [];
  $("classicGallery").innerHTML = images.map((e, i) => `
    <button class="thumb" data-i="${i}"><img src="samples/${esc(e.file)}" alt="" loading="lazy"><span>${esc(e.caption)}</span></button>`).join("") + `
    <label class="thumb local"><input type="file" id="cellsInput" accept="image/*"><div class="plus">＋</div><span>本地血涂片</span></label>`;
  const mark = (btn) => $("classicGallery").querySelectorAll(".thumb").forEach((t) => t.classList.toggle("active", t === btn));
  $("classicGallery").querySelectorAll("button.thumb").forEach((btn) => btn.addEventListener("click", async () => {
    mark(btn);
    runCellsCompare(await sampleFile(images[+btn.dataset.i]));
  }));
  $("cellsInput").addEventListener("change", (e) => { mark(null); e.target.files[0] && runCellsCompare(e.target.files[0]); });
}

async function runCellsCompare(file) {
  const img = await loadImage(URL.createObjectURL(file));
  const fd1 = new FormData(); fd1.append("file", file);
  const fd2 = new FormData(); fd2.append("file", file);
  const m = modelInfo();
  const groups = m?.count_groups || {};

  // the classic algorithm lives only in the Python backend
  const classic = api("/api/cells/classic", { backend: "python", method: "POST", body: fd1 })
    .then((r) => { drawDetections($("classicCanvas"), img, r.boxes); $("classicInfo").textContent = `${r.timing_ms.total} ms`; return r.counts; })
    .catch((e) => { $("classicInfo").textContent = e.message; return {}; });

  const yolo = m
    ? api("/api/detect", { method: "POST", body: fd2, params: { model: m.name, conf: m.recommended_conf || 0.25 } })
      .then((r) => {
        drawDetections($("yoloCanvas"), img, r.boxes);
        $("yoloInfo").textContent = `${r.timing_ms.total} ms · ${r.backend}`;
        const merged = {};
        Object.entries(r.counts).forEach(([k, n]) => { const g = groups[k] || k; merged[g] = (merged[g] || 0) + n; });
        return merged;
      })
      .catch((e) => { $("yoloInfo").textContent = e.message; return {}; })
    : Promise.resolve().then(() => { drawDetections($("yoloCanvas"), img, []); $("yoloInfo").textContent = "模型未安装"; return {}; });

  const [c1, c2] = await Promise.all([classic, yolo]);
  const labels = [...new Set([...Object.keys(c1), ...Object.keys(c2)])];
  $("compareBody").innerHTML = labels.map((l) => `<tr><td>${esc(zh(l))}</td><td>${c1[l] ?? 0}</td><td>${c2[l] ?? 0}</td></tr>`).join("");
}

// ---------------------------------------------------------------- wiring
async function init() {
  document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
    showTab(btn.dataset.tab);
    if (btn.dataset.tab === "classic" && !$("compareBody").children.length) {
      $("classicGallery").querySelector("button.thumb")?.click();
    }
  }));

  $("backendSelect").addEventListener("change", (e) => {
    state.backend = e.target.value;
    stopVideo();
    refreshBackend().then(runImageDetect);
  });

  const dz = $("dropZone");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("drag");
    markThumb(null);
    onImageFile(e.dataTransfer.files[0], "本地文件");
  });

  $("confSlider").addEventListener("input", (e) => { $("confValue").textContent = (+e.target.value).toFixed(2); renderImageResult(); });
  $("iouSlider").addEventListener("input", (e) => { $("iouValue").textContent = (+e.target.value).toFixed(2); });
  $("iouSlider").addEventListener("change", runImageDetect);
  $("wcInput").addEventListener("change", runImageDetect);
  $("wwInput").addEventListener("change", runImageDetect);

  $("downloadBtn").addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = $("imageCanvas").toDataURL("image/png");
    a.download = `medlens_${state.mod.key}_${Date.now()}.png`;
    a.click();
  });
  $("reportBtn").addEventListener("click", exportReport);

  $("videoInput").addEventListener("change", (e) => {
    if (!e.target.files[0]) return;
    $("videoSamples").querySelectorAll(".chip").forEach((c) => c.classList.remove("active"));
    $("videoCaption").textContent = e.target.files[0].name;
    setVideoSource(URL.createObjectURL(e.target.files[0]));
  });
  $("cameraBtn").addEventListener("click", async () => {
    try { setVideoSource(await navigator.mediaDevices.getUserMedia({ video: true })); } catch (e) { alert(`无法打开摄像头：${e.message}`); }
  });
  $("videoToggle").addEventListener("click", () => (video.running ? stopVideo() : startVideo()));
  $("vConfSlider").addEventListener("input", (e) => {
    $("vConfValue").textContent = (+e.target.value).toFixed(2);
    if (video.ws && video.ws.readyState === WebSocket.OPEN) video.ws.send(JSON.stringify({ conf: parseFloat(e.target.value) }));
  });

  window.addEventListener("hashchange", () => {
    const key = location.hash.slice(1);
    if (key !== state.mod.key && MODALITIES.some((m) => m.key === key)) selectModality(key);
  });

  try { // the Python service hosts the page and knows the C++ backend's port
    const h = await api("/api/health", { backend: "python" });
    if (h.cpp_port) state.cppPort = h.cpp_port;
  } catch { /* ignore */ }
  await Promise.all([loadManifest(), refreshBackend()]);
  selectModality(location.hash.slice(1));
}

init();
