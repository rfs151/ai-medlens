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
  {
    key: "volume3d", model: null, mode: "volume", icon: "🧊", title: "3D 体渲染",
    intro: "脑胶质瘤 MRI 的 3D 体渲染与三视图（MPR）。C++ / OpenGL 实现，编译成 WebAssembly 在浏览器 GPU 上实时渲染。",
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
  hardware: null,     // server CPU/GPU from the Python service's /api/health
};

// ---------------------------------------------------------------- 3D volume viewer (C++ -> WebAssembly)
// gl_viewer/ is compiled with Emscripten to frontend/gl/medlens_volume.{js,wasm}; the C++ side owns
// the WebGL2 canvas and input, this code only loads it, feeds the volume and wires the HTML controls.
const TUMOUR_CLASSES = [
  { name: "水肿", en: "edema", color: "#fad140" },
  { name: "非增强肿瘤", en: "non-enhancing", color: "#f27326" },
  { name: "增强肿瘤", en: "enhancing", color: "#e61f33" },
];
const CHANNEL_ZH = { FLAIR: "FLAIR（水肿最明显）", T1Gd: "T1 增强（强化肿瘤）", T2: "T2" };
const vol = { module: null, loading: null, timer: 0, state: null, error: "" };

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error(`无法加载 ${src}`));
    document.head.appendChild(s);
  });
}

function volStatus(text, isError = false) {
  const el = $("volStatus");
  el.hidden = !text;
  el.textContent = text || "";
  el.classList.toggle("err-text", isError);
}

async function fetchGunzip(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`数据文件缺失：${url}（运行 python scripts/fetch_volume.py）`);
  if (typeof DecompressionStream === "undefined") throw new Error("浏览器不支持 DecompressionStream，请使用新版 Chrome / Edge / Firefox / Safari");
  const stream = res.body.pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function ensureVolumeViewer() {
  if (vol.loading) return vol.loading;
  vol.loading = (async () => {
    if (!document.createElement("canvas").getContext("webgl2")) throw new Error("浏览器或显卡不支持 WebGL2，无法显示 3D 视图");
    const entry = (samplesOf().volumes || [])[0];
    if (!entry) throw new Error("没有 3D 示例数据（运行 python scripts/fetch_volume.py）");
    volStatus("加载 C++ 渲染模块（WebAssembly）…");
    await loadScript("gl/medlens_volume.js");
    const module = await window.createMedlensVolume({
      locateFile: (p) => `gl/${p}`,
      onViewerError: (msg) => { vol.error = msg; volStatus(`渲染模块出错：${msg}`, true); },
      print: (t) => console.log("[viewer]", t),
      printErr: (t) => console.warn("[viewer]", t),
    });
    if (vol.error) throw new Error(vol.error);
    volStatus(`下载 MRI 数据（${entry.caption}）…`);
    const bytes = await fetchGunzip(`samples/${entry.file}`);
    const ptr = module._malloc(bytes.length);
    module.HEAPU8.set(bytes, ptr);  // HEAPU8 is re-read after malloc: memory may have grown
    const rc = module._mv_load_volume(ptr, bytes.length);
    module._free(ptr);
    if (rc !== 0) throw new Error(vol.error || "体数据解析失败");
    vol.module = module;
    vol.setInt = module.cwrap("mv_set_int", null, ["string", "number"]);
    vol.setFloat = module.cwrap("mv_set_float", null, ["string", "number"]);
    vol.entry = entry;
    setupVolumeControls();
    volStatus("");
    refreshVolumeState();
  })().catch((e) => {
    vol.loading = null;  // allow a retry by re-selecting the menu
    volStatus(e.message, true);
  });
  return vol.loading;
}

function setupVolumeControls() {
  const st = JSON.parse(vol.module.UTF8ToString(vol.module._mv_state_json()));
  $("volChannel").innerHTML = (st.channels || []).map((c, i) => `<option value="${i}">${esc(CHANNEL_ZH[c] || c)}</option>`).join("");
  $("volTumour").innerHTML = TUMOUR_CLASSES.map((c, i) => `<label class="check">
      <input type="checkbox" data-cls="${i}" checked><span class="swatch" style="background:${c.color}"></span>
      ${c.name} <small>${c.en}</small><b id="volMl${i}"></b></label>`).join("");
  $("volTumour").querySelectorAll("input").forEach((cb) =>
    cb.addEventListener("change", () => vol.setInt(`tumour${cb.dataset.cls}`, cb.checked ? 1 : 0)));
  const e = vol.entry;
  $("volSource").innerHTML = `数据：${esc(e.caption)} · ${esc(e.source)} · ${esc(e.license)} ·
    <a href="${esc(e.url)}" target="_blank" rel="noopener">来源</a>`;
}

function refreshVolumeState() {
  if (!vol.module) return;
  const st = JSON.parse(vol.module.UTF8ToString(vol.module._mv_state_json()));
  const first = !vol.state?.loaded;
  vol.state = st;
  if (!st.loaded) return;
  if (first) renderVolumeCard();
  (st.tumour_ml || []).forEach((v, i) => { const el = $(`volMl${i}`); if (el) el.textContent = `${v.toFixed(1)} ml`; });
  const label = st.label > 0 ? TUMOUR_CLASSES[st.label - 1].name : "非肿瘤";
  $("volCursor").innerHTML = `体素 <b>[${st.cursor.join(", ")}]</b> · 强度 <b>${st.value}</b> · 标注 <b>${label}</b><br>
    3D 渲染分辨率 <b>${Math.round(st.render_scale * 100)}%</b>${st.render_scale < 1 ? "（交互中先降分辨率，停下后自动精绘）" : ""}`;
}

function renderVolumeCard() {
  const st = vol.state;
  const total = st?.loaded ? st.tumour_ml.reduce((a, b) => a + b, 0).toFixed(1) : "-";
  $("modelCard").innerHTML = `
    <div class="model-desc">GPU 光线投射体渲染 + 多平面重建（MPR）</div>
    <div class="kpis">
      <div><span>体数据</span><b>${st?.loaded ? st.dims.join("×") : "-"}</b></div>
      <div><span>体素间距</span><b>${st?.loaded ? `${st.spacing[0]} mm` : "-"}</b></div>
      <div><span>肿瘤总体积</span><b>${total} ml</b></div>
      <div><span>序列</span><b>${st?.loaded ? st.channels.length : "-"}</b></div>
    </div>
    <div class="hint">C++17 · OpenGL 3.3 / OpenGL ES 3.0 (WebGL2) · Emscripten → WebAssembly · 不依赖第三方图形库</div>
    <div class="hint">数据：Medical Segmentation Decathlon 脑肿瘤任务（BraTS）· CC BY-SA 4.0 · 标注为专家手工分割，不是模型预测</div>`;
}

// Start / stop the per-menu work: the wasm module is loaded on first use and kept afterwards.
function volumeActive(on) {
  clearInterval(vol.timer);
  vol.timer = 0;
  if (!on) return;
  ensureVolumeViewer();
  vol.timer = setInterval(refreshVolumeState, 250);
}

function bindVolumeControls() {
  const intCtl = { volChannel: "channel", volMode: "mode", volPreset: "preset", volLayout: "layout" };
  Object.entries(intCtl).forEach(([id, key]) => $(id).addEventListener("change", (e) => {
    if (!vol.module) return;
    vol.setInt(key, parseInt(e.target.value, 10));
    if (key === "layout") $("volWrap").classList.toggle("single", e.target.value === "1");
    if (key === "mode") $("volBrain").disabled = e.target.value === "1";
  }));
  const floatCtl = { volBrain: "brain", volClip: "clip", volWinLo: "winlo", volWinHi: "winhi", volTumourOpacity: "tumour" };
  Object.entries(floatCtl).forEach(([id, key]) => $(id).addEventListener("input", (e) => {
    if (vol.module) vol.setFloat(key, parseFloat(e.target.value));
  }));
  $("volPlanes").addEventListener("change", (e) => vol.module && vol.setInt("planes", e.target.checked ? 1 : 0));
  $("volReset").addEventListener("click", () => vol.module && vol.module._mv_reset_view());
}

// ---------------------------------------------------------------- environment / engine notes
// Explains which inference engine suits the server this page runs on (CPU vendor, AVX, GPU).
function renderEnvInfo() {
  const hw = state.hardware;
  if (!hw) return;
  const amd = hw.vendor === "AMD", intel = hw.vendor === "Intel";
  const simd = [hw.avx512 ? "AVX-512" : hw.avx2 ? "AVX2" : "", hw.vnni ? "VNNI" : ""].filter(Boolean).join(" + ") || "无 AVX2";
  $("envSummary").innerHTML = `运行环境：<b>${esc(hw.model)}</b> · ${hw.cores} 核 ${hw.threads} 线程 · ${esc(simd)} · ` +
    `<b>${hw.gpu ? "有 GPU" : "无 GPU，纯 CPU 推理"}</b><span class="more">推理引擎说明 ▸</span>`;

  const openvino = intel
    ? [`<span class="badge maybe">可扩展 · 有提升空间</span>`,
       `Intel 自家的推理引擎，专门针对 Intel CPU / 核显 / NPU 优化。本机是 Intel CPU，换用 OpenVINO 一般能比 ONNX Runtime 再快约 1.2–2 倍${hw.vnni ? "；CPU 支持 VNNI，做 INT8 量化后提升更明显" : ""}。`]
    : [`<span class="badge no">不适用 · 无提升空间</span>`,
       `Intel 自家的推理引擎，优化针对的是 Intel CPU / 核显 / NPU。本机是 <b>${esc(hw.vendor)}</b> CPU（${esc(hw.model)}），` +
       `OpenVINO 虽然也能运行，但速度和 ONNX Runtime 基本持平，所以这里不采用。`];
  const rows = [
    { key: "python", name: "Python · ONNX Runtime", status: `<span class="badge on">已启用</span>`,
      text: `计算全部在 ONNX Runtime 的 C++ 内核里完成，Python 只负责前后处理和 HTTP（几毫秒）。它会做计算图优化（合并卷积、BN 和激活层，预先算好常量），` +
            `并按 CPU 指令集自动选用 ${hw.avx512 ? "AVX-512" : "AVX2"} 版本的计算内核，所以是本机 CPU 上最快的一档。` },
    { key: "cpp", name: "C++ · Drogon + OpenCV DNN", status: `<span class="badge on">已启用</span>`,
      text: `同样的模型和前后处理，在纯 CPU 上推理时间约为上一行的 2–3 倍。差距来自推理引擎而不是编程语言：OpenCV DNN 默认的 CPU 实现比较通用，能合并的层少，计算内核也没调得那么深。` +
            `它的优势是轻量、依赖少，并且可以切换到 CUDA 后端。` },
    { key: "cpp-ort", name: "C++ · ONNX Runtime", status: `<span class="badge maybe">可扩展 · 推荐</span>`,
      text: `保留 Drogon 和前后处理代码，只把推理那十几行换成 ONNX Runtime 的 C++ 接口（同一个 .onnx 模型文件，镜像约增加 30 MB）。` +
            `推理时间可以降到和 Python 一样，再加上省掉了 Python 自身的开销，C++ 整体有望反超 Python。<b>这是在本机提升 C++ 推理速度的方向。</b>` },
    { key: "openvino", name: "OpenVINO", status: openvino[0], text: openvino[1] },
    { key: "gpu", name: "GPU · CUDA / TensorRT", status: hw.gpu ? `<span class="badge on">可用</span>` : `<span class="badge no">不可用</span>`,
      text: hw.gpu ? `检测到 GPU。ONNX Runtime 会优先使用 TensorRT / CUDA，推理速度通常比 CPU 快一个数量级。`
                   : `本机没有 GPU。换到有 NVIDIA GPU 的机器后，ONNX Runtime（CUDA / TensorRT）和 OpenCV DNN（CUDA 后端）都可以直接切换，推理速度通常比 CPU 快一个数量级。` },
  ];
  $("envBody").innerHTML = `<div class="env-body">
    <p>页面右上角可以切换推理后端。三种引擎用的是同一个 .onnx 模型文件和同样的前后处理，差别只在执行推理的引擎。下表是各引擎在这台服务器上的适用情况，高亮的一行是当前选择的后端。</p>
    <table class="engine-table">
      <tr><th>推理引擎</th><th>本机状态</th><th>说明</th></tr>
      ${rows.map((r) => `<tr class="${r.key === state.backend ? "current" : ""}"><td>${esc(r.name)}</td><td>${r.status}</td><td>${r.text}</td></tr>`).join("")}
    </table>
    <p>ONNX Runtime ${esc(hw.ort_version)} · CPU 厂商 ${esc(hw.vendor)}${amd ? "（AMD 平台上 ONNX Runtime 已是 CPU 推理的最佳选择）" : ""}</p>
  </div>`;
  $("envInfo").hidden = false;
}

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
    const needed = [...new Set(MODALITIES.map((m) => m.model).filter(Boolean))]; // video menus share their image menu's model
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
    const missing = m.model && !state.backendError && !modelInfo(m) ? `<span class="badge">未安装</span>` : "";
    const newGroup = i > 0 && (m.mode || "") !== (MODALITIES[i - 1].mode || "");
    return `${newGroup ? `<span class="menu-sep" aria-hidden="true"></span>` : ""}<button class="menu-item${m === state.mod ? " active" : ""}" data-key="${m.key}">
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
  if (mod.mode === "volume") {
    renderVolumeCard();
    return;
  }
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
  const isVideo = mod.mode === "video", isVolume = mod.mode === "volume";
  document.querySelector('.tab[data-tab="image"]').hidden = isVideo || isVolume;
  document.querySelector('.tab[data-tab="video"]').hidden = !isVideo;
  document.querySelector('.tab[data-tab="classic"]').hidden = !mod.classic;
  document.querySelector('.tab[data-tab="volume"]').hidden = !isVolume;
  $("subTabs").hidden = [...document.querySelectorAll(".tab")].filter((t) => !t.hidden).length < 2;
  showTab(isVideo ? "video" : isVolume ? "volume" : "image");
  volumeActive(isVolume);
  if (isVolume) {
    resetImage();  // no image detection in this menu (e.g. when the backend selector changes)
    renderMenu();
    renderModelCard();
    return;
  }

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

  bindVolumeControls();

  $("backendSelect").addEventListener("change", (e) => {
    state.backend = e.target.value;
    renderEnvInfo(); // highlight the selected engine
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
    state.hardware = h.hardware || null;
    renderEnvInfo();
  } catch { /* ignore */ }
  await Promise.all([loadManifest(), refreshBackend()]);
  selectModality(location.hash.slice(1));
}

init();
