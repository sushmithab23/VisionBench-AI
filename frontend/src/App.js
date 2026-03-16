import { useState, useRef, useEffect, useCallback } from "react";

const MODELS = [
  { id: "haar",      label: "Haar Cascade", desc: "Classic · Fast · CPU",       color: "#fb923c", year: "2001", type: "Classical CV" },
  { id: "mediapipe", label: "MediaPipe",    desc: "Google · Blazing · Mobile",   color: "#60a5fa", year: "2019", type: "Lightweight ML" },
  { id: "ssd",       label: "SSD-ResNet",   desc: "Deep · Accurate · DNN",       color: "#c084fc", year: "2016", type: "Deep Learning" },
  { id: "dlib",      label: "Dlib HOG",     desc: "Histogram · Robust · Retro",  color: "#34d399", year: "2009", type: "Classical ML" },
];

const API_BASE   = "http://localhost:8000";
const API_DETECT = `${API_BASE}/detect`;
const API_HEALTH = `${API_BASE}/health`;

export default function App() {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const overlayRef  = useRef(null);
  const streamRef   = useRef(null);
  const intervalRef = useRef(null);
  const fpsRef      = useRef({ count: 0, last: Date.now() });

  const [model,          setModel]          = useState("haar");
  const [camOn,          setCamOn]          = useState(false);
  const [faces,          setFaces]          = useState([]);
  const [latency,        setLatency]        = useState(null);
  const [latencyHistory, setLatencyHistory] = useState([]);
  const [fps,            setFps]            = useState(0);
  const [history,        setHistory]        = useState([]);
  const [error,          setError]          = useState(null);
  const [uploadedImage,  setUploadedImage]  = useState(null);
  const [mode,           setMode]           = useState("webcam");
  const [loading,        setLoading]        = useState(false);
  const [health,         setHealth]         = useState(null);
  const [totalDetected,  setTotalDetected]  = useState(0);
  const [dragOver,       setDragOver]       = useState(false);

  // ── HEALTH POLL ────────────────────────────────────────────────────────────
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(API_HEALTH);
        if (r.ok) setHealth(await r.json());
      } catch { setHealth(null); }
    };
    poll();
    const id = setInterval(poll, 10000);
    return () => clearInterval(id);
  }, []);

  // ── WEBCAM ─────────────────────────────────────────────────────────────────
  const startCam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setCamOn(true);
          setError(null);
        };
      }
    } catch { setError("Camera access denied or unavailable."); }
  }, []);

  const stopCam = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    clearInterval(intervalRef.current);
    setCamOn(false);
    setFaces([]);
    setFps(0);
  }, []);

  useEffect(() => () => stopCam(), [stopCam]);

  // ── FRAME CAPTURE ──────────────────────────────────────────────────────────
  const captureFrame = useCallback(() => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return null;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.8);
  }, []);

  // ── DETECTION ──────────────────────────────────────────────────────────────
  const runDetection = useCallback(async (imageData, modelOverride) => {
    const activeModel = modelOverride || model;
    try {
      const res  = await fetch(API_DETECT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData, model: activeModel }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data     = await res.json();
      const faceList = data.faces || [];
      setFaces(faceList);
      setLatency(data.latency);
      setLatencyHistory(h => [...h.slice(-19), data.latency]);
      setTotalDetected(t => t + faceList.length);
      setHistory(h => [
        { model: activeModel, count: faceList.length, latency: data.latency, ts: Date.now() },
        ...h.slice(0, 29),
      ]);
      const now = Date.now();
      fpsRef.current.count++;
      if (now - fpsRef.current.last >= 1000) {
        setFps(fpsRef.current.count);
        fpsRef.current = { count: 0, last: now };
      }
    } catch (e) { setError(`API error: ${e.message}`); }
  }, [model]);

  // ── AUTO LOOP ──────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!camOn) return;
    clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      const frame = captureFrame();
      if (frame) runDetection(frame, model);
    }, 500);
    return () => clearInterval(intervalRef.current);
  }, [camOn, model]); // eslint-disable-line

  // ── FILE HANDLING (PNG · JPG · HEIC) ──────────────────────────────────────
  const processFile = useCallback(async (file) => {
    if (!file) return;

    const name = file.name?.toLowerCase() || "";
    const type = file.type?.toLowerCase() || "";
    const isHeic = type === "image/heic" || type === "image/heif"
      || name.endsWith(".heic") || name.endsWith(".heif");
    const isImage = type.startsWith("image/") || isHeic;
    if (!isImage) {
      setError("Unsupported file type. Please upload PNG, JPG, or HEIC.");
      return;
    }

    setLoading(true);
    setFaces([]);

    try {
      let blob = file;

      // Convert HEIC → JPEG in-browser before display & detection
      if (isHeic) {
        // Lazy-load heic2any only when needed
        const heic2any = (await import("https://cdn.jsdelivr.net/npm/heic2any@0.0.4/dist/heic2any.min.js")).default
          || window.heic2any;
        blob = await heic2any({ blob: file, toType: "image/jpeg", quality: 0.92 });
      }

      const reader = new FileReader();
      reader.onload = (ev) => {
        setUploadedImage(ev.target.result);
        runDetection(ev.target.result, model).finally(() => setLoading(false));
      };
      reader.onerror = () => { setError("Failed to read file."); setLoading(false); };
      reader.readAsDataURL(blob);
    } catch (e) {
      setError(`HEIC conversion failed: ${e.message}`);
      setLoading(false);
    }
  }, [model, runDetection]);

  const handleUpload = (e) => processFile(e.target.files[0]);
  const handleDrop   = (e) => { e.preventDefault(); setDragOver(false); processFile(e.dataTransfer.files[0]); };

  useEffect(() => {
    if (mode === "upload" && uploadedImage) {
      setLoading(true);
      runDetection(uploadedImage, model).finally(() => setLoading(false));
    }
  }, [model]); // eslint-disable-line

  // ── EXPORT ─────────────────────────────────────────────────────────────────
  const exportImage = useCallback(() => {
    const src = mode === "webcam" ? captureFrame() : uploadedImage;
    if (!src) return;
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.width; c.height = img.height;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0);
      const color = MODELS.find(m => m.id === model)?.color || "#fff";
      faces.forEach((face, idx) => {
        const x = (face.x / 100) * img.width, y = (face.y / 100) * img.height;
        const fw = (face.w / 100) * img.width, fh = (face.h / 100) * img.height;
        ctx.strokeStyle = color; ctx.lineWidth = 3;
        ctx.shadowColor = color; ctx.shadowBlur = 14;
        ctx.strokeRect(x, y, fw, fh);
        ctx.shadowBlur = 0;
        ctx.fillStyle = color;
        ctx.font = `bold ${Math.max(12, fw * 0.12)}px monospace`;
        ctx.fillText(`#${idx + 1}`, x + 4, y + Math.max(14, fw * 0.12) + 2);
      });
      ctx.fillStyle = "rgba(8,14,26,0.8)";
      ctx.fillRect(0, img.height - 30, img.width, 30);
      ctx.fillStyle = "#e2e8f0"; ctx.font = "11px monospace";
      ctx.fillText(`VisionBench-AI · ${model.toUpperCase()} · ${faces.length} face(s) · ${latency}ms`, 8, img.height - 10);
      const a = document.createElement("a");
      a.href = c.toDataURL("image/png");
      a.download = `visionbench_${model}_${Date.now()}.png`;
      a.click();
    };
    img.src = src;
  }, [faces, model, mode, uploadedImage, captureFrame, latency]);

  // ── CANVAS OVERLAY ─────────────────────────────────────────────────────────
  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext("2d");
    const w = overlay.width, h = overlay.height;
    ctx.clearRect(0, 0, w, h);
    const color = MODELS.find(m => m.id === model)?.color || "#fff";

    faces.forEach((face, idx) => {
      const x = (face.x / 100) * w, y = (face.y / 100) * h;
      const fw = (face.w / 100) * w, fh = (face.h / 100) * h;
      const r = 6;

      ctx.strokeStyle = color; ctx.lineWidth = 2.5;
      ctx.shadowColor = color; ctx.shadowBlur = 18;
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.lineTo(x + fw - r, y);
      ctx.quadraticCurveTo(x + fw, y, x + fw, y + r);
      ctx.lineTo(x + fw, y + fh - r);
      ctx.quadraticCurveTo(x + fw, y + fh, x + fw - r, y + fh);
      ctx.lineTo(x + r, y + fh);
      ctx.quadraticCurveTo(x, y + fh, x, y + fh - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath(); ctx.stroke();

      // Corner ticks
      ctx.shadowBlur = 0; ctx.lineWidth = 3;
      const cs = Math.min(fw, fh) * 0.18;
      [[x,y,1,1],[x+fw,y,-1,1],[x,y+fh,1,-1],[x+fw,y+fh,-1,-1]].forEach(([cx,cy,dx,dy]) => {
        ctx.beginPath();
        ctx.moveTo(cx + dx * cs, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy + dy * cs);
        ctx.stroke();
      });

      // Numbered badge
      const label = `#${idx + 1}`;
      const tw = ctx.measureText(label).width + 10;
      ctx.fillStyle = color;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, y - 20, tw, 18, 3);
      else ctx.rect(x, y - 20, tw, 18);
      ctx.fill();
      ctx.fillStyle = "#000";
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      ctx.fillText(label, x + 5, y - 7);
    });
  }, [faces, model]);

  const syncOverlay = (el) => {
    if (!overlayRef.current || !el) return;
    overlayRef.current.width  = el.offsetWidth;
    overlayRef.current.height = el.offsetHeight;
  };

  // ── DERIVED ────────────────────────────────────────────────────────────────
  const sel = MODELS.find(m => m.id === model);
  const avg = latencyHistory.length ? Math.round(latencyHistory.reduce((a, b) => a + b, 0) / latencyHistory.length) : null;
  const mx  = latencyHistory.length ? Math.max(...latencyHistory) : null;
  const mn  = latencyHistory.length ? Math.min(...latencyHistory) : null;
  const lc  = (v) => v <= 50 ? "#34d399" : v < 150 ? "#fb923c" : "#f87171";

  return (
    <div style={S.root}>
      <style>{CSS}</style>

      {/* ── HEADER ── */}
      <header style={S.header}>
        <div style={S.headerLeft}>
          <svg width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="2" y="2" width="30" height="30" rx="7" fill="#fb923c" fillOpacity="0.12" stroke="#fb923c" strokeWidth="1.5"/>
            <circle cx="17" cy="17" r="5.5" fill="none" stroke="#fb923c" strokeWidth="2"/>
            <line x1="17" y1="2.5" x2="17" y2="9.5" stroke="#fb923c" strokeWidth="1.8" strokeLinecap="round"/>
            <line x1="17" y1="24.5" x2="17" y2="31.5" stroke="#fb923c" strokeWidth="1.8" strokeLinecap="round"/>
            <line x1="2.5" y1="17" x2="9.5" y2="17" stroke="#fb923c" strokeWidth="1.8" strokeLinecap="round"/>
            <line x1="24.5" y1="17" x2="31.5" y2="17" stroke="#fb923c" strokeWidth="1.8" strokeLinecap="round"/>
            <circle cx="17" cy="17" r="2.5" fill="#fb923c"/>
          </svg>
          <div>
            <div style={S.title}>VISIONBENCH<span style={{ color: sel?.color }}>-AI</span></div>
            <div style={S.subtitle}>Multi-Model Face Detection · Real-Time CV Benchmark</div>
          </div>
        </div>

        <div style={S.statusBar}>
          {fps > 0 && <span style={{ ...S.badge, color: "#c8d8e8" }}>🎞 <b>{fps}</b> FPS</span>}
          {latency !== null && (
            <span style={{ ...S.badge, color: lc(latency), borderColor: lc(latency) + "44" }}>⚡ <b>{latency}ms</b></span>
          )}
          <span style={{ ...S.badge, color: camOn ? "#34d399" : "#7c8fa8", borderColor: camOn ? "#34d39944" : "#2d3f55", background: camOn ? "#34d39910" : "#0f172a" }}>
            {camOn ? "● LIVE" : "○ IDLE"}
          </span>
          <span style={{ ...S.badge, color: sel?.color, borderColor: sel?.color + "44", background: sel?.color + "10" }}>
            {sel?.label}
          </span>
          <span style={{ ...S.badge, color: health ? "#34d399" : "#f87171", borderColor: health ? "#34d39944" : "#f8717144" }}
            title={health ? "Backend connected" : "Backend offline"}>
            {health ? "◉" : "◎"} API
          </span>
        </div>
      </header>

      <div style={S.body}>
        {/* ── VIEWPORT COL ── */}
        <div style={S.viewportCol}>

          <div style={S.modeToggle}>
            {["webcam", "upload"].map(m => (
              <button key={m} onClick={() => { setMode(m); setFaces([]); }}
                style={{ ...S.modeBtn, ...(mode === m ? { borderColor: sel?.color, color: sel?.color, background: sel?.color + "14" } : {}) }}>
                {m === "webcam" ? "📷  Webcam" : "🖼  Upload"}
              </button>
            ))}
          </div>

          <div style={{ ...S.viewport, borderColor: faces.length > 0 ? sel?.color + "77" : "#1e293b" }}
            onDragOver={e => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}>

            {dragOver && (
              <div style={S.dropOverlay}>
                <span style={{ fontSize: 44 }}>⬇</span>
                <span style={{ marginTop: 12, fontSize: 14, color: "#e2e8f0", fontWeight: 700, letterSpacing: 1 }}>Drop image to detect</span>
              </div>
            )}

            {mode === "webcam" ? (
              <>
                <video ref={videoRef} autoPlay playsInline muted
                  style={{ ...S.video, display: camOn ? "block" : "none" }}
                  onLoadedMetadata={e => syncOverlay(e.target)} />
                {!camOn && (
                  <div style={S.placeholder}>
                    <span style={{ fontSize: 54 }}>📷</span>
                    <p style={{ color: "#6b7f96", marginTop: 14, fontSize: 18, letterSpacing: 1 }}>Start camera — detection runs automatically</p>
                  </div>
                )}
              </>
            ) : (
              <>
                {uploadedImage
                  ? <img src={uploadedImage} alt="uploaded" style={S.video} onLoad={e => syncOverlay(e.target)} />
                  : (
                    <div style={S.placeholder}>
                      <span style={{ fontSize: 54 }}>🖼</span>
                      <p style={{ color: "#6b7f96", marginTop: 14, fontSize: 18, letterSpacing: 1 }}>Drop an image or click Choose below</p>
                      <p style={{ color: "#4a5f75", marginTop: 6, fontSize: 11 }}>JPG · PNG · HEIC (iPhone)</p>
                    </div>
                  )}
              </>
            )}

            <canvas ref={overlayRef} style={S.overlay} />

            {faces.length > 0 && (
              <div style={{ ...S.badge, position: "absolute", bottom: 12, right: 12, background: sel?.color, color: "#000", borderColor: "transparent", fontWeight: 900, fontSize: 11 }}>
                {faces.length} face{faces.length !== 1 ? "s" : ""} detected
              </div>
            )}
            {loading && (
              <div style={S.loadOverlay}>
                <span style={S.spinner} />
                <span style={{ marginTop: 12, color: "#94a3b8", fontSize: 12, letterSpacing: 1 }}>Analysing…</span>
              </div>
            )}
          </div>

          <canvas ref={canvasRef} style={{ display: "none" }} />

          <div style={S.controls}>
            {mode === "webcam" ? (
              <>
                <button onClick={camOn ? stopCam : startCam}
                  style={{ ...S.btn, ...(camOn
                    ? { color: "#f87171", borderColor: "#f8717144", background: "#f8717110" }
                    : { color: "#34d399", borderColor: "#34d39944", background: "#34d39910" }) }}>
                  {camOn ? "⏹  Stop Camera" : "▶  Start Camera"}
                </button>
                <button onClick={exportImage} disabled={!camOn}
                  style={{ ...S.btn, opacity: camOn ? 1 : 0.3 }}>
                  💾  Export Frame
                </button>
              </>
            ) : (
              <>
                <label style={{ ...S.btn, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", gap: 6 }}>
                  📂  Choose Image
                  <input type="file" accept="image/*,.heic,.heif" onChange={handleUpload} style={{ display: "none" }} />
                </label>
                <button onClick={exportImage} disabled={!uploadedImage}
                  style={{ ...S.btn, opacity: uploadedImage ? 1 : 0.3, color: sel?.color, borderColor: sel?.color + "44", background: sel?.color + "10" }}>
                  💾  Export Annotated
                </button>
              </>
            )}
          </div>

          {error && (
            <div style={S.errorBox} onClick={() => setError(null)}>
              ⚠ {error} <span style={{ float: "right", opacity: 0.6 }}>✕</span>
            </div>
          )}

          {/* Latency chart */}
          {latencyHistory.length > 1 && (
            <div style={S.card}>
              <div style={S.sectionLabel}>LATENCY HISTORY (ms)</div>
              <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 48, marginTop: 10 }}>
                {latencyHistory.map((v, i) => (
                  <div key={i} title={`${v}ms`} style={{
                    flex: 1, height: `${Math.max(8, (v / mx) * 100)}%`,
                    background: lc(v), borderRadius: 3,
                    opacity: 0.45 + (i / latencyHistory.length) * 0.55,
                    transition: "height 0.25s ease",
                  }} />
                ))}
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 7, fontSize: 10 }}>
                <span style={{ color: "#34d399" }}>min {mn}ms</span>
                <span style={{ color: "#8ba3bc" }}>avg {avg}ms</span>
                <span style={{ color: "#f87171" }}>max {mx}ms</span>
              </div>
            </div>
          )}

          {/* Stats row */}
          <div style={S.statsRow}>
            {[
              { val: history.length,                        label: "SCANS",       col: "#94a3b8" },
              { val: faces.length,                          label: "CURRENT",     col: sel?.color },
              { val: totalDetected,                         label: "TOTAL FACES", col: "#94a3b8" },
              { val: avg ? `${avg}ms` : "—",                label: "AVG LATENCY", col: avg ? lc(avg) : "#475569" },
            ].map(({ val, label, col }) => (
              <div key={label} style={S.statBox}>
                <div style={{ fontSize: 22, fontWeight: 900, color: col, letterSpacing: -1 }}>{val}</div>
                <div style={{ fontSize: 13, color: "#6b7f96", letterSpacing: 1.5, marginTop: 5 }}>{label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── RIGHT PANEL ── */}
        <div style={S.rightPanel}>

          {/* Model selector */}
          <div style={S.section}>
            <div style={S.sectionLabel}>SELECT MODEL</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
              {MODELS.map(m => {
                const active  = model === m.id;
                const isReady = health?.models?.[m.id] !== false;
                return (
                  <button key={m.id} onClick={() => setModel(m.id)}
                    style={{ display: "flex", alignItems: "center", gap: 11, padding: "11px 13px",
                      border: `1px solid ${active ? m.color : "transparent"}`,
                      borderRadius: 8, cursor: "pointer", textAlign: "left",
                      transition: "all 0.18s", 
                      background: active ? m.color + "14" : "transparent",
                      boxShadow: active ? `0 0 20px ${m.color}22` : "none",
                      opacity: isReady ? 1 : 1, fontFamily: "'Rajdhani', sans-serif" }}>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: m.color, flexShrink: 0, boxShadow: active ? `0 0 8px ${m.color}` : "none" }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ color: active ? m.color : "#cbd5e1", fontWeight: 700, fontSize: 13 }}>{m.label}</div>
                      <div style={{ color: "#6b7f96", fontSize: 14, marginTop: 2 }}>{m.desc}</div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: 10, color: "#4a5f75" }}>{m.year}</div>
                      {!isReady && <div style={{ fontSize: 8, color: "#f87171", marginTop: 2, letterSpacing: 1 }}>OFFLINE</div>}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Backend health */}
          {health && (
            <div style={S.section}>
              <div style={S.sectionLabel}>BACKEND STATUS</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                {Object.entries(health.models).map(([k, v]) => (
                  <div key={k} style={{ display: "flex", alignItems: "center", padding: "6px 10px",
                    background: v ? "#34d39908" : "#f8717108",
                    border: `1px solid ${v ? "#34d39933" : "#f8717133"}`, borderRadius: 5 }}>
                    <span style={{ color: v ? "#34d399" : "#f87171", fontSize: 12 }}>{v ? "●" : "○"}</span>
                    <span style={{ color: v ? "#a7f3d0" : "#fca5a5", fontSize: 11, marginLeft: 7, fontWeight: 600 }}>{k}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Model info */}
          <div style={S.section}>
            <div style={S.sectionLabel}>ABOUT THIS MODEL</div>
            <div style={S.card}>
              {[
                { id: "haar",      color: "#fb923c", tag: "Classical CV",   body: "Haar Cascade uses the Viola-Jones algorithm (2001). Extremely fast on CPU using a sliding window with boosted feature classifiers.", perf: "Precision: Medium · Speed: ⚡⚡⚡ · GPU: No", tip: "Best with well-lit, forward-facing subjects." },
                { id: "mediapipe", color: "#60a5fa", tag: "Lightweight ML",  body: "MediaPipe is Google's lightweight ML pipeline. Uses a custom short/full-range model tuned for near real-time inference on mobile hardware.", perf: "Precision: High · Speed: ⚡⚡⚡ · GPU: No",     tip: "Best for group photos and varied angles." },
                { id: "ssd",       color: "#c084fc", tag: "Deep Learning",   body: "SSD-ResNet is a Single Shot MultiBox Detector backed by ResNet-10. Handles occlusion, varying scales, and challenging scenes well.", perf: "Precision: Very High · Speed: ⚡⚡ · GPU: Optional", tip: "Best accuracy for challenging scenes." },
                { id: "dlib",      color: "#34d399", tag: "Classical ML",    body: "Dlib HOG+SVM uses Histogram of Oriented Gradients with a trained Support Vector Machine classifier. Very robust with few false positives.", perf: "Precision: High · Speed: ⚡⚡ · GPU: No",     tip: "Great all-round baseline for real-world images." },
              ].filter(x => x.id === model).map(x => (
                <div key={x.id}>
                  <span style={{ display: "inline-block", padding: "3px 10px", borderRadius: 4, fontSize: 9, fontWeight: 700, letterSpacing: 1, color: x.color, background: x.color + "18", border: `1px solid ${x.color}33` }}>{x.tag}</span>
                  <p style={{ marginTop: 10, color: "#c8d8e8", lineHeight: 1.75, fontSize: 12 }}>
                    <b style={{ color: x.color }}>{MODELS.find(m => m.id === model)?.label}</b> {x.body}
                  </p>
                  <p style={{ color: "#7c8fa8", marginTop: 8, fontSize: 11 }}>{x.perf}</p>
                  <p style={{ color: "#6b7f96", marginTop: 8, fontSize: 15 }}>💡 {x.tip}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Detection log */}
          <div style={S.section}>
            <div style={{ ...S.sectionLabel, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span>DETECTION LOG</span>
              {history.length > 0 && (
                <span onClick={() => setHistory([])}
                  style={{ cursor: "pointer", color: "#4a5f75", fontSize: 9, letterSpacing: 1, padding: "2px 7px", border: "1px solid #2d3f55", borderRadius: 3 }}>
                  CLEAR
                </span>
              )}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 5, maxHeight: 240, overflowY: "auto" }}>
              {history.length === 0 && (
                <div style={{ color: "#4a5f75", fontSize: 12, padding: "12px 0", textAlign: "center" }}>No detections yet.</div>
              )}
              {history.map((h, i) => {
                const m   = MODELS.find(x => x.id === h.model);
                const ago = Math.round((Date.now() - h.ts) / 1000);
                return (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", borderRadius: 5, background: "#0f172a", border: "1px solid #1e293b", opacity: Math.max(0.35, 1 - i * 0.03) }}>
                    <div style={{ width: 7, height: 7, borderRadius: "50%", background: m?.color, flexShrink: 0 }} />
                    <span style={{ color: m?.color, fontSize: 10, fontWeight: 700, minWidth: 76 }}>{m?.label}</span>
                    <span style={{ color: "#8ba3bc", fontSize: 11 }}>{h.count} face{h.count !== 1 ? "s" : ""}</span>
                    <span style={{ color: lc(h.latency), fontSize: 10, marginLeft: "auto", fontWeight: 700 }}>{h.latency}ms</span>
                    <span style={{ color: "#4a5f75", fontSize: 9, marginLeft: 8 }}>{ago}s</span>
                  </div>
                );
              })}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

// ── STYLES ───────────────────────────────────────────────────────────────────
const S = {
  root:       { minHeight: "100vh", background: "#080e1a", color: "#e2e8f0", fontFamily: "'Rajdhani', sans-serif", display: "flex", flexDirection: "column" },
  header:     { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "14px 28px", borderBottom: "1px solid #1e293b", background: "linear-gradient(90deg,#0a1628,#0d1f3c 60%,#0a1628)" },
  headerLeft: { display: "flex", alignItems: "center", gap: 14 },
  title:      { fontSize: 22, fontWeight: 700, letterSpacing: 5, color: "#f1f5f9" },
  subtitle:   { fontSize: 12, color: "#7c8fa8", letterSpacing: 2, marginTop: 3 },
  statusBar:  { display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" },
  badge:      { fontSize: 12, fontWeight: 600, letterSpacing: 1, padding: "4px 10px", borderRadius: 4, border: "1px solid #2d3f55", background: "#0f172a", color: "#8ba3bc", fontFamily: "'Rajdhani', sans-serif" },
  body:       { display: "flex", flex: 1 },
  viewportCol:{ flex: 1, padding: 22, display: "flex", flexDirection: "column", gap: 14, borderRight: "1px solid #1e293b" },
  modeToggle: { display: "flex", gap: 10 },
  modeBtn:    { flex: 1, padding: "10px 0", border: "1px solid #2d3f55", background: "#0f172a", color: "#7c8fa8", cursor: "pointer", borderRadius: 7, fontSize: 13, fontFamily: "'Rajdhani', sans-serif", fontWeight: 600, letterSpacing: 1, transition: "all 0.15s" },
  viewport:   { position: "relative", background: "linear-gradient(145deg,#0a1628,#0d1f3c)", borderRadius: 10, border: "1px solid", overflow: "hidden", minHeight: 320, display: "flex", alignItems: "center", justifyContent: "center", transition: "border-color 0.35s" },
  video:      { width: "100%", display: "block", borderRadius: 10 },
  overlay:    { position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none" },
  placeholder:{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 56 },
  loadOverlay:{ position: "absolute", inset: 0, background: "#080e1acc", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", backdropFilter: "blur(3px)" },
  dropOverlay:{ position: "absolute", inset: 0, background: "#080e1aee", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", zIndex: 10, border: "2px dashed #334155", borderRadius: 10 },
  controls:   { display: "flex", gap: 10 },
  btn:        { flex: 1, padding: "10px 0", border: "1px solid #1e293b", background: "#0f172a", color: "#94a3b8", cursor: "pointer", borderRadius: 7, fontSize: 13, fontFamily: "'Rajdhani', sans-serif", fontWeight: 700, letterSpacing: 1, transition: "all 0.15s", textAlign: "center" },
  errorBox:   { background: "#f8717110", border: "1px solid #f8717144", color: "#fca5a5", borderRadius: 6, padding: "10px 14px", fontSize: 12, cursor: "pointer" },
  card:       { background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, padding: "13px 14px" },
  statsRow:   { display: "flex", gap: 8 },
  statBox:    { flex: 1, background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, padding: "12px 6px", textAlign: "center" },
  rightPanel: { width: 300, padding: 22, display: "flex", flexDirection: "column", gap: 22, overflowY: "auto", background: "#080e1a" },
  section:    {},
  sectionLabel: { fontSize: 13, letterSpacing: 3, color: "#6b7f96", fontWeight: 900, marginBottom: 10, borderBottom: "1px solid #1e293b", paddingBottom: 6 },
  spinner:    { width: 28, height: 28, border: "3px solid #1e293b", borderTop: "3px solid #60a5fa", borderRadius: "50%", animation: "spin 0.7s linear infinite", display: "inline-block" },
};

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #080e1a;
    background-image:
      linear-gradient(rgba(96,165,250,0.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(96,165,250,0.035) 1px, transparent 1px);
    background-size: 40px 40px;
    background-attachment: fixed;
  }
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0f172a; }
  ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #334155; }
  button:hover:not(:disabled) { filter: brightness(1.15); transform: translateY(-1px); }
  button:active:not(:disabled) { transform: translateY(0px); }
  label:hover { filter: brightness(1.15); }
  @keyframes spin { to { transform: rotate(360deg); } }
`;