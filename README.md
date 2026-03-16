# ◈ VisionBench-AI

> **Multi-Model Real-Time Face Detection System**

![Python](https://img.shields.io/badge/Python-3.10+-10b981?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-3b82f6?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-a855f7?style=flat-square&logo=react&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-f97316?style=flat-square&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-888888?style=flat-square)

---

## Demo

https://github.com/YOUR-USERNAME/VisionBench-AI/assets/XXXXX/your-video.mp4

> Switch between Haar Cascade · MediaPipe · SSD-ResNet · Dlib HOG in real time.
> Each model runs live with color-coded bounding boxes and latency tracking.

---

## Overview

VisionBench-AI is a full-stack computer vision application that benchmarks **four face detection algorithms** in real time — side by side in one unified interface.

Built to compare classical CV techniques against modern deep learning approaches, it supports:
- **Live webcam detection** with auto-running inference
- **Static image analysis** with drag & drop support
- **Automatic re-detection** on model switch
- **Latency profiling** with a live histogram dashboard
- **Export** of annotated frames as PNG

This project demonstrates end-to-end ML system design: a FastAPI backend serving OpenCV / MediaPipe / Dlib models, a React frontend with live video processing, canvas-based bounding box overlay, and a real-time performance dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VISIONBENCH-AI                        │
│                                                          │
│  ┌────────────────┐   POST /detect   ┌────────────────┐ │
│  │   React SPA    │ ───────────────▶ │ FastAPI Backend│ │
│  │                │                  │                │ │
│  │  Webcam feed   │                  │  ┌──────────┐  │ │
│  │  Canvas overlay│ ◀─────────────── │  │   Haar   │  │ │
│  │  Model picker  │   JSON response  │  │MediaPipe │  │ │
│  │  Latency chart │                  │  │SSD-ResNet│  │ │
│  │  Export PNG    │   GET /health    │  │Dlib HOG  │  │ │
│  └────────────────┘ ───────────────▶ │  └──────────┘  │ │
│   localhost:3000                      └────────────────┘ │
│                                        localhost:8000     │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
VisionBench-AI/
├── backend/
│   ├── models/
│   │   ├── deploy.prototxt                        # SSD-ResNet architecture
│   │   └── res10_300x300_ssd_iter_140000.caffemodel  # Pretrained weights
│   ├── main.py                                    # FastAPI server + all 4 CV models
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── App.js                                 # Full React UI
│       ├── index.js                               # React entry point
│       └── index.css                              # Base reset styles
│
├── research_scripts/
│   ├── face_recognition.py                        # Experimental scripts
│   └── videoRead.py
│
└── README.md
```

---

## Files Reference

| File | Status | Purpose |
|------|--------|---------|
| `backend/main.py` | ✅ Core | FastAPI server — all 4 detection models, NMS, health endpoint |
| `backend/requirements.txt` | ✅ Required | Python dependencies |
| `models/deploy.prototxt` | ✅ Required | SSD-ResNet network architecture definition |
| `models/*.caffemodel` | ✅ Required | SSD-ResNet pretrained weights (~10MB) |
| `frontend/src/App.js` | ✅ Core | Full React UI — webcam, upload, overlay, export, charts |
| `frontend/src/index.js` | ✅ Keep | React entry point |
| `frontend/src/index.css` | ✅ Keep | Base body/font reset styles |
| `frontend/src/App.css` | ❌ Unused | Default CRA boilerplate — not used, safe to delete |
| `frontend/src/logo.svg` | ❌ Unused | Default CRA logo — not imported in App.js, safe to delete |
| `frontend/src/App.test.js` | ⚠️ Stale | Default CRA test — rewrite or delete |
| `research_scripts/` | 📁 Optional | Experimental scripts — not part of production app |

---

## Detection Models

| Model | Type | Year | Speed | Precision | Notes |
|-------|------|------|-------|-----------|-------|
| **Haar Cascade** | Classical CV | 2001 | ⚡⚡⚡ | Medium | Viola-Jones algorithm. Best with frontal, well-lit faces |
| **MediaPipe** | Lightweight ML | 2019 | ⚡⚡⚡ | High | Google's pipeline. Excellent for varied angles & groups |
| **SSD-ResNet10** | Deep Learning | 2016 | ⚡⚡ | Very High | Best accuracy. Handles occlusion and tough scenes |
| **Dlib HOG+SVM** | Classical ML | 2009 | ⚡⚡ | High | Robust baseline with very few false positives |

All models apply **Non-Maximum Suppression (NMS)** post-processing to eliminate duplicate bounding boxes. Bounding box coordinates are returned as percentages (0–100) for resolution-independence.

---

## Setup & Installation

### 1. Backend (The Brain)
The backend handles the AI inference logic and coordinate calculation.

**Prerequisites:**
- MacBook M-series users should run: `brew install cmake` (required for Dlib).

**Installation:**
```bash
# Navigate to backend folder
cd backend

# Create and activate the environment
conda create -n vision_env python=3.10 -y
conda activate vision_env

# Install all required AI packages
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm start

# App runs at http://localhost:3000
```

### 3. SSD Model Files

Place the following files in `backend/models/`:
- `deploy.prototxt` — network architecture (included in repo)
- `res10_300x300_ssd_iter_140000.caffemodel` — pretrained weights

```bash
# Download the caffemodel:
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel \
     -P backend/models/
```

---

## API Reference

### `POST /detect`

Runs face detection on a base64-encoded image using the specified model.

**Request:**
```json
{
  "image": "<base64 data URL>",
  "model": "haar" | "mediapipe" | "ssd" | "dlib"
}
```

**Response:**
```json
{
  "faces": [
    { "x": 12.5, "y": 8.3, "w": 22.1, "h": 29.8 }
  ],
  "count": 1,
  "latency": 14.2,
  "model": "haar"
}
```

> Coordinates are percentages (0–100) relative to image dimensions.

---

### `GET /health`

Returns the load status of all four models on the backend.

**Response:**
```json
{
  "status": "ok",
  "models": {
    "haar": true,
    "mediapipe": true,
    "ssd": false,
    "dlib": true
  }
}
```

---

## Features

- 📷 **Live webcam detection** — auto-runs at ~2 FPS, no button needed
- 🖼 **Static image upload** — Support for PNG, JPG, and iPhone HEIC formats
- 🔄 **Auto re-detection on model switch** — works in both webcam and upload modes
- 🎨 **Canvas bounding box overlay** — per-model color coding with face numbering (#1, #2…)
- 📊 **Latency histogram** — live bar chart of last 20 readings (min / avg / max), color-coded
- 💾 **Export annotated PNG** — saves the current frame with boxes and model watermark
- 🟢 **Backend health panel** — live `/health` poll every 10s showing model availability
- 🔢 **Session statistics** — scan count, total faces detected, average latency
- 📋 **Detection log** — last 30 detections with model, face count, latency, and timestamp
- 🎞 **FPS counter** — live frames-per-second display during webcam detection

---

## Requirements

### Python (`requirements.txt`)

```
fastapi
uvicorn
opencv-python-headless
mediapipe
numpy
python-multipart
# dlib — install separately
```

### Node

```
Node.js 16+
React 18
axios
react-webcam
heic2any (for HEIC support)
```

---

## Files You Can Safely Delete

These are default Create React App boilerplate files not used by this project:

- `frontend/src/App.css` — replaced entirely by inline styles in `App.js`
- `frontend/src/logo.svg` — not imported anywhere
- `frontend/src/App.test.js` — default CRA placeholder test

The `research_scripts/` folder contains experimental Python scripts and is not part of the production application. Keep locally for reference or remove before deployment.

---

## License

MIT — free to use, modify, and distribute.

---

*VisionBench-AI · Full-Stack Computer Vision Benchmark Project*