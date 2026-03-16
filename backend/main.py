import cv2
import numpy as np
import base64
import time
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_MODELS = {"haar", "mediapipe", "ssd", "dlib"}

# --- 1. HAAR CASCADE ---
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if haar_cascade.empty():
    print("⚠️  Haar Cascade failed to load")
else:
    print("✅ Haar Cascade Loaded")

# --- 2. MEDIAPIPE ---
mp_enabled = False
try:
    import mediapipe.python.solutions.face_detection as mp_face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    mp_enabled = True
    print("✅ MediaPipe Loaded")
except Exception as e:
    print(f"⚠️  MediaPipe Disabled: {e}")

# --- 3. OPENCV DNN (SSD-RESNET) ---
ssd_enabled = False
try:
    net = cv2.dnn.readNetFromCaffe(
        "models/deploy.prototxt",
        "models/res10_300x300_ssd_iter_140000.caffemodel"
    )
    ssd_enabled = True
    print("✅ SSD-ResNet Loaded")
except Exception as e:
    print(f"⚠️  SSD-ResNet Disabled: {e}")

# --- 4. DLIB ---
dlib_enabled = False
try:
    import dlib
    dlib_detector = dlib.get_frontal_face_detector()
    dlib_enabled = True
    print("✅ Dlib Loaded")
except Exception as e:
    print(f"⚠️  Dlib Disabled: {e}")


def decode_image(b64_string: str):
    """Decode a base64 image string to an OpenCV BGR image."""
    try:
        encoded_data = b64_string.split(',')[1] if "," in b64_string else b64_string
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"⚠️  Image decode failed: {e}")
        return None


def clamp_box(x, y, w, h, img_w, img_h):
    """Clamp bounding box to stay within image boundaries."""
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return x, y, w, h


def boxes_to_percent(boxes, img_w, img_h):
    """Convert pixel [x, y, w, h] boxes to percentage-based coords."""
    return [
        {
            "x": (x / img_w) * 100,
            "y": (y / img_h) * 100,
            "w": (w / img_w) * 100,
            "h": (h / img_h) * 100,
        }
        for (x, y, w, h) in boxes
    ]


def apply_nms(raw_boxes, confidences, score_threshold=0.3, nms_threshold=0.4):
    """
    Apply Non-Maximum Suppression and return surviving boxes.
    Uses a lower score_threshold (0.3) so that Haar (confidence=1.0)
    and MediaPipe/SSD boxes all pass through cleanly.
    """
    if not raw_boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        raw_boxes, confidences,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold
    )

    if len(indices) == 0:
        return []

    return [raw_boxes[i] for i in indices.flatten()]


@app.post("/detect")
async def detect_faces(data: dict = Body(...)):
    image_b64 = data.get("image")
    model_type = data.get("model", "").lower()

    # FIX 7: validate model name upfront with a clear error
    if model_type not in VALID_MODELS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown model '{model_type}'. Choose from: {sorted(VALID_MODELS)}"}
        )

    img = decode_image(image_b64)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Could not decode image."})

    h_img, w_img = img.shape[:2]

    # FIX 1: start timer right after decode — captures all processing time
    start_time = time.time()

    raw_boxes = []
    confidences = []

    # --- HAAR CASCADE ---
    if model_type == "haar":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Equalise histogram for better detection in varied lighting
        gray = cv2.equalizeHist(gray)
        detected = haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)  # ignore tiny false-positive boxes
        )
        for (x, y, w, h) in detected:
            x, y, w, h = clamp_box(x, y, w, h, w_img, h_img)
            raw_boxes.append([x, y, w, h])
            confidences.append(1.0)

        # FIX 2: Haar needs a lower score_threshold since all confs = 1.0
        # NMS here is purely spatial (overlap-based), not confidence-based
        final_boxes = apply_nms(raw_boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # --- MEDIAPIPE ---
    elif model_type == "mediapipe":
        if not mp_enabled:
            return JSONResponse(status_code=503, content={"error": "MediaPipe is not available on this server."})

        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            for d in results.detections:
                b = d.location_data.relative_bounding_box
                x = int(b.xmin * w_img)
                y = int(b.ymin * h_img)
                w = int(b.width * w_img)
                h = int(b.height * h_img)
                # FIX 5: clamp — MediaPipe can return negative coords
                x, y, w, h = clamp_box(x, y, w, h, w_img, h_img)
                raw_boxes.append([x, y, w, h])
                confidences.append(float(d.score[0]))

        # FIX 6: use 0.3 threshold so all MP detections that passed model filter survive NMS
        final_boxes = apply_nms(raw_boxes, confidences, score_threshold=0.3, nms_threshold=0.4)

    # --- SSD-RESNET ---
    elif model_type == "ssd":
        if not ssd_enabled:
            return JSONResponse(status_code=503, content={"error": "SSD-ResNet model files not found."})

        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
                x1, y1, x2, y2 = box.astype("int")
                x, y, w, h = x1, y1, x2 - x1, y2 - y1
                x, y, w, h = clamp_box(x, y, w, h, w_img, h_img)
                raw_boxes.append([x, y, w, h])
                confidences.append(confidence)

        final_boxes = apply_nms(raw_boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # --- DLIB HOG+SVM ---
    elif model_type == "dlib":
        if not dlib_enabled:
            return JSONResponse(status_code=503, content={"error": "Dlib is not available on this server."})

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = dlib_detector(gray, 1)
        for r in rects:
            x = r.left()
            y = r.top()
            w = r.right() - r.left()
            h = r.bottom() - r.top()
            x, y, w, h = clamp_box(x, y, w, h, w_img, h_img)
            raw_boxes.append([x, y, w, h])
            confidences.append(1.0)

        final_boxes = apply_nms(raw_boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    latency = round((time.time() - start_time) * 1000, 2)

    return {
        "faces": boxes_to_percent(final_boxes, w_img, h_img),
        "count": len(final_boxes),
        "latency": latency,
        "model": model_type,
    }


# Optional: health check endpoint
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "haar": not haar_cascade.empty(),
            "mediapipe": mp_enabled,
            "ssd": ssd_enabled,
            "dlib": dlib_enabled,
        }
    }
