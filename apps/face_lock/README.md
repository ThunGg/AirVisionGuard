# Face Lock

A real-time, face-recognition-based screen locker for Windows. The application continuously monitors your webcam and automatically locks the screen when the authorized user's face is no longer detected.

## How It Works

1. **Detection** — `YuNetDetector` (default) or `MTCNNDetector` (fallback) detects faces and 5-point facial landmarks in each webcam frame. YuNet is faster on CPU; MTCNN supports GPU acceleration via CUDA.
2. **Alignment & Preprocessing** — The detected face is geometrically aligned to ArcFace's canonical 112×112 landmark template using a similarity transform.
3. **Recognition** — The aligned face is passed through an ONNX face recognition model (`FaceRecognizer`) that produces a normalized 512-d embedding vector.
4. **Matching** — Cosine similarity is computed between the live embedding and all registered embeddings. A match is declared when the best similarity exceeds the threshold (default `0.6`).
5. **Lock / Unlock** — `ScreenLocker` (Tkinter fullscreen overlay) locks the screen when the authorized face has been missing for 3 consecutive frames and unlocks it the moment the face is recognized again.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.8+ | Tested on 3.10 |
| Webcam | Any USB or built-in camera (OpenCV device index `0`) |
| ONNX face recognition model | Export from the parent framework; see [Model Setup](#model-setup) |
| CUDA (optional) | Enables GPU-accelerated MTCNN detection |

---

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Replace `onnxruntime` with `onnxruntime-gpu` and install the matching CUDA-enabled PyTorch build from https://pytorch.org before running the command above.

---

## Model Setup

The app requires an ONNX face recognition model and a YuNet face detector model. By default it looks for:

```
Face recognition model: ../../models/face_rec.onnx
YuNet detector model: models/yunet/face_detection_yunet_2023mar.onnx
```

(i.e., `<repo_root>/models/face_rec.onnx` for the recognition model, and `./models/yunet/` relative to the app directory for the detector).

The YuNet model is automatically downloaded on first run. The MTCNN weights are cached inside `models/mtcnn/` when MTCNN detector is used.

---

## Usage

```bash
python main.py [--model PATH_TO_MODEL] [--detector {yunet|mtcnn}] [--detector-model PATH_TO_DETECTOR]
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `../../models/face_rec.onnx` | Path to the ONNX face recognition model |
| `--detector` | `yunet` | Face detector backend: `yunet` (faster on CPU) or `mtcnn` (GPU-accelerated) |
| `--detector-model` | `models/yunet/face_detection_yunet_2023mar.onnx` | Path to the YuNet ONNX detector model file |

### Example

```bash
# Default model path and YuNet detector (fastest on CPU)
python main.py

# Use MTCNN detector (requires GPU for best performance)
python main.py --detector mtcnn

# Custom model path
python main.py --model C:/models/my_face_rec.onnx

# Custom detector model
python main.py --detector yunet --detector-model C:/models/yunet.onnx
```

---

## Interactive Controls

Once the preview window opens, use the following keyboard shortcuts:

| Key | Action |
|---|---|
| `r` | **Register** the current face. Must be pressed 3 times (once per distance step). A face must be visible in the frame. |
| `t` | **Toggle** the face lock ON or OFF (available only after all 3 steps are registered). |
| `q` | **Quit** the application. |

---

## Face Registration Workflow

Registration is required before the lock activates. The app captures embeddings at **three distances** to improve robustness across varying positions:

```
Step 1 — Close up     →  press 'r'
Step 2 — Medium distance →  press 'r'
Step 3 — Far distance    →  press 'r'
```

The on-screen overlay will prompt you for each step. Once all three embeddings are saved, face monitoring begins automatically.

> **Tip:** Keep your face well-lit and centered in the frame when registering. Poor lighting during registration is the most common cause of false negatives.

---

## Lock Behaviour

| Situation | Result |
|---|---|
| Authorized face detected (similarity > 0.6) | Screen stays unlocked / unlocks immediately |
| Authorized face absent for **3 consecutive frames** | Screen locks |
| Face lock toggled OFF via `t` | Screen unlocks; locking is suspended |
| Face lock toggled back ON via `t` | Monitoring resumes |

When locked, the screen overlay:
- Takes a blurred screenshot as the background.
- Displays **"Screen Locked — Waiting for authorized face..."**
- Blocks `Esc`, `Alt+F4`, and `Alt+Tab` via Tkinter event interception.
- Continuously re-lifts itself to the foreground to resist focus loss.

> **Note:** `Ctrl+Alt+Del` (Windows Security screen) is **not** blocked by this application, as it is handled at the OS kernel level.

---

## Project Structure

```
apps/face_lock/
├── main.py           # Entry point; orchestrates detection, recognition, and lock logic
├── detector.py       # YuNetDetector (default) and MTCNNDetector — face & landmark detection
├── recognizer.py     # FaceRecognizer — ONNX inference + cosine similarity
├── lock_screen.py    # ScreenLocker — Tkinter fullscreen lock overlay
├── utils.py          # align_face(), preprocess_face() — ArcFace preprocessing helpers
├── requirements.txt  # Python dependencies
└── models/
    ├── yunet/        # YuNet ONNX model (auto-downloaded on first run)
    └── mtcnn/        # Auto-downloaded MTCNN weights (on first run)
```

---

## Tuning

Key constants in `main.py` that you can adjust:

| Constant | Default | Effect |
|---|---|---|
| `LOCK_THRESHOLD` | `3` | Frames without a match before locking. Lower = faster lock. |
| `UNLOCK_THRESHOLD` | `0.6` | Cosine similarity required to count as a match. Higher = stricter. |
| `registration_steps` | 3 distances | Add or remove steps to change how many embeddings are registered. |
| `DETECT_EVERY_N` | `3` | Run face detection every Nth frame to reduce CPU load. |
| `SKIP_AFTER_AUTH` | `15` | Skip recognition for N frames after successful authorization. |

The MTCNN detector parameters (min face size, confidence thresholds, scale factor) can be adjusted in `MTCNNDetector.__init__()` inside `detector.py`.

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `Error: Face recognition model not found` | Wrong/missing ONNX model | Supply the correct path via `--model` |
| `Error: Could not open camera` | Camera in use or wrong device index | Close other apps using the camera; change `cv2.VideoCapture(0)` index if needed |
| Face not detected during registration | Poor lighting or face too small | Improve lighting; move closer to the camera |
| Frequent false locks | Similarity threshold too strict or varied lighting | Lower `UNLOCK_THRESHOLD` or re-register under consistent lighting |
| YuNet model download failed | No internet connection | Manually download from OpenCV Zoo and place at `models/yunet/face_detection_yunet_2023mar.onnx` |
| High CPU usage | Using MTCNN on CPU (instead of YuNet) | Use default `--detector yunet` for better CPU performance |
