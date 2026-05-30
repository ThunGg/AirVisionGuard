import os
import numpy as np
import cv2
import torch
import requests
import onnxruntime as ort
from facenet_pytorch import MTCNN


class YuNetDetector:
    """
    Lightweight OpenCV YuNet detector with 5-point landmark output.
    This is substantially faster than MTCNN on CPU while preserving the
    landmark format expected by the face-alignment pipeline.
    """
    DEFAULT_MODEL_URL = (
        "https://github.com/ShiqiYu/libfacedetection.train/raw/master/onnx/"
        "yunet_n_640_640.onnx"
    )
    FULL_OUTPUT_NAMES = [
        "cls_8", "cls_16", "cls_32",
        "obj_8", "obj_16", "obj_32",
        "bbox_8", "bbox_16", "bbox_32",
        "kps_8", "kps_16", "kps_32",
    ]
    PRUNED_OUTPUT_NAMES = [
        "cls_16", "obj_16", "bbox_16", "kps_16",
        "cls_32", "obj_32", "bbox_32", "kps_32",
    ]

    def __init__(
        self,
        model_path="models/yunet/yunet_n_640_640.onnx",
        score_threshold=0.85,
        nms_threshold=0.3,
        top_k=5000,
    ):
        self.name = "YuNet"
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._ensure_model()
        self._backend = None
        self._session = None
        self._input_name = None
        self._input_size = None
        self._output_names = None
        self._output_name_set = None
        self._setup_backend()
        print(f"Initializing {self.name} detector with model: {self.model_path} ({self._backend})")

    def _ensure_model(self):
        if os.path.exists(self.model_path):
            return

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        filename = os.path.basename(self.model_path)
        if "2023mar" in filename:
            url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        else:
            url = self.DEFAULT_MODEL_URL

        print(f"YuNet model not found locally. Downloading to {self.model_path} ...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Unable to download the YuNet model automatically. "
                f"Please place the model file at {self.model_path}."
            ) from exc

        with open(self.model_path, "wb") as model_file:
            model_file.write(response.content)

    def _setup_backend(self):
        session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        output_names = [output.name for output in session.get_outputs()]
        self._session = session
        self._output_names = output_names
        self._output_name_set = set(output_names)
        input_meta = session.get_inputs()[0]
        shape = input_meta.shape
        if len(shape) != 4:
            raise RuntimeError(f"YuNet model input shape is unexpected: {shape}")
        self._input_name = input_meta.name
        self._input_size = (
            int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else None,
            int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else None,
        )

        if self._output_name_set.issuperset(self.FULL_OUTPUT_NAMES):
            self._backend = "opencv-dnn"
            self._session = None
            self.detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 320),
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )
            return

        if self._output_name_set.issuperset(self.PRUNED_OUTPUT_NAMES):
            self._backend = "onnxruntime-pruned"
            self.detector = None
            return

        raise RuntimeError(
            "Unsupported YuNet ONNX layout. Expected the full 12-output graph or the pruned 8-output graph, "
            f"but got outputs: {output_names}"
        )

    @staticmethod
    def _pad_to_divisor(img, divisor=32):
        src_h, src_w = img.shape[:2]
        pad_h = int((src_h - 1) / divisor + 1) * divisor
        pad_w = int((src_w - 1) / divisor + 1) * divisor
        bottom = pad_h - src_h
        right = pad_w - src_w
        if bottom == 0 and right == 0:
            return img, 0, 0
        padded = cv2.copyMakeBorder(
            img,
            0,
            bottom,
            0,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        return padded, 0, 0

    @staticmethod
    def _clip_faces(faces, width, height):
        if faces.size == 0:
            return faces
        faces[:, 0] = np.clip(faces[:, 0], 0, width - 1)
        faces[:, 1] = np.clip(faces[:, 1], 0, height - 1)
        faces[:, 2] = np.clip(faces[:, 2], 0, width - 1)
        faces[:, 3] = np.clip(faces[:, 3], 0, height - 1)
        for idx in range(5):
            faces[:, 4 + 2 * idx] = np.clip(faces[:, 4 + 2 * idx], 0, width - 1)
            faces[:, 4 + 2 * idx + 1] = np.clip(faces[:, 4 + 2 * idx + 1], 0, height - 1)
        return faces

    def _decode_outputs(self, output_blobs, strides, pad_left, pad_top, src_width, src_height):
        faces = []
        for scale_idx, stride in strides.items():
            cls, obj, bbox, kps = output_blobs[scale_idx]
            cls_v = np.asarray(cls, dtype=np.float32).reshape(-1)
            obj_v = np.asarray(obj, dtype=np.float32).reshape(-1)
            bbox_v = np.asarray(bbox, dtype=np.float32).reshape(-1, 4)
            kps_v = np.asarray(kps, dtype=np.float32).reshape(-1, 10)

            scores = np.sqrt(np.clip(cls_v, 0.0, 1.0) * np.clip(obj_v, 0.0, 1.0))
            keep_mask = scores >= self.score_threshold
            if not np.any(keep_mask):
                continue

            idx = np.flatnonzero(keep_mask)
            cols = int(round(src_width / stride))
            rows = int(round(src_height / stride))
            if idx.size > rows * cols:
                idx = idx[: rows * cols]

            c = (idx % cols).astype(np.float32)
            r = (idx // cols).astype(np.float32)
            box = bbox_v[idx]
            kp = kps_v[idx].reshape(-1, 5, 2)

            cx = (c + box[:, 0]) * stride
            cy = (r + box[:, 1]) * stride
            w = np.exp(box[:, 2]) * stride
            h = np.exp(box[:, 3]) * stride
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0

            face = np.zeros((idx.size, 15), dtype=np.float32)
            face[:, 0] = x1
            face[:, 1] = y1
            face[:, 2] = w
            face[:, 3] = h
            offsets = np.stack((c, r), axis=1)[:, None, :]
            face[:, 4:14] = ((kp + offsets) * stride).reshape(-1, 10)
            face[:, 14] = scores[idx]
            faces.append(face)

        if not faces:
            return np.empty((0, 15), dtype=np.float32)

        faces = np.concatenate(faces, axis=0)

        if pad_left != 0 or pad_top != 0:
            faces[:, 0] -= pad_left
            faces[:, 1] -= pad_top
            faces[:, 4:14:2] -= pad_left
            faces[:, 5:14:2] -= pad_top

        faces = self._clip_faces(faces, src_width, src_height)

        if faces.shape[0] <= 1:
            return faces

        face_boxes = faces[:, :4].tolist()
        face_scores = faces[:, 14].tolist()
        keep_idx = cv2.dnn.NMSBoxes(face_boxes, face_scores, self.score_threshold, self.nms_threshold, 1.0, self.top_k)
        if len(keep_idx) == 0:
            return np.empty((0, 15), dtype=np.float32)

        keep_idx = np.asarray(keep_idx).reshape(-1).astype(int)
        return faces[keep_idx]

    def _detect_onnxruntime(self, img):
        src_height, src_width = img.shape[:2]
        padded, pad_left, pad_top = self._pad_to_divisor(img, divisor=32)
        blob = cv2.dnn.blobFromImage(padded)

        if self._output_name_set.issuperset(self.PRUNED_OUTPUT_NAMES):
            ordered_outputs = self.PRUNED_OUTPUT_NAMES
            output_blobs = self._session.run(ordered_outputs, {self._input_name: blob})
            grouped = [output_blobs[0:4], output_blobs[4:8]]
            return self._decode_outputs(grouped, {0: 16, 1: 32}, pad_left, pad_top, src_width, src_height)

        raise RuntimeError("ORT backend was selected for a YuNet graph that does not match the pruned layout.")

    def detect(self, img):
        """
        Detect faces and landmarks in the input frame.

        Args:
            img (numpy.ndarray): Input image in BGR format (from OpenCV).

        Returns:
            tuple:
                - total_boxes (numpy.ndarray): Shape (N, 5) containing
                  [x1, y1, x2, y2, confidence_score]
                - landmarks (numpy.ndarray): Shape (N, 5, 2) containing
                  5 landmark points [(x, y)]
        """
        if img is None:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        if self._backend == "opencv-dnn":
            self.detector.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.detector.detect(img)
            if faces is None or len(faces) == 0:
                return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

            faces = np.asarray(faces, dtype=np.float32)
            x1 = faces[:, 0]
            y1 = faces[:, 1]
            x2 = x1 + faces[:, 2]
            y2 = y1 + faces[:, 3]
            scores = faces[:, 14]
            total_boxes = np.column_stack((x1, y1, x2, y2, scores)).astype(np.float32)
            landmarks = faces[:, 4:14].reshape(-1, 5, 2).astype(np.float32)
            return total_boxes, landmarks

        faces = self._detect_onnxruntime(img)
        if faces is None or len(faces) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)

        x1 = faces[:, 0]
        y1 = faces[:, 1]
        x2 = x1 + faces[:, 2]
        y2 = y1 + faces[:, 3]
        scores = faces[:, 14]
        total_boxes = np.column_stack((x1, y1, x2, y2, scores)).astype(np.float32)
        landmarks = faces[:, 4:14].reshape(-1, 5, 2).astype(np.float32)
        return total_boxes, landmarks


class MTCNNDetector:
    """
    Fast MTCNN face detector backend.
    This implementation wraps facenet-pytorch's MTCNN (the exact engine used by 
    DeepFace's 'fastmtcnn' backend) for robust, real-time, and GPU-accelerated 
    face and landmark detection.
    """
    def __init__(self, model_dir="models/mtcnn", minsize=20, threshold=[0.6, 0.7, 0.8], factor=0.709):
        self.name = "MTCNN"
        # Automatically detect and use CUDA GPU if available for maximum FPS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Fast MTCNN (facenet-pytorch) detector on: {self.device}")
        
        # Instantiate the optimized MTCNN detector
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=minsize,
            thresholds=threshold,
            factor=factor,
            post_process=False,
            keep_all=True,
            device=self.device
        )

    def detect(self, img):
        """
        Detect faces and landmarks in the input frame.
        
        Args:
            img (numpy.ndarray): Input image in BGR format (from OpenCV).
            
        Returns:
            tuple:
                - total_boxes (numpy.ndarray): Shape (N, 5) containing [x1, y1, x2, y2, confidence_score]
                - landmarks (numpy.ndarray): Shape (N, 5, 2) containing 5 landmark points [(x, y)]
        """
        if img is None:
            return np.empty((0, 5)), np.empty((0, 5, 2))
        
        # Convert BGR frame to RGB (as expected by facenet-pytorch)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Perform inference using optimized MTCNN
        boxes, probs, landmarks = self.mtcnn.detect(img_rgb, landmarks=True)
        
        # If no faces were detected, return empty arrays
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 5)), np.empty((0, 5, 2))
            
        # Explicitly cast to numpy float32 arrays to ensure correct data types for downstream skimage/numpy operations
        boxes = np.array(boxes, dtype=np.float32)
        landmarks = np.array(landmarks, dtype=np.float32)
        
        if probs is None:
            probs = np.ones(len(boxes), dtype=np.float32)
        else:
            probs = np.array(probs, dtype=np.float32)
            
        # Format boxes to (N, 5) by appending confidence scores
        total_boxes = np.hstack([boxes, np.expand_dims(probs, axis=1)])
        
        return total_boxes, landmarks
