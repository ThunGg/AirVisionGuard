import os
import numpy as np
import cv2
import torch
import requests
from facenet_pytorch import MTCNN


class YuNetDetector:
    """
    Lightweight OpenCV YuNet detector with 5-point landmark output.
    This is substantially faster than MTCNN on CPU while preserving the
    landmark format expected by the face-alignment pipeline.
    """
    DEFAULT_MODEL_URL = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx"
    )

    def __init__(
        self,
        model_path="models/yunet/face_detection_yunet_2023mar.onnx",
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
        print(f"Initializing {self.name} detector with model: {self.model_path}")
        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )

    def _ensure_model(self):
        if os.path.exists(self.model_path):
            return

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        print(f"YuNet model not found locally. Downloading to {self.model_path} ...")
        try:
            response = requests.get(self.DEFAULT_MODEL_URL, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Unable to download the YuNet model automatically. "
                "Please place face_detection_yunet_2023mar.onnx at "
                f"{self.model_path}."
            ) from exc

        with open(self.model_path, "wb") as model_file:
            model_file.write(response.content)

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
