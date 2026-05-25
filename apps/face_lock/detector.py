import os
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN

class MTCNNDetector:
    """
    Fast MTCNN face detector backend.
    This implementation wraps facenet-pytorch's MTCNN (the exact engine used by 
    DeepFace's 'fastmtcnn' backend) for robust, real-time, and GPU-accelerated 
    face and landmark detection.
    """
    def __init__(self, model_dir="models/mtcnn", minsize=20, threshold=[0.6, 0.7, 0.8], factor=0.709):
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
