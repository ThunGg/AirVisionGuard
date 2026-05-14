import cv2
import argparse
import time
import os
import sys

from detector import MTCNNDetector
from recognizer import FaceRecognizer
from lock_screen import ScreenLocker
from utils import align_face, preprocess_face

def main(args):
    print("Initializing components...")
    if not os.path.exists(args.model):
        print(f"Error: Face recognition model not found at {args.model}")
        print("Please provide a valid ONNX model path using --model")
        sys.exit(1)

    detector = MTCNNDetector()
    recognizer = FaceRecognizer(args.model)
    locker = ScreenLocker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    authorized_embedding = None
    print("Camera opened. Press 'r' to register your face as the authorized user.")
    print("Press 'q' to quit.")

    missing_frames = 0
    LOCK_THRESHOLD = 15 # lock after 15 consecutive frames without the authorized face
    UNLOCK_THRESHOLD = 0.6 # cosine similarity threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, landmarks = detector.detect(frame)
        
        face_found = False

        if boxes.shape[0] > 0:
            # Get the largest face
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            max_idx = areas.argmax()
            box = boxes[max_idx]
            landmark = landmarks[max_idx]

            # Align and preprocess
            aligned_face = align_face(frame, box, landmark)
            img_tensor = preprocess_face(aligned_face)

            # Get embedding
            emb = recognizer.get_embedding(img_tensor)

            # Draw box
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            if authorized_embedding is None:
                cv2.putText(frame, "Press 'r' to register", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                sim = recognizer.compute_similarity(emb, authorized_embedding)
                if sim > UNLOCK_THRESHOLD:
                    cv2.putText(frame, f"Authorized: {sim:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    face_found = True
                else:
                    cv2.putText(frame, f"Unknown: {sim:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Update logic for locking/unlocking
        if authorized_embedding is not None:
            if face_found:
                missing_frames = 0
                if locker.is_locked:
                    locker.unlock()
            else:
                missing_frames += 1
                if missing_frames > LOCK_THRESHOLD and not locker.is_locked:
                    locker.lock()
        
        # Only show preview window if not locked
        if not locker.is_locked:
            cv2.imshow("Face Lock Registration / Preview", frame)
        else:
            # Close preview window to avoid showing on top of lock screen
            if cv2.getWindowProperty("Face Lock Registration / Preview", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Face Lock Registration / Preview")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and boxes.shape[0] > 0 and authorized_embedding is None:
            authorized_embedding = emb
            print("Face registered successfully! Monitoring started.")
            # We don't lock immediately.
            missing_frames = 0

        # Important to update tkinter events
        locker.update()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../../models/face_rec.onnx", help="Path to the trained Face Recognition ONNX model")
    args = parser.parse_args()
    main(args)
