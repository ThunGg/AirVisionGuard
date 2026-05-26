import cv2
import argparse
import time
import os
import sys

from detector import MTCNNDetector, YuNetDetector
from recognizer import FaceRecognizer
from lock_screen import ScreenLocker
from utils import align_face, preprocess_face

def create_detector(args):
    if args.detector == "mtcnn":
        return MTCNNDetector()

    try:
        return YuNetDetector(model_path=args.detector_model)
    except Exception as exc:
        print(f"Warning: failed to initialize YuNet detector: {exc}")
        print("Falling back to MTCNN detector.")
        return MTCNNDetector()

def main(args):
    if not os.path.exists(args.model):
        print(f"Error: Face recognition model not found at {args.model}")
        print("Please provide a valid ONNX model path using --model")
        sys.exit(1)

    detector = create_detector(args)
    print(f"Initializing components (using {detector.name} face detector)...")
    recognizer = FaceRecognizer(args.model)
    locker = ScreenLocker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    authorized_embeddings = []
    registration_steps = ["Close up", "Medium distance", "Far distance"]
    print("Camera opened. Face registration required:")
    for step in registration_steps:
        print(f"  - {step}")
    print("Press 'r' to register the face at each distance step-by-step.")
    print("Press 't' to toggle face lock ON/OFF once registered.")
    print("Press 'q' to quit.")

    missing_frames = 0
    LOCK_THRESHOLD = 15  # lock after 15 consecutive frames without the authorized face
    UNLOCK_THRESHOLD = 0.6  # cosine similarity threshold
    lock_enabled = True
    TARGET_FPS = 10  # cap processing rate to reduce CPU usage
    FRAME_DURATION = 1.0 / TARGET_FPS

    while True:
        frame_start = time.time()
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

            if len(authorized_embeddings) == len(registration_steps):
                # Check similarity against all registered embeddings and use the maximum match
                similarities = [recognizer.compute_similarity(emb, auth_emb) for auth_emb in authorized_embeddings]
                max_sim = max(similarities)
                if max_sim > UNLOCK_THRESHOLD:
                    cv2.putText(frame, f"Authorized: {max_sim:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    face_found = True
                else:
                    cv2.putText(frame, f"Unknown: {max_sim:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Draw status text at the top-left of the frame
        if len(authorized_embeddings) < len(registration_steps):
            current_step = registration_steps[len(authorized_embeddings)]
            cv2.putText(frame, f"Register: {current_step} ('r')", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            status_text = "Face Lock: ACTIVE (t: turn off)" if lock_enabled else "Face Lock: INACTIVE (t: turn on)"
            color = (0, 255, 0) if lock_enabled else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Update logic for locking/unlocking
        if len(authorized_embeddings) == len(registration_steps) and lock_enabled:
            if face_found:
                missing_frames = 0
                if locker.is_locked:
                    locker.unlock()
            else:
                missing_frames += 1
                if missing_frames > LOCK_THRESHOLD and not locker.is_locked:
                    locker.lock()
        elif locker.is_locked:
            # If lock screen is active but face lock gets disabled, automatically unlock
            locker.unlock()
            missing_frames = 0
        
        # Only show preview window if not locked
        if not locker.is_locked:
            cv2.imshow("Face Lock Registration / Preview", frame)
        else:
            # Close preview window to avoid showing on top of lock screen
            if cv2.getWindowProperty("Face Lock Registration / Preview", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Face Lock Registration / Preview")

        # Throttle to TARGET_FPS to reduce CPU usage
        elapsed = time.time() - frame_start
        sleep_ms = max(1, int((FRAME_DURATION - elapsed) * 1000))
        key = cv2.waitKey(sleep_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and boxes.shape[0] > 0 and len(authorized_embeddings) < len(registration_steps):
            authorized_embeddings.append(emb)
            step_name = registration_steps[len(authorized_embeddings) - 1]
            print(f"Face registered successfully for {step_name}!")
            if len(authorized_embeddings) == len(registration_steps):
                print("All face distances registered successfully! Monitoring started.")
            # We don't lock immediately.
            missing_frames = 0
        elif key == ord('t') and len(authorized_embeddings) == len(registration_steps):
            lock_enabled = not lock_enabled
            if not lock_enabled:
                print("Face lock turned OFF (disabled).")
            else:
                print("Face lock turned ON (enabled).")
            missing_frames = 0

        # Important to update tkinter events and prevent Alt+Tab / focus loss bypasses
        if locker.is_locked:
            locker.root.lift()
            locker.root.focus_force()
        locker.update()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../../models/face_rec.onnx", help="Path to the trained Face Recognition ONNX model")
    parser.add_argument(
        "--detector",
        type=str,
        default="yunet",
        choices=["yunet", "mtcnn"],
        help="Face detector backend to use. 'yunet' is faster on CPU; 'mtcnn' is kept as a fallback.",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default="models/yunet/face_detection_yunet_2023mar.onnx",
        help="Path to the YuNet ONNX detector model file.",
    )
    args = parser.parse_args()
    main(args)
