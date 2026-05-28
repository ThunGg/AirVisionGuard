import cv2
import time
import psutil
import os

def main():
    # Limit OpenCV internal thread pool to reduce power draw like in main.py
    cv2.setNumThreads(2)

    # Initialize process to monitor
    p = psutil.Process(os.getpid())
    
    # Initialize camera as in main.py
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Set properties as in main.py
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Testing camera at {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print("Press 'q' to stop testing.")

    # Discard first CPU usage measurement
    p.cpu_percent(interval=None)

    frame_count = 0
    start_time = time.time()
    last_print = start_time

    FPS_ACTIVE = 8
    
    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_count += 1
        current_time = time.time()

        if current_time - last_print >= 1.0:
            cpu_usage = p.cpu_percent(interval=None)
            fps = frame_count / (current_time - last_print)
            print(f"CPU Usage: {cpu_usage:.1f}% | FPS: {fps:.1f}")
            frame_count = 0
            last_print = current_time

        cv2.imshow("Camera CPU Test", frame)
        
        elapsed = time.time() - frame_start
        frame_duration = 1.0 / FPS_ACTIVE
        sleep_ms = max(1, int((frame_duration - elapsed) * 1000))
        
        if cv2.waitKey(sleep_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
