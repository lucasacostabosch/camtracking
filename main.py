# main.py
import cv2
from src.trackManager import TrackManager
import concurrent.futures
import argparse

def main(use_webcam):
    if use_webcam:
        cap = cv2.VideoCapture(0)
    else:
        VIDEO_FILE = "./videos/videoTest.mp4"
        cap = cv2.VideoCapture(VIDEO_FILE)

    manager = TrackManager()
    skip_frames = 2
    frame_count = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            future = executor.submit(manager.predict_and_detect, frame)
            result_frame = future.result()

            cv2.imshow("Processed Frame", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesa video o webcam y detecta personas.")
    parser.add_argument("--webcam", action="store_true", help="Usar la cámara web en lugar de un archivo de video.")
    args = parser.parse_args()

    main(args.webcam)