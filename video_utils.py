import cv2
import os
from config import FRAME_INTERVAL, TEMP_DIR, FRAME_DIR, QDRANT_COLLECTION_NAME

def extract_frames(video_path: str, interval: int) -> list:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)

    frame_paths = []
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            frame_name = f"frame_{i}.jpg"
            frame_path = os.path.join("storage/frames", frame_name)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        i += 1
    cap.release()
    return frame_paths

def compute_color_histogram(image_path: str) -> list:
    image = cv2.imread(image_path)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()
