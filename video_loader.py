import cv2 as cv
import numpy as np
frames = 32

def load_video(video_path, frame_rate=frames, resize=(32, 32)):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Nie można otworzyć pliku wideo: {video_path}")
    all_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    if all_frames == 0:
        raise ValueError(f"Plik wideo {video_path} nie zawiera żadnych klatek.") #HOW
    equal_interval_idx = np.linspace(0, all_frames - 1, frame_rate, dtype=int)
    video_frames = []
    
    
    for idx in equal_interval_idx:
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if not ret:
            continue
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, resize)
        frame = frame.astype(np.float32) / 255.0  # Normalizacja 
        
        video_frames.append(frame)
        
    cap.release()
    
    if len(video_frames) == 0:
        raise ValueError(f"Nie można wczytać żadnych klatek z wideo: {video_path}") 
    while len(video_frames) < frame_rate:
        video_frames.append(video_frames[-1])  # Powtarzanie ostatniej klatki, jeśli jest ich mniej niż frame_rate
    video_frames = np.array(video_frames, dtype=np.float32)
    return video_frames