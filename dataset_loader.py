import os
import numpy as np

from video_loader import load_video

def get_class_names(dataset_dir):
    class_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_names.sort()
    return class_names
def index_classes(dataset_dir):
    class_names = get_class_names(dataset_dir)
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    index_to_class = {index: class_name for class_name, index in class_to_index.items()}
    return class_to_index, index_to_class

def load_dataset(dataset_dir, class_to_index=None):
    if class_to_index is None:
        class_to_index, _ = index_classes(dataset_dir)
    X = []
    Y = []
    for class_name, index in class_to_index.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for video_file in sorted(os.listdir(class_dir)):
            if not video_file.lower().endswith(".avi"):
                continue
            video_path = os.path.join(class_dir, video_file)
            try:
                print(f"Wczytywanie wideo: {video_path}")
                video = load_video(video_path)
                X.append(video)
                Y.append(index)
            except Exception as e:
                print(f"Nie można wczytać wideo {video_path}: {e}")
    return np.array(X), np.array(Y)
