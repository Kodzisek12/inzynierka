"""Ładowanie danych wideo.

Pełen zbiór jest dekodowany raz do RAM-u, a ``tf.data`` przekazuje modelowi
wyłącznie aktualny batch.
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from video_loader import frames as FRAME_COUNT
from video_loader import load_video


FRAME_HEIGHT = 32
FRAME_WIDTH = 32
VIDEO_SHAPE = (FRAME_COUNT, FRAME_HEIGHT, FRAME_WIDTH, 3)
SHUFFLE_SEED = 42


def get_class_names(dataset_dir):
    class_names = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    class_names.sort()
    return class_names


def index_classes(dataset_dir):
    class_names = get_class_names(dataset_dir)
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    index_to_class = {index: class_name for class_name, index in class_to_index.items()}
    return class_to_index, index_to_class


def get_video_samples(dataset_dir, class_to_index):
    """Zwraca wyłącznie ścieżki i etykiety, bez dekodowania filmów do RAM."""
    samples = []
    root = Path(dataset_dir)
    for class_name, index in class_to_index.items():
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for video_path in sorted(class_dir.glob("*.avi")):
            samples.append((video_path, np.int32(index)))
    return samples


def make_memory_dataset(x_data, y_data, batch_size, training=False):
    """Buduje dataset z danych w RAM-ie, pozostawiając pełne tablice na CPU.

    Pełen zbiór jest przechowywany w pamięci operacyjnej, ale dzięki
    ``tf.data`` GPU otrzymuje tylko aktualny batch. Kontekst CPU jest istotny:
    bez niego TensorFlow może spróbować skopiować całą tablicę na kartę.

    Podczas treningu mieszamy *wszystkie* próbki. Nie mieszamy samych filmów
    w buforze ``tf.data`` — dla tego zbioru zajęłoby to wiele dodatkowych GB
    RAM. Zamiast tego mieszamy małe indeksy, a właściwy film pobieramy dopiero
    dla aktualnej próbki. To odpowiada dawnemu ``model.fit(..., shuffle=True)``.
    """
    if len(x_data) == 0:
        raise ValueError("Nie można utworzyć datasetu z pustych danych.")

    with tf.device("/CPU:0"):
        x_tensor = tf.convert_to_tensor(x_data)
        y_tensor = tf.convert_to_tensor(y_data)
        indices = tf.range(len(y_data), dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices(indices)
    if training:
        dataset = dataset.shuffle(
            len(y_data), seed=SHUFFLE_SEED, reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        lambda index: (tf.gather(x_tensor, index), tf.gather(y_tensor, index)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)


def load_dataset(dataset_dir, class_to_index=None):
    """Wczytuje cały zbiór do RAM bez tworzenia drugiej pełnej kopii tablicy."""
    if class_to_index is None:
        class_to_index, _ = index_classes(dataset_dir)
    samples = get_video_samples(dataset_dir, class_to_index)
    x_data = np.empty((len(samples), *VIDEO_SHAPE), dtype=np.float32)
    y_data = np.empty(len(samples), dtype=np.int32)
    loaded_count = 0
    for video_path, label in samples:
        try:
            video = load_video(str(video_path))
            if video.shape != VIDEO_SHAPE:
                raise ValueError(f"Nieprawidłowy kształt {video.shape}.")
            x_data[loaded_count] = video
            y_data[loaded_count] = label
            loaded_count += 1
        except (ValueError, OSError) as error:
            print(f"Nie można wczytać wideo {video_path}: {error}")
    return x_data[:loaded_count], y_data[:loaded_count]
