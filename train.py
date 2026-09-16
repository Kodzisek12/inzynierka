import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import (
    VIDEO_SHAPE,
    get_class_names,
    load_dataset,
    make_memory_dataset,
)
from model import L2_WEIGHT_DECAY, create_model


TRAIN_DIR = Path("dataset_split/train")
VAL_DIR = Path("dataset_split/val")
MODEL_PATH = Path("model.keras")
BEST_MODEL_PATH = Path("best_model.keras")
HISTORY_PATH = Path("training_history.csv")
SUMMARY_PATH = Path("training_summary.json")
SUMMARY_TEXT_PATH = Path("training_summary.txt")
EPOCHS = 35
BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 2


def save_training_summary(history, model, class_names, train_size, val_size, input_shape):
    val_losses = history.history["val_loss"]
    val_accuracies = history.history["val_accuracy"]
    best_accuracy_index = int(np.argmax(val_accuracies))
    best_loss_index = int(np.argmin(val_losses))
    summary = {
        "model": {
            "parameters_total": int(model.count_params()),
            "input_shape": list(input_shape),
            "classes": len(class_names),
            "regularization": {
                "augmentation": "horizontal flip, brightness ±0.08, contrast 0.85–1.15",
                "normalization": "batch normalization",
                "spatial_dropout": [0.10, 0.15],
                "dropout": 0.45,
                "l2_weight_decay": L2_WEIGHT_DECAY,
                "gradient_clipnorm": 1.0,
            },
            "classifier": "Dense(256) → Flatten → Dropout(0.45)",
        },
        "training": {
            "train_videos": train_size,
            "validation_videos": val_size,
            "batch_size": BATCH_SIZE,
            "epochs_completed": len(history.history["loss"]),
            "best_val_accuracy_epoch": best_accuracy_index + 1,
            "best_val_accuracy": float(val_accuracies[best_accuracy_index]),
            "val_loss_at_best_accuracy": float(val_losses[best_accuracy_index]),
            "best_val_loss_epoch": best_loss_index + 1,
            "best_val_loss": float(val_losses[best_loss_index]),
            "val_accuracy_at_best_loss": float(val_accuracies[best_loss_index]),
            "final_loss": float(history.history["loss"][-1]),
            "final_accuracy": float(history.history["accuracy"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        },
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    training = summary["training"]
    SUMMARY_TEXT_PATH.write_text(
        "Podsumowanie treningu\n"
        f"Parametry modelu: {summary['model']['parameters_total']:,}\n"
        f"Klasy: {summary['model']['classes']}\n"
        f"Wymiary wejścia: {tuple(summary['model']['input_shape'])}\n"
        f"Filmy treningowe: {training['train_videos']}\n"
        f"Filmy walidacyjne: {training['validation_videos']}\n"
        f"Batch size: {training['batch_size']}\n"
        f"Najlepsza epoka (val_accuracy): {training['best_val_accuracy_epoch']}\n"
        f"Najlepsze val_accuracy: {training['best_val_accuracy']:.4f}\n"
        f"Najlepsza epoka (val_loss): {training['best_val_loss_epoch']}\n"
        f"Najlepsze val_loss: {training['best_val_loss']:.4f}\n",
        encoding="utf-8",
    )


def main():
    tf.keras.utils.set_random_seed(42)
    class_names = get_class_names(TRAIN_DIR)
    if not class_names:
        raise ValueError("Brak klas w dataset_split/train. Najpierw uruchom preprocessing.py --clean.")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    Path("class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print("Loading training data into RAM (GPU receives batches only)...")
    x_train, y_train = load_dataset(TRAIN_DIR, class_to_index)
    train_size = len(x_train)
    train_dataset = make_memory_dataset(x_train, y_train, BATCH_SIZE, training=True)
    del x_train, y_train

    print("Loading validation data into RAM (GPU receives batches only)...")
    x_val, y_val = load_dataset(VAL_DIR, class_to_index)
    val_size = len(x_val)
    val_dataset = make_memory_dataset(x_val, y_val, BATCH_SIZE)
    del x_val, y_val
    if train_size == 0 or val_size == 0:
        raise ValueError("Zbiór treningowy lub walidacyjny jest pusty.")

    input_shape = VIDEO_SHAPE
    train_steps = math.ceil(train_size / BATCH_SIZE)
    validation_steps = math.ceil(val_size / BATCH_SIZE)
    print(
        f"Training videos: {train_size} ({train_steps} batches per epoch), "
        f"validation videos: {val_size} ({validation_steps} batches)"
    )
    print(f"Input shape: {input_shape}, Number of classes: {len(class_names)}")

    model = create_model(input_shape, len(class_names))
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_MODEL_PATH), monitor="val_loss", mode="min", save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=REDUCE_LR_PATIENCE, min_lr=1e-5
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.CSVLogger(str(HISTORY_PATH)),
    ]
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        shuffle=False,
        callbacks=callbacks,
    )
    model.save(str(MODEL_PATH))
    save_training_summary(history, model, class_names, train_size, val_size, input_shape)
    print(
        f"Saved: {MODEL_PATH}, {BEST_MODEL_PATH}, {HISTORY_PATH} and {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
