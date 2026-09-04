import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from dataset_loader import get_class_names, load_dataset
from model import create_model


TRAIN_DIR = Path("dataset_split/train")
VAL_DIR = Path("dataset_split/val")
EPOCHS = 10
BATCH_SIZE = 8
PATIENCE = 3


def save_training_summary(history, model, class_names, train_size, val_size, input_shape):
    val_losses = history.history["val_loss"]
    best_index = int(np.argmin(val_losses))
    summary = {
        "model": {
            "parameters_total": int(model.count_params()),
            "input_shape": list(input_shape),
            "classes": len(class_names),
        },
        "training": {
            "train_videos": train_size,
            "validation_videos": val_size,
            "batch_size": BATCH_SIZE,
            "epochs_completed": len(history.history["loss"]),
            "best_epoch": best_index + 1,
            "best_val_loss": float(val_losses[best_index]),
            "best_val_accuracy": float(history.history["val_accuracy"][best_index]),
            "final_loss": float(history.history["loss"][-1]),
            "final_accuracy": float(history.history["accuracy"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]),
            "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        },
    }
    with Path("training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    training = summary["training"]
    Path("training_summary.txt").write_text(
        "Podsumowanie treningu\n"
        f"Parametry modelu: {summary['model']['parameters_total']:,}\n"
        f"Klasy: {summary['model']['classes']}\n"
        f"Wymiary wejścia: {tuple(summary['model']['input_shape'])}\n"
        f"Filmy treningowe: {training['train_videos']}\n"
        f"Filmy walidacyjne: {training['validation_videos']}\n"
        f"Batch size: {training['batch_size']}\n"
        f"Najlepsza epoka: {training['best_epoch']}\n"
        f"Najlepsze val_accuracy: {training['best_val_accuracy']:.4f}\n"
        f"Najlepsze val_loss: {training['best_val_loss']:.4f}\n",
        encoding="utf-8",
    )


def main():
    class_names = get_class_names(TRAIN_DIR)
    if not class_names:
        raise ValueError("Brak klas w dataset_split/train. Najpierw uruchom preprocessing.py --clean.")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    Path("class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print("Loading training data...")
    x_train, y_train = load_dataset(TRAIN_DIR, class_to_index)
    print("Loading validation data...")
    x_val, y_val = load_dataset(VAL_DIR, class_to_index)
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("Zbiór treningowy lub walidacyjny jest pusty.")

    print("Training data shape:", x_train.shape, y_train.shape)
    print("Validation data shape:", x_val.shape, y_val.shape)
    input_shape = x_train.shape[1:]
    print(f"Input shape: {input_shape}, Number of classes: {len(class_names)}")

    model = create_model(input_shape, len(class_names))
    model.summary()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger("training_history.csv"),
    ]
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=callbacks,
    )
    model.save("model.keras")
    save_training_summary(history, model, class_names, len(x_train), len(x_val), input_shape)
    print("Saved: model.keras, best_model.keras, training_history.csv and training_summary.json")


if __name__ == "__main__":
    main()
