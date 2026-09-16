import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from dataset_loader import get_class_names, load_dataset
from model import VideoAugmentation


TEST_DIR = Path("dataset_split/test")


def load_class_names():
    names_path = Path("class_names.json")
    if names_path.exists():
        return json.loads(names_path.read_text(encoding="utf-8"))
    return get_class_names(TEST_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Testuje wybrany wytrenowany model.")
    parser.add_argument(
        "--model",
        default="best_model.keras",
        help="Ścieżka do pliku .keras (domyślnie: best_model.keras).",
    )
    return parser.parse_args()


def main(model_path: Path):
    class_names = load_class_names()
    if not class_names:
        raise ValueError("Brak klas testowych lub class_names.json.")
    class_to_index = {name: index for index, name in enumerate(class_names)}
    if not model_path.exists():
        raise FileNotFoundError(f"Brakuje pliku modelu: {model_path}.")

    print(f"Loading test data from {TEST_DIR}...")
    x_test, y_test = load_dataset(TEST_DIR, class_to_index)
    if len(x_test) == 0:
        raise ValueError("Zbiór testowy jest pusty.")
    print("Test data shape:", x_test.shape, y_test.shape)

    model = tf.keras.models.load_model(
        model_path, custom_objects={"VideoAugmentation": VideoAugmentation}
    )
    loss, keras_accuracy = model.evaluate(x_test, y_test, verbose=1)
    probabilities = model.predict(x_test, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    labels = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(
        y_test, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    summary = {
        "model": str(model_path),
        "test_videos": int(len(y_test)),
        "classes": len(class_names),
        "parameters_total": int(model.count_params()),
        "loss": float(loss),
        "accuracy_keras": float(keras_accuracy),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "classification_report": report,
    }
    report_stem = model_path.stem
    summary_path = Path(f"test_summary_{report_stem}.json")
    matrix_npy_path = Path(f"confusion_matrix_{report_stem}.npy")
    matrix_csv_path = Path(f"confusion_matrix_{report_stem}.csv")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(matrix_npy_path, matrix)
    np.savetxt(matrix_csv_path, matrix, fmt="%d", delimiter=",")

    print("\nWyniki testu")
    print(f"loss: {loss:.4f}")
    print(f"accuracy: {summary['accuracy']:.4f}")
    print(f"macro precision: {precision:.4f}")
    print(f"macro recall: {recall:.4f}")
    print(f"macro F1: {f1:.4f}")
    print(f"Saved: {summary_path}, {matrix_npy_path} and {matrix_csv_path}")


if __name__ == "__main__":
    args = parse_args()
    main(Path(args.model))
