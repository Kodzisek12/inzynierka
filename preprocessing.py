"""Deterministyczny podział UCF101 na train, validation i test.

Przykład:
    python preprocessing.py --clean

Domyślnie skrypt tworzy twarde linki do filmów, więc nie dubluje wielkości
zbioru na tym samym dysku. Użyj --copy, gdy pliki muszą być fizycznie kopiowane.
"""

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")


def _add_file(source: Path, destination: Path, copy_files: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
        return "copy"

    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        # Twarde linki nie działają między różnymi dyskami lub systemami plików.
        shutil.copy2(source, destination)
        return "copy"


def _prepare_output(input_dir: Path, output_dir: Path, clean: bool) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Katalog wejściowy i wyjściowy muszą być różne.")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not clean:
            raise FileExistsError(
                f"{output_dir} nie jest pusty. Uruchom skrypt z --clean, "
                "aby usunąć stary podział i wygenerować nowy."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def verify_split(output_dir: Path) -> dict[str, int]:
    """Sprawdza, czy żaden film nie znajduje się w więcej niż jednym splicie."""
    seen: dict[str, str] = {}
    counts = {split: 0 for split in SPLIT_NAMES}

    for split in SPLIT_NAMES:
        split_dir = output_dir / split
        if not split_dir.is_dir():
            raise ValueError(f"Brakuje katalogu {split_dir}.")
        for video in split_dir.rglob("*.avi"):
            key = video.relative_to(split_dir).as_posix()
            if key in seen:
                raise ValueError(
                    f"Film {key} występuje zarówno w {seen[key]}, jak i w {split}."
                )
            seen[key] = split
            counts[split] += 1

    if not seen:
        raise ValueError("Podział nie zawiera żadnych plików .avi.")
    return counts


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    clean: bool = False,
    copy_files: bool = False,
) -> dict[str, int]:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Proporcje train, val i test muszą sumować się do 1.")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Nie znaleziono katalogu wejściowego: {input_dir}")

    _prepare_output(input_dir, output_dir, clean)
    rng = random.Random(seed)
    class_counts: dict[str, dict[str, int]] = {}
    storage_modes = {"hardlink": 0, "copy": 0}

    for class_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        videos = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".avi"
        )
        if not videos:
            continue

        rng.shuffle(videos)
        train_end = int(len(videos) * train_ratio)
        val_end = train_end + int(len(videos) * val_ratio)
        groups = {
            "train": videos[:train_end],
            "val": videos[train_end:val_end],
            "test": videos[val_end:],
        }
        class_counts[class_dir.name] = {}

        for split, split_videos in groups.items():
            class_counts[class_dir.name][split] = len(split_videos)
            for video in split_videos:
                mode = _add_file(video, output_dir / split / class_dir.name / video.name, copy_files)
                storage_modes[mode] += 1

    counts = verify_split(output_dir)
    manifest = {
        "source": str(input_dir),
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "counts": counts,
        "class_counts": class_counts,
        "storage": storage_modes,
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    print("Podział zakończony i zweryfikowany.")
    print(f"train: {counts['train']}, val: {counts['val']}, test: {counts['test']}")
    print(f"Pliki: {storage_modes['hardlink']} linków, {storage_modes['copy']} kopii")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generuje poprawny podział danych UCF101.")
    parser.add_argument("--input", default="dataset/UCF101", help="Katalog źródłowy UCF101.")
    parser.add_argument("--output", default="dataset_split", help="Katalog na podział danych.")
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowania.")
    parser.add_argument("--clean", action="store_true", help="Usuwa poprzedni katalog wyjściowy.")
    parser.add_argument("--copy", action="store_true", help="Kopiuje filmy zamiast tworzyć twarde linki.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(
        Path(args.input),
        Path(args.output),
        seed=args.seed,
        clean=args.clean,
        copy_files=args.copy,
    )
