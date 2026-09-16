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
import re
import shutil
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")
GROUP_PATTERN = re.compile(r"_g(?P<group>\d+)_c\d+$", re.IGNORECASE)


def _group_key(video_path: Path) -> str:
    """Zwraca identyfikator grupy UCF101 dla pojedynczego nagrania.

    Filmy z tym samym ``gXX`` dla jednej klasy pochodzą z podobnego materiału
    źródłowego. Muszą zawsze trafić do tego samego splitu.
    """
    match = GROUP_PATTERN.search(video_path.stem)
    if match is None:
        raise ValueError(
            f"Nazwa pliku nie ma formatu UCF101 z identyfikatorem grupy: {video_path}"
        )
    return f"{video_path.parent.name}:g{match.group('group')}"


def _split_grouped_videos(
    videos: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    """Dzieli filmy, nie rozdzielając żadnej grupy UCF101.

    Grupy mają różne liczby klipów, dlatego wyniki nie muszą mieć dokładnie
    docelowych proporcji co do jednego pliku. Wybór według największego
    względnego niedoboru daje zbliżone proporcje bez przecieku danych.
    """
    grouped_videos: dict[str, list[Path]] = {}
    for video in videos:
        grouped_videos.setdefault(_group_key(video), []).append(video)

    group_keys = sorted(grouped_videos)
    rng.shuffle(group_keys)
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    targets = {split: len(videos) * ratio for split, ratio in ratios.items()}
    current_counts = {split: 0 for split in SPLIT_NAMES}
    split_videos = {split: [] for split in SPLIT_NAMES}

    for group_key in group_keys:
        split = max(
            SPLIT_NAMES,
            key=lambda name: (targets[name] - current_counts[name]) / targets[name],
        )
        group = grouped_videos[group_key]
        split_videos[split].extend(group)
        current_counts[split] += len(group)

    return split_videos


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
    """Sprawdza brak duplikatów filmów i grup między splitami."""
    seen: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
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
            group_key = _group_key(video)
            if group_key in seen_groups and seen_groups[group_key] != split:
                raise ValueError(
                    f"Grupa {group_key} występuje zarówno w "
                    f"{seen_groups[group_key]}, jak i w {split}."
                )
            seen_groups[group_key] = split
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

        groups = _split_grouped_videos(
            videos, train_ratio, val_ratio, test_ratio, rng
        )
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
        "split_policy": "grouped_by_ucf101_source_group",
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
