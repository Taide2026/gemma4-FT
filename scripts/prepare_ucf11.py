#!/usr/bin/env python3
"""Convert extracted official UCF11 videos to the repo's messages JSON format."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


LABELS = {
    "basketball": "basketball shooting",
    "biking": "biking",
    "diving": "diving",
    "golf_swing": "golf swinging",
    "horse_riding": "horseback riding",
    "soccer_juggling": "soccer juggling",
    "swing": "swinging",
    "tennis_swing": "tennis swinging",
    "trampoline_jumping": "trampoline jumping",
    "volleyball_spiking": "volleyball spiking",
    "walking": "walking with a dog",
}
EXTENSIONS = {".mpg", ".mpeg", ".avi", ".mp4"}
QUESTION = "Which action is shown? Answer with exactly one of: " + ", ".join(LABELS.values()) + "."


def video_class(path: Path) -> str:
    matches = [part.lower() for part in path.parts if part.lower() in LABELS]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify one UCF11 class from {path}")
    return matches[0]


def video_group(path: Path) -> int:
    # Official updated archive: basketball/v_shooting_01/v_shooting_01_01.mpg
    match = re.search(r"_(\d{2})$", path.parent.name)
    if not match:
        raise ValueError(f"Cannot identify UCF11 group from {path.parent}")
    return int(match.group(1))


def make_sample(path: Path, root: Path, label: str) -> dict:
    return {
        "label": label,
        "messages": [
            {"role": "user", "content": [
                {"type": "video", "video": path.relative_to(root).as_posix()},
                {"type": "text", "text": QUESTION},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": label}]},
        ],
    }


def build_splits(root: Path, seed: int) -> tuple[dict[str, list], dict]:
    videos = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not videos:
        raise ValueError(f"No UCF11 videos found under {root}")

    groups = sorted({video_group(p) for p in videos})
    if len(groups) < 10:
        raise ValueError(f"Expected UCF11's grouped clips; found only {len(groups)} groups")
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_holdout = max(2, round(len(groups) * 0.2))
    val_groups = set(groups[:n_holdout])
    test_groups = set(groups[n_holdout:2 * n_holdout])
    splits = {name: [] for name in ("train", "val", "test")}
    counts = {name: Counter() for name in splits}

    for path in videos:
        group = video_group(path)
        label = LABELS[video_class(path.relative_to(root))]
        split = "val" if group in val_groups else "test" if group in test_groups else "train"
        splits[split].append(make_sample(path, root, label))
        counts[split][label] += 1

    for name in splits:
        missing = set(LABELS.values()) - counts[name].keys()
        if missing:
            raise ValueError(f"{name} has no examples for: {sorted(missing)}")

    metadata = {
        "source": "https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php",
        "seed": seed,
        "groups": {"train": sorted(set(groups) - val_groups - test_groups),
                   "val": sorted(val_groups), "test": sorted(test_groups)},
        "counts": {name: dict(sorted(counts[name].items())) for name in splits},
    }
    return splits, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Extracted UCF11 dataset directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"UCF11 directory not found: {root}")
    splits, metadata = build_splits(root, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, samples in splits.items():
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (args.output_dir / "labels.txt").write_text("\n".join(LABELS.values()) + "\n", encoding="utf-8")
    (args.output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print({name: len(samples) for name, samples in splits.items()})


if __name__ == "__main__":
    main()
