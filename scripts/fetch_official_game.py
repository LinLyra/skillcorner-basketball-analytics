from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "datasets.lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, entry: dict) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    actual_hash = sha256(path)
    if actual_size != entry["bytes"]:
        raise ValueError(f"Size mismatch for {path}: {actual_size} != {entry['bytes']}")
    if actual_hash != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual_hash}")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a pinned official SkillCorner game")
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    entries = lock["games"].get(str(args.game_id))
    if entries is None:
        available = ", ".join(sorted(lock["games"]))
        raise ValueError(f"Game {args.game_id} is not locked. Available: {available}")

    commit = lock["source_commit"]
    base_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata-basketball/{commit}"
    destination_dir = PROJECT_ROOT / "data" / "raw" / str(args.game_id)

    for entry in entries:
        filename = Path(entry["path"]).name
        destination = destination_dir / filename
        if not args.verify_only and not destination.exists():
            print(f"Downloading {filename}...")
            download(f"{base_url}/{entry['path']}", destination)
        verify(destination, entry)
        print(f"Verified {filename}")


if __name__ == "__main__":
    main()

