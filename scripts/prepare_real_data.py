from __future__ import annotations
from pathlib import Path
import json
import random

def main():
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # This is a project scaffold hook.
    # In a real extension, map DeepFashion2/ABO metadata into this unified JSONL format:
    # {
    #   "id": "...",
    #   "image": "relative/path/to/image.jpg",
    #   "title": "...",
    #   "description": "...",
    #   "attributes": { "category": "...", "primary_color": "...", ... }
    # }
    note = {
        "status": "placeholder_preprocessor",
        "message": "Add dataset-specific parsers here for DeepFashion2 and ABO."
    }
    with open(out_dir / "PREPARE_REAL_DATA_STATUS.json", "w", encoding="utf-8") as f:
        json.dump(note, f, indent=2)
    print("Prepared scaffold for real dataset conversion.")

if __name__ == "__main__":
    main()
