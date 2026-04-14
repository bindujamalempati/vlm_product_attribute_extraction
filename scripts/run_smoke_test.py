from __future__ import annotations
import subprocess
import sys
from pathlib import Path

def run(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    run([sys.executable, "-m", "vlm_attr_extraction.train", "--config", "configs/train.yaml"])
    run([
        sys.executable, "-m", "vlm_attr_extraction.predict",
        "--model-path", "outputs/run_01/best_model.pt",
        "--image-path", "data/sample/images/sample_006.jpg",
        "--text", "Yellow polka dot blouse with short sleeves in good condition"
    ])
    print("Smoke test complete.")

if __name__ == "__main__":
    main()
