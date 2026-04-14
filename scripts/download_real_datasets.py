from __future__ import annotations
from pathlib import Path
import textwrap

README = '''
This project is designed around real public multimodal datasets commonly used for product/fashion attribute learning.

Recommended real datasets:
1. DeepFashion / DeepFashion2
2. Amazon Berkeley Objects (ABO)

Why this script does not auto-download everything:
- Some sources are extremely large.
- Some require agreeing to the dataset terms on the source website.
- Some are better downloaded manually with authentication or browser-based approval.

What to do:
- Create data/raw/deepfashion2 and data/raw/abo
- Download the official archives from the dataset owners
- Unzip them into those folders
- Then run scripts/prepare_real_data.py

Expected final layout:
data/raw/
  deepfashion2/
  abo/
'''
Path("data/raw/DATASET_INSTRUCTIONS.txt").write_text(textwrap.dedent(README).strip() + "\n", encoding="utf-8")
print("Wrote data/raw/DATASET_INSTRUCTIONS.txt")
