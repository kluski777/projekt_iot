from pathlib import Path
from PIL import Image

removed = 0
for p in Path("plates_data").rglob("*.jpg"):
    try:
        Image.open(p).verify()
    except Exception:
        print(f"Removing: {p}")
        p.unlink()
        removed += 1

print(f"Done. Removed {removed} corrupt file(s).")
