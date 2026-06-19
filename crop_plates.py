from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

SRC = Path("plates_data")
DST = Path("plates_data_cropped")
MODEL = "best.pt"
CONF = 0.01

yolo = YOLO(MODEL)

images = list(SRC.rglob("*.jpg"))
print(f"Found {len(images)} images")

for src_path in tqdm(images):
    dst_path = DST / src_path.relative_to(SRC)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(src_path).convert("RGB")

    results = yolo(img, conf=CONF, verbose=False)
    cropped = img
    for r in results:
        if len(r.boxes):
            x1, y1, x2, y2 = map(int, r.boxes.xyxy[r.boxes.conf.argmax()].tolist())
            cropped = img.crop((x1, y1, x2, y2))
            break

    cropped.save(dst_path)

print("Done")
