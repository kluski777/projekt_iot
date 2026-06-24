from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from tqdm import tqdm


yolo = YOLO('./yolo_best.pt')
out_dir = Path("dataset/plates_data_cropped_2")


for img_path in tqdm(Path("dataset/plates_data").rglob("*")):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        continue
    try:
        results = yolo(str(img_path), verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0 and boxes[0].conf.item() > 0.5:
            box = boxes[0].xyxy[0].cpu().numpy()
            img = Image.open(img_path)
            crop = img.crop(box)
            # proporcje tablicy (szerszy niż wysoki)
            w, h = crop.size
            if w > h and w / h > 1.5:
                save_path = out_dir / img_path.parent.name / img_path.name
                save_path.parent.mkdir(parents=True, exist_ok=True)
                crop.save(save_path)
            else:
                print(f"Złe proporcje, pomijam: {img_path.name} ({w}x{h})")
        else:
            print(f"Brak detekcji, pomijam: {img_path.name}")
    except Exception as e:
        print(f"Błąd: {img_path.name} — {e}")