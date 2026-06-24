import glob, random
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import PlateModule, PlatesDataModule
import torchvision.transforms as T


model_name = 'model_4_synth_2.pt'

dm = PlatesDataModule(batch_size=32)
module = PlateModule()
model_state_dict = torch.load(model_name)
module.model.load_state_dict( model_state_dict )


CHARS = 'ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789'
BLANK = len(CHARS)                              # index 36 — blank / padding
NUM_CLASSES = len(CHARS) + 1
MAX_LEN = 8
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}

IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


_TRAIN_TRANSFORM = T.Compose([
    T.Resize((64, 224)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.RandomPerspective(distortion_scale=0.15, p=0.4),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_VAL_TRANSFORM = T.Compose([
    T.Resize((64, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def decode(indices: torch.Tensor) -> str:
    return "".join(IDX2CHAR.get(i.item(), "") for i in indices if i.item() != BLANK)


def show_plates_predictions(model, datamodule, n: int = 30):
    device = next(model.parameters()).device
    model.eval()

    datamodule.setup()
    loader = DataLoader(datamodule.test_ds, batch_size=n, shuffle=True)
    imgs, labels = next(iter(loader))

    with torch.no_grad():
        logits = model(imgs.to(device))        # (n, MAX_LEN, NUM_CLASSES)
    preds = logits.argmax(-1).cpu()            # (n, MAX_LEN)

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.8))
    axes = axes.flatten()

    for i in range(n):
        img = imgs[i].permute(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
        img = img.clamp(0, 1).numpy()

        gt   = decode(labels[i])
        pred = decode(preds[i])
        match = pred == gt

        axes[i].imshow(img)
        axes[i].set_title(f"GT:   {gt}\nPred: {pred}", fontsize=8,
                          color="green" if match else "red")
        axes[i].axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


dm = PlatesDataModule(batch_size=32)
show_plates_predictions(module.model, dm)


# ==================================


def show_cars_predictions():
    random.seed(42)
    _paths = random.sample(
        glob.glob("dataset/plates_data/**/*.jpg", recursive=True) +
        glob.glob("dataset/plates_data/**/*.png", recursive=True) +
        glob.glob("dataset/plates_data/**/*.webp", recursive=True),
        8
    )

    module.model.load_state_dict(torch.load(model_name, map_location='cuda'))
    module.eval()

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for ax, path in zip(axes.flatten(), _paths):
        gt   = Path(path).stem.split("_")[-1].upper()
        img  = Image.open(path).convert("RGB")
        w, h = img.size
        img960 = img.resize((960, 960))

        x = torch.from_numpy(np.array(img960)).permute(2, 0, 1).float().unsqueeze(0)  # [1,3,960,960] [0-255]

        with torch.no_grad():
            best_box, chars = module(x)

        pred = "".join(IDX2CHAR.get(i.item(), "") for i in chars.argmax(-1).squeeze(0) if i.item() != BLANK)

        x1, y1, x2, y2, _ = best_box[0].tolist()
        draw = ImageDraw.Draw(img)
        draw.rectangle([x1 * w/960, y1 * h/960, x2 * w/960, y2 * h/960], outline="red", width=4)

        ax.imshow(img)
        ax.set_title(f"GT: {gt}  |  Pred: {pred}", fontsize=11,
                     color="green" if pred == gt else "red")
        ax.axis("off")

    for ax in axes.flatten()[len(_paths):]:
        ax.axis("off")

    plt.suptitle("Full pipeline: YOLO crop → classifier OCR", fontsize=14)
    plt.tight_layout()
    plt.show()


show_cars_predictions()


# ================================================


def evaluate_test(model, datamodule):
    model.eval()
    device = next(model.parameters()).device
    datamodule.setup()
    loader = DataLoader(datamodule.test_ds, batch_size=64, shuffle=False)

    total_chars = 0
    correct_chars = 0
    total_plates = 0
    correct_plates = 0
    errors = []

    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            preds = logits.argmax(-1).cpu()

            for pred, label in zip(preds, labels):
                total_plates += 1
                pred_text = decode(pred)
                gt_text = decode(label)

                plate_correct = pred_text == gt_text
                if plate_correct:
                    correct_plates += 1
                else:
                    errors.append((gt_text, pred_text))

                for p, g in zip(pred, label):
                    total_chars += 1
                    if p.item() == g.item():
                        correct_chars += 1

    char_acc = correct_chars / total_chars
    plate_acc = correct_plates / total_plates

    print(f"Test set: {total_plates} tablic")
    print(f"Char accuracy:  {char_acc:.1%}")
    print(f"Plate accuracy: {plate_acc:.1%}")
    print(f"Błędne: {len(errors)}")
    print(f"\nPrzykłady błędów:")
    for gt, pred in errors[:20]:
        print(f"  GT: {gt:10s} → Pred: {pred}")


evaluate_test(module.model, dm)



