from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.ops import roi_align
import timm
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from ultralytics import YOLO as UltralyticsYOLO
import matplotlib.pyplot as plt
import wandb
from lightning.pytorch.loggers import WandbLogger


torch.set_float32_matmul_precision('high')

CHARS = 'ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789'
BLANK = len(CHARS)  # index 36 — blank / padding
NUM_CLASSES = len(CHARS) + 1
MAX_LEN = 8
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}

_TRAIN_TRANSFORM = T.Compose([
    T.Resize((64, 224)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_VAL_TRANSFORM = T.Compose([
    T.Resize((64, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PlatesDataset(Dataset):
    def __init__(self, samples: list[tuple[str, str]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def _encode(self, label: str) -> torch.Tensor:
        seq = [CHAR2IDX.get(c, BLANK) for c in label[:MAX_LEN]]
        seq += [BLANK] * (MAX_LEN - len(seq))
        return torch.tensor(seq, dtype=torch.long)

    def __getitem__(self, idx):
        for attempt in range(5):  # max 5 prób
            try:
                path, label_str = self.samples[idx]
                img = Image.open(path).convert("RGB")
                img.load()
                if self.transform:
                    img = self.transform(img)
                return img, self._encode(label_str)
            except Exception:
                print(f'Uszkodzony: {path}')
                idx = torch.randint(len(self), (1,)).item()
        # Fallback — czarny obraz
        return torch.zeros(3, 64, 224), self._encode("")


class PlatesDataModule(L.LightningDataModule):
    def __init__(self, root: str = "dataset/plates_data_cropped_3", batch_size: int = 32,
                 val_split: float = 0.1, num_workers: int = 16):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        all_samples = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for p in sorted(Path(self.root).rglob(ext)):
                try:
                    img = Image.open(p)
                    img.load()
                    img.close()
                    all_samples.append((str(p), p.stem.split("_")[-1].upper()))
                except Exception:
                    print(f"Uszkodzony, pomijam: {p}")

        # all_samples = sorted(
        #     (str(p), p.stem.split("_")[-1].upper())
        #     for p in Path(self.root).rglob("*.jpg")
        # )

        perm = torch.randperm(len(all_samples), generator=torch.Generator().manual_seed(42)).tolist()
        n_test = int(len(all_samples) * 0.1)
        n_val = int(len(all_samples) * 0.1)

        self.train_ds = PlatesDataset([all_samples[i] for i in perm[n_test + n_val:]], _TRAIN_TRANSFORM)
        self.val_ds = PlatesDataset([all_samples[i] for i in perm[n_test:n_test + n_val]], _VAL_TRANSFORM)
        self.test_ds = PlatesDataset([all_samples[i] for i in perm[:n_test]], _VAL_TRANSFORM)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, pin_memory=True)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


class PlateClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, max_len: int = MAX_LEN):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=True,
            num_classes=1,
            drop_rate=0.2,
        )
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(1024, max_len * num_classes)
        self.max_len = max_len
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.dropout(x)
        x = self.head(x)
        return x.view(-1, self.max_len, self.num_classes)


def _decode(t):
    return "".join(IDX2CHAR.get(i.item(), "") for i in t if i.item() != BLANK)


class PlateModule(L.LightningModule):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4, unfreeze_epoch: int = 5):
        super().__init__()
        self.model = PlateClassifier()

        _yolo_wrapper = UltralyticsYOLO('./yolo_best.pt')
        self.yolo = _yolo_wrapper.model.float()
        for m in self.yolo.modules():
            if hasattr(m, 'export'):
                m.export = True

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.lr = lr
        self.weight_decay = weight_decay
        self.unfreeze_epoch = unfreeze_epoch
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        for p in self.model.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):                                    # [1, 3, 960, 960] float32 NCHW [0-255]
        det  = self.yolo(x / 255.0)                         # [1, 300, 6]  x1 y1 x2 y2 conf cls
        conf = det[:, :, 4:5]
        best_idx = conf.argmax(dim=1, keepdim=True)
        best_det = torch.gather(det, 1, best_idx.expand(-1, -1, 6)).squeeze(1)

        x1, y1 = best_det[:, 0:1], best_det[:, 1:2]
        x2, y2 = best_det[:, 2:3], best_det[:, 3:4]
        best_conf = best_det[:, 4:5]  # ← confidence

        # crop
        batch_idx = torch.zeros_like(x1)
        rois    = torch.cat([batch_idx, x1, y1, x2, y2]).unsqueeze(0)   # [1, 5]
        cropped = roi_align(x, rois, output_size=(64, 224), spatial_scale=1.0) # [1, 3, 64, 224]

        # OCR
        normalized = (cropped / 255.0 - self.mean) / self.std
        chars    = self.model(normalized)                    # [1, 8, 37]
        best_box = torch.cat([x1, y1, x2, y2, best_conf], dim=1)  # [1, 5] zamiast [1, 4]

        return best_box, chars

    def on_train_epoch_start(self):
        if self.current_epoch == self.unfreeze_epoch:
            for p in self.model.backbone.parameters():
                p.requires_grad = True

    def _step(self, batch):
        imgs, labels = batch
        logits = self.model(imgs)
        loss = self.loss_fn(logits.permute(0, 2, 1), labels)
        preds = logits.argmax(-1)
        char_acc  = (preds == labels).float().mean()
        plate_acc = (preds == labels).all(dim=-1).float().mean()
        return loss, char_acc, plate_acc

    def training_step(self, batch, _):
        loss, char_acc, plate_acc = self._step(batch)
        self.log_dict({"train/loss": loss, "train/char_acc": char_acc, "train/plate_acc": plate_acc},
                      prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        loss, char_acc, plate_acc = self._step(batch)
        self.log_dict({"val/loss": loss, "val/char_acc": char_acc, "val/plate_acc": plate_acc},
                      prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, batch, _):
        loss, char_acc, plate_acc = self._step(batch)
        self.log_dict({"test/loss": loss, "test/char_acc": char_acc, "test/plate_acc": plate_acc})

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.estimated_stepping_batches)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


if __name__ == '__main__':

    model = PlateClassifier()
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Output shape: {model(torch.zeros(1, 3, 64, 224)).shape}")

    IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}

    _dm = PlatesDataModule(batch_size=16)
    _dm.setup()
    _imgs, _labels = next(iter(_dm.val_dataloader()))

    fig, axes = plt.subplots(4, 2, figsize=(10, 20))
    for ax, img, lbl in zip(axes.flatten(), _imgs, _labels):
        rgb = (img.permute(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).numpy()
        ax.imshow(rgb)
        ax.set_title(_decode(lbl), fontsize=14)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


    wandb.login()
    project = "IoT-plates-recognition"

    wandb_logger = WandbLogger(
        project=project,
        name="MobileNetV3-2.5M",
        log_model=True
    )

    dm = PlatesDataModule(batch_size=2048)
    module = PlateModule(lr=1e-3, unfreeze_epoch=5)  # = 1.2e-2, lr=3e-3 * 4

    trainer = L.Trainer(
        max_epochs=200,
        accelerator="auto",
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(
                monitor="val/loss",
                patience=30,
                mode='min',
            ),
            ModelCheckpoint(
                monitor='val/loss',
                mode='min',
                save_top_k=1,
                filename='best-{epoch}-{val/loss:.3f}',
            )
        ]
    )

    trainer.fit(module, dm)
    trainer.test(module, dm)
    torch.save(module.model.state_dict(), 'model_4_synth_2.pt')

