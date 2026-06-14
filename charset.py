import os
import numpy as np
import cv2
from IPython.display import Image
import random
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
from codes import regular, police, customs
from icecream import ic


@dataclass
class PlateRenderConfig:
    out_h: int = 64
    out_w: int = 256
    grayscale: bool = True
    font_size_range: tuple[int, int] = (140, 144)
    char_spacing_range: tuple[int, int] = (1, 3)
    fonts: list[str] = field(default_factory=list)
    euroband: bool = True  # draw the blue strip on the left
    euroband_prob: float = 0.95
    max_rotation_deg: float = 8.0
    max_perspective: float = 0.11


_FALLBACK_FONTS = [
    './arklatrs-webfont.ttf'
]

def resolve_fonts(user_fonts: list[str]) -> list[str]:
    fonts = [f for f in user_fonts if os.path.exists(f)]
    fonts += [f for f in _FALLBACK_FONTS if os.path.exists(f)]

    seen, out = set(), []
    for f in fonts:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


_EU_BLUE = (0, 51, 153)


def _draw_euroband(draw: ImageDraw.ImageDraw, x0: int, y0: int, w: int, h: int) -> None:
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=_EU_BLUE)
    # twelve stars (dots)
    cx = x0 + w / 2
    cy = y0 + h * 0.38
    r = min(w, h) * 0.26
    for k in range(12):
        ang = 2 * np.pi * k / 12
        sx = cx + r * np.cos(ang)
        sy = cy + r * np.sin(ang)
        draw.ellipse([sx - 1, sy - 1, sx + 1, sy + 1], fill=(255, 204, 0))
    try:
        f = ImageFont.truetype(_FALLBACK_FONTS[-1], int(h * 0.20))
        tb = f.getbbox("PL")
        draw.text((cx - (tb[2] - tb[0]) / 2, y0 + h * 0.72), "PL",
                  font=f, fill=(255, 255, 255))
    except Exception:
        pass


def render_plate(label: str, cfg: PlateRenderConfig, rng) -> np.ndarray:
    font_path = rng.choice(cfg.fonts)
    font_size = rng.integers(*cfg.font_size_range, endpoint=True)
    spacing = rng.integers(*cfg.char_spacing_range, endpoint=True)
    font = ImageFont.truetype(font_path, font_size)
    ascent, descent = font.getmetrics()

    def cw(c: str) -> int:
        if c == ' ':
            return int(font_size * 0.6)
        return font.getbbox(c)[2] + spacing

    text_w = sum(cw(c) for c in label)
    text_h = ascent + descent
    pad_y = rng.integers(25, 30, endpoint=True)

    use_band = cfg.euroband and rng.random() < cfg.euroband_prob
    band_w = int(text_h * 0.9) if use_band else 0
    pad_x = rng.integers(19, 23, endpoint=True)

    W = band_w + text_w + 2 * pad_x
    H = text_h + 2 * pad_y

    # white reflective base with slight tint + grain
    base = rng.integers(125, 252, endpoint=True)
    img = Image.new("RGB", (W, H), (base, base, base))
    draw = ImageDraw.Draw(img)

    if use_band:
        _draw_euroband(draw, 0, 0, band_w, H)

    # black lettering with mild ink variation
    fg = rng.integers(5, 45, endpoint=True)
    x = band_w + pad_x
    y = pad_y + rng.integers(-2, 2, endpoint=True)
    for c in label:
        if c == " ":
            x += cw(c)
            continue
        # jy = rng.integers(-1, 1, endpoint=True)
        draw.text((x, y), c, font=font, fill=(fg, fg, fg))
        x += cw(c)

    # czarna obwódka blachy
    if rng.random() < 0.98:
        bw = int(rng.integers(3, 4, endpoint=True))      # grubość ramki
        inset = int(rng.integers(1, 2, endpoint=True))   # odstęp od krawędzi
        frame_fg = int(rng.integers(35, 40))             # ciemna, lekko zmienna

        try:
            draw.rounded_rectangle(
                [inset, inset, W - 1 - inset, H - 1 - inset],
                radius=int(rng.integers(2, 6, endpoint=True)),
                outline=(frame_fg, frame_fg, frame_fg), width=bw)
        except AttributeError:
            draw.rectangle(
                [inset, inset, W - 1 - inset, H - 1 - inset],
                outline=(frame_fg, frame_fg, frame_fg), width=bw)

    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # paint grain / reflective speckle
    noise = int(rng.integers(2, 8, endpoint=True))
    gray = np.clip(gray.astype(np.int16) +
                   rng.integers(-noise, noise + 1, gray.shape), 0, 255
                   ).astype(np.uint8)
    return gray


def aug_rotate(img: np.ndarray, cfg: PlateRenderConfig,
               rng: random.Random) -> np.ndarray:
    h, w = img.shape
    angle = rng.uniform(-cfg.max_rotation_deg, cfg.max_rotation_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = int(np.median(img))
    return cv2.warpAffine(img, M, (w, h), borderValue=border,
                          flags=cv2.INTER_LINEAR)


def aug_perspective(img: np.ndarray, cfg: PlateRenderConfig,
                    rng: random.Random) -> np.ndarray:
    h, w = img.shape
    m = rng.uniform(0.04, cfg.max_perspective)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    d = lambda: rng.uniform(-m, m)
    dst = np.float32([[w * d(), h * d()], [w * (1 + d()), h * d()],
                      [w * (1 + d()), h * (1 + d())], [w * d(), h * (1 + d())]])
    M = cv2.getPerspectiveTransform(src, dst)
    border = int(np.median(img))
    return cv2.warpPerspective(img, M, (w, h), borderValue=border,
                               flags=cv2.INTER_LINEAR)


def aug_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    r = rng.random()
    if r < 0.4:
        k = int(rng.choice([3, 5]))
        return cv2.GaussianBlur(img, (k, k), 0)
    if r < 0.6:
        k = rng.choice([5, 7, 9])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1.0 / k
        if rng.random() < 0.5:
            kern = kern.T
        return cv2.filter2D(img, -1, kern)
    return img


def aug_noise(img: np.ndarray, rng) -> np.ndarray:
    if rng.random() < 0.6:
        sigma = rng.uniform(2, 16)
        img = np.clip(img.astype(np.float32) +
                      np.random.randn(*img.shape) * sigma, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        q = rng.integers(20, 60)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return img


def aug_photometric(img: np.ndarray, rng) -> np.ndarray:
    alpha = rng.uniform(0.65, 1.35)
    beta = rng.uniform(-35, 35)
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        h, w = img.shape
        gx = np.linspace(rng.uniform(0.55, 1.0), rng.uniform(0.55, 1.0), w)
        gy = np.linspace(rng.uniform(0.65, 1.0), 1.0, h)
        img = np.clip(img.astype(np.float32) * np.outer(gy, gx), 0, 255
                      ).astype(np.uint8)
    return img


def augment(img: np.ndarray, cfg: PlateRenderConfig, rng) -> np.ndarray:
    if rng.random() < 0.85:
        img = aug_rotate(img, cfg, rng)
    if rng.random() < 0.85:
        img = aug_perspective(img, cfg, rng)
    img = aug_photometric(img, rng)
    img = aug_blur(img, rng)
    img = aug_noise(img, rng)
    return img


def finalize(img: np.ndarray, cfg: PlateRenderConfig) -> np.ndarray:
    return cv2.resize(img, (cfg.out_w, cfg.out_h), interpolation=cv2.INTER_AREA)


def synth_plates(chars, batch, rng) -> str:
    all_plates_prefixes = np.array([regular, police, customs], dtype=np.object_)
    a, b, c = rng.choice(all_plates_prefixes, 3, p=[0.9, 0.07, 0.03])
    prefix = np.array(list(a | b | c))
    prefix = rng.choice(prefix, batch, replace=True)

    chars = np.array([*chars])
    suffix = rng.choice(chars, (len(prefix), 5), replace=True)
    n = suffix.shape[1]
    suffix = np.ascontiguousarray(suffix).view(f'<U{n}').ravel()

    return np.char.add(np.char.add(prefix, ' '), suffix)


if __name__ == '__main__':
    rng = np.random.default_rng()
    CHARS = 'ACEFGHJKLMNPRSTUVWXY012345678901234567890123456789'  # bez indywidualnych
    batch_size = 64

    plates = synth_plates(CHARS, batch_size, rng)
    plate = plates[0]

    cfg = PlateRenderConfig(fonts=resolve_fonts([]))
    img = render_plate(plate, cfg, rng)
    img = augment(img, cfg, rng)
    img = finalize(img, cfg)
    cv2.imwrite("plate_sample.png", img)
    ic(plate)