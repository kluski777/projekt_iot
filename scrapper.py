# pyright: basic
import os
import re
import time
import random
from playwright.sync_api import sync_playwright  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from codes import regular, police, customs  # type: ignore

BASE_URL = "https://platesmania.com"
GALLERY_BASE = "https://platesmania.com/pl/gallery.php"
CDP_URL = "http://localhost:9222"

ALL_CODES = sorted(regular | police | customs)


def gallery_url(code, start):
    if start == 0:
        return f"{GALLERY_BASE}?gal=pl&nomer={code}"
    return f"{GALLERY_BASE}?&nomer={code}&start={start}"


def fetch_html(page, url):
    while True:
        try:
            page.goto(url, wait_until="load", timeout=5000)
        except Exception as e:
            print(f"  [retry goto] {e} — waiting 1s")
            time.sleep(1)
            continue
        try:
            page.wait_for_selector("div.panel.panel-grey", timeout=2000)
            return page.content()
        except Exception:
            raise ValueError('Page content is empty')


def parse_gallery(html):
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for panel in soup.select("div.panel.panel-grey"):
        try:
            body = panel.find("div", class_="panel-body")  # type: ignore
            car_img_tag = body.select_one("div.row > a > img")  # type: ignore
            car_img_url = car_img_tag["src"]  # type: ignore
            model_tag = body.select_one("h4.text-center > a")  # type: ignore
            car_model = model_tag.get_text(strip=True)  # type: ignore
            detail_url = BASE_URL + model_tag["href"]  # type: ignore
            plate_img_tag = body.select_one('img[src*="/inf/"]')  # type: ignore
            plate_img_url = plate_img_tag["src"]  # type: ignore
            plate_number = plate_img_tag["alt"]  # type: ignore
        except (TypeError, KeyError):
            continue

        results.append({
            "plate_number": plate_number,
            "plate_img_url": plate_img_url,
            "car_img_url": car_img_url,
            "car_model": car_model,
            "detail_url": detail_url,
        })

    return results


def scrape_code(page, code, output_dir):
    code_dir = os.path.join(output_dir, code)
    os.makedirs(code_dir, exist_ok=True)
    start = 0

    while True:
        url = gallery_url(code, start)
        print(f"  [{code}] page start={start} {url}")
        try:
            html = fetch_html(page, url)
        except Exception:
            print(f"  [{code}] no panels visible, skipping to next code")
            break
        plates = parse_gallery(html)

        if not plates or start >= 100:
            print(f"  [{code}] done (start={start}), next code")
            break

        for plate in plates:
            print(f"    {plate['plate_number']:<15} {plate['car_model']}")
            model = plate["car_model"] or "unknown unknown"
            model_slug = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").lower()
            plate_slug = re.sub(r"\s+", "", plate["plate_number"])
            ext = os.path.splitext(plate["car_img_url"])[1] or ".jpg"
            dest = os.path.join(code_dir, f"{model_slug}_{plate_slug}{ext}")
            if not os.path.exists(dest):
                while True:
                    try:
                        data = page.request.get(plate["car_img_url"]).body()
                        with open(dest, "wb") as f:
                            f.write(data)
                        break
                    except Exception as e:
                        print(f"    [img retry] {e} — waiting 3s")
                        time.sleep(3)

        start += 1


def scrape(output_dir="plates_data"):
    os.makedirs(output_dir, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.connect_over_cdp(CDP_URL)
        ctx = browser.contexts[0]
        page = ctx.new_page()

        for code in ALL_CODES[424:]:
            print(f"\n[code {code}]")
            scrape_code(page, code, output_dir)

        page.close()


if __name__ == "__main__":
    scrape()