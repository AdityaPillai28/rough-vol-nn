"""Record assets/demo.gif: drive the Streamlit app's sliders with Playwright and capture
frames. Uses an installed Chromium-family browser (Edge or Chrome) via ``channel``, so no
browser download is needed.

Usage: python scripts/record_demo.py [--channel msedge] [--out assets/demo.gif]
Requires: pip install -e ".[app,demo]"
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765

# (slider index in the sidebar, key, presses): H, eta, rho, xi0 are sliders 0..3.
SCRIPT = [
    (0, "ArrowRight", 24),
    (0, "ArrowLeft", 34),
    (2, "ArrowRight", 30),
    (1, "ArrowRight", 22),
    (2, "ArrowLeft", 30),
    (0, "ArrowRight", 10),
]


def wait_for_server(url: str, timeout: float = 90.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("streamlit did not start")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--out", default=str(ROOT / "assets" / "demo.gif"))
    ap.add_argument("--every", type=int, default=2, help="capture a frame every N key presses")
    a = ap.parse_args()

    from playwright.sync_api import sync_playwright

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app" / "streamlit_app.py"),
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    frames: list[Image.Image] = []
    try:
        wait_for_server(f"http://localhost:{PORT}")
        with sync_playwright() as p:
            browser = p.chromium.launch(channel=a.channel, headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 860}, device_scale_factor=1)
            page.goto(f"http://localhost:{PORT}")
            page.wait_for_function(
                "document.querySelectorAll('.js-plotly-plot').length >= 3", timeout=90000
            )
            page.wait_for_timeout(2500)

            def grab():
                img = Image.open(io.BytesIO(page.screenshot()))
                frames.append(img.convert("RGB").resize((960, 645), Image.LANCZOS))

            grab()
            sliders = page.locator('[data-testid="stSidebar"] [role="slider"]')
            for idx, key, n in SCRIPT:
                sliders.nth(idx).focus()
                for i in range(n):
                    page.keyboard.press(key)
                    if i % a.every == a.every - 1:
                        page.wait_for_timeout(350)
                        grab()
                page.wait_for_timeout(500)
                grab()
            browser.close()
    finally:
        proc.terminate()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=140, loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
