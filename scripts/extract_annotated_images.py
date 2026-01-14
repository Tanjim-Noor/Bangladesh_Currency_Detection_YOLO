"""
Extract base64-encoded `annotated_image` PNGs from JSON responses and save
them into `DISCUSSION AND SCREENSHOTS/api responses/Annotated Images from API response`.

Usage: python scripts/extract_annotated_images.py
"""
from pathlib import Path
import json
import base64
import sys

ROOT = Path(__file__).resolve().parents[1]
API_RESPONSES_DIR = ROOT / "DISCUSSION AND SCREENSHOTS" / "api responses"
OUTPUT_DIR = API_RESPONSES_DIR / "Annotated Images from API response"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

found = 0
for p in sorted(API_RESPONSES_DIR.glob("*.json")):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Skipping {p.name}: failed to parse JSON: {e}")
        continue

    b64 = data.get("annotated_image")
    if not b64:
        # try nested or alternative keys (be liberal)
        for alt in ("annotatedImage", "annotated_image_base64", "image_annotated"):
            b64 = data.get(alt)
            if b64:
                break

    if not b64:
        # nothing to do for this file
        continue

    # strip data URI prefix if present
    if isinstance(b64, str) and b64.startswith("data:"):
        try:
            b64 = b64.split(",", 1)[1]
        except Exception:
            pass

    try:
        img_bytes = base64.b64decode(b64)
    except Exception as e:
        print(f"Failed to decode base64 image in {p.name}: {e}")
        continue

    out_path = OUTPUT_DIR / (p.stem + ".png")
    with open(out_path, "wb") as fh:
        fh.write(img_bytes)

    print(f"Wrote: {out_path}")
    found += 1

print(f"Done. Extracted {found} image(s) to: {OUTPUT_DIR}")
if found == 0:
    sys.exit(1)

