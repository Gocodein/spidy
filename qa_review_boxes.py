"""
qa_review_boxes.py
------------------------------------------------------------
Efficient QA for MegaDetector-generated YOLO labels.

Instead of reviewing every image one by one, this:
  1. Flags boxes that are statistically likely to be wrong
     (low-mid confidence, extreme size, too many boxes per image)
  2. Builds a single browser-viewable HTML contact sheet of
     CROPPED detections (not full images) so you can scan
     dozens per screen and click to reject bad ones
  3. Parses your rejections back into a clean label set

Run locally after setup_tiger_dataset.py has populated
tiger_dataset/images/all and tiger_dataset/labels/all.

Usage:
  python qa_review_boxes.py flag        # Step A: generate flagged crops + HTML sheet
  python qa_review_boxes.py apply       # Step B: after you've edited rejected.txt, clean labels
------------------------------------------------------------
"""

import sys
import json
from pathlib import Path
from PIL import Image

ROOT = Path("tiger_dataset")
IMG_DIR = ROOT / "images" / "all"
LBL_DIR = ROOT / "labels" / "all"
QA_DIR = ROOT / "qa_review"
CROPS_DIR = QA_DIR / "crops"
REJECTED_LIST = QA_DIR / "rejected.txt"   # you fill this in during review

# --- Risk thresholds (tune based on your data) ---
MIN_CONF_TO_TRUST = 0.85       # boxes >= this are auto-approved without review
MIN_AREA_FRACTION = 0.015      # boxes smaller than this % of frame = suspicious
MAX_AREA_FRACTION = 0.85       # boxes larger than this % of frame = suspicious
MAX_BOXES_PER_IMAGE = 3        # more boxes than this in one frame = suspicious


def yolo_line_to_box(line, img_w, img_h):
    parts = line.strip().split()
    cls_id = int(parts[0])
    xc, yc, w, h = map(float, parts[1:5])
    conf = float(parts[5]) if len(parts) > 5 else None  # optional 6th field
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    area_frac = w * h
    return {"cls": cls_id, "box": (x1, y1, x2, y2), "area_frac": area_frac, "conf": conf}


def flag_step():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(exist_ok=True)

    flagged = []  # list of dicts: {crop_path, image_name, line_index, reason}
    label_files = sorted(LBL_DIR.glob("*.txt"))

    for label_path in label_files:
        img_path = IMG_DIR / f"{label_path.stem}.jpg"
        if not img_path.exists():
            continue

        lines = label_path.read_text().strip().splitlines()
        if not lines:
            continue

        try:
            img = Image.open(img_path)
        except Exception:
            continue
        img_w, img_h = img.size

        boxes = [yolo_line_to_box(l, img_w, img_h) for l in lines]

        too_many = len(boxes) > MAX_BOXES_PER_IMAGE

        for i, b in enumerate(boxes):
            reasons = []
            if b["conf"] is not None and b["conf"] < MIN_CONF_TO_TRUST:
                reasons.append(f"low_confidence({b['conf']:.2f})")
            if b["area_frac"] < MIN_AREA_FRACTION:
                reasons.append("too_small")
            if b["area_frac"] > MAX_AREA_FRACTION:
                reasons.append("too_large")
            if too_many:
                reasons.append(f"too_many_boxes({len(boxes)})")

            if not reasons:
                continue  # looks fine, skip review

            # Save a crop with a little padding for context
            x1, y1, x2, y2 = b["box"]
            pad = int(0.15 * max(x2 - x1, y2 - y1, 20))
            crop_box = (max(0, x1 - pad), max(0, y1 - pad),
                        min(img_w, x2 + pad), min(img_h, y2 + pad))
            crop = img.crop(crop_box)
            crop.thumbnail((300, 300))
            crop_name = f"{label_path.stem}_{i}.jpg"
            crop.save(CROPS_DIR / crop_name)

            flagged.append({
                "crop": crop_name,
                "image": img_path.name,
                "label_file": label_path.name,
                "line_index": i,
                "reasons": reasons,
            })

    print(f"Flagged {len(flagged)} boxes out of "
          f"{sum(len(open(f).readlines()) for f in label_files)} total boxes for review.")

    (QA_DIR / "flagged.json").write_text(json.dumps(flagged, indent=2))
    build_html_sheet(flagged)


def build_html_sheet(flagged):
    html_path = QA_DIR / "review.html"
    rows = []
    for idx, item in enumerate(flagged):
        rows.append(f"""
        <div class="card">
          <img src="crops/{item['crop']}">
          <div class="meta">{item['image']}<br>{', '.join(item['reasons'])}</div>
          <label><input type="checkbox" data-idx="{idx}" class="reject"> Reject this box</label>
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ font-family: sans-serif; background:#111; color:#eee; }}
  .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(180px,1fr)); gap:12px; padding:16px; }}
  .card {{ background:#222; padding:6px; border-radius:6px; text-align:center; }}
  .card img {{ width:100%; border-radius:4px; }}
  .meta {{ font-size:11px; margin:6px 0; word-break: break-all; }}
  button {{ margin:16px; padding:10px 16px; font-size:14px; }}
</style></head>
<body>
  <h2>Review {len(flagged)} flagged boxes -- check the ones that are WRONG, then click Export</h2>
  <button onclick="exportRejected()">Export rejected.txt</button>
  <div class="grid">{''.join(rows)}</div>
  <script>
    function exportRejected() {{
      const checked = Array.from(document.querySelectorAll('.reject:checked'))
                            .map(el => el.dataset.idx);
      const blob = new Blob([checked.join('\\n')], {{type: 'text/plain'}});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'rejected.txt';
      a.click();
    }}
  </script>
</body></html>"""
    html_path.write_text(html)
    print(f"Open {html_path} in a browser, check boxes to reject, click 'Export rejected.txt',")
    print(f"then save the downloaded file into {REJECTED_LIST} before running 'apply'.")


def apply_step():
    if not REJECTED_LIST.exists():
        print(f"{REJECTED_LIST} not found -- export it from review.html first.")
        return

    flagged = json.loads((QA_DIR / "flagged.json").read_text())
    rejected_indices = {int(x) for x in REJECTED_LIST.read_text().split() if x.strip()}

    # Group rejections by label file so we rewrite each file once
    to_remove_by_file = {}
    for idx in rejected_indices:
        item = flagged[idx]
        to_remove_by_file.setdefault(item["label_file"], set()).add(item["line_index"])

    removed_count = 0
    for label_file, line_indices in to_remove_by_file.items():
        path = LBL_DIR / label_file
        lines = path.read_text().strip().splitlines()
        kept = [l for i, l in enumerate(lines) if i not in line_indices]
        removed_count += len(lines) - len(kept)
        if kept:
            path.write_text("\n".join(kept) + "\n")
        else:
            path.unlink()  # no valid boxes left -> remove label file
            img_path = IMG_DIR / f"{path.stem}.jpg"
            # keep the image as a hard-negative/empty example if you want,
            # or uncomment below to remove it entirely from the training set
            # if img_path.exists(): img_path.unlink()

    print(f"Removed {removed_count} bad boxes from {len(to_remove_by_file)} label files.")
    print("Re-run the train/val/test split step before training.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("flag", "apply"):
        print("Usage: python qa_review_boxes.py [flag|apply]")
        sys.exit(1)

    if sys.argv[1] == "flag":
        flag_step()
    else:
        apply_step()
