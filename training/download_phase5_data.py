#!/usr/bin/env python
"""
DETECTOR AI — Phase 5: Expanded Species & Human Data Download
===============================================================
Downloads data from 3 sources to expand from 4 to 9 classes + color morphs.

Sources:
  1. WCS Camera Traps  — jaguar, human, cheetah (with bounding boxes)
  2. iNaturalist API   — cheetah, snow_leopard, sloth_bear, melanistic leopard, white tiger
  3. Open Images V7    — snow_leopard supplement

Class mapping (9 classes):
  0: bengal_tiger       (existing + white tiger morph)
  1: asian_elephant     (existing)
  2: leopard            (existing + melanistic morph)
  3: rhinoceros         (existing)
  4: person             (NEW)
  5: cheetah            (NEW)
  6: jaguar             (NEW)
  7: snow_leopard       (NEW)
  8: sloth_bear         (NEW)

Usage:
    python training/download_phase5_data.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import requests
from tqdm import tqdm

# ── Paths ──
BASE_DIR = Path(r"D:\Spidy\Detector\multispecies_dataset")
IMAGES_DIR = BASE_DIR / "images"
LABELS_DIR = BASE_DIR / "labels"
RAW_DIR = BASE_DIR / "raw_phase5"
WCS_JSON = Path(r"D:\Spidy\Detector\tiger_dataset\raw\wcs_bbox_extracted\wcs_20220205_bboxes_with_classes.json")
LILA_BASE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/wcs-unzipped/"

# ── Class mapping ──
CLASS_MAP = {
    0: "bengal_tiger",
    1: "asian_elephant",
    2: "leopard",
    3: "rhinoceros",
    4: "person",
    5: "cheetah",
    6: "jaguar",
    7: "snow_leopard",
    8: "sloth_bear",
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_MAP.items()}

# WCS category IDs → our class IDs
# NOTE: WCS human images (cat_id=75) return 404 — removed from public blob for privacy
WCS_CATEGORY_MAP = {
    24: 6,    # panthera onca → jaguar
    122: 5,   # acinonyx jubatus → cheetah
}

# iNaturalist targets
INAT_TARGETS = {
    "cheetah":           {"taxon_id": 41955, "target": 300, "class_id": 5, "query_params": {}},
    "snow_leopard":      {"taxon_id": 41968, "target": 250, "class_id": 7, "query_params": {}},
    "sloth_bear":        {"taxon_id": 41651, "target": 300, "class_id": 8, "query_params": {}},
    "melanistic_leopard": {"taxon_id": 41963, "target": 100, "class_id": 2, "query_params": {"q": "melanistic"}},
    "white_tiger":       {"taxon_id": 41966, "target": 80,  "class_id": 0, "query_params": {"q": "white"}},
}

# ── Session ──
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "DetectorAI/2.0 (wildlife-research)"})


# =====================================================================
#  PART 1: WCS Camera Traps Download (with bounding boxes)
# =====================================================================

def download_wcs_data(max_jaguar: int = 500, max_cheetah: int = 50):
    """Download jaguar and cheetah from WCS with COCO bboxes."""
    print("\n" + "=" * 60)
    print("  PART 1: WCS Camera Traps — Jaguar, Cheetah")
    print("=" * 60)

    print(f"\nLoading WCS metadata from {WCS_JSON.name}...")
    with open(WCS_JSON, "r", encoding="utf-8") as f:
        wcs = json.load(f)

    # Build lookups
    img_lookup = {img["id"]: img for img in wcs["images"]}

    # Group annotations by WCS category
    caps = {24: max_jaguar, 122: max_cheetah}
    anns_by_cat: dict[int, list] = defaultdict(list)
    for ann in wcs["annotations"]:
        cid = ann["category_id"]
        if cid in caps:
            anns_by_cat[cid].append(ann)

    # Collect unique images per category
    wcs_downloads = []  # (url, local_path, class_id, bboxes)
    total_new = 0

    for wcs_cid, our_cid in WCS_CATEGORY_MAP.items():
        species_name = CLASS_MAP[our_cid]
        anns = anns_by_cat.get(wcs_cid, [])
        cap = caps[wcs_cid]

        # Group by image
        img_anns: dict[str, list] = defaultdict(list)
        for ann in anns:
            img_id = ann["image_id"]
            img_anns[img_id].append(ann)

        # Limit
        selected_imgs = list(img_anns.keys())[:cap]

        print(f"\n  {species_name} (wcs_cat={wcs_cid}): {len(selected_imgs)} images (cap={cap})")

        for img_id in selected_imgs:
            img_meta = img_lookup.get(img_id)
            if not img_meta:
                continue
            fname = img_meta.get("file_name", "")
            url = LILA_BASE_URL + fname
            local_name = f"wcs_{species_name}_{img_id}.jpg"
            local_path = RAW_DIR / species_name / local_name

            # Collect bboxes (COCO format: [x, y, w, h] in pixels)
            bboxes = []
            img_w = img_meta.get("width", 0)
            img_h = img_meta.get("height", 0)
            for ann in img_anns[img_id]:
                bbox = ann.get("bbox", [])
                if len(bbox) == 4 and img_w > 0 and img_h > 0:
                    x, y, w, h = bbox
                    # COCO → YOLO normalized
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    # Clamp
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))
                    bboxes.append((our_cid, cx, cy, nw, nh))

            wcs_downloads.append((url, local_path, our_cid, bboxes, img_w, img_h))
            total_new += 1

    print(f"\n  Total WCS images to download: {total_new}")

    # Download
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for species in CLASS_MAP.values():
        (RAW_DIR / species).mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = 0

    def _dl(item):
        url, path, cid, bboxes, w, h = item
        if path.exists() and path.stat().st_size > 1000:
            return True, item
        try:
            r = SESSION.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(r.content)
                return True, item
        except Exception:
            pass
        return False, item

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_dl, item) for item in wcs_downloads]
        for f in tqdm(as_completed(futures), total=len(futures), desc="  WCS download"):
            ok, item = f.result()
            if ok:
                downloaded += 1
            else:
                failed += 1

    print(f"  Downloaded: {downloaded}, Failed: {failed}")
    return wcs_downloads


# =====================================================================
#  PART 2: iNaturalist Download (auto-annotate with YOLO)
# =====================================================================

def fetch_inat_urls(taxon_id: int, target: int, query_params: dict = None) -> list[str]:
    """Fetch photo URLs from iNaturalist API."""
    urls = []
    page = 1
    per_page = min(target, 200)

    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
        "photos": "true",
        "per_page": per_page,
        "order": "desc",
        "order_by": "votes",
    }
    # For white tiger, also include 'needs_id' quality to get more results
    if query_params and query_params.get("q") == "white":
        params["quality_grade"] = "any"
        params["identifications"] = "most_agree"

    if query_params:
        params.update(query_params)

    while len(urls) < target and page <= 20:
        params["page"] = page
        try:
            r = SESSION.get("https://api.inaturalist.org/v1/observations",
                           params=params, timeout=15)
            results = r.json().get("results", [])
            if not results:
                break
            for obs in results:
                for photo in obs.get("photos", []):
                    url = photo.get("url", "")
                    if url:
                        # Get medium-size image
                        url = url.replace("square", "medium").replace("/square.", "/medium.")
                        urls.append(url)
                        if len(urls) >= target:
                            break
                if len(urls) >= target:
                    break
        except Exception as e:
            print(f"    API error page {page}: {e}")
            break
        page += 1
        time.sleep(1.0)

    return urls[:target]


def download_inat_data():
    """Download images from iNaturalist for new species and color morphs."""
    print("\n" + "=" * 60)
    print("  PART 2: iNaturalist — New Species & Color Morphs")
    print("=" * 60)

    all_inat = {}  # species_key → [(local_path, class_id), ...]

    for species_key, cfg in INAT_TARGETS.items():
        taxon_id = cfg["taxon_id"]
        target = cfg["target"]
        class_id = cfg["class_id"]
        class_name = CLASS_MAP[class_id]
        qp = cfg.get("query_params", {})

        print(f"\n  {species_key} (taxon_id={taxon_id}, class={class_name})")

        # Check existing
        species_dir = RAW_DIR / class_name
        species_dir.mkdir(parents=True, exist_ok=True)
        existing = len([f for f in species_dir.glob("inat_*") if f.stat().st_size > 1000])
        need = max(0, target - existing)
        print(f"    Target: {target} | Existing: {existing} | Need: {need}")

        if need == 0:
            print("    Skipping (already have enough)")
            continue

        # Fetch URLs
        urls = fetch_inat_urls(taxon_id, need, qp)
        print(f"    Found {len(urls)} photo URLs")

        # Download
        downloaded_paths = []
        prefix = f"inat_{species_key}"

        def _dl_inat(args):
            idx, url = args
            fname = f"{prefix}_{idx:04d}.jpg"
            path = species_dir / fname
            if path.exists() and path.stat().st_size > 1000:
                return path
            try:
                r = SESSION.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    path.write_bytes(r.content)
                    return path
            except Exception:
                pass
            return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_dl_inat, (i, u)) for i, u in enumerate(urls)]
            for f in tqdm(as_completed(futures), total=len(futures),
                         desc=f"    {species_key}"):
                path = f.result()
                if path:
                    downloaded_paths.append((path, class_id))

        print(f"    Downloaded: {len(downloaded_paths)}/{len(urls)}")
        all_inat[species_key] = downloaded_paths

    return all_inat


# =====================================================================
#  PART 3: Auto-Annotate iNaturalist images with YOLOv8
# =====================================================================

def auto_annotate_inat(inat_data: dict):
    """Run YOLOv8 on iNat images to generate bounding boxes."""
    print("\n" + "=" * 60)
    print("  PART 3: Auto-Annotate iNaturalist Images with YOLOv8")
    print("=" * 60)

    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # COCO pretrained for detection

    # COCO animal class IDs
    coco_animals = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24}

    annotated = 0
    total = 0

    for species_key, items in inat_data.items():
        if not items:
            continue
        class_id = items[0][1]
        class_name = CLASS_MAP[class_id]

        print(f"\n  Annotating {species_key} ({len(items)} images) → class {class_id} ({class_name})")

        for img_path, cid in tqdm(items, desc=f"    {species_key}"):
            total += 1
            label_path = img_path.with_suffix(".txt")

            if label_path.exists():
                annotated += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Run YOLO
            results = model(img, verbose=False)
            boxes = results[0].boxes

            labels = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id in coco_animals and conf > 0.25:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        labels.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Fallback: full image as bbox (common for close-up wildlife photos)
            if not labels:
                labels.append(f"{cid} 0.500000 0.500000 0.900000 0.900000")

            label_path.write_text("\n".join(labels))
            annotated += 1

    print(f"\n  Total annotated: {annotated}/{total}")


# =====================================================================
#  PART 4: Open Images V7 — Snow Leopard Supplement
# =====================================================================

def download_openimages_snow_leopard(target: int = 150):
    """Download snow leopard images from Open Images V7."""
    print("\n" + "=" * 60)
    print("  PART 4: Open Images V7 — Snow Leopard Supplement")
    print("=" * 60)

    # Open Images V7 class MID for Snow Leopard: /m/0dftk
    # Bbox annotations CSV: https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-bbox.csv
    # Image IDs CSV: https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv

    # Try downloading the class descriptions to find Snow Leopard MID
    class_desc_url = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
    print("  Fetching Open Images class list...")

    try:
        r = SESSION.get(class_desc_url, timeout=30)
        r.raise_for_status()

        snow_leopard_mid = None
        leopard_mid = None
        for line in r.text.strip().split("\n"):
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                mid, label = parts
                label_lower = label.lower().strip()
                if "snow leopard" in label_lower:
                    snow_leopard_mid = mid
                    print(f"  Found Snow Leopard: MID={mid}")
                if label_lower == "leopard":
                    leopard_mid = mid

        if not snow_leopard_mid:
            print("  Snow Leopard not found in Open Images boxable classes.")
            if leopard_mid:
                print(f"  Found generic 'Leopard' (MID={leopard_mid}), but skipping to avoid duplicates.")
            print("  Will rely on iNaturalist data for snow leopard.")
            return

        # Download bbox annotations (train set has the most data)
        print("  Downloading train bbox annotations (large file, may take a moment)...")
        bbox_url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
        r = SESSION.get(bbox_url, timeout=120, stream=True)

        # Parse for snow leopard entries
        snow_leopard_entries = []
        first_line = True
        for line in r.iter_lines(decode_unicode=True):
            if first_line:
                first_line = False
                continue
            if snow_leopard_mid in line:
                parts = line.split(",")
                if len(parts) >= 7:
                    image_id = parts[0]
                    x_min = float(parts[4])
                    x_max = float(parts[5])
                    y_min = float(parts[6])
                    y_max = float(parts[7])
                    snow_leopard_entries.append({
                        "image_id": image_id,
                        "bbox": (x_min, y_min, x_max, y_max),
                    })

        print(f"  Found {len(snow_leopard_entries)} snow leopard annotations in Open Images")

        if not snow_leopard_entries:
            print("  No annotations found. Relying on iNaturalist data.")
            return

        # Limit
        random.shuffle(snow_leopard_entries)
        selected = snow_leopard_entries[:target]

        # Download images
        oi_dir = RAW_DIR / "snow_leopard"
        oi_dir.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        for entry in tqdm(selected, desc="  Open Images download"):
            img_id = entry["image_id"]
            # Open Images URL pattern
            url = f"https://s3.amazonaws.com/open-images-dataset/train/{img_id}.jpg"
            path = oi_dir / f"oi_{img_id}.jpg"

            if path.exists() and path.stat().st_size > 1000:
                downloaded += 1
                continue

            try:
                r = SESSION.get(url, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    path.write_bytes(r.content)

                    # Write label (OI bboxes are already normalized 0-1)
                    xmin, ymin, xmax, ymax = entry["bbox"]
                    cx = (xmin + xmax) / 2
                    cy = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin
                    label_path = path.with_suffix(".txt")
                    label_path.write_text(f"7 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    downloaded += 1
            except Exception:
                pass

        print(f"  Downloaded: {downloaded}/{len(selected)}")

    except Exception as e:
        print(f"  Open Images download failed: {e}")
        print("  Will rely on iNaturalist data for snow leopard.")


# =====================================================================
#  PART 5: Merge into Main Dataset & Re-split
# =====================================================================

def merge_and_split(wcs_data: list, inat_data: dict, coco_person_data: list = None):
    """Merge all data into the main dataset and re-split."""
    print("\n" + "=" * 60)
    print("  PART 5: Merge All Data & Re-split Dataset")
    print("=" * 60)

    # ── Step 1: Backup existing data to staging dir ──
    staging_dir = BASE_DIR / "_staging_backup"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    existing_count = 0
    for split in ["train", "val", "test"]:
        img_dir = IMAGES_DIR / split
        lbl_dir = LABELS_DIR / split
        if img_dir.is_dir():
            stg_img = staging_dir / "images" / split
            stg_lbl = staging_dir / "labels" / split
            stg_img.mkdir(parents=True, exist_ok=True)
            stg_lbl.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(img_dir.glob("*.*")):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    shutil.copy2(str(img_path), str(stg_img / img_path.name))
                    lbl_path = lbl_dir / img_path.with_suffix(".txt").name
                    if lbl_path.exists():
                        shutil.copy2(str(lbl_path), str(stg_lbl / lbl_path.name))
                    existing_count += 1

    print(f"  Backed up {existing_count} existing images to staging")

    # ── Step 2: Collect ALL image+label pairs ──
    all_pairs = []  # (img_path, label_path, class_ids_set)

    # Existing data (from staging backup)
    for split in ["train", "val", "test"]:
        stg_img = staging_dir / "images" / split
        stg_lbl = staging_dir / "labels" / split
        if stg_img.is_dir():
            for img_path in sorted(stg_img.glob("*.*")):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    lbl_path = stg_lbl / img_path.with_suffix(".txt").name
                    classes = set()
                    if lbl_path.exists():
                        for line in lbl_path.read_text().strip().split("\n"):
                            if line.strip():
                                classes.add(int(line.split()[0]))
                    all_pairs.append((img_path, lbl_path if lbl_path.exists() else None, classes))

    print(f"  Existing images collected: {len(all_pairs)}")

    # WCS new data
    wcs_added = 0
    for url, local_path, class_id, bboxes, img_w, img_h in wcs_data:
        if not local_path.exists() or local_path.stat().st_size < 1000:
            continue
        # Write label
        label_path = local_path.with_suffix(".txt")
        if bboxes and not label_path.exists():
            lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                     for cid, cx, cy, w, h in bboxes]
            label_path.write_text("\n".join(lines))

        classes = {b[0] for b in bboxes} if bboxes else {class_id}
        all_pairs.append((local_path, label_path if label_path.exists() else None, classes))
        wcs_added += 1

    print(f"  WCS new images: {wcs_added}")

    # iNat new data (already have labels from auto-annotate)
    inat_added = 0
    for species_key, items in inat_data.items():
        for img_path, class_id in items:
            if not img_path.exists() or img_path.stat().st_size < 1000:
                continue
            label_path = img_path.with_suffix(".txt")
            classes = {class_id}
            all_pairs.append((img_path, label_path if label_path.exists() else None, classes))
            inat_added += 1

    print(f"  iNat new images: {inat_added}")

    # COCO person data
    coco_added = 0
    if coco_person_data:
        for img_path, lbl_path, classes in coco_person_data:
            if img_path.exists() and img_path.stat().st_size > 1000:
                all_pairs.append((img_path, lbl_path if lbl_path.exists() else None, classes))
                coco_added += 1

    print(f"  COCO person images: {coco_added}")

    # Open Images snow leopard data
    oi_dir = RAW_DIR / "snow_leopard"
    oi_added = 0
    if oi_dir.is_dir():
        for img_path in oi_dir.glob("oi_*.jpg"):
            if img_path.stat().st_size < 1000:
                continue
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                all_pairs.append((img_path, label_path, {7}))
                oi_added += 1

    print(f"  Open Images new images: {oi_added}")
    print(f"  TOTAL images: {len(all_pairs)}")

    # ── Step 3: Shuffle and split 70/20/10 ──
    random.seed(42)
    random.shuffle(all_pairs)

    n = len(all_pairs)
    train_end = int(n * 0.70)
    val_end = int(n * 0.90)

    splits = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:],
    }

    # ── Step 4: Clear old dirs and copy to final locations ──
    for split in ["train", "val", "test"]:
        for subdir in [IMAGES_DIR / split, LABELS_DIR / split]:
            if subdir.exists():
                shutil.rmtree(subdir)
            subdir.mkdir(parents=True, exist_ok=True)

    copied = 0
    failed = 0
    for split, pairs in splits.items():
        img_out = IMAGES_DIR / split
        lbl_out = LABELS_DIR / split

        for i, (img_path, lbl_path, classes) in enumerate(pairs):
            ext = img_path.suffix
            new_name = f"p5_{split}_{i:05d}{ext}"

            dst_img = img_out / new_name
            dst_lbl = lbl_out / new_name.replace(ext, ".txt")

            try:
                shutil.copy2(str(img_path), str(dst_img))
                if lbl_path and lbl_path.exists():
                    shutil.copy2(str(lbl_path), str(dst_lbl))
                else:
                    dst_lbl.write_text("")
                copied += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Warning: failed to copy {img_path.name}: {e}")

    if failed > 5:
        print(f"  ... and {failed - 5} more copy failures")
    print(f"  Copied: {copied}, Failed: {failed}")

    # Cleanup staging
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
        print("  Staging backup cleaned up")

    # ── Count per class ──
    class_counts = defaultdict(int)
    for split in splits:
        lbl_dir = LABELS_DIR / split
        for lbl_path in lbl_dir.glob("*.txt"):
            content = lbl_path.read_text().strip()
            if content:
                for line in content.split("\n"):
                    parts = line.split()
                    if parts:
                        cid = int(parts[0])
                        class_counts[cid] += 1

    # ── Write data.yaml ──
    yaml_content = f"""# DETECTOR AI — Phase 5 Multi-Species Dataset (9 classes)
# Auto-generated by download_phase5_data.py

path: {BASE_DIR}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.values())}
"""
    yaml_path = BASE_DIR / "data.yaml"
    yaml_path.write_text(yaml_content)

    # ── Summary ──
    print(f"\n  {'=' * 50}")
    print(f"  DATASET SUMMARY")
    print(f"  {'=' * 50}")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    print(f"  Test:  {len(splits['test'])} images")
    print(f"  Total: {sum(len(v) for v in splits.values())} images")
    print(f"\n  Per-class annotation counts:")
    for cid in sorted(class_counts.keys()):
        name = CLASS_MAP.get(cid, f"class_{cid}")
        print(f"    {cid}: {name:<20} {class_counts[cid]:>6} boxes")
    print(f"\n  data.yaml: {yaml_path}")


# =====================================================================
#  PART 1B: COCO 2017 Person Download (replaces WCS human — 404'd)
# =====================================================================

def download_coco_person(target: int = 600):
    """Download person images from COCO 2017 val set with bounding boxes."""
    print("\n" + "=" * 60)
    print("  PART 1B: COCO 2017 — Person Images with Bounding Boxes")
    print("=" * 60)

    person_dir = RAW_DIR / "person"
    person_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = len([f for f in person_dir.glob("coco_*.jpg") if f.stat().st_size > 1000])
    if existing >= target:
        print(f"  Already have {existing} person images, skipping download")
        return _collect_coco_person_data(person_dir)

    # Download COCO 2017 val annotations
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_cache = RAW_DIR / "coco_annotations.zip"

    import zipfile
    import io

    print("  Downloading COCO 2017 annotations...")
    try:
        # Try to download the instances_val2017.json directly from a mirror
        val_ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        if not ann_cache.exists():
            r = SESSION.get(val_ann_url, timeout=120, stream=True)
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            print(f"  Annotations zip size: {total_size / 1024 / 1024:.1f} MB")

            with open(ann_cache, "wb") as f:
                for chunk in tqdm(r.iter_content(chunk_size=8192),
                                  total=total_size // 8192 if total_size else None,
                                  desc="  Downloading annotations"):
                    f.write(chunk)

        # Extract instances_val2017.json
        print("  Extracting annotations...")
        with zipfile.ZipFile(str(ann_cache)) as zf:
            # Find the instances file
            for name in zf.namelist():
                if "instances_val2017" in name:
                    with zf.open(name) as f:
                        coco_data = json.load(f)
                    break
            else:
                print("  ERROR: instances_val2017.json not found in zip")
                return []

    except Exception as e:
        print(f"  Failed to download COCO annotations: {e}")
        print("  Falling back to iNaturalist human-like approach...")
        return []

    # Parse person annotations (COCO person category_id = 1)
    img_lookup = {img["id"]: img for img in coco_data["images"]}

    # Group person annotations by image
    person_img_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        if ann["category_id"] == 1 and not ann.get("iscrowd", 0):
            person_img_anns[ann["image_id"]].append(ann)

    print(f"  Found {len(person_img_anns)} COCO val images with person annotations")

    # Select images (prefer ones with 1-3 people, clear images)
    selected = []
    for img_id, anns in person_img_anns.items():
        if len(anns) <= 5:  # Skip very crowded scenes
            # Filter out tiny annotations
            valid_anns = [a for a in anns if a["bbox"][2] > 30 and a["bbox"][3] > 30]
            if valid_anns:
                selected.append((img_id, valid_anns))

    random.shuffle(selected)
    selected = selected[:target]
    print(f"  Selected {len(selected)} images for download")

    # Download images
    coco_items = []  # Return data for merge
    downloaded = 0

    def _dl_coco(item):
        img_id, anns = item
        img_meta = img_lookup[img_id]
        fname = img_meta["file_name"]
        url = f"http://images.cocodataset.org/val2017/{fname}"
        local_path = person_dir / f"coco_{img_id:012d}.jpg"

        if local_path.exists() and local_path.stat().st_size > 1000:
            return local_path, anns, img_meta

        try:
            r = SESSION.get(url, timeout=15)
            if r.status_code == 200 and len(r.content) > 1000:
                local_path.write_bytes(r.content)
                return local_path, anns, img_meta
        except Exception:
            pass
        return None, None, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_dl_coco, item) for item in selected]
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc="  COCO person download"):
            path, anns, meta = f.result()
            if path:
                # Write YOLO label
                img_w = meta["width"]
                img_h = meta["height"]
                label_path = path.with_suffix(".txt")
                lines = []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))
                    lines.append(f"4 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                label_path.write_text("\n".join(lines))

                coco_items.append((path, label_path, {4}))
                downloaded += 1

    print(f"  Downloaded: {downloaded}/{len(selected)}")
    return coco_items


def _collect_coco_person_data(person_dir: Path):
    """Collect already-downloaded COCO person data."""
    items = []
    for img_path in person_dir.glob("coco_*.jpg"):
        if img_path.stat().st_size > 1000:
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                items.append((img_path, label_path, {4}))
    return items


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("=" * 60)
    print("  DETECTOR AI — Phase 5 Expanded Data Download")
    print("  Classes: 9 (4 existing + person + cheetah + jaguar")
    print("           + snow_leopard + sloth_bear)")
    print("  Color morphs: melanistic leopard, white tiger")
    print("=" * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1: WCS (jaguar + cheetah only — human images 404'd)
    wcs_data = download_wcs_data(max_jaguar=500, max_cheetah=50)

    # Part 1B: COCO person images (replaces WCS human)
    coco_person_data = download_coco_person(target=600)

    # Part 2: iNaturalist (new species + color morphs)
    inat_data = download_inat_data()

    # Part 3: Auto-annotate iNat images
    auto_annotate_inat(inat_data)

    # Part 4: Open Images snow leopard supplement
    download_openimages_snow_leopard(target=150)

    # Part 5: Merge and split
    merge_and_split(wcs_data, inat_data, coco_person_data)

    print("\n" + "=" * 60)
    print("  PHASE 5 DATA DOWNLOAD COMPLETE")
    print("=" * 60)
    print("\n  Next steps:")
    print("    1. python training/train_detector.py \\")
    print("         --data D:\\Spidy\\Detector\\multispecies_dataset\\data.yaml \\")
    print("         --epochs 120 --name phase5_9class")
    print("    2. python training/extract_crops.py")
    print("    3. python training/train_classifier.py \\")
    print("         --data D:\\Spidy\\Detector\\multispecies_dataset\\crops --epochs 50")


if __name__ == "__main__":
    main()
