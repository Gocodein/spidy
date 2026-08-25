"""
setup_multispecies_dataset.py
------------------------------------------------------------
Downloads and prepares a multi-species endangered wildlife
dataset for YOLOv8 training from the LILA BC WCS Camera Traps
bounding-box archive (already cached locally from Phase 3).

Target species:
  0: bengal_tiger    (Panthera tigris)         — WCS cat_id 154
  1: asian_elephant  (Elephas maximus)         — WCS cat_id 149
  2: leopard         (Panthera pardus)         — WCS cat_id 104
  3: rhinoceros      (Rhinoceros / Diceros)    — WCS cat_ids 260, 261

Pipeline:
  1. Parse WCS bounding-box JSON (already extracted locally)
  2. Download images from LILA BC Azure Blob Storage
  3. Convert COCO-format boxes → YOLO format labels
  4. Mine empty/background images as negative samples
  5. Train / Val / Test split (70/20/10)
  6. Generate data.yaml for YOLOv8

Usage:
  python training/setup_multispecies_dataset.py
------------------------------------------------------------
"""

import json
import shutil
import requests
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WCS_JSON_DIR = PROJECT_ROOT / "tiger_dataset" / "raw" / "wcs_bbox_extracted"
DATASET_ROOT = PROJECT_ROOT / "multispecies_dataset"
RAW_DIR = DATASET_ROOT / "raw"
IMG_DIR = DATASET_ROOT / "images" / "all"
LBL_DIR = DATASET_ROOT / "labels" / "all"

# ------------------------------------------------------------------
# LILA BC Azure Blob Storage base URL (publicly accessible, no SAS token)
# ------------------------------------------------------------------
LILA_BASE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/wcs-unzipped/"

# ------------------------------------------------------------------
# Species mapping: WCS category IDs → our YOLO class IDs
# ------------------------------------------------------------------
SPECIES_CONFIG = {
    "bengal_tiger": {
        "yolo_class_id": 0,
        "wcs_cat_ids": [154],
        "scientific": "Panthera tigris",
        "max_images": 500,
    },
    "asian_elephant": {
        "yolo_class_id": 1,
        "wcs_cat_ids": [149],
        "scientific": "Elephas maximus",
        "max_images": 500,
    },
    "leopard": {
        "yolo_class_id": 2,
        "wcs_cat_ids": [104],
        "scientific": "Panthera pardus",
        "max_images": 500,
    },
    "rhinoceros": {
        "yolo_class_id": 3,
        "wcs_cat_ids": [260, 261],
        "scientific": "Diceros bicornis / Ceratotherium simum",
        "max_images": 500,
    },
}

# How many empty/background images to include (negative mining)
MAX_EMPTY_IMAGES = 200

# Download parallelism
MAX_WORKERS = 8
DOWNLOAD_TIMEOUT = 15  # seconds per image


# ------------------------------------------------------------------
# STEP 0: Setup directories
# ------------------------------------------------------------------
def setup_dirs():
    for split in ["train", "val", "test"]:
        (DATASET_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    LBL_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# STEP 1: Parse WCS metadata and build download manifest
# ------------------------------------------------------------------
def load_wcs_metadata():
    """Load the WCS bounding-box JSON (already extracted from Phase 3)."""
    json_files = list(WCS_JSON_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No WCS JSON found in {WCS_JSON_DIR}. "
            "Run setup_tiger_dataset.py first to download the WCS bbox archive."
        )

    print(f"Loading WCS metadata from {json_files[0].name}...")
    with open(json_files[0]) as f:
        meta = json.load(f)

    n_cat = len(meta["categories"])
    n_img = len(meta["images"])
    n_ann = len(meta["annotations"])
    print(f"  {n_cat} categories, {n_img} images, {n_ann} annotations")
    return meta


def build_download_manifest(meta):
    """
    Build a per-image manifest mapping image_id -> download + annotation info.
    Returns manifest dict and species_counts.
    """
    images_by_id = {img["id"]: img for img in meta["images"]}

    # Build reverse map: WCS cat_id -> (species_name, yolo_class_id)
    wcs_to_yolo = {}
    for name, cfg in SPECIES_CONFIG.items():
        for wcs_id in cfg["wcs_cat_ids"]:
            wcs_to_yolo[wcs_id] = (name, cfg["yolo_class_id"])

    # Collect all annotations for target species
    manifest = {}
    species_counts = {name: 0 for name in SPECIES_CONFIG}

    for ann in meta["annotations"]:
        cat_id = ann.get("category_id")
        if cat_id not in wcs_to_yolo:
            continue

        species_name, yolo_id = wcs_to_yolo[cat_id]
        img_id = ann["image_id"]

        if img_id not in images_by_id:
            continue

        img_meta = images_by_id[img_id]

        # Skip corrupt images
        if img_meta.get("corrupt", False):
            continue

        # Check max_images limit per species
        if species_counts[species_name] >= SPECIES_CONFIG[species_name]["max_images"]:
            continue

        if img_id not in manifest:
            manifest[img_id] = {
                "file_name": img_meta["file_name"],
                "width": img_meta.get("width", 0),
                "height": img_meta.get("height", 0),
                "annotations": [],
            }

        # COCO bbox format: [x_topleft, y_topleft, width, height] in pixels
        manifest[img_id]["annotations"].append({
            "yolo_class_id": yolo_id,
            "bbox": ann["bbox"],
            "species": species_name,
        })

        species_counts[species_name] += 1

    print("\n=== Download Manifest ===")
    for name, count in species_counts.items():
        cfg = SPECIES_CONFIG[name]
        print(f"  {name}: {count} annotations (max {cfg['max_images']})")
    print(f"  Total unique images: {len(manifest)}")

    return manifest, species_counts


def collect_empty_images(meta, manifest):
    """
    Collect empty/background images for negative mining.
    """
    images_by_id = {img["id"]: img for img in meta["images"]}

    # Get all image IDs that have ANY annotation
    annotated_ids = set()
    for ann in meta["annotations"]:
        annotated_ids.add(ann["image_id"])

    # Find empty images: WCS category 0 is 'empty'
    empty_candidates = []
    for ann in meta["annotations"]:
        if ann.get("category_id") == 0:
            img_id = ann["image_id"]
            if img_id in images_by_id and img_id not in manifest:
                img = images_by_id[img_id]
                if not img.get("corrupt", False):
                    empty_candidates.append(img)

    # Deduplicate
    seen = set()
    unique_empty = []
    for img in empty_candidates:
        if img["id"] not in seen:
            seen.add(img["id"])
            unique_empty.append(img)

    # Sample up to MAX_EMPTY_IMAGES
    random.seed(42)
    if len(unique_empty) > MAX_EMPTY_IMAGES:
        unique_empty = random.sample(unique_empty, MAX_EMPTY_IMAGES)

    print(f"  Empty/background images for negative mining: {len(unique_empty)}")
    return unique_empty


# ------------------------------------------------------------------
# STEP 2: Download images from LILA BC Azure Blob
# ------------------------------------------------------------------
def download_single_image(args):
    """Download a single image. Returns (local_path, success)."""
    file_name, dest_path = args
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        return dest_path, True  # already downloaded

    url = LILA_BASE_URL + file_name
    try:
        r = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        if r.status_code == 200 and len(r.content) > 1000:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(r.content)
            return dest_path, True
        return dest_path, False
    except Exception:
        return dest_path, False


def download_images(manifest, empty_images):
    """Download all images using parallel workers."""
    tasks = []

    # Species images
    for img_id, info in manifest.items():
        safe_name = info["file_name"].replace("/", "_").replace("\\", "_")
        dest = RAW_DIR / safe_name
        tasks.append((info["file_name"], dest))
        # Store the local filename back for label generation
        info["local_name"] = safe_name

    # Empty/background images
    for img in empty_images:
        safe_name = "bg_" + img["file_name"].replace("/", "_").replace("\\", "_")
        dest = RAW_DIR / safe_name
        tasks.append((img["file_name"], dest))
        img["local_name"] = safe_name

    total = len(tasks)
    print(f"\nDownloading {total} images from LILA BC Azure Blob...")
    print(f"  Workers: {MAX_WORKERS} | Timeout: {DOWNLOAD_TIMEOUT}s each")

    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single_image, t): t for t in tasks}
        with tqdm(total=total, desc="Downloading") as pbar:
            for future in as_completed(futures):
                _, ok = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                pbar.update(1)

    print(f"  Downloaded: {success} | Failed: {failed}")
    return success


# ------------------------------------------------------------------
# STEP 3: Convert COCO boxes -> YOLO format labels & copy to dataset
# ------------------------------------------------------------------
def generate_labels_and_copy(manifest, empty_images):
    """
    Convert COCO bbox [x_tl, y_tl, w, h] (pixels) -> YOLO [class cx cy w h] (normalized).
    Copy images and labels to IMG_DIR / LBL_DIR.
    """
    copied = 0
    skipped = 0

    for img_id, info in tqdm(manifest.items(), desc="Generating labels"):
        local_name = info.get("local_name")
        if not local_name:
            continue

        src_img = RAW_DIR / local_name
        if not src_img.exists() or src_img.stat().st_size < 1000:
            skipped += 1
            continue

        img_w = info["width"]
        img_h = info["height"]

        if img_w <= 0 or img_h <= 0:
            # Try to read dimensions from the image itself
            try:
                from PIL import Image
                with Image.open(src_img) as im:
                    img_w, img_h = im.size
            except Exception:
                skipped += 1
                continue

        label_lines = []
        for ann in info["annotations"]:
            # COCO: [x_topleft, y_topleft, box_width, box_height] in pixels
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / img_w
            cy = (by + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h

            # Clamp to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0.001, min(1, nw))
            nh = max(0.001, min(1, nh))

            yolo_id = ann["yolo_class_id"]
            label_lines.append(f"{yolo_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if label_lines:
            # Copy image
            dest_img = IMG_DIR / f"{src_img.stem}.jpg"
            if not dest_img.exists():
                shutil.copy(src_img, dest_img)

            # Write label
            dest_lbl = LBL_DIR / f"{src_img.stem}.txt"
            dest_lbl.write_text("\n".join(label_lines))
            copied += 1

    # Copy empty/background images (no labels -> acts as negative sample)
    bg_copied = 0
    for img in empty_images:
        local_name = img.get("local_name")
        if not local_name:
            continue
        src_img = RAW_DIR / local_name
        if not src_img.exists() or src_img.stat().st_size < 1000:
            continue
        dest_img = IMG_DIR / f"{src_img.stem}.jpg"
        if not dest_img.exists():
            shutil.copy(src_img, dest_img)
        # Write empty label file (required by YOLO for background images)
        dest_lbl = LBL_DIR / f"{src_img.stem}.txt"
        dest_lbl.write_text("")
        bg_copied += 1

    print(f"\n  Species images with labels: {copied}")
    print(f"  Background negatives: {bg_copied}")
    print(f"  Skipped (download failed/corrupt): {skipped}")
    return copied + bg_copied


# ------------------------------------------------------------------
# STEP 4: Train / Val / Test split
# ------------------------------------------------------------------
def split_dataset(train_ratio=0.7, val_ratio=0.2):
    images = sorted(IMG_DIR.glob("*.jpg"))
    if not images:
        print("ERROR: No images found to split!")
        return

    train_imgs, temp_imgs = train_test_split(
        images, train_size=train_ratio, random_state=42
    )
    rel_val = val_ratio / (1 - train_ratio)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, train_size=rel_val, random_state=42
    )

    split_map = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
    for split, imgs in split_map.items():
        for img_path in imgs:
            label_path = LBL_DIR / f"{img_path.stem}.txt"
            dest_img = DATASET_ROOT / "images" / split / img_path.name
            if not dest_img.exists():
                shutil.copy(img_path, dest_img)
            if label_path.exists():
                dest_lbl = DATASET_ROOT / "labels" / split / label_path.name
                if not dest_lbl.exists():
                    shutil.copy(label_path, dest_lbl)

    print(f"\nSplit: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")


# ------------------------------------------------------------------
# STEP 5: Generate data.yaml
# ------------------------------------------------------------------
def write_data_yaml():
    class_names = {}
    for name, cfg in SPECIES_CONFIG.items():
        class_names[cfg["yolo_class_id"]] = name

    names_yaml = "\n".join(f"  {k}: {v}" for k, v in sorted(class_names.items()))

    content = f"""path: {DATASET_ROOT.resolve()}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names:
{names_yaml}
"""
    yaml_path = DATASET_ROOT / "data.yaml"
    yaml_path.write_text(content)
    print(f"\nWrote {yaml_path}")


# ------------------------------------------------------------------
# STEP 6: Print summary
# ------------------------------------------------------------------
def print_summary():
    print("\n" + "=" * 60)
    print("  MULTI-SPECIES DATASET READY!")

    for split in ["train", "val", "test"]:
        n_imgs = len(list((DATASET_ROOT / "images" / split).glob("*.jpg")))
        n_lbls = len(list((DATASET_ROOT / "labels" / split).glob("*.txt")))
        print(f"  {split:6s}: {n_imgs} images, {n_lbls} labels")

    # Count per-species annotations
    print("\n  Per-species annotation counts:")
    species_counts = {name: 0 for name in SPECIES_CONFIG}
    bg_count = 0
    for split in ["train", "val", "test"]:
        for lbl_path in (DATASET_ROOT / "labels" / split).glob("*.txt"):
            content = lbl_path.read_text().strip()
            if not content:
                bg_count += 1
                continue
            for line in content.split("\n"):
                parts = line.strip().split()
                if parts:
                    cid = int(parts[0])
                    for name, cfg in SPECIES_CONFIG.items():
                        if cfg["yolo_class_id"] == cid:
                            species_counts[name] += 1

    for name, count in species_counts.items():
        print(f"    {name}: {count} boxes")
    print(f"    background (negative): {bg_count} images")

    print("=" * 60)
    yaml_path = DATASET_ROOT / "data.yaml"
    print("\nNext step: train multi-species YOLOv8:")
    print(f"  python training/train_detector.py --data {yaml_path}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  DETECTOR AI -- Multi-Species Dataset Setup")
    print("  Target: Bengal Tiger, Asian Elephant, Leopard, Rhinoceros")
    print("=" * 60)

    setup_dirs()

    # Step 1: Load WCS metadata
    meta = load_wcs_metadata()
    manifest, species_counts = build_download_manifest(meta)
    empty_images = collect_empty_images(meta, manifest)

    # Step 2: Download images
    download_images(manifest, empty_images)

    # Step 3: Generate YOLO labels and copy
    generate_labels_and_copy(manifest, empty_images)

    # Step 4: Split
    split_dataset()

    # Step 5: data.yaml
    write_data_yaml()

    # Step 6: Summary
    print_summary()
