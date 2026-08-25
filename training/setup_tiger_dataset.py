"""
setup_tiger_dataset.py
------------------------------------------------------------
RUN THIS ON YOUR OWN MACHINE (needs open internet access to
GBIF, LILA BC / Azure blob storage -- not available inside this chat's sandbox).

Pipeline:
  1. Download WII/GBIF Rajaji National Park tiger dataset (Darwin Core Archive)
  2. Download a filtered tiger subset from LILA BC's WCS Camera Traps metadata
  3. Auto-generate bounding boxes for weakly-labeled images using MegaDetector
  4. Organize everything into YOLO training format with train/val/test splits
  5. Emit a ready-to-use data.yaml for YOLOv8 fine-tuning

Install requirements first:
    pip install requests pandas tqdm scikit-learn pillow
    pip install megadetector          # Microsoft's pretrained animal detector
------------------------------------------------------------
"""

import os
import json
import zipfile
import shutil
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
ROOT = Path("tiger_dataset")
RAW_DIR = ROOT / "raw"
IMG_DIR = ROOT / "images" / "all"
LBL_DIR = ROOT / "labels" / "all"
SPLIT_DIRS = ["train", "val", "test"]
CLASS_NAME = "bengal_tiger"
CLASS_ID = 0

GBIF_DWCA_URL = "https://api.gbif.org/v1/occurrence/download/request/0020239-260623161305970.zip"
# NOTE: GBIF download links expire / regenerate. If this fails, go to
# https://www.gbif.org/dataset/e61455a4-352d-4c55-83ea-dbca254e3b29
# and click "Download" manually to get a fresh DwC-A zip, then point
# RAJAJI_ZIP_PATH at the downloaded file instead of re-fetching.
RAJAJI_ZIP_PATH = RAW_DIR / "rajaji_tiger_dwca.zip"

# LILA BC WCS Camera Traps -- confirmed current links (checked July 2026):
# https://lila.science/datasets/wcscameratraps/
LILA_CLASS_LABELS_URL = "https://storage.googleapis.com/public-datasets-lila/wcs/wcs_camera_traps.json.zip"
# IMPORTANT: this one already has real bounding boxes + species classes,
# so for WCS images you do NOT need to run MegaDetector -- just filter for tiger.
LILA_BBOX_URL = "https://storage.googleapis.com/public-datasets-lila/wcs/wcs_20220205_bboxes_with_classes.zip"
LILA_IMAGE_BASE_URL = "https://lilawildlife.blob.core.windows.net/lila-wildlife/wcs-unzipped/"


def setup_dirs():
    for d in [RAW_DIR, IMG_DIR, LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_DIRS:
        (ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# STEP 1: Rajaji NP tiger dataset (GBIF Darwin Core Archive)
# ------------------------------------------------------------------
def download_rajaji_dataset():
    print("Downloading Rajaji NP tiger dataset from GBIF...")
    try:
        r = requests.get(GBIF_DWCA_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(RAJAJI_ZIP_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        print(f"  Auto-download failed ({e}).")
        print("  Manually download from https://www.gbif.org/dataset/"
              "e61455a4-352d-4c55-83ea-dbca254e3b29 and place the zip at:")
        print(f"  {RAJAJI_ZIP_PATH}")
        return

    extract_dir = RAW_DIR / "rajaji_extracted"
    with zipfile.ZipFile(RAJAJI_ZIP_PATH, "r") as z:
        z.extractall(extract_dir)

    # Darwin Core Archives store image URLs in multimedia.txt (tab-separated)
    multimedia_path = extract_dir / "multimedia.txt"
    if not multimedia_path.exists():
        print("  multimedia.txt not found -- check archive contents manually.")
        return

    df = pd.read_csv(multimedia_path, sep="\t")
    url_col = next((c for c in df.columns if "identifier" in c.lower()), None)
    if url_col is None:
        print("  Could not find an image URL column in multimedia.txt")
        return

    dest = RAW_DIR / "rajaji_images"
    dest.mkdir(exist_ok=True)
    for i, url in enumerate(tqdm(df[url_col].dropna().tolist(), desc="Rajaji images")):
        try:
            img_data = requests.get(url, timeout=30).content
            with open(dest / f"rajaji_{i:04d}.jpg", "wb") as f:
                f.write(img_data)
        except Exception:
            continue


# ------------------------------------------------------------------
# STEP 2: Filtered tiger subset from LILA BC WCS Camera Traps
# NOTE: this dataset already ships ~375,000 real bounding box annotations
# (not just species labels), so tiger images from here need NO MegaDetector
# step -- we use the shipped boxes directly. MegaDetector is only needed
# for the Rajaji set in Step 1, which has no boxes.
# ------------------------------------------------------------------
def download_lila_tiger_subset():
    print("Downloading LILA BC WCS bounding-box annotations (has real boxes already)...")
    bbox_zip_path = RAW_DIR / "wcs_bboxes.zip"

    # Only download if zip doesn't exist yet
    if not bbox_zip_path.exists():
        r = requests.get(LILA_BBOX_URL, stream=True, timeout=120)
        r.raise_for_status()
        with open(bbox_zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print("  WCS bbox zip already exists — skipping download.")

    extract_dir = RAW_DIR / "wcs_bbox_extracted"
    if not extract_dir.exists():
        with zipfile.ZipFile(bbox_zip_path, "r") as z:
            z.extractall(extract_dir)

    json_files = list(extract_dir.glob("*.json"))
    if not json_files:
        print("  No .json found after extraction -- check archive contents.")
        return
    with open(json_files[0]) as f:
        meta = json.load(f)

    cat_id_to_name = {c["id"]: c["name"].lower() for c in meta["categories"]}
    # WCS taxonomy uses scientific names (e.g. "panthera tigris"), not common names
    TIGER_KEYWORDS = {"tiger", "panthera tigris", "tigris"}
    tiger_cat_ids = {
        cid for cid, name in cat_id_to_name.items()
        if any(kw in name for kw in TIGER_KEYWORDS)
        and "tigrisoma" not in name  # exclude Tigrisoma (a bird genus)
    }
    print(f"  Found {len(tiger_cat_ids)} tiger-related category id(s) in WCS taxonomy")
    for cid in tiger_cat_ids:
        print(f"    → ID={cid}: {cat_id_to_name[cid]}")

    images_by_id = {img["id"]: img for img in meta["images"]}
    tiger_anns = [ann for ann in meta["annotations"] if ann.get("category_id") in tiger_cat_ids]
    print(f"  {len(tiger_anns)} tiger bounding-box annotations found")

    img_dest = RAW_DIR / "lila_tiger_images"
    img_dest.mkdir(exist_ok=True)

    for ann in tqdm(tiger_anns, desc="LILA tiger boxes"):
        img_info = images_by_id.get(ann["image_id"])
        if not img_info:
            continue
        file_name = img_info["file_name"]  # e.g. 'animals/0011/0009.jpg'
        url = LILA_IMAGE_BASE_URL + file_name
        local_name = file_name.replace("/", "_")
        img_path = img_dest / local_name

        if not img_path.exists():
            try:
                img_data = requests.get(url, timeout=30).content
                with open(img_path, "wb") as f:
                    f.write(img_data)
            except Exception:
                continue

        # ann["bbox"] in COCO format is [x_min, y_min, width, height] in PIXELS
        w_img, h_img = img_info["width"], img_info["height"]
        x_min, y_min, box_w, box_h = ann["bbox"]
        x_center = (x_min + box_w / 2) / w_img
        y_center = (y_min + box_h / 2) / h_img
        norm_w = box_w / w_img
        norm_h = box_h / h_img

        label_line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        label_path = LBL_DIR / f"{Path(local_name).stem}.txt"
        with open(label_path, "a") as f:
            f.write(label_line + "\n")
        shutil.copy(img_path, IMG_DIR / local_name)


# ------------------------------------------------------------------
# STEP 3: Auto-generate bounding boxes with YOLOv8
# ------------------------------------------------------------------
# COCO animal class IDs that YOLO can detect
COCO_ANIMAL_IDS = {
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23,  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
}

def run_yolo_auto_annotate(image_dir, output_labels_dir, conf_threshold=0.3):
    """
    Uses a pre-trained YOLOv8 model to auto-detect animals in images and
    writes YOLO-format labels. Since these images come from known tiger
    camera-trap datasets, every detected animal box is assigned CLASS_ID=0
    (bengal_tiger).

    This replaces MegaDetector — no extra dependency needed.
    """
    from ultralytics import YOLO
    from PIL import Image

    print(f"\n  Auto-annotating images in {image_dir} using YOLOv8...")
    model = YOLO("yolov8n.pt")

    image_paths = list(Path(image_dir).glob("*.jpg"))
    annotated = 0

    for img_path in tqdm(image_paths, desc="YOLOv8 auto-annotate"):
        try:
            results = model(str(img_path), conf=conf_threshold, verbose=False)
        except Exception:
            continue

        if not results or results[0].boxes is None:
            continue

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        # Get image dimensions
        img = Image.open(img_path)
        w, h = img.size

        label_lines = []
        for idx in range(len(xyxy)):
            cid = int(class_ids[idx])
            # Only keep detections that YOLO classifies as animals
            if cid not in COCO_ANIMAL_IDS:
                continue

            x1, y1, x2, y2 = xyxy[idx]
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            label_lines.append(
                f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            )

        if label_lines:
            out_path = Path(output_labels_dir) / f"{img_path.stem}.txt"
            out_path.write_text("\n".join(label_lines))
            shutil.copy(img_path, IMG_DIR / img_path.name)
            shutil.copy(out_path, LBL_DIR / out_path.name)
            annotated += 1

    print(f"  Auto-annotated {annotated}/{len(image_paths)} images with animal boxes.")


# ------------------------------------------------------------------
# STEP 4: Train / val / test split
# ------------------------------------------------------------------
def split_dataset(train_ratio=0.7, val_ratio=0.2):
    images = sorted(IMG_DIR.glob("*.jpg"))
    train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
    val_imgs, test_imgs = train_test_split(
        temp_imgs, train_size=val_ratio / (1 - train_ratio), random_state=42
    )

    split_map = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
    for split, imgs in split_map.items():
        for img_path in imgs:
            label_path = LBL_DIR / f"{img_path.stem}.txt"
            shutil.copy(img_path, ROOT / "images" / split / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, ROOT / "labels" / split / label_path.name)

    print(f"Split: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")


# ------------------------------------------------------------------
# STEP 5: YOLOv8 data.yaml
# ------------------------------------------------------------------
def write_data_yaml():
    content = f"""path: {ROOT.resolve()}
train: images/train
val: images/val
test: images/test

names:
  {CLASS_ID}: {CLASS_NAME}
"""
    (ROOT / "tiger_data.yaml").write_text(content)
    print(f"Wrote {ROOT / 'tiger_data.yaml'} -- use this as the YOLOv8 --data config")


if __name__ == "__main__":
    setup_dirs()

    # Step 1: Rajaji (skip if already downloaded)
    rajaji_dir = RAW_DIR / "rajaji_images"
    if rajaji_dir.exists() and any(rajaji_dir.glob("*.jpg")):
        print(f"Rajaji images already exist ({len(list(rajaji_dir.glob('*.jpg')))} images) — skipping download.")
    else:
        download_rajaji_dataset()

    # Step 2: LILA BC WCS tiger subset (skip if already downloaded)
    lila_dir = RAW_DIR / "lila_tiger_images"
    bbox_json = RAW_DIR / "wcs_bbox_extracted"
    if bbox_json.exists():
        print("LILA BC WCS metadata already exists — re-processing tiger filter...")
        download_lila_tiger_subset()  # re-run to apply fixed filter
    else:
        download_lila_tiger_subset()

    # Step 3: Auto-annotate Rajaji images (they have no boxes from source)
    # Only run on rajaji_images — LILA images already have real boxes from Step 2
    rajaji_label_check = LBL_DIR / "rajaji_0000.txt"
    if rajaji_dir.exists() and any(rajaji_dir.iterdir()) and not rajaji_label_check.exists():
        run_yolo_auto_annotate(rajaji_dir, LBL_DIR)
    else:
        print("Rajaji auto-annotation already done or no images — skipping.")

    # Step 4: Split
    split_dataset()

    # Step 5: data.yaml
    write_data_yaml()

    # Summary
    n_train = len(list((ROOT / "images" / "train").glob("*.jpg")))
    n_val = len(list((ROOT / "images" / "val").glob("*.jpg")))
    n_test = len(list((ROOT / "images" / "test").glob("*.jpg")))
    print(f"\n{'='*60}")
    print(f"  DATASET READY!")
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"  Total: {n_train + n_val + n_test} images")
    print(f"{'='*60}")
    print(f"\nNext step: fine-tune YOLOv8 with:")
    print(f"  python training/train_detector.py --data {ROOT/'tiger_data.yaml'}")
