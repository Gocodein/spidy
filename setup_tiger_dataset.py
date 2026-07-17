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
    r = requests.get(LILA_BBOX_URL, stream=True, timeout=120)
    r.raise_for_status()
    with open(bbox_zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    extract_dir = RAW_DIR / "wcs_bbox_extracted"
    with zipfile.ZipFile(bbox_zip_path, "r") as z:
        z.extractall(extract_dir)

    json_files = list(extract_dir.glob("*.json"))
    if not json_files:
        print("  No .json found after extraction -- check archive contents.")
        return
    with open(json_files[0]) as f:
        meta = json.load(f)

    cat_id_to_name = {c["id"]: c["name"].lower() for c in meta["categories"]}
    tiger_cat_ids = {cid for cid, name in cat_id_to_name.items() if "tiger" in name}
    print(f"  Found {len(tiger_cat_ids)} tiger-related category id(s) in WCS taxonomy")

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
# STEP 3: Auto-generate bounding boxes with MegaDetector
# ------------------------------------------------------------------
def run_megadetector(image_dir, output_labels_dir):
    """
    Runs MegaDetector on every image in image_dir and writes YOLO-format
    labels (class_id x_center y_center width height, normalized 0-1) to
    output_labels_dir. Every detected 'animal' box is assigned CLASS_ID
    since we already know these images are tiger-labeled from the source dataset.
    """
    from megadetector.detection import run_detector
    from PIL import Image

    detector = run_detector.load_detector("MDV5A")  # or "MDV6" if available

    for img_path in tqdm(list(Path(image_dir).glob("*.jpg")), desc="MegaDetector"):
        try:
            img = Image.open(img_path).convert("RGB")
            result = detector.generate_detections_one_image(img, img_path.name)
        except Exception:
            continue

        w, h = img.size
        label_lines = []
        for det in result.get("detections", []):
            # MegaDetector category '1' = animal
            if det["category"] != "1" or det["conf"] < 0.5:
                continue
            x, y, box_w, box_h = det["bbox"]  # already normalized [0-1] top-left x,y,w,h
            x_center = x + box_w / 2
            y_center = y + box_h / 2
            label_lines.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        if label_lines:
            out_path = Path(output_labels_dir) / f"{img_path.stem}.txt"
            out_path.write_text("\n".join(label_lines))
            shutil.copy(img_path, IMG_DIR / img_path.name)
            shutil.copy(out_path, LBL_DIR / out_path.name)


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
    download_rajaji_dataset()
    download_lila_tiger_subset()

    # Run MegaDetector on whichever raw image folders downloaded successfully
    for folder in ["rajaji_images", "lila_tiger_images"]:
        p = RAW_DIR / folder
        if p.exists() and any(p.iterdir()):
            run_megadetector(p, LBL_DIR)

    split_dataset()
    write_data_yaml()
    print("\nDone. Next step: fine-tune YOLOv8 with:")
    print(f"  yolo detect train data={ROOT/'tiger_data.yaml'} model=yolov8n.pt epochs=100 imgsz=640")
