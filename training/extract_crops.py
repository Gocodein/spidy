import os
import glob
import cv2
from collections import defaultdict

def extract_crops():
    base_dir = r"d:\Spidy\Detector\multispecies_dataset"
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    crops_dir = os.path.join(base_dir, "crops")
    
    classes = {
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
    
    # Create output directories
    for class_name in classes.values():
        os.makedirs(os.path.join(crops_dir, class_name), exist_ok=True)
        
    splits = ["train", "val", "test"]
    
    crop_counts = defaultdict(int)
    
    for split in splits:
        split_img_dir = os.path.join(images_dir, split)
        split_lbl_dir = os.path.join(labels_dir, split)
        
        if not os.path.exists(split_img_dir):
            continue
            
        # Find all images in this split
        image_paths = glob.glob(os.path.join(split_img_dir, "*.*"))
        
        for img_path in image_paths:
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_name = os.path.basename(img_path)
            base_name, _ = os.path.splitext(img_name)
            
            lbl_path = os.path.join(split_lbl_dir, f"{base_name}.txt")
            
            if not os.path.exists(lbl_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img_h, img_w = img.shape[:2]
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                if class_id not in classes:
                    continue
                    
                cx, cy, w_norm, h_norm = map(float, parts[1:5])
                
                # Convert to pixel coordinates
                box_w = w_norm * img_w
                box_h = h_norm * img_h
                box_cx = cx * img_w
                box_cy = cy * img_h
                
                # Add 10% padding
                pad_w = box_w * 0.1
                pad_h = box_h * 0.1
                
                x1 = int(max(0, box_cx - box_w/2 - pad_w))
                y1 = int(max(0, box_cy - box_h/2 - pad_h))
                x2 = int(min(img_w, box_cx + box_w/2 + pad_w))
                y2 = int(min(img_h, box_cy + box_h/2 + pad_h))
                
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                if crop_w < 32 or crop_h < 32:
                    continue
                    
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                class_name = classes[class_id]
                crop_counts[class_name] += 1
                
                # Save crop
                crop_name = f"{base_name}_{idx}.jpg"
                save_path = os.path.join(crops_dir, class_name, crop_name)
                cv2.imwrite(save_path, crop)

    print("Crop extraction complete.")
    print("-" * 30)
    print("Summary of crops per species:")
    for class_id, class_name in classes.items():
        count = crop_counts[class_name]
        print(f"{class_name}: {count} crops")
    print("-" * 30)

if __name__ == "__main__":
    extract_crops()
