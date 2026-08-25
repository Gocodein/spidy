# ==============================================================================
# 🐾 DETECTOR AI / SPIDY — Hugging Face Spaces Entry Point
# ==============================================================================
# Multi-Species Wildlife Detection, Fine-Grained Classification & Threat Analysis
# Protected under Indian Patent Application No. 202531071175 A
# ==============================================================================

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import gradio as gr
from ultralytics import YOLO

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector_ai.config import DEFAULT_YOLO_WEIGHTS, DEFAULT_CLASSIFIER_WEIGHTS
from detector_ai.stage2_classifier import SpeciesClassifier

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO(str(DEFAULT_YOLO_WEIGHTS))

classifier = None
if Path(DEFAULT_CLASSIFIER_WEIGHTS).exists():
    try:
        classifier = SpeciesClassifier(weights_path=str(DEFAULT_CLASSIFIER_WEIGHTS), conf_threshold=0.30, device=device)
    except Exception as e:
        print(f"Classifier load warning: {e}")

# ZeroGPU support for Hugging Face Spaces
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False


def detect_wildlife(image: np.ndarray, conf_threshold: float, enable_stage2: bool):
    if image is None:
        return None, pd.DataFrame(), "ℹ️ Please upload or capture an image."
    
    # Image in RGB from Gradio
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    results = yolo_model(img_bgr, conf=conf_threshold, verbose=False)
    annotated = img_bgr.copy()

    records = []
    has_human = False
    animal_count = 0

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for idx in range(len(xyxy)):
            bx1, by1, bx2, by2 = xyxy[idx]
            x1 = int(max(0, min(w - 1, bx1)))
            y1 = int(max(0, min(h - 1, by1)))
            x2 = int(max(0, min(w, bx2)))
            y2 = int(max(0, min(h, by2)))

            cid = int(classes[idx])
            conf = float(confs[idx])
            yolo_label = yolo_model.names.get(cid, f"class_{cid}")

            final_label = yolo_label
            final_conf = conf
            stage2_applied = False

            if (x2 > x1) and (y2 > y1):
                crop = img_bgr[y1:y2, x1:x2]
                if enable_stage2 and classifier and classifier.is_available and crop.size > 0:
                    cls_res = classifier.classify(crop)
                    if cls_res:
                        final_label, final_conf = cls_res
                        stage2_applied = True

            is_human = final_label.lower() in ("person", "human")
            if is_human:
                has_human = True
            else:
                animal_count += 1

            color = (0, 165, 255) if is_human else (0, 255, 100)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            tag = f"{final_label} ({final_conf:.1%})"
            if stage2_applied:
                tag += " [S2]"

            cv2.putText(annotated, tag, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            records.append({
                "Entity": final_label.replace("_", " ").title(),
                "Stage": "Stage 2 (EffNetV2-S)" if stage2_applied else "Stage 1 (YOLOv8)",
                "Confidence": f"{final_conf:.1%}",
                "Category": "Human / Intruder" if is_human else "Wildlife",
                "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]",
            })

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    df_out = pd.DataFrame(records)

    if has_human and animal_count > 0:
        status = f"🚨 DISTURBANCE WARNING: {animal_count} Animal(s) & Human detected in proximity!"
    elif has_human:
        status = "⚠️ Human / Ranger Detected"
    elif animal_count > 0:
        status = f"✅ Wildlife Verified: {animal_count} animal(s) detected"
    else:
        status = "ℹ️ No wildlife or humans detected at current confidence threshold"

    return annotated_rgb, df_out, status


if HAS_SPACES:
    detect_wildlife = spaces.GPU(detect_wildlife)


# Gradio Interface
with gr.Blocks(title="DETECTOR AI — Wildlife Surveillance", theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.Markdown(
        """
        # 🐾 DETECTOR AI (Spidy) — 9-Class Multi-Species Wildlife Surveillance
        ### *Arachnid Research Companion (ARC): A Biomimetic Hexapod Robot for Ground-Level Environmental Monitoring*
        **🏛️ Indian Patent Application No. 202531071175 A** | **Apache 2.0 License**
        
        *Detects Bengal Tiger (inc. White Tiger), Asian Elephant, Leopard (inc. Black Panther), Rhinoceros, Cheetah, Jaguar, Snow Leopard, Sloth Bear, & Humans.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="numpy", label="Input Wildlife Photo / Video Frame")
            conf_slider = gr.Slider(minimum=0.10, maximum=0.90, value=0.25, step=0.05, label="Confidence Threshold")
            s2_check = gr.Checkbox(value=True, label="Enable Stage 2 Classifier (EfficientNetV2-S, 97.32% Acc)")
            submit_btn = gr.Button("🚀 Run AI Detection & Classification", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(type="numpy", label="Annotated Output with Bounding Boxes")
            status_box = gr.Textbox(label="Threat & Sanctuary Status", interactive=False)
            output_df = gr.DataFrame(label="Detected Entities Breakdown", interactive=False)

    submit_btn.click(
        fn=detect_wildlife,
        inputs=[input_img, conf_slider, s2_check],
        outputs=[output_img, output_df, status_box],
    )

    # Sample examples if available
    sample_dir = PROJECT_ROOT / "multispecies_dataset" / "images" / "val"
    sample_imgs = [str(f) for f in list(sample_dir.glob("*.jpg"))[:5]] if sample_dir.exists() else []
    if sample_imgs:
        gr.Examples(
            examples=[[img, 0.25, True] for img in sample_imgs],
            inputs=[input_img, conf_slider, s2_check],
            outputs=[output_img, output_df, status_box],
            fn=detect_wildlife,
            cache_examples=False,
        )

if __name__ == "__main__":
    demo.launch()
