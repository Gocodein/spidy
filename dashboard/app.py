# Run: streamlit run dashboard/app.py
# Or:  python -m streamlit run dashboard/app.py
"""
🐾 DETECTOR AI / SPIDY — Wildlife Research & Threat Monitoring Dashboard
========================================================================
Interactive Streamlit dashboard for exploring multi-species detection sessions,
fine-grained species classification, kinematics behavior patterns,
and human-wildlife disturbance events.

Protected under Indian Patent No. 202531071175 A
"Arachnid Research Companion (ARC): A Biomimetic Hexapod Robot for Ground-Level Environmental Monitoring"
"""

from __future__ import annotations

import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Resolve project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector_ai.config import (
    DEFAULT_DB_PATH,
    DEFAULT_YOLO_WEIGHTS,
    DEFAULT_CLASSIFIER_WEIGHTS,
    MULTISPECIES_CLASS_MAP,
    SPECIES_DISTURBANCE_THRESHOLDS,
    PROJECT_ROOT,
)

# ──────────────────────────────────────────────── Page Configuration ──

st.set_page_config(
    page_title="DETECTOR AI — Wildlife Dashboard",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────── Custom Styling ──

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern Glassmorphism & Gradient Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(249, 115, 22, 0.4);
    }
    .metric-card h2 {
        color: #f97316;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .metric-card p {
        color: #94a3b8;
        font-size: 0.82rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 5px 0 0 0;
    }
    
    .patent-badge {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    
    .species-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Severity Badges */
    .badge-CRITICAL { background: #ef4444; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; }
    .badge-HIGH     { background: #f97316; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; }
    .badge-MEDIUM   { background: #eab308; color: #1e293b; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; }
    .badge-LOW      { background: #22c55e; color: white; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────── Fast Cached Database Helpers ──

@st.cache_data(ttl=60, show_spinner=False)
def load_sessions_cached(path_str: str) -> pd.DataFrame:
    """Load session summary rows into a DataFrame (cached in RAM)."""
    try:
        with sqlite3.connect(path_str) as conn:
            df = pd.read_sql_query("SELECT * FROM sessions ORDER BY id DESC", conn)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_detections_cached(path_str: str, session_id: Optional[int] = None) -> pd.DataFrame:
    """Load detection rows into a DataFrame (cached in RAM)."""
    try:
        query = "SELECT * FROM detections"
        if session_id is not None:
            query += f" WHERE session_id = {session_id}"
        query += " ORDER BY timestamp ASC"
        with sqlite3.connect(path_str) as conn:
            df = pd.read_sql_query(query, conn)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_disturbances_cached(path_str: str, session_id: Optional[int] = None) -> pd.DataFrame:
    """Load disturbance events into a DataFrame (cached in RAM)."""
    try:
        query = "SELECT * FROM disturbance_events"
        if session_id is not None:
            query += f" WHERE session_id = {session_id}"
        query += " ORDER BY timestamp ASC"
        with sqlite3.connect(path_str) as conn:
            df = pd.read_sql_query(query, conn)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────── Species Icons & Formatting ──

SPECIES_EMOJIS = {
    "bengal_tiger": "🐯",
    "asian_elephant": "🐘",
    "leopard": "🐆",
    "rhinoceros": "🦏",
    "person": "👤",
    "cheetah": "🐆",
    "jaguar": "🐆",
    "snow_leopard": "❄️",
    "sloth_bear": "🐻",
    "default": "🐾",
}

def format_species(name: str) -> str:
    if not name:
        return "Unknown"
    emoji = SPECIES_EMOJIS.get(name.lower(), "🐾")
    clean = name.replace("_", " ").title()
    return f"{emoji} {clean}"


# ──────────────────────────────────────────────── Sidebar ──

st.sidebar.markdown('<span class="patent-badge">🏛️ Indian Patent No. 202531071175 A</span>', unsafe_allow_html=True)
st.sidebar.title("🐾 DETECTOR AI")
st.sidebar.caption("Arachnid Research Companion (ARC / Spidy) — Wildlife Monitoring")
st.sidebar.markdown("---")

# Database selector
db_path = st.sidebar.text_input("Database path", value=str(DEFAULT_DB_PATH))

if st.sidebar.button("🔄 Fast Cache Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

sessions_df = load_sessions_cached(db_path)

# Session Filter
selected_session: Optional[int] = None
if not sessions_df.empty:
    session_options = ["All Sessions"] + [
        f"Session #{row['id']} — {row['video_source']} ({row['total_detections']} det)"
        for _, row in sessions_df.iterrows()
    ]
    chosen = st.sidebar.selectbox("Filter Session", session_options, index=0)
    if chosen != "All Sessions":
        selected_session = int(chosen.split("#")[1].split(" ")[0])

page = st.sidebar.radio(
    "Navigation",
    options=[
        "📊 Overview",
        "🦁 9-Species Analysis",
        "🚶 Behavior Kinematics",
        "⚠️ Disturbance Threat Radar",
        "🔬 Live AI Inference Playground",
        "📥 Export & Field Reports",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Hardware Status**: RTX 4050 GPU (6 GB VRAM)\n\n"
    "**Pipeline**: 9-Class YOLOv8 + EfficientNetV2-S (97.32% Acc)"
)

# Load data based on filter with RAM caching
det_df = load_detections_cached(db_path, session_id=selected_session)
dist_df = load_disturbances_cached(db_path, session_id=selected_session)
no_data = det_df.empty


def metric_card(label: str, value, icon: str = ""):
    st.markdown(
        f'<div class="metric-card">'
        f'<h2>{icon} {value}</h2>'
        f'<p>{label}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# PAGE 1: Overview
# ═══════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("📊 Multi-Species Surveillance Overview")
    if selected_session:
        st.caption(f"Showing metrics for **Session #{selected_session}**")
    else:
        st.caption("Aggregated analytics across all surveillance sessions")

    if no_data:
        st.info("ℹ️ No detection data found. Run `python run_detector.py --show` to generate live telemetry.")
        st.stop()

    # KPI Top Bar
    total_det = len(det_df)
    unique_tracks = det_df["track_id"].nunique() if "track_id" in det_df.columns else 0
    total_dist = len(dist_df)
    critical_dist = len(dist_df[dist_df["severity"] == "CRITICAL"]) if "severity" in dist_df.columns else 0
    unique_species = det_df["species"].nunique() if "species" in det_df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Total Detections", f"{total_det:,}", "📍")
    with c2:
        metric_card("Tracked Entities", f"{unique_tracks:,}", "🔗")
    with c3:
        metric_card("Species Detected", f"{unique_species}/9", "🦁")
    with c4:
        metric_card("Disturbance Alerts", f"{total_dist:,}", "⚠️")
    with c5:
        metric_card("Critical Threats", f"{critical_dist:,}", "🚨")

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("📈 Detection Timeline Activity")
        if "timestamp" in det_df.columns and not det_df["timestamp"].isna().all():
            timeline = det_df.set_index("timestamp").resample("1min").size().reset_index(name="count")
            fig = px.area(
                timeline, x="timestamp", y="count",
                labels={"timestamp": "Surveillance Time", "count": "Detections / min"},
                color_discrete_sequence=["#f97316"],
            )
            fig.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        elif "frame_num" in det_df.columns:
            frame_counts = det_df.groupby("frame_num").size().reset_index(name="count")
            fig = px.area(
                frame_counts, x="frame_num", y="count",
                labels={"frame_num": "Video Frame", "count": "Active Detections"},
                color_discrete_sequence=["#f97316"],
            )
            fig.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🐾 Species Representation")
        if "species" in det_df.columns:
            sp_counts = det_df["species"].value_counts().reset_index()
            sp_counts.columns = ["species", "count"]
            sp_counts["display"] = sp_counts["species"].apply(format_species)
            fig = px.pie(
                sp_counts, names="display", values="count",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # Live Event Stream Table
    st.subheader("⏱️ Recent Telemetry Stream")
    display_cols = [c for c in ["id", "frame_num", "species", "behavior", "confidence", "disturbance_flag"] if c in det_df.columns]
    st.dataframe(det_df[display_cols].tail(25).iloc[::-1], use_container_width=True, height=260)


# ═══════════════════════════════════════════════════════════════
# PAGE 2: 9-Species Analysis
# ═══════════════════════════════════════════════════════════════

elif page == "🦁 9-Species Analysis":
    st.title("🦁 9-Class Multi-Species Analytics")
    st.caption("Deep-dive metrics across Bengal Tiger, Asian Elephant, Leopard, Rhinoceros, Cheetah, Jaguar, Snow Leopard, Sloth Bear, & Humans")

    if no_data:
        st.info("No detection data available for species analysis.")
        st.stop()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Species Confidence Distribution")
        if "species" in det_df.columns and "confidence" in det_df.columns:
            fig = px.box(
                det_df, x="species", y="confidence", color="species",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                labels={"confidence": "Confidence Score", "species": "Species"},
            )
            fig.update_layout(template="plotly_dark", height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Spatial Detection Density (Field Coordinates)")
        if "x" in det_df.columns and "y" in det_df.columns and not det_df["x"].isna().all():
            fig = px.density_heatmap(
                det_df, x="x", y="y", nbinsx=30, nbinsy=25,
                color_continuous_scale="Viridis",
                labels={"x": "X Position (pixels)", "y": "Y Position (pixels)"},
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Spatial coordinates not recorded for this dataset.")

    st.subheader("Species Breakdown Summary")
    if "species" in det_df.columns:
        summary_df = det_df.groupby("species").agg(
            total_detections=("id", "count"),
            unique_tracks=("track_id", "nunique"),
            avg_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
        ).reset_index()
        summary_df["avg_confidence"] = summary_df["avg_confidence"].map(lambda x: f"{x:.2%}")
        summary_df["max_confidence"] = summary_df["max_confidence"].map(lambda x: f"{x:.2%}")
        st.dataframe(summary_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3: Behavior Kinematics
# ═══════════════════════════════════════════════════════════════

elif page == "🚶 Behavior Kinematics":
    st.title("🚶 Wildlife Behavior Kinematics")
    st.caption("State estimation (*Resting, Walking, Running, Alert, Stalking, Observing*) derived from velocity and directional variance")

    if no_data or "behavior" not in det_df.columns:
        st.info("No behavior data available in database.")
        st.stop()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Behavioral State Distribution")
        beh_counts = det_df["behavior"].value_counts().reset_index()
        beh_counts.columns = ["behavior", "count"]
        fig = px.bar(
            beh_counts, x="behavior", y="count", color="behavior",
            color_discrete_sequence=px.colors.qualitative.Prism,
            labels={"behavior": "Behavior State", "count": "Instance Count"},
        )
        fig.update_layout(template="plotly_dark", height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Track Trajectory Reconstruction (2D Field View)")
        if "track_id" in det_df.columns and "x" in det_df.columns and "y" in det_df.columns:
            track_ids = sorted([t for t in det_df["track_id"].unique() if t is not None])
            if track_ids:
                selected_track = st.selectbox("Select Animal Track ID", track_ids, index=0)
                t_df = det_df[det_df["track_id"] == selected_track].sort_values("frame_num")
                fig = px.line(
                    t_df, x="x", y="y", markers=True,
                    color="behavior",
                    title=f"Path History for Track #{selected_track}",
                )
                fig.update_yaxes(autorange="reversed")
                fig.update_layout(template="plotly_dark", height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No tracked object IDs available.")

    # State Transition Matrix
    st.subheader("Behavior Transition Probability Matrix")
    if "track_id" in det_df.columns:
        transitions: Dict[tuple[str, str], int] = {}
        for _, grp in det_df.sort_values("frame_num").groupby("track_id"):
            behaviors = grp["behavior"].dropna().tolist()
            for prev, curr in zip(behaviors[:-1], behaviors[1:]):
                if prev != curr:
                    transitions[(prev, curr)] = transitions.get((prev, curr), 0) + 1

        if transitions:
            all_behaviors = sorted({b for pair in transitions for b in pair})
            matrix = pd.DataFrame(0, index=all_behaviors, columns=all_behaviors)
            for (prev, curr), count in transitions.items():
                matrix.loc[prev, curr] = count

            fig = px.imshow(
                matrix, text_auto=True,
                color_continuous_scale="YlOrRd",
                labels=dict(x="Transitioned To", y="Transitioned From", color="Count"),
            )
            fig.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No inter-behavior transitions recorded yet.")


# ═══════════════════════════════════════════════════════════════
# PAGE 4: Disturbance Threat Radar
# ═══════════════════════════════════════════════════════════════

elif page == "⚠️ Disturbance Threat Radar":
    st.title("⚠️ Human-Wildlife Disturbance Threat Radar")
    st.caption("Automated spatial proximity and alert severity calculation")

    if dist_df.empty:
        st.info("No disturbance incidents logged in the database.")
        st.stop()

    # Severity Filter
    severities = dist_df["severity"].unique().tolist() if "severity" in dist_df.columns else []
    sel_sev = st.multiselect("Filter by Threat Severity", severities, default=severities)
    filtered_dist = dist_df[dist_df["severity"].isin(sel_sev)] if sel_sev else dist_df

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Severity Distribution")
        if "severity" in filtered_dist.columns:
            sev_counts = filtered_dist["severity"].value_counts()
            fig = px.pie(
                values=sev_counts.values, names=sev_counts.index,
                color=sev_counts.index,
                color_discrete_map={
                    "CRITICAL": "#ef4444",
                    "HIGH": "#f97316",
                    "MEDIUM": "#eab308",
                    "LOW": "#22c55e",
                },
                hole=0.4,
            )
            fig.update_layout(template="plotly_dark", height=340)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Human-to-Wildlife Distance Distribution")
        if "distance_px" in filtered_dist.columns:
            fig = px.histogram(
                filtered_dist, x="distance_px", nbins=25,
                color_discrete_sequence=["#ef4444"],
                labels={"distance_px": "Proximity Distance (Pixels)"},
            )
            fig.add_vline(x=250, line_dash="dash", line_color="#f97316", annotation_text="Standard Threshold (250px)")
            fig.add_vline(x=100, line_dash="dash", line_color="#ef4444", annotation_text="Critical Threshold (100px)")
            fig.update_layout(template="plotly_dark", height=340)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Disturbance Incident Log")
    st.dataframe(filtered_dist, use_container_width=True, height=350)


# ═══════════════════════════════════════════════════════════════
# PAGE 5: Live AI Inference Playground (Optimized)
# ═══════════════════════════════════════════════════════════════

elif page == "🔬 Live AI Inference Playground":
    st.title("🔬 Live AI Model Inference Playground")
    st.caption("Upload any wildlife image or test built-in samples to evaluate Stage 1 (YOLOv8) & Stage 2 (EfficientNetV2-S).")

    # Dynamic Model Loader
    @st.cache_resource(show_spinner=False)
    def load_ai_models_cached(yolo_path: str, cls_path: str):
        try:
            import torch
            from ultralytics import YOLO
            from detector_ai.stage2_classifier import SpeciesClassifier

            device = "cuda" if torch.cuda.is_available() else "cpu"
            yolo_model = YOLO(yolo_path)
            
            classifier = None
            if Path(cls_path).exists():
                try:
                    classifier = SpeciesClassifier(weights_path=cls_path, conf_threshold=0.30, device=device)
                except Exception as e:
                    st.warning(f"Classifier load warning: {e}")
            return yolo_model, classifier, device
        except Exception as err:
            return None, None, f"Error: {err}"

    yolo_model, classifier, dev_status = load_ai_models_cached(
        str(DEFAULT_YOLO_WEIGHTS),
        str(DEFAULT_CLASSIFIER_WEIGHTS),
    )

    if yolo_model is None:
        st.error(f"Failed to load detection models: {dev_status}")
        st.stop()

    # Controls Bar
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 2])
    with ctrl1:
        conf_thresh = st.slider("Detection Confidence Threshold", 0.10, 0.90, 0.25, 0.05)
    with ctrl2:
        enable_stage2 = st.checkbox("Enable Stage 2 Classifier (EfficientNetV2-S)", value=True)
    with ctrl3:
        st.markdown(f"**Compute Device:** `{dev_status.upper()}`")

    st.markdown("---")

    # Mode Selector: Upload vs Dataset Samples
    input_mode = st.radio(
        "Select Image Input Mode:",
        options=["📁 Upload Your Own Image", "🐾 Test with Dataset Samples"],
        horizontal=True,
    )
    
    img_bgr: Optional[np.ndarray] = None
    image_source_label = ""

    if input_mode == "📁 Upload Your Own Image":
        uploaded_file = st.file_uploader(
            "Upload Image (JPG, PNG, JPEG)", 
            type=["jpg", "jpeg", "png"],
            help="Drag & drop or browse any wildlife photo from your device"
        )
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_source_label = f"Uploaded File: {uploaded_file.name}"
            if img_bgr is None:
                st.error("❌ Failed to decode uploaded image. Please try a standard JPG/PNG file.")
    else:
        sample_dir = PROJECT_ROOT / "data" / "samples"
        if not sample_dir.exists() or len(list(sample_dir.glob("*.jpg"))) == 0:
            sample_dir = PROJECT_ROOT / "multispecies_dataset" / "images" / "val"
        sample_files = list(sample_dir.glob("*.jpg"))[:18] if sample_dir.exists() else []
        if sample_files:
            sample_names = [f.name for f in sample_files]
            chosen_sample = st.selectbox("Choose a sample validation image:", sample_names, index=0)
            if chosen_sample:
                sample_path = sample_dir / chosen_sample
                img_bgr = cv2.imread(str(sample_path))
                image_source_label = f"Sample: {chosen_sample}"
        else:
            st.info("ℹ️ No sample validation images found locally.")

    if img_bgr is not None:
        st.markdown(f"**Current Input:** `{image_source_label}`")
        col1, col2 = st.columns(2)
        h, w = img_bgr.shape[:2]

        with col1:
            st.subheader("Original Input Image")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Run Real-Time Inference
        with st.spinner("Running 2-Stage Cascaded Neural Network (YOLOv8 + EfficientNetV2-S)..."):
            results = yolo_model(img_bgr, conf=conf_thresh, verbose=False)
            annotated = img_bgr.copy()

            results_summary = []
            crops_list = []

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

                    crop = None
                    # Stage 2 Fine-Grained Classifier
                    if (x2 > x1) and (y2 > y1):
                        crop = img_bgr[y1:y2, x1:x2]
                        if enable_stage2 and classifier and classifier.is_available and crop.size > 0:
                            cls_res = classifier.classify(crop)
                            if cls_res:
                                final_label, final_conf = cls_res
                                stage2_applied = True

                    # Color coding: Green for animals, Cyan for human
                    is_human = final_label.lower() in ("person", "human")
                    color = (255, 200, 0) if is_human else (0, 255, 100)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                    tag = f"{final_label} ({final_conf:.1%})"
                    if stage2_applied:
                        tag += " [S2]"

                    cv2.putText(annotated, tag, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    if crop is not None and crop.size > 0:
                        crops_list.append((final_label, final_conf, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

                    results_summary.append({
                        "Entity": format_species(final_label),
                        "Classification Stage": "Stage 2 (EffNetV2-S)" if stage2_applied else "Stage 1 (YOLOv8)",
                        "Confidence": f"{final_conf:.2%}",
                        "Bounding Box [x1, y1, x2, y2]": f"[{x1}, {y1}, {x2}, {y2}]",
                    })

        with col2:
            st.subheader(f"Detections ({len(results_summary)} Found)")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        if results_summary:
            st.subheader("🔍 Detection Breakdown Table")
            st.dataframe(pd.DataFrame(results_summary), use_container_width=True)

            if crops_list:
                st.subheader("📸 Extracted Entity Crops (Stage 2 Inputs)")
                crop_cols = st.columns(min(len(crops_list), 4))
                for c_idx, (c_label, c_conf, c_img) in enumerate(crops_list):
                    col_target = crop_cols[c_idx % len(crop_cols)]
                    with col_target:
                        st.image(c_img, caption=f"{format_species(c_label)} ({c_conf:.1%})", use_container_width=True)
        else:
            st.warning(f"No objects detected at confidence threshold {conf_thresh:.0%}. Try lowering the slider above.")


# ═══════════════════════════════════════════════════════════════
# PAGE 6: Export & Field Reports
# ═══════════════════════════════════════════════════════════════

elif page == "📥 Export & Field Reports":
    st.title("📥 Field Data Export & Summary Reports")
    st.caption("Generate structured CSV/JSON downloads for wildlife researchers")

    if no_data and dist_df.empty:
        st.info("No telemetry data to export.")
        st.stop()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Detections Telemetry")
        if not det_df.empty:
            csv = det_df.to_csv(index=False)
            st.download_button("📄 Download Detections CSV", csv, "wildlife_detections.csv", "text/csv", use_container_width=True)
            json_str = det_df.to_json(orient="records", indent=2, date_format="iso")
            st.download_button("📄 Download Detections JSON", json_str, "wildlife_detections.json", "application/json", use_container_width=True)

    with c2:
        st.subheader("Disturbance Incident Reports")
        if not dist_df.empty:
            csv_d = dist_df.to_csv(index=False)
            st.download_button("⚠️ Download Disturbances CSV", csv_d, "wildlife_disturbances.csv", "text/csv", use_container_width=True)
            json_d = dist_df.to_json(orient="records", indent=2, date_format="iso")
            st.download_button("⚠️ Download Disturbances JSON", json_d, "wildlife_disturbances.json", "application/json", use_container_width=True)

    st.markdown("---")
    st.caption(f"Source Database: `{db_path}`")
    st.caption(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")


