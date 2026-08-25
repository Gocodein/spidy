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
from typing import Optional, List, Dict

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
)
from detector_ai.stage1_detector import AnimalDetector
from detector_ai.stage2_classifier import SpeciesClassifier

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
        padding: 20px 16px;
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
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .metric-card p {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 6px 0 0 0;
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


# ──────────────────────────────────────────────── Database Helpers ──

@st.cache_resource
def get_db_connection(db_path: str):
    """Return a sqlite3 connection (cached across reruns)."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn


def load_sessions(conn) -> pd.DataFrame:
    """Load session summary rows into a DataFrame."""
    try:
        df = pd.read_sql_query("SELECT * FROM sessions ORDER BY id DESC", conn)
        return df
    except Exception:
        return pd.DataFrame()


def load_detections(conn, session_id: Optional[int] = None) -> pd.DataFrame:
    """Load detection rows into a DataFrame, optionally filtered by session_id."""
    try:
        query = "SELECT * FROM detections"
        if session_id is not None:
            query += f" WHERE session_id = {session_id}"
        query += " ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def load_disturbances(conn, session_id: Optional[int] = None) -> pd.DataFrame:
    """Load disturbance events into a DataFrame, optionally filtered by session_id."""
    try:
        query = "SELECT * FROM disturbance_events"
        if session_id is not None:
            query += f" WHERE session_id = {session_id}"
        query += " ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────── Species Icons & Theme ──

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

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

conn = get_db_connection(db_path)
sessions_df = load_sessions(conn)

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

# Load data based on filter
det_df = load_detections(conn, session_id=selected_session)
dist_df = load_disturbances(conn, session_id=selected_session)
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
        metric_card("Unique Tracked Animals", f"{unique_tracks:,}", "🔗")
    with c3:
        metric_card("Species Detected", f"{unique_species}/9", "🦁")
    with c4:
        metric_card("Disturbance Incidents", f"{total_dist:,}", "⚠️")
    with c5:
        metric_card("Critical Alerts", f"{critical_dist:,}", "🚨")

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
# PAGE 5: Live AI Inference Playground
# ═══════════════════════════════════════════════════════════════

elif page == "🔬 Live AI Inference Playground":
    st.title("🔬 Live AI Model Inference Playground")
    st.caption("Upload any wildlife image to test the 9-Class YOLOv8 Detector and EfficientNetV2-S Classifier in real-time.")

    uploaded_file = st.file_uploader("Upload Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    @st.cache_resource
    def load_ai_models():
        detector = AnimalDetector(weights=DEFAULT_YOLO_WEIGHTS, conf_threshold=0.30, tracker="bytetrack.yaml")
        classifier = SpeciesClassifier(weights_path=DEFAULT_CLASSIFIER_WEIGHTS, conf_threshold=0.50, device="cuda")
        return detector, classifier

    try:
        detector, classifier = load_ai_models()
        model_loaded = True
    except Exception as e:
        st.error(f"Failed to load AI models: {e}")
        model_loaded = False

    if uploaded_file and model_loaded:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Run Inference
        with st.spinner("Running YOLOv8 Localization & EfficientNetV2-S Classification..."):
            detections = detector.detect_no_track(img_bgr)
            annotated = img_bgr.copy()

            results_summary = []
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                crop = img_bgr[y1:y2, x1:x2]

                species_label = det.class_name
                species_conf = det.confidence

                if classifier.is_available and crop.size > 0:
                    cls_res = classifier.classify(crop)
                    if cls_res:
                        species_label, species_conf = cls_res

                # Draw bounding box
                color = (0, 255, 0) if det.category == "animal" else (255, 0, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                label_text = f"{species_label} ({species_conf:.1%})"
                cv2.putText(annotated, label_text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                results_summary.append({
                    "Detected Entity": format_species(species_label),
                    "Category": det.category.capitalize(),
                    "Confidence": f"{species_conf:.2%}",
                    "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]",
                })

        with col2:
            st.subheader(f"Detections ({len(detections)} Found)")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        if results_summary:
            st.subheader("Inference Breakdown")
            st.dataframe(pd.DataFrame(results_summary), use_container_width=True)
        else:
            st.warning("No wildlife or human detected in this image at 30% confidence.")


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

