# Run: streamlit run dashboard/app.py
# Or:  python -m streamlit run dashboard/app.py
"""
DETECTOR AI — Research Dashboard
====================================
Interactive Streamlit dashboard for exploring detection sessions,
species analytics, behavior patterns, and disturbance events.

Reads from the SQLite database written by the pipeline (Stage 6).
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Resolve project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector_ai.config import DEFAULT_DB_PATH

# ──────────────────────────────────────────────── Page Config ──

st.set_page_config(
    page_title="DETECTOR AI Dashboard",
    page_icon="🐯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────── Custom CSS ──

st.markdown("""
<style>
    /* Dark-themed tweaks */
    .stApp { }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card h2 {
        color: #f97316;
        font-size: 2.2rem;
        margin: 0;
    }
    .metric-card p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 4px 0 0 0;
    }
    .severity-CRITICAL { color: #ef4444; font-weight: bold; }
    .severity-HIGH     { color: #f97316; font-weight: bold; }
    .severity-MEDIUM   { color: #eab308; }
    .severity-LOW      { color: #22c55e; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────── DB Helpers ──

@st.cache_resource
def get_db_connection(db_path: str):
    """Return a sqlite3 connection (cached across reruns)."""
    import sqlite3
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn


def load_detections(conn) -> pd.DataFrame:
    """Load all detection rows into a DataFrame."""
    try:
        df = pd.read_sql_query(
            "SELECT * FROM detections ORDER BY timestamp ASC", conn,
        )
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def load_disturbances(conn) -> pd.DataFrame:
    """Load all disturbance events into a DataFrame."""
    try:
        df = pd.read_sql_query(
            "SELECT * FROM disturbance_events ORDER BY timestamp ASC", conn,
        )
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────── Sidebar ──

st.sidebar.image("https://img.icons8.com/emoji/96/tiger-face.png", width=64)
st.sidebar.title("🐯 DETECTOR AI")
st.sidebar.caption("Endangered Species Monitoring Dashboard")
st.sidebar.markdown("---")

# Database selector
db_path = st.sidebar.text_input(
    "Database path", value=str(DEFAULT_DB_PATH),
)

page = st.sidebar.radio(
    "Navigation",
    options=[
        "📊 Overview",
        "🦁 Species Analysis",
        "🚶 Behavior Analysis",
        "⚠️ Disturbance Events",
        "📥 Export",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with [Streamlit](https://streamlit.io) & "
    "[Plotly](https://plotly.com)",
)

# ── Connect ──
conn = get_db_connection(db_path)
det_df = load_detections(conn)
dist_df = load_disturbances(conn)

no_data = det_df.empty


# ──────────────────────────────── Helper: metric card ──

def metric_card(label: str, value, icon: str = ""):
    st.markdown(
        f'<div class="metric-card">'
        f'<h2>{icon} {value}</h2>'
        f'<p>{label}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("📊 Session Overview")

    if no_data:
        st.info("No detection data found. Run the pipeline first to populate the database.")
        st.stop()

    # Key metrics
    total_det = len(det_df)
    unique_tracks = det_df["track_id"].nunique() if "track_id" in det_df.columns else 0
    total_dist = len(dist_df)
    unique_species = det_df["species"].nunique() if "species" in det_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Detections", f"{total_det:,}", "📍")
    with c2:
        metric_card("Unique Tracks", f"{unique_tracks:,}", "🔗")
    with c3:
        metric_card("Disturbance Events", f"{total_dist:,}", "⚠️")
    with c4:
        metric_card("Species Detected", f"{unique_species:,}", "🦁")

    st.markdown("---")

    # Detection timeline
    if "timestamp" in det_df.columns and not det_df["timestamp"].isna().all():
        st.subheader("Detection Timeline")
        timeline = det_df.set_index("timestamp").resample("1min").size().reset_index(name="count")
        fig = px.area(
            timeline, x="timestamp", y="count",
            labels={"timestamp": "Time", "count": "Detections"},
            color_discrete_sequence=["#f97316"],
        )
        fig.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig, width="stretch")
    elif "frame_num" in det_df.columns:
        st.subheader("Detection Timeline (by frame)")
        frame_counts = det_df.groupby("frame_num").size().reset_index(name="count")
        fig = px.area(
            frame_counts, x="frame_num", y="count",
            labels={"frame_num": "Frame", "count": "Detections"},
            color_discrete_sequence=["#f97316"],
        )
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, width="stretch")

    # Recent detections table
    st.subheader("Recent Detections")
    st.dataframe(det_df.tail(50).iloc[::-1], width="stretch", height=300)


# ═══════════════════════════════════════════════════════════════
# PAGE: Species Analysis
# ═══════════════════════════════════════════════════════════════

elif page == "🦁 Species Analysis":
    st.title("🦁 Species Analysis")

    if no_data:
        st.info("No detection data available.")
        st.stop()

    col_left, col_right = st.columns(2)

    # Species distribution (pie)
    if "species" in det_df.columns:
        with col_left:
            st.subheader("Species Distribution")
            species_counts = det_df["species"].value_counts().reset_index()
            species_counts.columns = ["species", "count"]
            fig = px.pie(
                species_counts, names="species", values="count",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig, width="stretch")

    # Confidence histogram
    if "confidence" in det_df.columns:
        with col_right:
            st.subheader("Confidence Distribution")
            fig = px.histogram(
                det_df, x="confidence", nbins=40,
                color_discrete_sequence=["#06b6d4"],
                labels={"confidence": "Detection Confidence"},
            )
            fig.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig, width="stretch")

    # Detection heatmap (bbox centers)
    if all(c in det_df.columns for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")):
        st.subheader("Detection Heatmap (bbox centres)")
        det_df["cx"] = (det_df["bbox_x1"] + det_df["bbox_x2"]) / 2
        det_df["cy"] = (det_df["bbox_y1"] + det_df["bbox_y2"]) / 2
        fig = px.density_heatmap(
            det_df, x="cx", y="cy", nbinsx=40, nbinsy=30,
            color_continuous_scale="Inferno",
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, width="stretch")


# ═══════════════════════════════════════════════════════════════
# PAGE: Behavior Analysis
# ═══════════════════════════════════════════════════════════════

elif page == "🚶 Behavior Analysis":
    st.title("🚶 Behavior Analysis")

    if no_data or "behavior" not in det_df.columns:
        st.info("No behavior data available.")
        st.stop()

    col_left, col_right = st.columns(2)

    # Behavior distribution (pie)
    with col_left:
        st.subheader("Behavior Distribution")
        beh_counts = det_df["behavior"].value_counts().reset_index()
        beh_counts.columns = ["behavior", "count"]
        fig = px.pie(
            beh_counts, names="behavior", values="count",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4,
        )
        fig.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig, width="stretch")

    # Behavior over time per track
    with col_right:
        st.subheader("Behavior Over Time")
        if "track_id" in det_df.columns:
            track_ids = sorted(det_df["track_id"].unique())
            selected = st.selectbox("Select Track ID", track_ids, index=0)
            track_df = det_df[det_df["track_id"] == selected]
            time_col = "timestamp" if "timestamp" in track_df.columns else "frame_num"
            if time_col in track_df.columns:
                fig = px.scatter(
                    track_df, x=time_col, y="behavior",
                    color="behavior",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(template="plotly_dark", height=380)
                st.plotly_chart(fig, width="stretch")

    # Behavior transition matrix
    st.subheader("Behavior Transition Matrix")
    if "track_id" in det_df.columns:
        transitions: dict[tuple[str, str], int] = {}
        for _, grp in det_df.sort_values("frame_num").groupby("track_id"):
            behaviors = grp["behavior"].tolist()
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
                labels=dict(x="To", y="From", color="Count"),
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No behavior transitions recorded yet.")


# ═══════════════════════════════════════════════════════════════
# PAGE: Disturbance Events
# ═══════════════════════════════════════════════════════════════

elif page == "⚠️ Disturbance Events":
    st.title("⚠️ Disturbance Events")

    if dist_df.empty:
        st.info("No disturbance events recorded.")
        st.stop()

    # Severity filter
    severities = dist_df["severity"].unique().tolist() if "severity" in dist_df.columns else []
    sel_sev = st.multiselect("Filter by severity", severities, default=severities)
    filtered = dist_df[dist_df["severity"].isin(sel_sev)] if sel_sev else dist_df

    # Styled event log
    st.subheader("Event Log")

    def highlight_severity(val):
        colors = {
            "CRITICAL": "background-color: #7f1d1d; color: #fca5a5;",
            "HIGH":     "background-color: #78350f; color: #fdba74;",
            "MEDIUM":   "background-color: #713f12; color: #fde047;",
            "LOW":      "background-color: #14532d; color: #86efac;",
        }
        return colors.get(val, "")

    styled = filtered.style.applymap(
        highlight_severity, subset=["severity"] if "severity" in filtered.columns else [],
    )
    st.dataframe(styled, use_container_width=True, height=400)

    col_left, col_right = st.columns(2)

    # Disturbance frequency chart
    with col_left:
        st.subheader("Disturbance Frequency")
        time_col = "timestamp" if "timestamp" in filtered.columns else "frame_num"
        if time_col in filtered.columns:
            if time_col == "timestamp" and not filtered["timestamp"].isna().all():
                freq = filtered.set_index("timestamp").resample("5min").size().reset_index(name="count")
                fig = px.bar(
                    freq, x="timestamp", y="count",
                    color_discrete_sequence=["#ef4444"],
                )
            else:
                freq = filtered.groupby("frame_num").size().reset_index(name="count")
                fig = px.bar(freq, x="frame_num", y="count", color_discrete_sequence=["#ef4444"])
            fig.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Severity distribution
    with col_right:
        st.subheader("Severity Distribution")
        if "severity" in filtered.columns:
            sev_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            sev_counts = filtered["severity"].value_counts().reindex(sev_order, fill_value=0)
            fig = px.bar(
                x=sev_counts.index, y=sev_counts.values,
                labels={"x": "Severity", "y": "Count"},
                color=sev_counts.index,
                color_discrete_map={
                    "LOW": "#22c55e", "MEDIUM": "#eab308",
                    "HIGH": "#f97316", "CRITICAL": "#ef4444",
                },
            )
            fig.update_layout(template="plotly_dark", height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE: Export
# ═══════════════════════════════════════════════════════════════

elif page == "📥 Export":
    st.title("📥 Export Data")

    if no_data and dist_df.empty:
        st.info("No data to export.")
        st.stop()

    st.subheader("Detections")
    if not det_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            csv = det_df.to_csv(index=False)
            st.download_button(
                "📄 Download Detections CSV",
                csv, "detections.csv", "text/csv",
            )
        with col2:
            json_str = det_df.to_json(orient="records", indent=2, date_format="iso")
            st.download_button(
                "📄 Download Detections JSON",
                json_str, "detections.json", "application/json",
            )
        st.dataframe(det_df.head(20), use_container_width=True)

    st.markdown("---")

    st.subheader("Disturbance Events")
    if not dist_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            csv = dist_df.to_csv(index=False)
            st.download_button(
                "📄 Download Disturbances CSV",
                csv, "disturbances.csv", "text/csv",
            )
        with col2:
            json_str = dist_df.to_json(orient="records", indent=2, date_format="iso")
            st.download_button(
                "📄 Download Disturbances JSON",
                json_str, "disturbances.json", "application/json",
            )
        st.dataframe(dist_df.head(20), use_container_width=True)

    st.markdown("---")
    st.caption(f"Database: `{db_path}`")
    st.caption(f"Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
