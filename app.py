"""
# ==============================================================================
# 🐾 DETECTOR AI / SPIDY — Hugging Face Spaces Entry Point
# ==============================================================================
# Multi-Species Wildlife Detection, Behavior Tracking & Disturbance Analytics
# Protected under Indian Patent Application No. 202531071175 A
# ==============================================================================
"""

import sys
import runpy
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Launch the full research dashboard
dashboard_file = PROJECT_ROOT / "dashboard" / "app.py"
runpy.run_path(str(dashboard_file), run_name="__main__")
