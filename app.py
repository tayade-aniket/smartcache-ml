"""
Root entry point for SmartCache ML Streamlit Dashboard.
Delegates execution to dashboard/app.py for seamless cloud hosting
on platforms such as Hugging Face Spaces and Streamlit Community Cloud.
"""

import sys
from pathlib import Path
import runpy

# Ensure repository root and src/ are in Python path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Execute the dashboard application
target_file = ROOT_DIR / "dashboard" / "app.py"
runpy.run_path(str(target_file), run_name="__main__")
