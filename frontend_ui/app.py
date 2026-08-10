"""
Frontend entry point redirecting to root app.py for Streamlit deployment compatibility.
"""
import os
import sys

# Add project root to Python module search path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import main application module from root
import app