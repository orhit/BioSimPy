#!/usr/bin/env python3
"""
BioSimPy - Quick Start Script
"""

import os
import sys
import subprocess

def main():
    print("🧬 BioSimPy - BioMEMS Simulation Platform")
    print("=" * 50)
    
    # Check if streamlit is available
    try:
        import streamlit
        print("✅ Streamlit found!")
    except ImportError:
        print("❌ Streamlit not found. Please install dependencies:")
        print("pip install -r requirements.txt")
        return
    
    # Launch the dashboard
    print("🚀 Launching BioSimPy Dashboard...")
    dashboard_path = os.path.join(os.path.dirname(__file__), 'dashboard', 'app.py')
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])
    except KeyboardInterrupt:
        print("\n👋 BioSimPy closed")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()