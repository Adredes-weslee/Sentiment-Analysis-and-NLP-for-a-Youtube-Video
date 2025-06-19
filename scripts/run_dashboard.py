"""Main script to launch the Streamlit dashboard application."""
import os
import sys
from pathlib import Path

def main():
    """Sets up the Python path and launches the Streamlit app."""
    # Add project root to the Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    app_path = project_root / "dashboard" / "app.py"
    command = f"streamlit run {app_path}"

    print(f"Running command: {command}")
    os.system(command)

if __name__ == "__main__":
    main()
