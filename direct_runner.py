#!/usr/bin/env python3
"""
Direct pipeline execution that doesn't rely on shell initialization.
Runs all 5 milestones and captures the results.
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

def main():
    project_dir = Path(r"C:\Users\BIDHAN\Desktop\Projects\Paper project 1\bangla-hatespeech-fragility")
    os.chdir(str(project_dir))
    
    # Run using the bat file through cmd.exe
    bat_file = project_dir / "run_pipeline.bat"
    
    print(f"Project directory: {project_dir}")
    print(f"Batch file: {bat_file}")
    print(f"Batch file exists: {bat_file.exists()}")
    print(f"\nStarting pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not bat_file.exists():
        # Fallback to direct Python execution
        print("Batch file not found, using direct Python execution...\n")
        result = subprocess.call([sys.executable, str(project_dir / "execute_pipeline.py")])
    else:
        # Run through cmd
        result = subprocess.call(['cmd.exe', '/c', str(bat_file)])
    
    return result

if __name__ == "__main__":
    sys.exit(main())
