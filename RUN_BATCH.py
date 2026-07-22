#!/usr/bin/env python3
"""Execute the pipeline batch file."""
import subprocess
import sys

batch_file = r"C:\Users\BIDHAN\Desktop\Projects\Paper project 1\bangla-hatespeech-fragility\RUN_FULL_PIPELINE.bat"

print(f"Executing: {batch_file}\n")
result = subprocess.run([batch_file], shell=True)
sys.exit(result.returncode)
