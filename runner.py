import subprocess
import sys

# Try to run the pipeline
result = subprocess.call([sys.executable, r"C:\Users\BIDHAN\Desktop\Projects\Paper project 1\bangla-hatespeech-fragility\EXECUTE_FULL_PIPELINE.py"])
print(f"\nExit code: {result}")
