#!/usr/bin/env python3
"""Debug script to test environment variable passing."""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables like the GUI does
os.environ['THEME'] = "L'IA ci ruberà il lavoro?"
os.environ['SEED'] = '1748682400'

print("Environment variables set:")
print(f"THEME: {repr(os.environ.get('THEME'))}")
print(f"SEED: {repr(os.environ.get('SEED'))}")

# Import settings to see what they read
try:
    from src.config.settings import THEME, SEED
    print("\nValues from settings.py:")
    print(f"THEME: {repr(THEME)}")
    print(f"SEED: {repr(SEED)}")
    
    from src.utils import sanitize_filename
    output_dir = f"output/{sanitize_filename(THEME)}_{SEED}_iterative"
    print(f"\nCalculated output dir: {output_dir}")
    print(f"Directory exists: {os.path.exists(output_dir)}")
    
    ordered_file = os.path.join(output_dir, "ordered_sentences.json")
    print(f"Ordered file path: {ordered_file}")
    print(f"File exists: {os.path.exists(ordered_file)}")
    
    if os.path.exists(ordered_file):
        import json
        with open(ordered_file) as f:
            data = json.load(f)
        print(f"Total clips in file: {data.get('total_clips', 'Unknown')}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
