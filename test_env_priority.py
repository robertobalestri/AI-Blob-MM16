#!/usr/bin/env python3
"""Test script to verify environment variable priority."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_env_priority():
    """Test that runtime environment variables take precedence over .env file."""
    
    print("🧪 Testing Environment Variable Priority")
    print("=" * 50)
    
    # Set environment variables at runtime (simulating GUI behavior)
    test_theme = "L'IA ci ruberà il lavoro?"
    test_seed = "1748682400"
    
    os.environ['THEME'] = test_theme
    os.environ['SEED'] = test_seed
    
    print(f"✅ Set runtime environment variables:")
    print(f"   THEME = '{test_theme}'")
    print(f"   SEED = '{test_seed}'")
    
    # Check .env file contents
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        print(f"\n📄 .env file contents:")
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    print(f"   {line.strip()}")
    
    # Import settings to see which values are loaded
    print(f"\n🔄 Importing settings...")
    from src.config.settings import THEME, SEED
    
    print(f"\n📊 Final loaded values:")
    print(f"   THEME = '{THEME}'")
    print(f"   SEED = '{SEED}'")
    
    # Verify the fix worked
    success = True
    if THEME != test_theme:
        print(f"❌ THEME mismatch! Expected '{test_theme}', got '{THEME}'")
        success = False
    else:
        print(f"✅ THEME correct: '{THEME}'")
    
    if SEED != test_seed:
        print(f"❌ SEED mismatch! Expected '{test_seed}', got '{SEED}'")
        success = False
    else:
        print(f"✅ SEED correct: '{SEED}'")
    
    print("=" * 50)
    if success:
        print("🎉 Environment variable priority fix SUCCESSFUL!")
        print("   Runtime variables correctly override .env file values.")
    else:
        print("❌ Environment variable priority fix FAILED!")
        print("   .env file values are still overriding runtime variables.")
    
    return success

if __name__ == "__main__":
    test_env_priority()
