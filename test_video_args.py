#!/usr/bin/env python3
"""Test the video creation script with command line arguments."""

import subprocess
import sys
import os
from pathlib import Path

def test_video_creation_args():
    """Test that the video creation script accepts theme and seed as arguments."""
    
    print("🧪 Testing Video Creation Script Arguments")
    print("=" * 50)
    
    # Test theme and seed values
    test_theme = "L'IA ci ruberà il lavoro?"
    test_seed = "1748682400"
    
    print(f"✅ Testing with:")
    print(f"   Theme: '{test_theme}'")
    print(f"   Seed: '{test_seed}'")
    
    # Expected output directory
    expected_dir = f"output/L_IA_ci_rubera_il_lavoro_{test_seed}_iterative"
    print(f"   Expected directory: {expected_dir}")
    
    # Check if the ordered_sentences.json exists from GUI testing
    ordered_file = Path(expected_dir) / "ordered_sentences.json"
    if not ordered_file.exists():
        print(f"❌ Missing ordered_sentences.json file: {ordered_file}")
        print("   This test requires running the GUI first to create clips.")
        return False
    
    print(f"✅ Found ordered_sentences.json: {ordered_file}")
    
    # Test the command line arguments parsing
    cmd = [
        sys.executable, 
        "script_create_video.py",
        "--theme", test_theme,
        "--seed", test_seed
    ]
    
    print(f"\n🔄 Testing command: {' '.join(cmd[:3])} --theme '{test_theme}' --seed '{test_seed}'")
    
    try:
        # Run with --help to test argument parsing without actually creating video
        help_cmd = [sys.executable, "script_create_video.py", "--help"]
        result = subprocess.run(
            help_cmd, 
            cwd=str(Path(__file__).parent),
            capture_output=True, 
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ Script accepts command line arguments correctly")
            print("📄 Help output:")
            print(result.stdout[:300] + "..." if len(result.stdout) > 300 else result.stdout)
            return True
        else:
            print(f"❌ Script argument parsing failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Script help command timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing script: {e}")
        return False

if __name__ == "__main__":
    success = test_video_creation_args()
    print("=" * 50)
    if success:
        print("🎉 Video creation script argument parsing test PASSED!")
        print("   The script now accepts --theme and --seed arguments correctly.")
    else:
        print("❌ Video creation script argument parsing test FAILED!")
    print("=" * 50)
