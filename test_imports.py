#!/usr/bin/env python3
"""
Test script to verify that all GUI imports work correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all GUI-related imports."""
    print("🔍 Testing GUI imports...")
    
    try:
        print("  ✓ Testing basic imports...")
        import streamlit as st
        from src.config.settings import VECTOR_STORE_DIR
        print("  ✓ Basic imports successful")
        
        print("  ✓ Testing GUI config...")
        from src.config.gui_config import GUI_SETTINGS, BUTTON_SETTINGS, STATUS_MESSAGES
        print("  ✓ GUI config imports successful")
        
        print("  ✓ Testing AI models...")
        from src.ai_models import AIModelsService, LLMType
        print("  ✓ AI models imports successful")
        
        print("  ✓ Testing GUI service...")
        from src.gui.gui_service import GUIService, ClipCandidate, SelectionState
        print("  ✓ GUI service imports successful")
        
        print("  ✓ Testing GUI components...")
        from src.gui.components import render_clip_card, render_progress_bar, render_theme_input
        print("  ✓ GUI components imports successful")
        
        print("\n🎉 All imports successful! The GUI should work correctly.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("The GUI may not work correctly.")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
