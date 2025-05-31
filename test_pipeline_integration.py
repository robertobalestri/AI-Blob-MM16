#!/usr/bin/env python3
"""
Test Pipeline Integration

This script tests that the GUI creates the correct JSON format
that can be consumed by the video creation pipeline.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.gui.gui_service import GUIService, SelectionState, ClipCandidate

def create_mock_clip(order: int, query: str) -> ClipCandidate:
    """Create a mock clip for testing."""
    return ClipCandidate(
        clip_id=f"test_clip_{order}",
        page_content=f"This is test content for clip {order}",
        metadata={
            "video_id": f"video_{order}",
            "sentence_number": order,
            "duration": 5.0,
            "start_time": order * 10,
            "end_time": (order * 10) + 5
        },
        score=0.8 + (order * 0.01),
        original_query_phrase=query,
        previous_sentence={"text": f"Previous sentence {order}"},
        next_sentence={"text": f"Next sentence {order}"}
    )

def test_json_format():
    """Test that the GUI exports the correct JSON format."""
    print("🧪 Testing GUI JSON export format...")
    
    # Create a test selection state
    state = SelectionState(
        theme="Test Theme for AI",
        seed=12345,
        max_iterations=3
    )
    
    # Add some mock clips
    clips = [
        create_mock_clip(1, "test query 1"),
        create_mock_clip(2, "test query 2"), 
        create_mock_clip(3, "test query 3")
    ]
    
    gui_service = GUIService()
    for clip in clips:
        gui_service.add_selected_clip(state, clip)
    
    # Export to a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "Test_Theme_for_AI_12345_iterative")
        
        try:
            ordered_file_path = gui_service.export_selection_to_ordered_file(state, output_dir)
            print(f"✅ Export successful: {ordered_file_path}")
            
            # Verify the file exists
            assert os.path.exists(ordered_file_path), "Ordered file does not exist"
            
            # Load and verify the JSON structure
            with open(ordered_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required top-level keys
            required_keys = ["theme", "total_clips", "ordered_phrases"]
            for key in required_keys:
                assert key in data, f"Missing required key: {key}"
            
            print(f"✅ JSON structure valid")
            print(f"   - Theme: {data['theme']}")
            print(f"   - Total clips: {data['total_clips']}")
            print(f"   - Ordered phrases count: {len(data['ordered_phrases'])}")
            
            # Check ordered phrases structure
            for i, phrase in enumerate(data['ordered_phrases']):
                required_phrase_keys = [
                    "matched_phrase", "order", "query_phrase_that_led_to_this_clip",
                    "selection_justification", "retrieval_score", "source",
                    "metadata", "previous_sentence_obj", "next_sentence_obj"
                ]
                
                for key in required_phrase_keys:
                    assert key in phrase, f"Missing required phrase key: {key} in phrase {i}"
                
                # Verify order is correct
                assert phrase["order"] == i + 1, f"Incorrect order for phrase {i}"
            
            print(f"✅ All phrase structures valid")
            
            # Display sample output
            print(f"\n📋 Sample JSON structure:")
            print(json.dumps(data, indent=2, ensure_ascii=False)[:500] + "...")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

def test_directory_structure():
    """Test that the directory structure matches the original pipeline."""
    print("\n🧪 Testing directory structure...")
    
    state = SelectionState(
        theme="L'Intelligenza Artificiale ci ruberà il lavoro?",
        seed=1000
    )
    
    gui_service = GUIService()
    
    # Test directory name generation
    output_dir = gui_service.get_output_directory(state)
    expected_dir = "output/L_Intelligenza_Artificiale_ci_rubera_il_lavoro__1000_iterative"
    
    print(f"   Generated: {output_dir}")
    print(f"   Expected:  {expected_dir}")
    
    assert output_dir == expected_dir, f"Directory mismatch!\nGot: {output_dir}\nExpected: {expected_dir}"
    
    print(f"✅ Directory structure matches original pipeline")
    return True

if __name__ == "__main__":
    print("🚀 Testing Pipeline Integration\n")
    
    success = True
    
    try:
        success &= test_directory_structure()
        success &= test_json_format()
        
        if success:
            print(f"\n🎉 All tests passed! GUI is ready for pipeline integration.")
        else:
            print(f"\n❌ Some tests failed.")
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        success = False
    
    sys.exit(0 if success else 1)
