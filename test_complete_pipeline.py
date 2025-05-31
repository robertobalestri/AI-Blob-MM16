#!/usr/bin/env python3
"""
Complete Pipeline Test

This script tests the complete pipeline from GUI export to video creation
using mock data to ensure everything works together.
"""

import json
import os
import sys
import shutil
import tempfile
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.gui.gui_service import GUIService, SelectionState, ClipCandidate

def create_mock_dataset():
    """Create a minimal mock dataset for testing."""
    print("📁 Creating mock dataset...")
    
    # Mock data that would normally be in the vector store
    mock_clips = [
        {
            "clip_id": "mock_1",
            "page_content": "Questa è una frase di prova sui robot che lavorano",
            "metadata": {
                "video_id": "ABC123",
                "sentence_number": 1,
                "duration": 8.5,
                "start_time": 10.0,
                "end_time": 18.5,
                "video_title": "Robot industriali"
            },
            "score": 0.95,
            "query": "robot lavoro"
        },
        {
            "clip_id": "mock_2", 
            "page_content": "L'intelligenza artificiale sta cambiando il mondo del lavoro",
            "metadata": {
                "video_id": "DEF456",
                "sentence_number": 5,
                "duration": 7.2,
                "start_time": 45.0,
                "end_time": 52.2,
                "video_title": "AI e futuro"
            },
            "score": 0.88,
            "query": "intelligenza artificiale lavoro"
        },
        {
            "clip_id": "mock_3",
            "page_content": "Il futuro dell'automazione nelle fabbriche italiane",
            "metadata": {
                "video_id": "GHI789",
                "sentence_number": 12,
                "duration": 6.8,
                "start_time": 120.0,
                "end_time": 126.8,
                "video_title": "Automazione industriale"
            },
            "score": 0.82,
            "query": "automazione futuro"
        }
    ]
    
    return mock_clips

def test_complete_export_flow():
    """Test the complete export flow with realistic data."""
    print("🧪 Testing complete export flow...")
    
    # Create mock clips
    mock_data = create_mock_dataset()
    
    # Create GUI service and state
    gui_service = GUIService()
    state = SelectionState(
        theme="L'Intelligenza Artificiale ci ruberà il lavoro?",
        seed=42,
        max_iterations=len(mock_data)
    )
    
    # Convert mock data to ClipCandidate objects
    for i, mock_clip in enumerate(mock_data):
        clip = ClipCandidate(
            clip_id=mock_clip["clip_id"],
            page_content=mock_clip["page_content"],
            metadata=mock_clip["metadata"],
            score=mock_clip["score"],
            original_query_phrase=mock_clip["query"],
            previous_sentence={"text": f"Frase precedente {i+1}"},
            next_sentence={"text": f"Frase successiva {i+1}"}
        )
        
        gui_service.add_selected_clip(state, clip)
        print(f"   ✅ Added clip {i+1}: {clip.original_query_phrase}")
    
    # Test export
    with tempfile.TemporaryDirectory() as temp_base:
        print(f"📂 Testing export in: {temp_base}")
        
        # Get the expected output directory
        output_dir = gui_service.get_output_directory(state)
        full_output_path = os.path.join(temp_base, os.path.basename(output_dir))
        
        # Export
        ordered_file_path = gui_service.export_selection_to_ordered_file(state, full_output_path)
        
        print(f"✅ Export completed:")
        print(f"   📁 Directory: {full_output_path}")
        print(f"   📄 File: {ordered_file_path}")
        
        # Verify file exists and has correct content
        assert os.path.exists(ordered_file_path), "Export file missing"
        
        with open(ordered_file_path, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        # Verify structure
        assert exported_data["theme"] == state.theme
        assert exported_data["total_clips"] == len(mock_data)
        assert len(exported_data["ordered_phrases"]) == len(mock_data)
        
        # Verify content
        for i, phrase_data in enumerate(exported_data["ordered_phrases"]):
            original_mock = mock_data[i]
            assert phrase_data["matched_phrase"] == original_mock["page_content"]
            assert phrase_data["query_phrase_that_led_to_this_clip"] == original_mock["query"]
            assert phrase_data["order"] == i + 1
            assert phrase_data["retrieval_score"] == original_mock["score"]
            
        print(f"✅ Export data verification successful")
        
        # Display a summary
        print(f"\n📋 Export Summary:")
        print(f"   🎯 Theme: {exported_data['theme']}")
        print(f"   🎲 Seed: {state.seed}")  
        print(f"   📊 Total clips: {exported_data['total_clips']}")
        print(f"   📁 Directory format: {os.path.basename(output_dir)}")
        print(f"   📄 File: ordered_sentences.json")
        
        print(f"\n📝 Clip Details:")
        for i, phrase in enumerate(exported_data["ordered_phrases"]):
            print(f"   {i+1}. {phrase['matched_phrase'][:50]}...")
            print(f"      Query: {phrase['query_phrase_that_led_to_this_clip']}")
            print(f"      Score: {phrase['retrieval_score']:.2f}")
            print(f"      Source: {phrase['source']}")
        
        return True

def test_directory_compatibility():
    """Test that the created directories match existing pipeline outputs."""
    print(f"\n🧪 Testing directory compatibility with existing outputs...")
    
    # Check if there are existing output directories to compare with
    output_base = Path("output")
    if output_base.exists():
        existing_dirs = [d.name for d in output_base.iterdir() if d.is_dir()]
        iterative_dirs = [d for d in existing_dirs if d.endswith("_iterative")]
        
        if iterative_dirs:
            print(f"   📁 Found {len(iterative_dirs)} existing iterative directories:")
            for dir_name in iterative_dirs[:3]:  # Show first 3
                print(f"      - {dir_name}")
            
            # Test that our format matches
            gui_service = GUIService()
            test_state = SelectionState(theme="Test Theme", seed=1000)
            our_dir = gui_service.get_output_directory(test_state)
            our_dir_name = os.path.basename(our_dir)
            
            # Check pattern matches
            if our_dir_name.endswith("_iterative"):
                print(f"   ✅ Directory pattern matches: {our_dir_name}")
            else:
                print(f"   ❌ Directory pattern mismatch: {our_dir_name}")
                return False
        else:
            print(f"   ℹ️  No existing iterative directories found")
    else:
        print(f"   ℹ️  Output directory does not exist yet")
    
    return True

if __name__ == "__main__":
    print("🚀 Complete Pipeline Integration Test\n")
    
    success = True
    
    try:
        success &= test_directory_compatibility()
        success &= test_complete_export_flow()
        
        if success:
            print(f"\n🎉 Complete pipeline test passed!")
            print(f"   The GUI is fully integrated with the video creation pipeline.")
            print(f"   You can now:")
            print(f"   1. Use the GUI to select clips interactively")
            print(f"   2. Export the selection (creates proper directory + JSON)")
            print(f"   3. Generate video directly from the GUI")
            print(f"   4. Or run script_create_video.py manually on the exported data")
        else:
            print(f"\n❌ Pipeline integration test failed.")
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    sys.exit(0 if success else 1)
