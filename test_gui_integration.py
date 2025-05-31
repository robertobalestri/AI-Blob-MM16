#!/usr/bin/env python3
"""
Integration Test for GUI → Video Creation Pipeline

This script tests the complete workflow from GUI session export
to final video creation using the existing pipeline.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.gui.gui_service import GUIService, SelectionState, ClipCandidate
from src.ai_models import AIModelsService
from src.config.settings import VECTOR_STORE_DIR, VECTOR_STORE_SETTINGS
from langchain_chroma import Chroma

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gui_service_initialization():
    """Test GUI service can be initialized properly."""
    print("🔧 Testing GUI Service Initialization...")
    
    try:
        gui_service = GUIService()
        ai_service = AIModelsService()
        
        # Test vector store connection
        embedding_model = ai_service.get_embedding_model()
        vector_store = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embedding_model,
            collection_name=VECTOR_STORE_SETTINGS["collection_name"]
        )
        
        # Test basic query
        results = vector_store.similarity_search("test query", k=1)
        
        print("✅ GUI Service initialized successfully")
        print(f"✅ Vector store contains {vector_store._collection.count()} documents")
        return True
        
    except Exception as e:
        print(f"❌ GUI Service initialization failed: {e}")
        return False

def test_session_creation_and_management():
    """Test session creation, saving, and loading."""
    print("\n📝 Testing Session Management...")
    
    try:
        gui_service = GUIService()
        
        # Create test session
        theme = "Test Theme for Integration"
        state = gui_service.create_session(theme, target_clips=5)
        
        print(f"✅ Created session: {state.session_id}")
        
        # Load session
        loaded_state = gui_service.load_session(state.session_id)
        
        if loaded_state and loaded_state.session_id == state.session_id:
            print("✅ Session saved and loaded successfully")
            return True, state
        else:
            print("❌ Session load failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        return False, None

def test_candidate_generation():
    """Test candidate clip generation."""
    print("\n🔍 Testing Candidate Generation...")
    
    try:
        gui_service = GUIService()
        theme = "La pizza napoletana"
        state = gui_service.create_session(theme, target_clips=3)
        
        # Get candidates for first iteration
        candidates = gui_service.get_candidate_clips(state)
        
        if candidates and len(candidates) > 0:
            print(f"✅ Generated {len(candidates)} candidates")
            
            # Test candidate structure
            first_candidate = candidates[0]
            required_fields = ['doc_id', 'content', 'metadata', 'similarity_score', 'original_query_phrase']
            
            for field in required_fields:
                if not hasattr(first_candidate, field):
                    print(f"❌ Candidate missing field: {field}")
                    return False, None
            
            print("✅ Candidate structure validated")
            return True, (state, candidates)
        else:
            print("❌ No candidates generated")
            return False, None
            
    except Exception as e:
        print(f"❌ Candidate generation test failed: {e}")
        return False, None

def test_auto_selection():
    """Test automatic clip selection."""
    print("\n🤖 Testing Auto Selection...")
    
    try:
        gui_service = GUIService()
        theme = "La pizza napoletana"
        state = gui_service.create_session(theme, target_clips=3)
        
        # Get candidates
        candidates = gui_service.get_candidate_clips(state)
        
        if not candidates:
            print("❌ No candidates available for auto selection test")
            return False, None
        
        # Test auto selection
        selected = gui_service.auto_select_clip(candidates, state)
        
        if selected and hasattr(selected, 'selection_justification'):
            print("✅ Auto selection successful")
            print(f"   Selected: {selected.doc_id}")
            print(f"   Reason: {selected.selection_justification[:100]}...")
            return True, (state, selected)
        else:
            print("❌ Auto selection failed")
            return False, None
            
    except Exception as e:
        print(f"❌ Auto selection test failed: {e}")
        return False, None

def test_complete_workflow():
    """Test complete workflow from session to export."""
    print("\n🔄 Testing Complete Workflow...")
    
    try:
        gui_service = GUIService()
        theme = "Test Workflow Theme"
        state = gui_service.create_session(theme, target_clips=3)
        
        # Simulate 3 selections
        for i in range(3):
            print(f"   Iteration {i+1}/3...")
            
            # Get candidates
            candidates = gui_service.get_candidate_clips(state)
            
            if not candidates:
                print(f"❌ No candidates for iteration {i+1}")
                return False
            
            # Auto-select (simulating GUI interaction)
            selected = gui_service.auto_select_clip(candidates, state)
            
            if not selected:
                print(f"❌ Selection failed for iteration {i+1}")
                return False
            
            # Update state
            state = gui_service.select_clip(selected, state, f"Test selection {i+1}")
        
        # Test export
        export_dir = gui_service.export_to_video_pipeline(state)
        export_path = Path(export_dir)
        
        # Verify export files
        ordered_file = export_path / "ordered_sentences.json"
        iteration_file = export_path / "iteration_state.json"
        
        if ordered_file.exists() and iteration_file.exists():
            print("✅ Complete workflow successful")
            print(f"   Exported to: {export_dir}")
            
            # Verify export format
            with open(ordered_file, 'r', encoding='utf-8') as f:
                ordered_data = json.load(f)
            
            if len(ordered_data) == 3:
                print("✅ Export format validated")
                return True, export_dir
            else:
                print(f"❌ Export format error: expected 3 clips, got {len(ordered_data)}")
                return False, None
        else:
            print("❌ Export files not created")
            return False, None
            
    except Exception as e:
        print(f"❌ Complete workflow test failed: {e}")
        return False, None

def test_video_creation_integration():
    """Test integration with existing video creation script."""
    print("\n🎬 Testing Video Creation Integration...")
    
    try:
        # This would require running the actual video creation script
        # For now, we'll just verify the export format compatibility
        
        gui_service = GUIService()
        theme = "Integration Test"
        state = gui_service.create_session(theme, target_clips=2)
        
        # Create minimal test session
        candidates = gui_service.get_candidate_clips(state)
        if candidates:
            selected = gui_service.auto_select_clip(candidates, state)
            if selected:
                state = gui_service.select_clip(selected, state)
                
                # Second iteration
                candidates = gui_service.get_candidate_clips(state)
                if candidates:
                    selected = gui_service.auto_select_clip(candidates, state)
                    if selected:
                        state = gui_service.select_clip(selected, state)
        
        # Export
        export_dir = gui_service.export_to_video_pipeline(state)
        
        # Verify compatibility with expected format
        ordered_file = Path(export_dir) / "ordered_sentences.json"
        with open(ordered_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check required fields for video creation
        required_fields = ['page_content', 'metadata', 'sentence_number']
        
        for item in data:
            for field in required_fields:
                if field not in item:
                    print(f"❌ Missing required field for video creation: {field}")
                    return False
        
        print("✅ Video creation integration format validated")
        print(f"   Compatible export at: {export_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Video creation integration test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 AI Blob GUI Integration Tests")
    print("================================")
    
    tests = [
        ("GUI Service Initialization", test_gui_service_initialization),
        ("Session Management", lambda: test_session_creation_and_management()[0]),
        ("Candidate Generation", lambda: test_candidate_generation()[0]),
        ("Auto Selection", lambda: test_auto_selection()[0]),
        ("Complete Workflow", lambda: test_complete_workflow()[0]),
        ("Video Creation Integration", test_video_creation_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=====================")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! GUI is ready for use.")
        print("\n🚀 To start the GUI:")
        print("   ./launch_gui.sh")
        print("   OR")
        print("   streamlit run src/gui/streamlit_app.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
