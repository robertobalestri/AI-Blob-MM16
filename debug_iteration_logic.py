#!/usr/bin/env python3
"""Debug script to test the iteration logic."""

# Simulate the iteration logic
def test_iteration_logic():
    print("Testing iteration logic for different max_clips values")
    
    for max_clips in [1, 2, 3, 5]:
        print(f"\n=== Testing with max_clips = {max_clips} ===")
        
        current_iteration = 0
        max_iterations = max_clips
        selected_clips = []
        
        print(f"Initial: current_iteration={current_iteration}, max_iterations={max_iterations}")
        
        for clip_num in range(1, max_clips + 2):  # Test one extra to see when it stops
            print(f"\n--- Attempting to select clip {clip_num} ---")
            
            # Display what user sees
            display_iteration = current_iteration + 1
            print(f"UI shows: Iteration {display_iteration}/{max_iterations}")
            
            # Check if complete before selection
            is_complete = current_iteration >= max_iterations
            print(f"is_complete before selection: {is_complete}")
            
            if is_complete:
                print("❌ Should show completion screen, not allow more selections!")
                break
                
            # Simulate selecting a clip
            selected_clips.append(f"clip_{clip_num}")
            current_iteration += 1
            
            print(f"After selection: current_iteration={current_iteration}")
            
            # Check completion after selection
            is_complete = current_iteration >= max_iterations
            print(f"is_complete after selection: {is_complete}")
            
            if is_complete:
                print("✅ Should show completion screen next")

if __name__ == "__main__":
    test_iteration_logic()
