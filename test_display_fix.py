#!/usr/bin/env python3
"""Test the corrected iteration display logic."""

def test_corrected_display():
    print("Testing corrected iteration display logic")
    
    for max_iterations in [1, 2, 3, 5]:
        print(f"\n=== Testing with max_iterations = {max_iterations} ===")
        
        for selected_clips_count in range(0, max_iterations + 1):
            current_iteration = selected_clips_count
            is_complete = current_iteration >= max_iterations
            
            # OLD logic (buggy)
            old_display = f"{current_iteration + 1}/{max_iterations}"
            
            # NEW logic (fixed)
            if is_complete:
                new_display = f"{max_iterations}/{max_iterations}"
                progress = 1.0
            else:
                new_display = f"{current_iteration + 1}/{max_iterations}"
                progress = current_iteration / max_iterations
            
            print(f"  After {selected_clips_count} clips: OLD='{old_display}' NEW='{new_display}' complete={is_complete} progress={progress:.1f}")

if __name__ == "__main__":
    test_corrected_display()
