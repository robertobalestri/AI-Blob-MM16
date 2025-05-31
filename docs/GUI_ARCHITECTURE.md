# GUI Architecture for Interactive Clip Selection

## Overview
Web-based interface for interactive clip selection during video montage generation, built with Streamlit.

## Architecture Components

### 1. Frontend (Streamlit Web App)
- **Main Interface**: `src/gui/streamlit_app.py`
- **Components**: `src/gui/components/`
  - Clip selection widgets
  - Video preview components
  - Progress tracking
  - Theme input and configuration

### 2. Backend Services
- **GUI Service**: `src/gui/gui_service.py`
  - Manages clip selection workflow
  - Interfaces with existing AI services
  - Handles session state and progress
- **Integration Layer**: `src/gui/integration.py`
  - Connects GUI with existing scripts
  - Manages data flow between GUI and core logic

### 3. Data Flow
```
User Input (Theme) → Generate Phrases → Search Clips → [GUI SELECTION] → Assemble Video
                                                      ↗              ↘
                                              Manual Selection   Auto Selection
                                             (Interactive GUI)  ("I'm Feeling Lucky")
```

### 4. User Experience Flow
1. **Theme Input**: User enters video theme and settings
2. **Iterative Selection**: For each narrative segment:
   - Display candidate clips with metadata
   - Show narrative context
   - Allow manual selection or auto-selection
   - Preview selected clips
3. **Progress Tracking**: Visual progress bar and selected clips summary
4. **Final Assembly**: Trigger video creation with selected clips

## Key Features

### Interactive Clip Selection
- Grid/card layout for candidate clips
- Rich metadata display (duration, score, context)
- Text preview with syntax highlighting
- Quick selection buttons

### "I'm Feeling Lucky" Mode
- Preserves existing automatic selection
- One-click complete automation
- Option to review auto-selections

### Video Preview Integration
- Thumbnail generation for clips
- Optional video preview (if feasible)
- Timeline visualization

### Progress Management
- Session persistence
- Resume interrupted sessions
- Export/import selection state

## Technical Implementation

### Session State Management
- Use Streamlit's session state for workflow persistence
- Save intermediate results to prevent data loss
- Track selection history and allow undo

### Performance Optimization
- Lazy loading of clip data
- Pagination for large candidate sets
- Caching of vector search results
- Background processing for video operations

### Integration Points
- Modify `script_generate_plot.py` to support GUI mode
- Create GUI wrapper for `script_create_video.py`
- Extend configuration system for GUI settings
