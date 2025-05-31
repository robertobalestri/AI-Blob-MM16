# Interactive GUI for AI Blob Video Generation

## Overview
The Interactive GUI provides a web-based interface for manually selecting clips during video montage generation. Instead of relying entirely on AI selection, users can now review candidate clips and make informed choices while preserving the option for automatic "I'm feeling lucky" selection.

## Quick Start

### 1. Launch the Application
```bash
cd "/home/rbale/dev/AI BLOB from JSON dataset/AI-Blob-MM16"
streamlit run src/gui/streamlit_app.py
```

The application will be available at: http://localhost:8501

### 2. Create a New Session
1. Enter your video theme (e.g., "L'Intelligenza Artificiale ci ruberà il lavoro")
2. Set target number of clips (default: 15)
3. Click "Avvia Nuova Sessione" to begin

### 3. Interactive Clip Selection
For each narrative phrase:
1. **Review Candidates**: Browse through 10 candidate clips with their:
   - Text content and metadata
   - Similarity scores
   - Duration and source information
2. **Choose Selection Method**:
   - **Manual**: Click "✅ Seleziona" on your preferred clip
   - **Auto**: Click "🎲 Mi Sento Fortunato!" for AI selection
3. **Continue**: Progress automatically to the next phrase

### 4. Complete and Export
1. **Review Progress**: Track selected clips and narrative flow
2. **Export**: Click "🎬 Crea Video" when complete
3. **Video Creation**: Use exported data with existing video pipeline

## Features

### 🎯 Interactive Clip Selection
- **Manual Choice**: Review and select from k=10 candidate clips
- **AI Assistance**: "I'm feeling lucky" mode for automatic selection
- **Detailed Preview**: View clip content, metadata, and similarity scores
- **Context Awareness**: Narrative context displayed for informed decisions

### 📊 Progress Tracking
- **Visual Progress Bar**: Shows completion percentage
- **Clip Counter**: Current selection vs. target clips
- **Narrative Flow**: Context of previously selected clips
- **Session State**: Automatic saving and resume capability

### 🔧 Flexible Configuration
- **Theme Suggestions**: Pre-configured Italian themes
- **Target Length**: Adjustable number of clips (5-30)
- **Export Options**: Compatible with existing video creation pipeline
- **Session Management**: Save, load, and resume sessions

### 🎨 User-Friendly Interface
- **Modern Design**: Clean, responsive Streamlit interface
- **Italian Localization**: Native Italian language support
- **Intuitive Navigation**: Clear workflow and action buttons
- **Real-time Updates**: Immediate feedback and state updates

## Architecture Integration

### Data Flow
```
[GUI Session] → [Phrase Generation] → [Vector Search] → [Manual Selection] → [Export]
      ↓                 ↓                    ↓               ↓              ↓
[State Management] [AI Service] [ChromaDB] [User Choice] [Video Pipeline]
```

### File Structure
```
src/gui/
├── streamlit_app.py     # Main web application
├── gui_service.py       # Backend service logic
├── components.py        # Reusable UI components
└── config/
    └── gui_config.py    # GUI-specific configuration
```

### Integration Points
- **AI Models**: Reuses existing `src/ai_models.py` service
- **Vector Store**: Direct ChromaDB integration
- **Configuration**: Extends `src/config/settings.py`
- **Export Format**: Compatible with `script_create_video.py`

## Session Management

### Session Data Structure
```json
{
  "theme": "L'Intelligenza Artificiale ci ruberà il lavoro",
  "target_clips": 15,
  "selected_clips": [...],
  "current_iteration": 5,
  "narrative_context": "AI → lavoro → futuro",
  "session_id": "AI_lavoro_20241201_143022",
  "timestamp": "2024-12-01T14:30:22"
}
```

### Persistence
- **Auto-Save**: Sessions saved automatically after each selection
- **Resume**: Load and continue interrupted sessions
- **Export**: Generate `ordered_sentences.json` for video creation
- **Backup**: Session files stored in `output/gui_sessions/`

## Usage Examples

### Example 1: Manual Selection Workflow
1. **Start**: Enter theme "La pizza napoletana"
2. **Iteration 1**: 
   - Generated phrase: "La tradizione della pizza napoletana"
   - Review 10 candidates about pizza traditions
   - Manually select clip about authentic ingredients
3. **Iteration 2**: 
   - Generated phrase: "Gli ingredienti della vera pizza"
   - Review candidates about ingredients
   - Select clip about San Marzano tomatoes
4. **Continue**: Repeat until 15 clips selected
5. **Export**: Create video with selected clips

### Example 2: Mixed Manual/Auto Selection
1. **Critical Selections**: Manually choose key narrative clips
2. **Supporting Content**: Use "I'm feeling lucky" for background clips
3. **Quality Control**: Review auto-selections before proceeding
4. **Final Review**: Ensure narrative coherence before export

### Example 3: Session Resume
1. **Interruption**: Session interrupted at clip 8/15
2. **Resume**: Load session from GUI interface
3. **Continue**: Pick up where left off
4. **Complete**: Finish remaining 7 clips

## Best Practices

### 🎯 Selection Strategy
- **Start Strong**: Manually select opening clips for narrative impact
- **Maintain Flow**: Consider narrative progression between clips
- **Use Context**: Leverage displayed narrative context for decisions
- **Quality Focus**: Prioritize content quality over similarity scores

### 🔍 Evaluation Criteria
- **Relevance**: How well does the clip match the current phrase?
- **Narrative Flow**: Does it follow logically from previous clips?
- **Content Quality**: Is the speech clear and impactful?
- **Ironic Potential**: Does it create satirical/ironic effect?

### ⚡ Efficiency Tips
- **Theme Preparation**: Use clear, specific themes for better candidates
- **Quick Review**: Scan similarity scores to identify top candidates
- **Mixed Approach**: Use auto-selection for less critical clips
- **Context Awareness**: Keep narrative context in mind throughout

## Troubleshooting

### Common Issues

#### No Candidates Found
- **Cause**: Theme too specific or vector store empty
- **Solution**: Try broader themes or check vector store data

#### Low Similarity Scores
- **Cause**: Theme mismatch with available content
- **Solution**: Adjust theme or lower similarity threshold

#### Session Not Saving
- **Cause**: Insufficient permissions or disk space
- **Solution**: Check `output/gui_sessions/` directory permissions

#### GUI Not Loading
- **Cause**: Missing dependencies or port conflicts
- **Solution**: Install requirements and try different port

### Debug Mode
Enable detailed logging:
```bash
LOG_LEVEL=DEBUG streamlit run src/gui/streamlit_app.py
```

### Support
- **Logs**: Check application logs for detailed error information
- **Sessions**: Session files contain full state for debugging
- **Config**: Verify configuration in `src/config/` files

## Advanced Features

### Customization Options
- **Theme Suggestions**: Modify `THEME_SUGGESTIONS` in gui_config.py
- **Display Settings**: Adjust clip preview and metadata display
- **Workflow Settings**: Configure automatic advancement and features
- **Button Labels**: Customize interface text and styling

### Integration Extensions
- **Video Preview**: Add thumbnail generation and preview
- **Batch Operations**: Multi-clip selection and bulk actions
- **Collaborative Features**: Multi-user session management
- **Analytics**: Selection pattern analysis and optimization

### Performance Optimization
- **Lazy Loading**: Load candidates on-demand for large datasets
- **Caching**: Cache search results and AI responses
- **Pagination**: Handle large candidate sets efficiently
- **Background Processing**: Async operations for better UX

## Future Enhancements

### Phase 1: Core Improvements
- [ ] Video thumbnail previews
- [ ] Advanced filtering and sorting
- [ ] Batch selection operations
- [ ] Enhanced progress visualization

### Phase 2: Advanced Features
- [ ] Real-time video preview
- [ ] Collaborative multi-user sessions
- [ ] Template and preset management
- [ ] Advanced analytics dashboard

### Phase 3: Professional Features
- [ ] API integration for external tools
- [ ] Advanced editing capabilities
- [ ] Export to multiple formats
- [ ] Professional deployment options

## Contributing

### Development Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables in `.env`
3. Run development server: `streamlit run src/gui/streamlit_app.py`
4. Access at http://localhost:8501

### Code Structure
- Follow existing patterns in `script_generate_plot.py`
- Maintain Italian localization throughout
- Add comprehensive logging and error handling
- Test with various themes and scenarios

This interactive GUI transforms the AI Blob video generation from a fully automated process into a collaborative human-AI experience, giving users control while preserving the power of AI assistance.
