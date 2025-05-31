"""
GUI Configuration Settings

Configuration settings specific to the web-based GUI interface.
"""

from typing import Dict, Any, List

# Main GUI Settings
GUI_SETTINGS: Dict[str, Any] = {
    "app_title": "AI Blob - Interactive Video Generator",
    "page_icon": "🎬",
    "layout": "wide",
    "theme": {
        "primaryColor": "#FF6B6B",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F0F2F6",
        "textColor": "#262730"
    }
}

# Clip Display Settings
CLIP_DISPLAY_SETTINGS: Dict[str, Any] = {
    "cards_per_row": 2,
    "max_clip_text_preview": 150,
    "show_metadata_details": True,
    "metadata_fields": {
        "duration": "Durata",
        "video_id": "Video ID", 
        "sentence_number": "Frase #",
        "start_time": "Tempo Inizio",
        "end_time": "Tempo Fine"
    },
    "field_formatters": {
        "duration": lambda x: f"{x:.1f}s" if isinstance(x, (int, float)) else str(x),
        "start_time": lambda x: f"{x:.2f}s" if isinstance(x, (int, float)) else str(x),
        "end_time": lambda x: f"{x:.2f}s" if isinstance(x, (int, float)) else str(x)
    }
}

# Button Settings
BUTTON_SETTINGS: Dict[str, Any] = {
    "select_clip": {
        "label": "✅ Seleziona",
        "type": "primary",
        "help": "Seleziona questa clip per il video"
    },
    "lucky_mode": {
        "label": "🎲 Mi Sento Fortunato!",
        "type": "secondary", 
        "help": "Lascia che l'AI scelga la clip migliore"
    },
    "next_iteration": {
        "label": "➡️ Prossima Frase",
        "type": "primary",
        "help": "Continua con la prossima frase narrativa"
    },
    "export_video": {
        "label": "🎬 Crea Video",
        "type": "primary",
        "help": "Esporta le clip selezionate e crea il video finale"
    },
    "new_session": {
        "label": "🆕 Nuova Sessione",
        "type": "secondary",
        "help": "Inizia una nuova sessione di selezione clip"
    },
    "load_session": {
        "label": "📂 Carica Sessione",
        "type": "secondary",
        "help": "Carica una sessione salvata"
    }
}

# Status Messages
STATUS_MESSAGES: Dict[str, str] = {
    "loading_candidates": "🔍 Cercando clip candidate...",
    "processing_selection": "⚙️ Elaborando selezione...",
    "auto_selecting": "🤖 L'AI sta selezionando la clip migliore...",
    "exporting_video": "🎬 Esportando per creazione video...",
    "session_saved": "💾 Sessione salvata con successo",
    "session_loaded": "📂 Sessione caricata con successo",
    "error_loading": "❌ Errore durante il caricamento",
    "error_processing": "❌ Errore durante l'elaborazione",
    "no_candidates": "⚠️ Nessuna clip candidata trovata",
    "session_complete": "🎉 Sessione completata! Tutte le clip sono state selezionate."
}

# Theme Suggestions
THEME_SUGGESTIONS: List[str] = [
    "L'Intelligenza Artificiale ci ruberà il lavoro",
    "La dieta mediterranea", 
    "La pizza napoletana",
    "Il caffè italiano",
    "La burocrazia italiana",
    "Il calcio in Italia",
    "La pasta fatta in casa",
    "I trasporti pubblici",
    "La politica italiana",
    "Il turismo in Italia",
    "L'arte italiana",
    "La moda italiana",
    "Il cinema italiano",
    "La musica italiana",
    "La famiglia italiana"
]

# Selection Workflow Settings
WORKFLOW_SETTINGS: Dict[str, Any] = {
    "default_target_clips": 15,
    "auto_advance_after_selection": True,
    "show_narrative_context": True,
    "enable_clip_preview": True,
    "save_session_automatically": True
}

# Progress Display Settings
PROGRESS_SETTINGS: Dict[str, Any] = {
    "show_progress_bar": True,
    "show_clip_counter": True,
    "show_narrative_context": True,
    "progress_bar_color": "#FF6B6B",
    "completed_color": "#4CAF50"
}

# Export Settings
EXPORT_SETTINGS: Dict[str, Any] = {
    "output_directory": "output",
    "session_directory": "gui_sessions",
    "auto_create_directories": True,
    "include_selection_metadata": True,
    "create_backup": True
}
