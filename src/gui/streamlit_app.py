"""
Streamlit Web Application for Interactive Clip Selection

This is the main web interf        theme = st.text_input(
            "🎯 Tema del video",
            value="", for the AI Blob video generation system,
allowing users to interactively select clips for their video montages.
"""

import streamlit as st
import asyncio
import json
import os
import sys
import logging
from typing import List, Optional
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import GUI components
from src.gui.gui_service import GUIService, SelectionState, ClipCandidate
from src.gui.components import render_clip_card, render_progress_bar, render_theme_input, render_all_candidates_with_pagination, render_seed_input
from src.config.settings import VECTOR_STORE_DIR
from src.config.gui_config import GUI_SETTINGS, BUTTON_SETTINGS, STATUS_MESSAGES

logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Blob - Interactive Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'gui_service' not in st.session_state:
        st.session_state.gui_service = GUIService()
    
    if 'selection_state' not in st.session_state:
        st.session_state.selection_state = None
    
    if 'current_candidates' not in st.session_state:
        st.session_state.current_candidates = []
    
    if 'selected_candidate_index' not in st.session_state:
        st.session_state.selected_candidate_index = None
    
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

def render_header():
    """Render the application header."""
    st.title("🎬 AI Blob - Interactive Video Generator")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("**Crea video satirici e ironici** selezionando clip da un archivio televisivo italiano")
    
    with col2:
        if st.session_state.selection_state:
            state = st.session_state.selection_state
            current = state.current_iteration
            total = state.max_iterations
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Progresso", f"{current}/{total}", f"{(current/total)*100:.1f}%")
            with col2b:
                st.metric("Seed", state.seed)
    
    with col3:
        if st.button("🔄 Reset", help="Ricomincia da capo"):
            st.session_state.selection_state = None
            st.session_state.current_candidates = []
            st.session_state.selected_candidate_index = None
            st.rerun()

def render_theme_setup():
    """Render theme input and setup section."""
    st.header("🎯 Configurazione Sessione")
    
    # Theme input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        theme = st.text_input(
            "Tema del video",
            value="",
            placeholder="Es: L'intelligenza artificiale ci ruberà il lavoro",
            help="Descrivi il tema satirico del tuo video"
        )
    
    with col2:
        max_clips = st.number_input(
            "Max clip",
            value=15,
            help="Numero massimo di clip nel video finale"
        )
    
    # Seed input (new feature) - Non usa il default dal .env per permettere customizzazione
    if "user_seed" not in st.session_state:
        st.session_state.user_seed = ""
    seed = render_seed_input(default_seed=st.session_state.user_seed, key="session_seed")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 Inizia Selezione", type="primary", disabled=not theme.strip()):
            if theme.strip():
                state = st.session_state.gui_service.initialize_session(
                    theme.strip(), 
                    seed.strip() if seed.strip() else None
                )
                state.max_iterations = max_clips
                st.session_state.selection_state = state
                st.success(f"✅ Sessione iniziata!")
                st.info(f"🎯 Tema: {theme}")
                st.info(f"🎲 Seed: {state.seed}")
                st.rerun()
    
    with col2:
        lucky_btn = BUTTON_SETTINGS["lucky_mode"]
        if st.button(lucky_btn["label"], help="Genera automaticamente tutto il video"):
            if theme.strip():
                st.session_state.is_processing = True
                # This will trigger auto-generation mode
                st.session_state.auto_mode = True
                state = st.session_state.gui_service.initialize_session(
                    theme.strip(),
                    seed.strip() if seed.strip() else None
                )
                state.max_iterations = max_clips
                st.session_state.selection_state = state
                st.success(f"✅ Modalità automatica attivata!")
                st.info(f"🎯 Tema: {theme}")
                st.info(f"🎲 Seed: {state.seed}")
                st.rerun()
    
    with col3:
        # Show current configuration preview
        if theme.strip():
            st.markdown("**🔍 Anteprima Configurazione:**")
            st.caption(f"Tema: {theme}...")
            st.caption(f"Seed: {seed if seed.strip() else 'Auto-generato'}")
            st.caption(f"Max clip: {max_clips}")

def render_clip_selection():
    """Render the main clip selection interface."""
    state = st.session_state.selection_state
    
    if state.is_complete:
        render_completion_screen()
        return
    
    st.header(f"📝 Selezione Clip - Iterazione {state.current_iteration + 1}/{state.max_iterations}")
    
    # Show narrative context if available
    if state.narrative_context:
        with st.expander("📖 Contesto Narrativo", expanded=False):
            st.text_area(
                "Frasi selezionate precedentemente:",
                value=state.narrative_context,
                height=100,
                disabled=True
            )
    
    # Generate candidates if not already available
    if not st.session_state.current_candidates:
        with st.spinner("🔍 Generazione frasi narrative e ricerca clip..."):
            phrases = st.session_state.gui_service.generate_narrative_phrases(state)
            excluded_ids = st.session_state.gui_service.get_excluded_doc_ids(state)
            candidates = st.session_state.gui_service.search_candidate_clips(
                phrases, excluded_ids
            )
            st.session_state.current_candidates = candidates
    
    candidates = st.session_state.current_candidates
    
    if not candidates:
        st.error("❌ Nessuna clip trovata per questa iterazione. Prova a modificare il tema.")
        return
    
    # Handle auto-mode
    if hasattr(st.session_state, 'auto_mode') and st.session_state.auto_mode:
        with st.spinner("🤖 Selezione automatica in corso..."):
            selected_clip = st.session_state.gui_service.auto_select_best_clip(candidates, state)
            if selected_clip:
                st.session_state.gui_service.add_selected_clip(state, selected_clip)
                st.session_state.current_candidates = []
                if state.is_complete:
                    st.success("✅ Selezione automatica completata!")
                st.rerun()
        return
    
    # Show candidates for manual selection
    total_found = len(candidates)
    st.subheader(f"🎭 Clip Candidate ({total_found} trovate)")
    
    with st.expander("ℹ️ Come funziona la ricerca", expanded=False):
        st.info(f"""
        **Processo di ricerca completato:**
        1. 🎯 Sistema ha generato 3 frasi narrative diverse per esplorare il tema
        2. 🔍 Per ogni frase ha cercato 10 clip nel database (totale: {total_found} clip)
        3. 📊 Tutte le clip sono state ordinate per punteggio di rilevanza
        4. � **Ora puoi vedere e selezionare da TUTTE le {total_found} clip trovate!**
        
        **Novità:** Non siamo più limitati alle prime 10 - ora hai accesso completo!
        """)
    
    # Add debugging info
    if st.checkbox("🔍 Mostra info di debug", key=f"debug_iteration_{state.current_iteration}"):
        # Show unique query phrases generated by LLM
        unique_phrases = list(set([c.original_query_phrase for c in candidates])) if candidates else []
        # Show first few actual clip contents to verify diversity
        clip_previews = [c.page_content for c in candidates] if candidates else []
        
        st.json({
            "total_candidates": len(candidates),
            "iteration": state.current_iteration,
            "selected_clips_so_far": len(state.selected_clips),
            "unique_query_phrases": unique_phrases,
            "sample_clip_contents": clip_previews,
            "top_scores": [f"{c.score:.3f}" for c in candidates[:5]] if candidates else []
        })
    
    # Use the new component that can handle all candidates
    selected_index = render_all_candidates_with_pagination(candidates, state.current_iteration)
    
    # Handle selection with improved logging
    if selected_index is not None:
        selected_clip = candidates[selected_index]
        
        # Add detailed logging for debugging
        st.info(f"""
        **🎯 Selezione confermata:**
        - Clip selezionata: #{selected_index + 1} di {len(candidates)}
        - Query originale: {selected_clip.original_query_phrase}
        - Score: {selected_clip.score:.3f}
        - Primi 100 caratteri: {selected_clip.page_content}...
        """)
        
        # Add to selection
        st.session_state.gui_service.add_selected_clip(state, selected_clip)
        st.session_state.current_candidates = []
        
        # Show success and continue
        st.success(f"✅ Clip {selected_index + 1} aggiunta con successo!")
        
        # Auto-advance to next iteration
        if not state.is_complete:
            st.info("🔄 Caricamento prossima iterazione...")
            st.rerun()
        else:
            st.balloons()
            st.success("🎉 Selezione completa!")
            
            # Auto-export and start video creation
            with st.spinner("📤 Esportazione selezione..."):
                output_dir = st.session_state.gui_service.get_output_directory(state)
                ordered_file_path = st.session_state.gui_service.export_selection_to_ordered_file(state, output_dir)
                st.success(f"✅ Selezione esportata in: {output_dir}")
            
            # Auto-start video creation
            st.info("🎬 Avvio automatico creazione video...")
            st.session_state.auto_create_video = True
            st.rerun()
    
    # Auto-select option
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🍀 Selezione Automatica per questa clip", help="Lascia che l'AI scelga la migliore"):
            with st.spinner("🤖 Selezione automatica..."):
                selected_clip = st.session_state.gui_service.auto_select_best_clip(candidates, state)
                if selected_clip:
                    st.session_state.gui_service.add_selected_clip(state, selected_clip)
                    st.session_state.current_candidates = []
                    st.success("✅ Clip selezionata automaticamente!")
                    st.rerun()
    
    with col2:
        if st.button("⏭️ Salta questa iterazione"):
            state.current_iteration += 1
            st.session_state.current_candidates = []
            if state.current_iteration >= state.max_iterations:
                state.is_complete = True
            st.rerun()

def render_candidate_details(candidate: ClipCandidate, index: int):
    """Render detailed information for a candidate clip."""
    
    # Main clip text
    st.markdown(f"**Testo della clip:**")
    st.text_area(
        "Contenuto della clip",
        value=candidate.page_content,
        height=120,
        disabled=True,
        key=f"clip_text_{index}",
        label_visibility="collapsed"
    )
    
    # Metadata in columns
    metadata = candidate.formatted_metadata
    if metadata:
        st.markdown("**Metadata:**")
        cols = st.columns(min(len(metadata), 4))
        for i, (key, value) in enumerate(metadata.items()):
            with cols[i % len(cols)]:
                st.metric(key, value)
    
    # Context sentences if available
    if candidate.previous_sentence or candidate.next_sentence:
        with st.expander("🔍 Contesto (frasi precedenti/successive)", expanded=False):
            if candidate.previous_sentence:
                st.markdown("**Frase precedente:**")
                prev_content = candidate.previous_sentence.get('page_content', 'N/A')
                st.caption(prev_content)
            
            if candidate.next_sentence:
                st.markdown("**Frase successiva:**")
                next_content = candidate.next_sentence.get('page_content', 'N/A') 
                st.caption(next_content)

def render_completion_screen():
    """Render the completion screen with final options."""
    state = st.session_state.selection_state
    
    # Check if we should automatically start video creation
    if getattr(st.session_state, 'auto_create_video', False):
        st.session_state.auto_create_video = False  # Reset flag
        
        # Automatically start video creation
        st.info("🎬 Creazione automatica del video in corso...")
        
        try:
            # Export first
            output_dir = st.session_state.gui_service.get_output_directory(state)
            ordered_file_path = st.session_state.gui_service.export_selection_to_ordered_file(state, output_dir)
            
            # Then create video
            with st.spinner("📹 Generazione video... Questo potrebbe richiedere alcuni minuti."):
                final_video_path = st.session_state.gui_service.run_video_creation(state)
                
            st.success(f"✅ Video creato automaticamente!")
            st.balloons()
            st.info(f"📁 Percorso: {final_video_path}")
            
            # Show download button if video exists
            if os.path.exists(final_video_path):
                with open(final_video_path, 'rb') as video_file:
                    st.download_button(
                        "📥 Scarica Video",
                        data=video_file.read(),
                        file_name=f"{state.theme}_{state.seed}_video.mp4",
                        mime="video/mp4"
                    )
                    
        except Exception as e:
            st.error(f"❌ Errore nella creazione automatica del video: {str(e)}")
            st.info("Puoi provare a creare il video manualmente usando il pulsante qui sotto.")
    
    st.header("🎉 Selezione Completata!")
    st.success(f"Hai selezionato {len(state.selected_clips)} clip per il tuo video sul tema: **{state.theme}**")
    
    # Show selected clips summary
    with st.expander("📋 Riepilogo Clip Selezionate", expanded=True):
        for i, clip in enumerate(state.selected_clips):
            st.markdown(f"**{i+1}.** {clip.display_text}")
            st.caption(f"Query: {clip.original_query_phrase} | Score: {clip.score:.3f}")
    
    # Add export preview and pipeline info
    render_export_preview(state, st.session_state.gui_service)
    render_pipeline_info()
    
    # Export and video generation options
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Esporta Selezione", type="primary"):
            # Use the same directory structure as the original pipeline
            output_dir = st.session_state.gui_service.get_output_directory(state)
            
            ordered_file_path = st.session_state.gui_service.export_selection_to_ordered_file(state, output_dir)
            st.success(f"✅ Selezione esportata in: {ordered_file_path}")
            st.info(f"📁 Directory: {output_dir}")
            
            # Provide download link
            with open(ordered_file_path, 'r', encoding='utf-8') as f:
                st.download_button(
                    "📥 Scarica JSON",
                    data=f.read(),
                    file_name="ordered_sentences.json",
                    mime="application/json"
                )
    
    with col2:
        if st.button("🎬 Genera Video", help="Avvia la creazione del video con le clip selezionate"):
            # First export the selection if not already done
            output_dir = st.session_state.gui_service.get_output_directory(state)
            ordered_file_path = st.session_state.gui_service.export_selection_to_ordered_file(state, output_dir)
            
            st.info("� Avvio creazione video... Questo potrebbe richiedere alcuni minuti.")
            
            # Create a progress placeholder
            progress_placeholder = st.empty()
            
            try:
                with progress_placeholder.container():
                    st.write("📹 Scaricamento e elaborazione clip...")
                    
                # Run video creation
                final_video_path = st.session_state.gui_service.run_video_creation(state)
                
                progress_placeholder.empty()
                st.success(f"✅ Video creato con successo!")
                st.info(f"📁 Percorso: {final_video_path}")
                
                # Show download button if video exists
                if os.path.exists(final_video_path):
                    with open(final_video_path, 'rb') as video_file:
                        st.download_button(
                            "📥 Scarica Video",
                            data=video_file.read(),
                            file_name=f"{state.theme}_{state.seed}_video.mp4",
                            mime="video/mp4"
                        )
                        
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"❌ Errore nella creazione del video: {str(e)}")
                logger.error(f"Video creation error: {e}")
                st.info("🔧 Controlla i log per maggiori dettagli")
    
    with col3:
        if st.button("🔄 Nuova Selezione"):
            st.session_state.selection_state = None
            st.session_state.current_candidates = []
            st.rerun()

def render_export_preview(state: SelectionState, gui_service: GUIService):
    """Render a preview of what will be exported."""
    st.markdown("### 📁 Anteprima Esportazione")
    
    # Get the output directory that will be created
    output_dir = gui_service.get_output_directory(state)
    
    # Show directory structure
    with st.expander("📋 Struttura File che Sarà Creata", expanded=False):
        st.code(f"""
📁 {output_dir}/
├── 📄 ordered_sentences.json  (metadati delle clip selezionate)
└── 📁 clips/                  (creata durante generazione video)
    ├── 🎬 clip_001.mp4
    ├── 🎬 clip_002.mp4
    └── 🎬 ...
        """, language="text")
        
        st.info("""
        **📝 Nota**: Il file `ordered_sentences.json` contiene:
        - Testo di ogni clip selezionata
        - Query utilizzate per trovarla  
        - Punteggi di rilevanza
        - Metadati del video originale
        - Contesto (frasi precedenti/successive)
        """)

def render_pipeline_info():
    """Render information about the pipeline integration.""" 
    st.markdown("### 🔗 Integrazione Pipeline")
    
    with st.expander("ℹ️ Come Funziona l'Integrazione", expanded=False):
        st.markdown("""
        **🎯 Flusso Completo**:
        1. **Selezione Interattiva**: Usa questa GUI per selezionare clip
        2. **Esportazione**: Crea il file `ordered_sentences.json` 
        3. **Generazione Video**: 
           - 🚀 **Automatica**: Clicca "Genera Video" qui sotto
           - 🛠️ **Manuale**: Esegui `python script_create_video.py` 
           
        **📁 Compatibilità**:
        - Format JSON identico agli script originali
        - Stessa struttura directory (`theme_seed_iterative`)
        - File `ordered_sentences.json` standard
        
        **🎬 Cosa Succede Durante la Generazione**:
        1. Download automatico delle clip da YouTube
        2. Taglio preciso dei segmenti selezionati  
        3. Analisi LLM per ottimizzazione contenuti
        4. Assemblaggio finale in `final_montage.mp4`
        """)

def render_sidebar():
    """Render the sidebar with additional options and information."""
    with st.sidebar:
        st.header("ℹ️ Informazioni")
        
        if st.session_state.selection_state:
            state = st.session_state.selection_state
            
            st.metric("Tema", state.theme)
            st.metric("Seed", state.seed)
            st.metric("Clip Selezionate", len(state.selected_clips))
            
            # Fix iteration display to not go beyond max when complete
            if state.is_complete:
                st.metric("Iterazione Corrente", f"{state.max_iterations}/{state.max_iterations}")
                progress = 1.0
            else:
                st.metric("Iterazione Corrente", f"{state.current_iteration + 1}/{state.max_iterations}")
                progress = state.current_iteration / state.max_iterations
            
            # Progress bar
            st.progress(progress)
            
            st.markdown("---")
            
            # Quick actions
            st.subheader("🛠️ Azioni Rapide")
            
            if st.button("⚡ Completa Automaticamente"):
                st.session_state.auto_mode = True
                st.rerun()
            
            if st.button("📊 Mostra Statistiche"):
                st.session_state.show_stats = True
        
        else:
            st.markdown("""
            ### 🎯 Come funziona:
            
            1. **Inserisci un tema** per il tuo video satirico
            2. **Seleziona manualmente** le clip che preferisci
            3. **Oppure usa "Sono Fortunato"** per la selezione automatica
            4. **Esporta** la selezione per generare il video
            
            ### 💡 Suggerimenti:
            - Sii specifico nel tema per risultati migliori
            - Usa il contesto narrativo per mantenere coerenza
            - Prova la selezione automatica se sei indeciso
            """)
        
        st.markdown("---")
        st.caption("AI Blob - Media Mutations 16")

def main():
    """Main application entry point."""
    initialize_session_state()
    
    render_header()
    render_sidebar()
    
    # Main content area
    if st.session_state.selection_state is None:
        render_theme_setup()
    else:
        render_clip_selection()

if __name__ == "__main__":
    main()
