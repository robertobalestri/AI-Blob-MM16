"""
Reusable UI components for the Streamlit application.
"""

import streamlit as st
import sys
from typing import Dict, Any, Optional
from pathlib import Path

# Add the project root to Python path for imports  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gui.gui_service import ClipCandidate

def render_clip_card(candidate: ClipCandidate, index: int, is_selected: bool = False) -> bool:
    """
    Render a clip candidate as a card with selection option.
    
    Args:
        candidate: The clip candidate to render
        index: Index of the candidate in the list
        is_selected: Whether this candidate is currently selected
        
    Returns:
        True if the clip was selected, False otherwise
    """
    # Create container with selection styling
    container_style = "border: 2px solid #00ff00;" if is_selected else "border: 1px solid #ccc;"
    
    with st.container():
        st.markdown(f'<div style="{container_style} padding: 10px; border-radius: 5px; margin: 5px 0;">', 
                   unsafe_allow_html=True)
        
        # Header with score and selection button
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**Clip {index + 1}**")
            st.caption(f"Query: {candidate.original_query_phrase}")
        
        with col2:
            st.metric("Score", f"{candidate.score:.3f}")
        
        with col3:
            selected = st.button(
                "✅ Seleziona" if not is_selected else "✓ Selezionata",
                key=f"select_clip_{index}",
                disabled=is_selected,
                type="primary" if not is_selected else "secondary"
            )
        
        # Clip content
        st.markdown("**Contenuto:**")
        st.text_area(
            "Contenuto della clip",
            value=candidate.page_content,
            height=100,
            disabled=True,
            key=f"content_{index}",
            label_visibility="collapsed"
        )
        
        # Metadata in expandable section
        if candidate.formatted_metadata:
            with st.expander("📊 Dettagli", expanded=False):
                metadata_cols = st.columns(3)
                for i, (key, value) in enumerate(candidate.formatted_metadata.items()):
                    with metadata_cols[i % 3]:
                        st.caption(f"**{key}:** {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return selected

def render_progress_bar(current: int, total: int, label: str = "Progresso") -> None:
    """
    Render a progress bar with current status.
    
    Args:
        current: Current step
        total: Total steps
        label: Label for the progress bar
    """
    progress = current / total if total > 0 else 0
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(progress)
    
    with col2:
        st.caption(f"{current}/{total}")
    
    st.caption(f"{label}: {progress*100:.1f}% completato")

def render_theme_input(default_theme: str = "", key: str = "theme_input") -> str:
    """
    Render theme input with validation and suggestions.
    
    Args:
        default_theme: Default theme value
        key: Unique key for the input widget
        
    Returns:
        The entered theme
    """
    st.markdown("### 🎯 Tema del Video")
    
    theme = st.text_input(
        "Descrivi il tema satirico per il tuo video:",
        value=default_theme,
        placeholder="Es: L'intelligenza artificiale ci ruberà il lavoro",
        help="Sii specifico per ottenere risultati migliori",
        key=key
    )
    
    # Theme suggestions
    with st.expander("💡 Suggerimenti di temi", expanded=False):
        suggestions = [
            "L'intelligenza artificiale ci ruberà il lavoro",
            "La dieta mediterranea e i suoi benefici",
            "Il riscaldamento globale",
            "I social media e la privacy",
            "L'economia italiana",
            "La politica contemporanea",
            "La tecnologia nella vita quotidiana"
        ]
        
        for suggestion in suggestions:
            if st.button(f"📝 {suggestion}", key=f"suggest_{suggestion}"):
                st.session_state[key] = suggestion
                st.rerun()
    
    return theme

def render_clip_summary(clips: list, theme: str) -> None:
    """
    Render a summary of selected clips.
    
    Args:
        clips: List of selected clips
        theme: Video theme
    """
    st.markdown(f"### 📋 Riepilogo - {len(clips)} clip selezionate")
    st.markdown(f"**Tema:** {theme}")
    
    if not clips:
        st.info("Nessuna clip selezionata ancora.")
        return
    
    # Create summary table
    data = []
    for i, clip in enumerate(clips):
        data.append({
            "#": i + 1,
            "Anteprima": clip.display_text[:50] + "..." if len(clip.display_text) > 50 else clip.display_text,
            "Query": clip.original_query_phrase[:30] + "..." if len(clip.original_query_phrase) > 30 else clip.original_query_phrase,
            "Score": f"{clip.score:.3f}"
        })
    
    st.table(data)

def render_narrative_context(context: str, max_height: int = 150) -> None:
    """
    Render the narrative context area.
    
    Args:
        context: The narrative context text
        max_height: Maximum height of the text area
    """
    if not context:
        st.info("📝 Nessun contesto narrativo disponibile (prima iterazione)")
        return
    
    st.markdown("### 📖 Contesto Narrativo")
    st.text_area(
        "Clip selezionate precedentemente:",
        value=context,
        height=max_height,
        disabled=True,
        help="Questo testo rappresenta il contesto narrativo costruito dalle clip selezionate finora"
    )

def render_candidate_grid(candidates: list, max_candidates: int = 30) -> Optional[int]:
    """
    Render candidates in a grid layout for easier comparison.
    
    Args:
        candidates: List of clip candidates
        max_candidates: Maximum number of candidates to show
        
    Returns:
        Index of selected candidate or None
    """
    if not candidates:
        st.warning("Nessuna clip candidata disponibile.")
        return None
    
    # Don't artificially limit candidates - show all found
    display_candidates = candidates[:max_candidates]
    total_candidates = len(candidates)
    
    st.markdown(f"**🎭 Mostrando {len(display_candidates)} di {total_candidates} clip trovate**")
    
    # Create grid (3 columns)
    cols_per_row = 3
    selected_index = None
    
    for row_start in range(0, len(display_candidates), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for col_idx, candidate_idx in enumerate(range(row_start, min(row_start + cols_per_row, len(display_candidates)))):
            candidate = display_candidates[candidate_idx]
            
            with cols[col_idx]:
                # Mini card for each candidate
                with st.container():
                    st.markdown(f"**Clip {candidate_idx + 1}**")
                    st.caption(f"Score: {candidate.score:.3f}")
                    st.caption(f"Query: {candidate.original_query_phrase}")
                    
                    # Truncated content
                    preview = candidate.page_content[:100] + "..." if len(candidate.page_content) > 100 else candidate.page_content
                    st.text(preview)
                    
                    # Use unique key with iteration info to avoid conflicts
                    unique_key = f"grid_select_{candidate_idx}_{hash(candidate.page_content)}"
                    if st.button(f"✅ Seleziona", key=unique_key):
                        selected_index = candidate_idx
    
    return selected_index

def render_all_candidates_with_pagination(candidates: list, iteration: int = 0) -> Optional[int]:
    """
    Render all candidates with enhanced grid visualization including YouTube embeds and metadata.
    
    Args:
        candidates: List of clip candidates
        iteration: Current iteration number for unique keys
        
    Returns:
        Index of selected candidate or None
    """
    if not candidates:
        st.warning("Nessuna clip candidata disponibile.")
        return None
    
    total_candidates = len(candidates)
    st.markdown(f"**🎭 {total_candidates} Clip Candidate Trovate**")
    
    # Always use enhanced grid view
    return _render_enhanced_grid(candidates, iteration)

def _render_compact_list(candidates: list, iteration: int) -> Optional[int]:
    """Render candidates as a compact scrollable list."""
    st.markdown("**Lista completa ordinata per rilevanza:**")
    
    # Create container for scrollable content
    with st.container():
        for i, candidate in enumerate(candidates):
            with st.expander(
                f"🎬 Clip {i+1} | Score: {candidate.score:.3f} | Query: {candidate.original_query_phrase[:50]}...",
                expanded=False
            ):
                st.text_area(
                    "Contenuto:",
                    value=candidate.page_content,
                    height=80,
                    disabled=True,
                    key=f"compact_content_{i}_{iteration}"
                )
                
                # Metadata
                if candidate.formatted_metadata:
                    cols = st.columns(len(candidate.formatted_metadata))
                    for j, (key, value) in enumerate(candidate.formatted_metadata.items()):
                        with cols[j]:
                            st.caption(f"**{key}:** {value}")
                
                # Selection button with unique key
                unique_key = f"compact_select_{i}_{iteration}_{hash(candidate.page_content)}"
                if st.button(f"✅ Seleziona Clip {i+1}", key=unique_key, type="primary"):
                    st.success(f"Clip {i+1} selezionata!")
                    return i
    
    return None

def _render_detailed_tabs(candidates: list, iteration: int) -> Optional[int]:
    """Render candidates in tabs with pagination if too many."""
    max_tabs_per_page = 10
    total_pages = (len(candidates) + max_tabs_per_page - 1) // max_tabs_per_page
    
    if total_pages > 1:
        # Add page selector
        page = st.selectbox(
            "Pagina:",
            range(1, total_pages + 1),
            format_func=lambda x: f"Pagina {x} (Clip {(x-1)*max_tabs_per_page + 1}-{min(x*max_tabs_per_page, len(candidates))})",
            key=f"page_selector_{iteration}"
        )
        start_idx = (page - 1) * max_tabs_per_page
        end_idx = min(start_idx + max_tabs_per_page, len(candidates))
        page_candidates = candidates[start_idx:end_idx]
        offset = start_idx
    else:
        page_candidates = candidates
        offset = 0
    
    # Create tabs for current page
    tab_names = [f"Clip {offset+i+1} ({c.score:.2f})" for i, c in enumerate(page_candidates)]
    tabs = st.tabs(tab_names)
    
    for i, (tab, candidate) in enumerate(zip(tabs, page_candidates)):
        actual_index = offset + i
        with tab:
            st.markdown("**Contenuto della clip:**")
            st.text_area(
                "Testo:",
                value=candidate.page_content,
                height=120,
                disabled=True,
                key=f"tab_content_{actual_index}_{iteration}"
            )
            
            st.markdown(f"**Query di ricerca:** {candidate.original_query_phrase}")
            
            # Metadata
            if candidate.formatted_metadata:
                st.markdown("**Metadata:**")
                cols = st.columns(min(len(candidate.formatted_metadata), 4))
                for j, (key, value) in enumerate(candidate.formatted_metadata.items()):
                    with cols[j % len(cols)]:
                        st.metric(key, value)
            
            # Context sentences
            if candidate.previous_sentence or candidate.next_sentence:
                with st.expander("🔍 Contesto (frasi adiacenti)", expanded=False):
                    if candidate.previous_sentence:
                        st.markdown("**⬅️ Frase precedente:**")
                        st.caption(candidate.previous_sentence.get('page_content', 'N/A'))
                    if candidate.next_sentence:
                        st.markdown("**➡️ Frase successiva:**")
                        st.caption(candidate.next_sentence.get('page_content', 'N/A'))
            
            # Selection button
            col1, col2 = st.columns([1, 3])
            with col1:
                unique_key = f"tab_select_{actual_index}_{iteration}_{hash(candidate.page_content)}"
                if st.button(f"✅ Seleziona", key=unique_key, type="primary"):
                    st.success(f"Clip {actual_index+1} selezionata!")
                    return actual_index
            with col2:
                st.caption(f"Clip {actual_index+1} di {len(candidates)}")
    
    return None

def _render_fast_grid(candidates: list, iteration: int) -> Optional[int]:
    """Render candidates in a fast grid format."""
    st.markdown("**Vista rapida - clicca per selezionare:**")
    
    cols_per_row = 3
    for row_start in range(0, len(candidates), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for col_idx in range(cols_per_row):
            candidate_idx = row_start + col_idx
            if candidate_idx >= len(candidates):
                break
                
            candidate = candidates[candidate_idx]
            
            with cols[col_idx]:
                with st.container():
                    st.markdown(f"**🎬 Clip {candidate_idx + 1}**")
                    st.caption(f"Score: {candidate.score:.3f}")
                    st.caption(f"Query: {candidate.original_query_phrase[:30]}...")
                    
                    # Preview text
                    preview = candidate.page_content
                    st.text(preview)
                    
                    # Selection button
                    unique_key = f"grid_select_{candidate_idx}_{iteration}_{hash(candidate.page_content)}"
                    if st.button(f"✅ Clip {candidate_idx+1}", key=unique_key, use_container_width=True):
                        st.success(f"Clip {candidate_idx+1} selezionata!")
                        return candidate_idx
    
    return None

def _render_enhanced_grid(candidates: list, iteration: int) -> Optional[int]:
    """Render candidates in an enhanced grid with YouTube embeds and metadata."""
    st.markdown("**🎬 Grid con Player YouTube e Metadati**")
    
    cols_per_row = 2  # Riduciamo a 2 per più spazio per i video
    for row_start in range(0, len(candidates), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for col_idx in range(cols_per_row):
            candidate_idx = row_start + col_idx
            if candidate_idx >= len(candidates):
                break
                
            candidate = candidates[candidate_idx]
            
            with cols[col_idx]:
                with st.container():
                    # Header with clip number and score
                    st.markdown(f"### 🎬 Clip {candidate_idx + 1}")
                    
                    # Metadata row
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.metric("Score", f"{candidate.score:.3f}")
                    with meta_col2:
                        duration = candidate.metadata.get('duration', 'N/A')
                        if isinstance(duration, (int, float)):
                            st.metric("Durata", f"{duration:.1f}s")
                        else:
                            st.metric("Durata", str(duration))
                    
                    # Video info
                    video_id = candidate.metadata.get('video_id', '')
                    start_time = candidate.metadata.get('start_time', 0)
                    
                    if video_id:
                        # YouTube embed with start time
                        if isinstance(start_time, (int, float)):
                            youtube_url = f"https://www.youtube.com/embed/{video_id}?start={int(start_time)}"
                        else:
                            youtube_url = f"https://www.youtube.com/embed/{video_id}"
                        
                        st.markdown(f"""
                        <iframe width="100%" height="200" 
                                src="{youtube_url}" 
                                frameborder="0" 
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                                allowfullscreen>
                        </iframe>
                        """, unsafe_allow_html=True)
                        
                        # Video link
                        st.caption(f"📺 [Video {video_id}](https://youtube.com/watch?v={video_id}&t={int(start_time)}s)")
                    else:
                        st.info("🚫 Video ID non disponibile")
                    
                    # Content preview
                    st.markdown("**📝 Contenuto:**")
                    preview = candidate.page_content[:200] + "..." if len(candidate.page_content) > 200 else candidate.page_content
                    st.text_area(
                        "Testo:",
                        value=preview,
                        height=80,
                        key=f"preview_{candidate_idx}_{iteration}",
                        disabled=True
                    )
                    
                    # Query info
                    st.caption(f"🔍 Query: {candidate.original_query_phrase}")
                    
                    # Additional metadata
                    sentence_num = candidate.metadata.get('sentence_number', 'N/A')
                    end_time = candidate.metadata.get('end_time', 'N/A')
                    
                    with st.expander("📊 Dettagli Tecnici", expanded=False):
                        st.text(f"Sentence #: {sentence_num}")
                        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                            st.text(f"Timing: {start_time:.2f}s - {end_time:.2f}s")
                        else:
                            st.text(f"Start: {start_time} | End: {end_time}")
                        
                        # Previous/Next context if available
                        if candidate.previous_sentence:
                            st.text_area("Frase Precedente:", candidate.previous_sentence, height=60, disabled=True)
                        if candidate.next_sentence:
                            st.text_area("Frase Successiva:", candidate.next_sentence, height=60, disabled=True)
                    
                    # Selection button
                    unique_key = f"enhanced_grid_select_{candidate_idx}_{iteration}_{hash(candidate.page_content)}"
                    if st.button(
                        f"✅ Seleziona Clip {candidate_idx+1}", 
                        key=unique_key, 
                        use_container_width=True,
                        type="primary"
                    ):
                        st.success(f"🎯 Clip {candidate_idx+1} selezionata!")
                        return candidate_idx
                    
                    st.markdown("---")  # Separatore tra clip
    
    return None

def render_status_message(message_type: str, **kwargs) -> None:
    """
    Render standardized status messages.
    
    Args:
        message_type: Type of message from STATUS_MESSAGES config
        **kwargs: Format arguments for the message
    """
    from src.config.gui_config import STATUS_MESSAGES
    
    if message_type not in STATUS_MESSAGES:
        st.error(f"Unknown message type: {message_type}")
        return
    
    message = STATUS_MESSAGES[message_type].format(**kwargs)
    
    if message_type in ["loading", "processing"]:
        st.info(message)
    elif message_type in ["clip_selected"]:
        st.success(message)
    elif message_type in ["no_selection"]:
        st.warning(message)
    else:
        st.info(message)

def render_action_buttons(
    show_lucky: bool = True,
    show_skip: bool = True,
    show_auto_this: bool = True,
    disabled: bool = False
) -> Dict[str, bool]:
    """
    Render standardized action buttons.
    
    Args:
        show_lucky: Whether to show "I'm feeling lucky" button
        show_skip: Whether to show skip button
        show_auto_this: Whether to show "auto select this clip" button
        disabled: Whether buttons should be disabled
        
    Returns:
        Dictionary with button states
    """
    from src.config.gui_config import BUTTON_SETTINGS
    
    button_states = {}
    
    cols = st.columns(4)
    
    if show_lucky:
        with cols[0]:
            button_states['lucky'] = st.button(
                BUTTON_SETTINGS["lucky_button_text"],
                disabled=disabled,
                help="Completa automaticamente la selezione"
            )
    
    if show_auto_this:
        with cols[1]:
            button_states['auto_this'] = st.button(
                "🤖 Auto-selezione",
                disabled=disabled,
                help="Selezione automatica solo per questa clip"
            )
    
    if show_skip:
        with cols[2]:
            button_states['skip'] = st.button(
                "⏭️ Salta",
                disabled=disabled,
                help="Salta questa iterazione"
            )
    
    with cols[3]:
        button_states['cancel'] = st.button(
            BUTTON_SETTINGS["cancel_button_text"],
            disabled=disabled,
            help="Annulla e torna alla configurazione"
        )
    
    return button_states

def render_seed_input(default_seed: str = "", key: str = "seed_input") -> str:
    """
    Render seed input with validation and suggestions.
    
    Args:
        default_seed: Default seed value
        key: Unique key for the input widget
        
    Returns:
        The entered seed
    """
    st.markdown("### 🎲 Seed (Controllo Casualità)")
    
    # Handle random seed generation with persistent storage
    import time
    random_key = f"random_seed_generated_{key}"
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🎲 Random", key=f"random_seed_{key}", help="Genera seed casuale"):
            random_seed = str(int(time.time()))
            st.session_state[random_key] = random_seed
            # Update user_seed in session state to persist the value
            if "user_seed" in st.session_state:
                st.session_state.user_seed = random_seed
            st.rerun()
    
    # Use generated random seed if available, otherwise use default
    current_value = default_seed
    if random_key in st.session_state and st.session_state[random_key]:
        current_value = st.session_state[random_key]
    
    with col1:
        seed = st.text_input(
            "Seed per controllo della casualità:",
            value=current_value,
            placeholder="Es: 12345 o lascia vuoto per seed automatico",
            help="Il seed controlla la casualità della generazione AI. Stesso seed = stessi risultati",
            key=key
        )
    
    # Update session state with the current seed value
    if seed != current_value and "user_seed" in st.session_state:
        st.session_state.user_seed = seed
    
    # Seed information and suggestions
    with st.expander("ℹ️ Cosa è il Seed?", expanded=False):
        st.info("""
        **Il Seed controlla la casualità del sistema AI:**
        
        🎯 **Stesso seed = Stessi risultati**
        - Utile per riprodurre esattamente lo stesso video
        - Permette di condividere configurazioni specifiche
        
        🎲 **Seed diversi = Risultati diversi**
        - Ogni seed esplora il database in modo leggermente diverso
        - Stesso tema, frasi narrative potenzialmente diverse
        
        💡 **Suggerimenti:**
        - Lascia vuoto per seed automatico basato su timestamp
        - Usa numeri semplici (es: 1, 2, 3) per test
        - Salva seed interessanti per riutilizzarli
        """)
        
        # Quick seed suggestions
        st.markdown("**⚡ Seed Rapidi:**")
        quick_seeds = ["1", "42", "2025", "12345", str(int(time.time()))]
        cols = st.columns(len(quick_seeds))
        for i, quick_seed in enumerate(quick_seeds):
            with cols[i]:
                quick_key = f"quick_seed_{quick_seed}_{key}"
                if st.button(f"🎯 {quick_seed}", key=quick_key):
                    st.session_state[random_key] = quick_seed
                    # Update user_seed in session state to persist the value
                    if "user_seed" in st.session_state:
                        st.session_state.user_seed = quick_seed
                    st.rerun()
    
    return seed


