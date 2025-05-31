"""
GUI Service for Interactive Clip Selection

This service manages the interactive clip selection workflow,
integrating with the existing AI-powered video generation pipeline.
"""

import json
import logging
import asyncio
import sys
import os
import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import streamlit as st

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import existing services
from src.ai_models import AIModelsService, LLMType
from src.config.settings import VECTOR_STORE_DIR, VECTOR_STORE_SETTINGS
from src.config.gui_config import GUI_SETTINGS, CLIP_DISPLAY_SETTINGS, STATUS_MESSAGES
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

def sanitize_filename(name: str) -> str:
    """Sanitize filename to match the original pipeline format."""
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^\w\-.]', '_', name)

@dataclass
class ClipCandidate:
    """Represents a candidate clip for selection."""
    clip_id: str
    page_content: str
    metadata: Dict[str, Any]
    score: float
    original_query_phrase: str
    previous_sentence: Optional[Dict] = None
    next_sentence: Optional[Dict] = None
    
    @property
    def display_text(self) -> str:
        """Get truncated text for display."""
        max_length = CLIP_DISPLAY_SETTINGS.get("max_clip_text_preview", 100)
        if len(self.page_content) <= max_length:
            return self.page_content
        return self.page_content[:max_length] + "..."
    
    @property
    def formatted_metadata(self) -> Dict[str, str]:
        """Get formatted metadata for display."""
        formatted = {}
        fields = CLIP_DISPLAY_SETTINGS.get("metadata_fields", {})
        formatters = CLIP_DISPLAY_SETTINGS.get("field_formatters", {})
        
        for key, display_name in fields.items():
            if key in self.metadata:
                value = self.metadata[key]
                if key in formatters:
                    formatter = formatters[key]
                    if callable(formatter):
                        value = formatter(value)
                    elif formatter == "timestamp":
                        value = self._format_timestamp(value)
                formatted[display_name] = str(value)
        
        return formatted
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

@dataclass 
class SelectionState:
    """Manages the state of the clip selection process."""
    theme: str
    seed: str = None
    current_iteration: int = 0
    max_iterations: int = 15
    selected_clips: List[ClipCandidate] = None
    narrative_context: str = ""
    is_complete: bool = False
    
    def __post_init__(self):
        if self.selected_clips is None:
            self.selected_clips = []
        if self.seed is None:
            import time
            self.seed = str(int(time.time()))

class GUIService:
    """Service for managing interactive clip selection."""
    
    def __init__(self):
        self.ai_service = AIModelsService()
        self.vector_store = None
        self._initialize_vector_store()
        
    def _initialize_vector_store(self):
        """Initialize the vector store connection."""
        try:
            embedding_model = self.ai_service.get_embedding_model()
            self.vector_store = Chroma(
                collection_name=VECTOR_STORE_SETTINGS["collection_name"],
                persist_directory=str(VECTOR_STORE_DIR),
                embedding_function=embedding_model
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def initialize_session(self, theme: str, seed: str = None) -> SelectionState:
        """Initialize a new clip selection session."""
        return SelectionState(theme=theme, seed=seed)
    
    def generate_narrative_phrases(self, 
                                 state: SelectionState, 
                                 num_phrases: int = 3) -> List[str]:
        """Generate narrative phrases for the current iteration."""
        # Raccoglie le ultime 10 frasi di query per dare più contesto all'LLM
        context_queries = []
        if state.selected_clips:
            # Prende le ultime 10 frasi (o tutte se ce ne sono meno di 10)
            recent_clips = state.selected_clips[-10:]
            context_queries = [clip.original_query_phrase for clip in recent_clips]
            logger.info(f"📝 Using {len(context_queries)} context queries from last {len(recent_clips)} clips")
            logger.debug(f"Context queries: {context_queries}")
            
        return self._llm_generate_narrative_phrases(
            state.theme, 
            context_queries, 
            num_phrases
        )
    
    def search_candidate_clips(self, 
                             phrases: List[str], 
                             excluded_doc_ids: set, 
                             k_per_phrase: int = 10) -> List[ClipCandidate]:
        """Search for candidate clips based on narrative phrases."""
        candidates = []
        
        logger.info(f"🔍 Searching for clips with {len(phrases)} phrases, k={k_per_phrase} each")
        logger.info(f"📝 Phrases: {phrases}")
        logger.info(f"🚫 Excluded doc IDs: {len(excluded_doc_ids)} items")
        
        for phrase_idx, phrase in enumerate(phrases):
            logger.info(f"🔍 Searching phrase {phrase_idx + 1}: '{phrase}'")
            
            results = self._search_vector_store(
                query=phrase,
                k=k_per_phrase,
                excluded_doc_ids=excluded_doc_ids
            )
            
            logger.info(f"📊 Found {len(results)} results for phrase '{phrase}'")
            
            for result_idx, result in enumerate(results):
                candidate = ClipCandidate(
                    clip_id=result.get("doc_id", f"phrase{phrase_idx}_result{result_idx}"),
                    page_content=result.get("page_content", ""),
                    metadata=result.get("metadata", {}),
                    score=result.get("score", 0.0),
                    original_query_phrase=phrase,
                    previous_sentence=result.get("previous_sentence"),
                    next_sentence=result.get("next_sentence")
                )
                candidates.append(candidate)
        
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"✅ Total candidates found: {len(candidates)}")
        logger.info(f"📈 Score range: {candidates[0].score:.3f} - {candidates[-1].score:.3f}")
        
        return candidates
    
    def auto_select_best_clip(self, 
                            candidates: List[ClipCandidate], 
                            state: SelectionState) -> ClipCandidate:
        """Automatically select the best clip using LLM."""
        if not candidates:
            return None
            
        # Convert candidates to format expected by LLM
        candidate_data = []
        for i, candidate in enumerate(candidates):
            candidate_data.append({
                "id": i,
                "query_di_recupero": candidate.original_query_phrase,
                "testo_clip": candidate.page_content,
                **candidate.metadata
            })
        
        selected_clip = self._llm_select_best_clip(
            candidate_data, 
            state.theme, 
            state.narrative_context
        )
        
        if selected_clip:
            return candidates[selected_clip.get("selected_clip_id", 0)]
        
        # Fallback to highest scoring clip
        return candidates[0] if candidates else None
    
    def add_selected_clip(self, state: SelectionState, clip: ClipCandidate):
        """Add a selected clip to the session state."""
        logger.info(f"✅ Adding selected clip:")
        logger.info(f"   📎 Clip ID: {clip.clip_id}")
        logger.info(f"   📝 Query phrase: {clip.original_query_phrase}")
        logger.info(f"   📊 Score: {clip.score:.3f}")
        logger.info(f"   📄 Content (first 100 chars): {clip.page_content[:100]}...")
        
        state.selected_clips.append(clip)
        old_iteration = state.current_iteration
        state.current_iteration += 1
        
        # Update narrative context
        context_clips = state.selected_clips[-3:]  # Last 3 clips
        state.narrative_context = " ".join([c.page_content for c in context_clips])
        
        # Check if we've reached the maximum iterations
        if state.current_iteration >= state.max_iterations:
            state.is_complete = True
            logger.info(f"🎉 Session complete! Selected {len(state.selected_clips)} clips total.")
        else:
            logger.info(f"🔄 Advanced from iteration {old_iteration} to {state.current_iteration}")
            logger.info(f"📖 Updated narrative context with {len(context_clips)} recent clips")
    
    def get_excluded_doc_ids(self, state: SelectionState) -> set:
        """Get doc IDs that should be excluded from search."""
        excluded = set()
        for clip in state.selected_clips:
            if clip.clip_id:
                excluded.add(clip.clip_id)
        return excluded
    
    def get_output_directory(self, state: SelectionState) -> str:
        """Generate output directory path matching the original pipeline format."""
        # Use the same format as the original pipeline: output/{sanitized_theme}_{seed}_iterative
        sanitized_theme = sanitize_filename(state.theme)
        output_dir = f"output/{sanitized_theme}_{state.seed}_iterative"
        return output_dir
    
    def export_selection_to_ordered_file(self, state: SelectionState, output_dir: str):
        """Export the selected clips to the format expected by video creation pipeline."""
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the ordered_sentences.json file in the correct format
        ordered_phrases_output = []
        
        for i, clip in enumerate(state.selected_clips):
            ordered_phrases_output.append({
                "matched_phrase": clip.page_content,
                "order": i + 1,
                "query_phrase_that_led_to_this_clip": clip.original_query_phrase,
                "selection_justification": "User selected via GUI",
                "retrieval_score": clip.score,
                "source": f"{clip.metadata.get('video_id', 'N/A')}/{clip.metadata.get('sentence_number', 'N/A')}",
                "metadata": clip.metadata,
                "previous_sentence_obj": clip.previous_sentence,
                "next_sentence_obj": clip.next_sentence
            })
        
        # Format according to the original pipeline structure
        output_data = {
            "theme": state.theme,
            "total_clips": len(ordered_phrases_output),
            "ordered_phrases": ordered_phrases_output
        }
        
        # Write to ordered_sentences.json (the filename expected by script_create_video.py)
        ordered_file_path = os.path.join(output_dir, "ordered_sentences.json")
        with open(ordered_file_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Exported {len(state.selected_clips)} selected clips to {ordered_file_path}")
        logger.info(f"Theme: {state.theme}, Seed: {state.seed}")
        logger.info(f"Output directory: {output_dir}")
        
        return ordered_file_path
    
    def run_video_creation(self, state: SelectionState) -> str:
        """Run the video creation script with the exported clips."""
        import subprocess
        import sys
        
        # Get the output directory
        output_dir = self.get_output_directory(state)
        
        # Check if ordered_sentences.json exists
        ordered_file = os.path.join(output_dir, "ordered_sentences.json")
        if not os.path.exists(ordered_file):
            raise FileNotFoundError(f"Ordered sentences file not found: {ordered_file}")
        
        try:
            # Run the video creation script with theme and seed as command line arguments
            cmd = [
                sys.executable, 
                "script_create_video.py",
                "--theme", state.theme,
                "--seed", str(state.seed)
            ]
            logger.info(f"Running video creation with command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                cwd=str(Path(__file__).parent.parent.parent), 
                capture_output=True, 
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                final_video = os.path.join(output_dir, "final_montage.mp4")
                logger.info(f"Video creation successful! Output: {final_video}")
                return final_video
            else:
                error_msg = f"Video creation failed with return code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            error_msg = "Video creation timed out after 1 hour"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Error running video creation: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _llm_generate_narrative_phrases(self, 
                                      theme: str, 
                                      context_queries: List[str], 
                                      num_phrases: int) -> List[str]:
        """Generate narrative phrases using LLM with enhanced context from last 10 queries."""
        if context_queries:
            # Costruisce il contesto dalle ultime query utilizzate
            context_list = "\n".join([f"- {query}" for query in context_queries])
            logger.info(f"🎯 Generating phrases with {len(context_queries)} context queries")
            context_prompt = (
                f"Le frasi chiave utilizzate finora per trovare clip nel video su '{theme}' sono state:\n"
                f"{context_list}\n\n"
                f"Basandoti su questa progressione narrativa e sul tema satirico generale di '{theme}', "
                f"genera {num_phrases} nuove e distinte frasi di ricerca per trovare le successive clip video. "
                f"Queste frasi dovrebbero:\n"
                f"- Continuare o sviluppare la narrazione esistente\n"
                f"- Esplorare nuovi aspetti ironici del tema\n"
                f"- Evitare di ripetere concetti già coperti dalle frasi precedenti\n"
                f"- Mantenere coerenza con la direzione satirica intrapresa"
            )
        else:
            logger.info("🎯 Generating initial phrases (no context available)")
            context_prompt = (
                f"Per un video IRONICO e SATIRICO sul tema '{theme}', "
                f"genera {num_phrases} frasi di ricerca iniziali e diversificate per trovare clip video avvincenti. "
                f"Queste frasi dovrebbero toccare diverse sfaccettature o potenziali ironie legate al tema."
            )

        prompt = (
            f"{context_prompt}\n"
            f"Le frasi dovrebbero essere adatte per la ricerca semantica in un database video. "
            f"Concentrati su frasi che, quando utilizzate per trovare clip del mondo reale, potrebbero produrre contenuti che appaiono "
            f"ironici o assurdi se inseriti nel contesto del tema '{theme}'.\n"
            f"Restituisci un oggetto JSON con una singola chiave 'phrases' contenente una lista di stringhe. Esempio:\n"
            f'{{"phrases": ["frase 1", "frase 2", "frase 3"]}}'
        )

        try:
            response_str = self.ai_service.call_llm(prompt, llm_type=LLMType.CHEAP)
            response_json = json.loads(response_str)
            phrases = response_json.get("phrases", [])
            
            if isinstance(phrases, list) and all(isinstance(p, str) for p in phrases):
                return phrases[:num_phrases]
            else:
                logger.error(f"Invalid LLM response for phrase generation: {response_str}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in phrase generation: {e}")
            return []
    
    def _search_vector_store(self, 
                           query: str, 
                           k: int = 8, 
                           excluded_doc_ids: set = None) -> List[Dict]:
        """Search vector store for matching clips."""
        filters = {"duration": {"$lte": 20}}
        if excluded_doc_ids:
            filters = {
                "$and": [
                    {"doc_id": {"$nin": list(excluded_doc_ids)}},
                    {"duration": {"$lte": 20}}
                ]
            }
        
        results_with_scores = self.vector_store.similarity_search_with_score(
            query, k=k, filter=filters
        )
        
        # Filter by minimum similarity threshold
        min_similarity = 0.3
        filtered_results = [r for r in results_with_scores if r[1] >= min_similarity]
        
        extracted_results = []
        for item in filtered_results:
            score = item[1]
            document = item[0]
            page_content = document.page_content
            metadata = document.metadata.copy()

            doc_id = metadata.get("doc_id")
            if not doc_id:
                video_id = metadata.get("video_id", "unknown_video")
                sentence_idx = metadata.get("sentence_number", "unknown_sentence")
                doc_id = f"{video_id}_{sentence_idx}"
                metadata["doc_id"] = doc_id
            
            # Add previous/next sentence context if available
            sentence_idx = metadata.get("sentence_number", None)
            video_id = metadata.get("video_id", None)
            previous_sentence_obj = None
            next_sentence_obj = None

            if sentence_idx is not None and video_id is not None and sentence_idx > 0:
                # Could implement context retrieval here if needed
                pass

            extracted_results.append({
                "page_content": page_content,
                "metadata": metadata,
                "doc_id": doc_id,
                "score": score,
                "previous_sentence": previous_sentence_obj,
                "next_sentence": next_sentence_obj
            })
        
        return extracted_results
    
    def _llm_select_best_clip(self, 
                            candidate_clips: List[Dict], 
                            theme: str, 
                            narrative_context: str) -> Optional[Dict]:
        """Use LLM to select the best clip from candidates."""
        if not candidate_clips:
            return None

        simplified_candidates = []
        for idx, clip_data in enumerate(candidate_clips):
            simplified_candidates.append({
                "id": idx,
                "query_di_recupero": clip_data.get('query_di_recupero', 'N/D'),
                "testo_clip": clip_data.get('testo_clip', ''),
            })

        prompt = (
            f"Sei un esperto video editor che sta creando un video SATIRICO e IRONICO sul tema: '{theme}'.\n"
            f"La narrazione finora è composta da clip il cui testo implica: '{narrative_context}'.\n\n"
            f"Dalle seguenti {len(simplified_candidates)} clip candidate, seleziona la SINGOLA MIGLIORE clip da aggiungere alla sequenza. "
            f"Considera:\n"
            f"1. Rilevanza rispetto al tema '{theme}'.\n"
            f"2. Coerenza narrativa: quanto bene segue o sviluppa la storia/ironia da '{narrative_context}'.\n"
            f"3. Potenziale ironico: come il contenuto serio della clip potrebbe essere percepito come ironico o assurdo nel contesto di '{theme}'.\n"
            f"4. Coinvolgimento e impatto.\n\n"
            f"Clip Candidate:\n"
            f"{json.dumps(simplified_candidates, indent=2, ensure_ascii=False)}\n\n"
            f"Fornisci la tua selezione come un oggetto JSON con 'selected_clip_id' (l'id numerico dalla lista sopra) e 'justification' (una breve spiegazione per la tua scelta). Esempio:\n"
            f'{{"selected_clip_id": 0, "justification": "Questa clip offre un buon contrasto..."}}'
        )

        try:
            response_str = self.ai_service.call_llm(prompt, llm_type=LLMType.INTELLIGENT)
            response_json = json.loads(response_str)
            selected_id = response_json.get("selected_clip_id")
            justification = response_json.get("justification", "")

            if isinstance(selected_id, int) and 0 <= selected_id < len(candidate_clips):
                return {
                    "selected_clip_id": selected_id,
                    "justification": justification
                }
            else:
                logger.error(f"Invalid selected_clip_id: {selected_id}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in clip selection: {e}")
            return None
