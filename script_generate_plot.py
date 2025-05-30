import json
import logging
import os
import math
import re
import unicodedata
from langchain_chroma import Chroma
from src.ai_models import AIModelsService, LLMType
from src.config.settings import VECTOR_STORE_DIR, VECTOR_STORE_SETTINGS, LOG_LEVEL, THEME, SEED

# Impostazione del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)

# --- CONSTANTS FOR THE NEW ITERATIVE PIPELINE ---
NUM_PHRASES_TO_EXPAND = 3
K_CLIPS_PER_EXPANSION_PHRASE = 10
MAX_VIDEO_CLIPS = 15
NARRATIVE_CONTEXT_WINDOW_SIZE = 3
MIN_SIMILARITY_THRESHOLD = 0.3


def sanitize_filename(name: str) -> str:
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^\w\-.]', '_', name)

OUTPUT_DIR = f"output/{sanitize_filename(THEME)}_{SEED}_iterative"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORDERED_FILE = os.path.join(OUTPUT_DIR, "ordered_sentences.json")
ITERATION_STATE_FILE = os.path.join(OUTPUT_DIR, "iteration_state.json")

def load_json_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {filepath}: {e}")
                return None
    return None

def save_json_file(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- LLM Interaction Functions for Iterative Pipeline (Italian Prompts) ---

def llm_generate_narrative_phrases(theme: str, last_query_phrase: str | None, ai_service: AIModelsService, num_phrases: int = NUM_PHRASES_TO_EXPAND) -> list[str]:
    """
    Richiede a un LLM di generare un set di frasi semanticamente correlate per il recupero di clip video.
    Se last_query_phrase è fornita, viene utilizzata per la continuità narrativa.
    """
    if last_query_phrase:
        context_prompt = (
            f"La frase chiave precedente utilizzata per trovare una clip era: '{last_query_phrase}'.\n"
            f"Basandoti su questo, e sul tema satirico generale di '{theme}', "
            f"genera {num_phrases} nuove e distinte frasi di ricerca per trovare le successive clip video. "
            f"Queste frasi dovrebbero mirare a continuare o sviluppare la narrazione o l'angolazione ironica, "
            f"potenzialmente esplorando sotto-temi correlati, contrasti o sviluppi inaspettati."
        )
    else: # Prima iterazione
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
    logger.debug(f"Prompt generazione frasi:\n{prompt}")

    response_str = ai_service.call_llm(prompt, llm_type=LLMType.CHEAP)
    try:
        response_json = json.loads(response_str)
        phrases = response_json.get("phrases", [])
        if isinstance(phrases, list) and all(isinstance(p, str) for p in phrases):
            logger.info(f"LLM ha generato le frasi: {phrases}")
            return phrases[:num_phrases]
        else:
            logger.error(f"La risposta dell'LLM per la generazione di frasi non era una lista di stringhe: {response_str}")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"Errore nel decodificare JSON dall'LLM per la generazione di frasi: {e}\nRisposta: {response_str}")
        return []

def llm_select_best_clip(
    candidate_clips: list[dict],
    theme: str,
    narrative_context_summary: str, # This will be in Italian if generated from Italian clips
    ai_service: AIModelsService
) -> dict | None:
    """
    Utilizza un LLM per selezionare la singola clip più appropriata dal punto di vista contestuale da una lista di candidati.
    """
    if not candidate_clips:
        return None

    simplified_candidates = []
    for idx, clip_data in enumerate(candidate_clips):
        simplified_candidates.append({
            "id": idx,
            "query_di_recupero": clip_data.get('original_query_phrase', 'N/D'), # "retrieval_query"
            "testo_clip": clip_data.get('page_content', ''), # "clip_text"
        })

    prompt = (
        f"Sei un esperto video editor che sta creando un video SATIRICO e IRONICO sul tema: '{theme}'.\n"
        f"La narrazione finora è composta da clip il cui testo implica: '{narrative_context_summary}'.\n\n"
        f"Dalle seguenti {len(simplified_candidates)} clip candidate, seleziona la SINGOLA MIGLIORE clip da aggiungere alla sequenza. "
        f"Considera:\n"
        f"1. Rilevanza rispetto al tema '{theme}'.\n"
        f"2. Coerenza narrativa: quanto bene segue o sviluppa la storia/ironia da '{narrative_context_summary}'.\n"
        f"3. Potenziale ironico: come il contenuto serio della clip potrebbe essere percepito come ironico o assurdo nel contesto di '{theme}'.\n"
        f"4. Coinvolgimento e impatto.\n\n"
        f"Clip Candidate:\n" # "Candidate Clips:"
        f"{json.dumps(simplified_candidates, indent=2, ensure_ascii=False)}\n\n"
        f"Fornisci la tua selezione come un oggetto JSON con 'selected_clip_id' (l'id numerico dalla lista sopra) e 'justification' (una breve spiegazione per la tua scelta). Esempio:\n"
        f'{{"selected_clip_id": 0, "justification": "Questa clip offre un buon contrasto..."}}' # Justification example can remain in English or be Italian
    )
    logger.debug(f"Prompt selezione clip:\n{prompt[:1000]}...")

    response_str = ai_service.call_llm(prompt, llm_type=LLMType.INTELLIGENT)
    try:
        response_json = json.loads(response_str)
        selected_id = response_json.get("selected_clip_id")
        justification = response_json.get("justification", "")

        if isinstance(selected_id, int) and 0 <= selected_id < len(candidate_clips):
            selected_clip_data = candidate_clips[selected_id]
            selected_clip_data['selection_justification'] = justification
            logger.info(f"LLM ha selezionato la clip ID {selected_id}. Motivazione: {justification}")
            return selected_clip_data
        else:
            logger.error(f"LLM ha restituito un selected_clip_id non valido: {selected_id}. Risposta: {response_str}")
            if candidate_clips:
                 logger.warning("Selezione LLM fallita, utilizzando il primo candidato come fallback.")
                 candidate_clips[0]['selection_justification'] = "Fallback: Selezione LLM fallita."
                 return candidate_clips[0]
            return None
    except json.JSONDecodeError as e:
        logger.error(f"Errore nel decodificare JSON dall'LLM per la selezione della clip: {e}\nRisposta: {response_str}")
        if candidate_clips:
            logger.warning("Parsing JSON della selezione LLM fallito, utilizzando il primo candidato come fallback.")
            candidate_clips[0]['selection_justification'] = "Fallback: Parsing JSON della selezione LLM fallito."
            return candidate_clips[0]
        return None

# --- Vector Store Interaction (unchanged from previous refactor) ---
def search_vector_store(query, vector_store, k=8, min_similarity=MIN_SIMILARITY_THRESHOLD, excluded_doc_ids=None):
    filters = {"duration": {"$lte": 20}}
    if excluded_doc_ids:
        filters = {
            "$and": [
                {"doc_id": {"$nin": list(excluded_doc_ids)}},
                {"duration": {"$lte": 20}}
            ]
        }
    
    results_with_scores = vector_store.similarity_search_with_score(
        query, k=k, filter=filters
    )
    
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
        
        sentence_idx = metadata.get("sentence_number", None)
        video_id = metadata.get("video_id", None)
        previous_sentence_obj = None
        next_sentence_obj = None

        if sentence_idx is not None and video_id is not None:
            if sentence_idx > 0:
                prev_sent_num = sentence_idx - 1
                prev_doc_id = f"{video_id}_{prev_sent_num}"
                if not (excluded_doc_ids and prev_doc_id in excluded_doc_ids):
                    try:
                        prev_docs = vector_store.get(ids=[prev_doc_id], include=["metadatas", "documents"])
                        if prev_docs["documents"] and prev_docs["documents"][0] is not None:
                             previous_sentence_obj = {
                                "page_content": prev_docs["documents"][0],
                                "metadata": prev_docs["metadatas"][0]
                            }
                    except Exception as e:
                        logger.debug(f"Non è stato possibile recuperare la frase precedente {prev_doc_id}: {e}")
            
            next_sent_num = sentence_idx + 1
            next_doc_id = f"{video_id}_{next_sent_num}"
            if not (excluded_doc_ids and next_doc_id in excluded_doc_ids):
                try:
                    next_docs = vector_store.get(ids=[next_doc_id], include=["metadatas", "documents"])
                    if next_docs["documents"] and next_docs["documents"][0] is not None:
                        next_sentence_obj = {
                            "page_content": next_docs["documents"][0],
                            "metadata": next_docs["metadatas"][0]
                        }
                except Exception as e:
                    logger.debug(f"Non è stato possibile recuperare la frase successiva {next_doc_id}: {e}")

        extracted_results.append({
            "page_content": page_content,
            "metadata": metadata,
            "doc_id": doc_id,
            "score": score,
            "previous_sentence": previous_sentence_obj,
            "next_sentence": next_sentence_obj
        })
    return extracted_results


# --- Final Output Formatting (unchanged from previous refactor) ---
def format_final_output(selected_clips: list[dict], theme: str) -> dict:
    ordered_phrases_output = []
    for i, clip_data in enumerate(selected_clips):
        metadata = clip_data.get("metadata", {})
        ordered_phrases_output.append({
            "matched_phrase": clip_data.get("page_content", ""),
            "order": i + 1,
            "query_phrase_that_led_to_this_clip": clip_data.get("original_query_phrase", "N/A"),
            "selection_justification": clip_data.get("selection_justification", ""),
            "retrieval_score": clip_data.get("score", 0.0),
            "source": f"{metadata.get('video_id', 'N/A')}/{metadata.get('sentence_number', 'N/A')}",
            "metadata": metadata,
            "previous_sentence_obj": clip_data.get("previous_sentence"),
            "next_sentence_obj": clip_data.get("next_sentence")
        })
    return {
        "theme": theme,
        "total_clips": len(ordered_phrases_output),
        "ordered_phrases": ordered_phrases_output
    }

# --- Main Iterative Pipeline (logic unchanged, but narrative_summary will be Italian if clips are Italian) ---
def main_iterative():
    logger.info(f"Avvio costruzione video iterativa per il tema: '{THEME}'")

    ai_service = AIModelsService()
    embedding_model = ai_service.get_embedding_model()
    try:
        chroma_db = Chroma(
            collection_name=VECTOR_STORE_SETTINGS["collection_name"],
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embedding_model
        )
        logger.info(f"Connesso alla collezione ChromaDB: {VECTOR_STORE_SETTINGS['collection_name']}")
    except Exception as e:
        logger.error(f"Inizializzazione ChromaDB fallita: {e}")
        return

    selected_clips_sequence: list[dict] = []
    excluded_doc_ids: set[str] = set()
    current_query_phrase_for_expansion: str | None = None

    for clip_num in range(MAX_VIDEO_CLIPS):
        logger.info(f"--- Iterazione {clip_num + 1} / {MAX_VIDEO_CLIPS} ---")

        narrative_phrases = llm_generate_narrative_phrases(
            THEME,
            current_query_phrase_for_expansion,
            ai_service,
            num_phrases=NUM_PHRASES_TO_EXPAND
        )
        if not narrative_phrases:
            logger.warning("Generazione frasi narrative fallita. Interruzione iterazione.")
            break

        all_candidate_clips_for_this_iteration: list[dict] = []
        for phrase in narrative_phrases:
            logger.info(f"Recupero clip per la frase: '{phrase}'")
            retrieved_clips = search_vector_store(
                query=phrase,
                vector_store=chroma_db,
                k=K_CLIPS_PER_EXPANSION_PHRASE,
                min_similarity=MIN_SIMILARITY_THRESHOLD,
                excluded_doc_ids=excluded_doc_ids
            )
            for clip_data in retrieved_clips:
                clip_data['original_query_phrase'] = phrase
                all_candidate_clips_for_this_iteration.append(clip_data)
            logger.info(f"Recuperate {len(retrieved_clips)} clip per la frase '{phrase}'. Candidati totali finora: {len(all_candidate_clips_for_this_iteration)}")

        if not all_candidate_clips_for_this_iteration:
            logger.warning("Nessuna clip candidata trovata in questa iterazione. Salto al prossimo tentativo o interruzione.")
            current_query_phrase_for_expansion = None
            if clip_num > 0 and len(narrative_phrases) > 0 :
                 current_query_phrase_for_expansion = narrative_phrases[0]
            continue

        logger.info(f"Totale {len(all_candidate_clips_for_this_iteration)} clip candidate per la selezione.")

        narrative_context_texts = [
            clip['page_content'] for clip in selected_clips_sequence[-NARRATIVE_CONTEXT_WINDOW_SIZE:]
        ]
        # If the source clips are Italian, narrative_summary will naturally be in Italian
        narrative_summary = " ".join(narrative_context_texts) if narrative_context_texts else "Questo è l'inizio del video."
        
        selected_clip = llm_select_best_clip(
            all_candidate_clips_for_this_iteration,
            THEME,
            narrative_summary, # This will be Italian
            ai_service
        )

        if not selected_clip:
            logger.warning("LLM non è riuscito a selezionare una clip in questa iterazione. Tentativo di continuare.")
            continue

        selected_clips_sequence.append(selected_clip)
        
        selected_doc_id = selected_clip.get("doc_id")
        if selected_doc_id:
            excluded_doc_ids.add(selected_doc_id)
        else:
            logger.error(f"La clip selezionata non ha un 'doc_id': {selected_clip.get('page_content', '')[:50]}...")
            excluded_doc_ids.add(selected_clip.get('page_content', ''))

        current_query_phrase_for_expansion = selected_clip.get('original_query_phrase', None)
        
        logger.info(f"Clip {len(selected_clips_sequence)} selezionata: '{selected_clip.get('page_content', '')[:70]}...' (dalla query: '{current_query_phrase_for_expansion}')")
        
    if not selected_clips_sequence:
        logger.error("Nessuna clip selezionata per il video. Uscita.")
        return

    final_video_structure = format_final_output(selected_clips_sequence, THEME)
    save_json_file(ORDERED_FILE, final_video_structure)
    logger.info(f"Costruzione video iterativa completata. {len(selected_clips_sequence)} clip selezionate.")
    logger.info(f"Script video finale salvato in: {ORDERED_FILE}")


if __name__ == "__main__":
    main_iterative()