import json
import logging
import os
import math
from langchain_chroma import Chroma
from src.ai_models import AIModelsService, LLMType
from src.config.settings import VECTOR_STORE_DIR, VECTOR_STORE_SETTINGS, LOG_LEVEL, THEME, SEED

# Impostazione del logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)

# COSTANTI
IRONY_LEVEL = 6  # Su una scala da 1 a 10, dove 10 è il massimo dell'ironia
RELEVANCE_LEVEL = 5
TOP_IRONY_LIMIT = 30
TOP_RELEVANCE_LIMIT = 30



import re
import unicodedata

def sanitize_filename(name: str) -> str:
    """
    Rende sicuro un nome di file rimuovendo caratteri non compatibili con i filesystem
    o con tool come FFmpeg (es. virgolette, apostrofi, virgole, accenti, ecc.).
    """
    # Rimuove accenti/segni diacritici
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
    
    # Sostituisce caratteri non validi con underscore
    return re.sub(r'[^\w\-.]', '_', name)

OUTPUT_DIR = f"output/{sanitize_filename(THEME)}_{SEED}"  # Cartella dedicata al tema e al seed
MAX_PHRASES_PER_BATCH = 20  # Numero massimo di frasi per batch per l'ordinamento
MIN_SIMILARITY_THRESHOLD = 0.3  # Soglia minima di similarità per le ricerche vettoriali

# Creazione cartella di output se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)



# File intermedi
TOPICS_FILE = os.path.join(OUTPUT_DIR, "topics.json")
PHRASES_FILE = os.path.join(OUTPUT_DIR, "phrases.json")
VECTORS_FILE = os.path.join(OUTPUT_DIR, "vector_results.json")
FILTERED_FILE = os.path.join(OUTPUT_DIR, "filtered_phrases.json")
TAGGED_FILE = os.path.join(OUTPUT_DIR, "tagged_phrases.json")
BATCHES_FILE = os.path.join(OUTPUT_DIR, "batch_ordering.json")
ORDERED_FILE = os.path.join(OUTPUT_DIR, "ordered_sentences.json")

def load_json_file(filepath):
    """Carica un file JSON se esiste, altrimenti restituisce None."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_json_file(filepath, data):
    """Salva i dati in un file JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def get_topics(theme):
    """Ottiene o carica una lista di argomenti correlati al tema con enfasi sull'ironia."""
    topics = load_json_file(TOPICS_FILE)
    if topics is None:
        prompt = (
            f"Immagina di dover creare un video IRONICO e SATIRICO sul tema '{theme}'. "
            f"Genera una lista di 6 argomenti laterali, assurdi, sorprendenti ma in qualche modo collegati al tema. "
            f"Pensa a prospettive contrastanti, situazioni paradossali o accostamenti inaspettati. "
            f"La risposta deve essere un JSON valido con questa struttura:\n"
            f'{{"topics": ["argomento1", "argomento2", "argomento3", ...]}}'
        )
        response = AIModelsService().call_llm(prompt, llm_type=LLMType.CHEAP)
        try:
            topics = json.loads(response).get("topics", [])
            save_json_file(TOPICS_FILE, topics)
        except json.JSONDecodeError as e:
            logger.error(f"Errore di parsing JSON nei topics: {e}")
            topics = []
    return topics

def get_phrases_for_topic(topic):
    """Ottiene o genera una lista di frasi per un argomento con maggiore potenziale ironico."""
    phrases_data = load_json_file(PHRASES_FILE) or {}

    if topic not in phrases_data:
        prompt = (
            f"Per un video IRONICO e SATIRICO sul tema '{topic}', genera 8 frasi che contengano "
            f"CONTRADDIZIONI, PARADOSSI o SITUAZIONI ASSURDE. Queste frasi serviranno per trovare contenuti reali "
            f"nel database che possano sembrare ironici quando decontestualizzati.\n"
            f"Crea frasi che, se pronunciate seriamente ma inserite in un contesto diverso, risulterebbero comiche o assurde. "
            f"Evita riferimenti espliciti a specifici brand, persone reali o eventi troppo specifici. "
            f"La risposta deve essere un JSON valido con questa struttura:\n"
            f'{{"phrases": ["frase1", "frase2", "frase3", ...]}}'
        )

        response = AIModelsService().call_llm(prompt, llm_type=LLMType.CHEAP)
        try:
            phrases = json.loads(response).get("phrases", [])
            phrases_data[topic] = phrases
            save_json_file(PHRASES_FILE, phrases_data)
        except json.JSONDecodeError as e:
            logger.error(f"Errore di parsing JSON nelle frasi per '{topic}': {e}")
            phrases = []
    else:
        phrases = phrases_data[topic]

    return phrases

def search_vector_store(query, vector_store, k=8, min_similarity=MIN_SIMILARITY_THRESHOLD, excluded_doc_ids=None):
    # 1. Set up filters to exclude already seen documents
    filters = None
    if excluded_doc_ids:
        filters = {
            "$and": [
                {
                    "doc_id": {
                       "$nin": list(excluded_doc_ids)  
                    }
                },
                {
                    "duration": {
                        "$lte": 30
                    }
                }
            ]
        }
    else:
        filters = {
                "duration": {
                    "$lte": 30
                }
        }

    # 2. Perform the similarity search with exclusion filter
    results = vector_store.similarity_search_with_score(
        query, 
        k=k, 
        filter=filters  # Applies document exclusion here
    )
    
    # 3. Filter results by similarity score threshold
    filtered_results = [r for r in results if r[1] >= min_similarity]

    extracted_results = []
    for item in filtered_results:
        score = item[1]
        document = item[0]
        page_content = document.page_content
        metadata = document.metadata.copy()

        sentence_idx = metadata.get("sentence_number", None)
        video_id = metadata.get("video_id", None)

        previous_sentence = None
        next_sentence = None

        if sentence_idx is not None:
            # 4. Get previous sentence ONLY if not excluded
            if sentence_idx > 1:
                previous_sentence_number = sentence_idx - 1
                doc_id_prev = f"{video_id}_{previous_sentence_number}"
                
                # Check if previous sentence is in excluded IDs
                if excluded_doc_ids and doc_id_prev in excluded_doc_ids:
                    previous_sentence = None
                else:
                    previous_sentence_vect = vector_store.get(
                        where={"doc_id": doc_id_prev},
                        include=["metadatas", "documents"]
                    )
                    if previous_sentence_vect["documents"]:
                        previous_sentence = {
                            "page_content": previous_sentence_vect["documents"][0],
                            "metadata": previous_sentence_vect["metadatas"][0]
                        }

            # 5. Get next sentence ONLY if not excluded
            try:
                next_sentence_number = sentence_idx + 1
                doc_id_next = f"{video_id}_{next_sentence_number}"
                
                # Check if next sentence is in excluded IDs
                if excluded_doc_ids and doc_id_next in excluded_doc_ids:
                    next_sentence = None
                else:
                    next_sentence_vect = vector_store.get(
                        where={"doc_id": doc_id_next},
                        include=["metadatas", "documents"]
                    )
                    if next_sentence_vect["documents"]:
                        next_sentence = {
                            "page_content": next_sentence_vect["documents"][0],
                            "metadata": next_sentence_vect["metadatas"][0]
                        }
            except Exception as e:
                logger.debug(f"Error getting next sentence: {str(e)}")
                next_sentence = None

        # 6. Build final result with context checks
        extracted_results.append({
            "page_content": page_content,
            "metadata": metadata,
            "score": score,
            "previous_sentence": previous_sentence if previous_sentence else None,
            "next_sentence": next_sentence if next_sentence else None,
            # Add explicit doc_id reference
            "doc_id": metadata.get("doc_id", f"{video_id}_{sentence_idx}")
        })

    return extracted_results

def score_irony(all_phrases_with_context, theme, batch_size=50):
    """
    Valuta il potenziale ironico di ciascuna frase, restituendo
    all_phrases_with_context con un campo 'irony_score' per ogni voce.
    Non scarta alcuna frase: si limita ad assegnare un punteggio.
    """
    phrases_batches = [
        all_phrases_with_context[i : i+batch_size]
        for i in range(0, len(all_phrases_with_context), batch_size)
    ]

    for batch_index, batch in enumerate(phrases_batches):
        # Prepariamo un JSON con le frasi da valutare:
        prompt_batch = json.dumps(batch, ensure_ascii=False)

        prompt = (
            f"Stai valutando una serie di frasi reali da usare in un video ironico sul tema '{theme}'.\n"
            f"Per ogni frase, assegna un punteggio 'irony_score' da 1 a 10, e una brevissima spiegazione del perché.\n\n"
            f"Restituisci un JSON valido del tipo:\n"
            f'{{"irony_scored_phrases": ['
            f'{{"phrase_id": "...", "irony_score": 7.5, "irony_explanation": "..."}}, ...]}}\n\n'
            f"Ecco il batch {batch_index+1}/{len(phrases_batches)} da valutare:\n"
            f"{prompt_batch}"
        )

        response = AIModelsService().call_llm(prompt, llm_type=LLMType.CHEAP)

        try:
            result = json.loads(response)
            irony_scored = result.get("irony_scored_phrases", [])

            # Reintegriamo i punteggi di ironia nei dati principali
            for item in irony_scored:
                pid = item.get("phrase_id")
                for original in batch:  # batch è un subset di all_phrases_with_context
                    if original["phrase_id"] == pid:
                        original["irony_score"] = item.get("irony_score", 0.0)
                        original["irony_explanation"] = item.get("irony_explanation", "")
                        break

        except json.JSONDecodeError as e:
            logger.error(f"Errore di parsing JSON nel batch {batch_index+1}: {e}")

    return all_phrases_with_context


def score_relevance(all_phrases_with_context, theme, batch_size=50):
    """
    Valuta quanto ciascuna frase è rilevante per il tema,
    aggiungendo 'relevance_to_theme' a tutte le frasi.
    """
    phrases_batches = [
        all_phrases_with_context[i : i+batch_size]
        for i in range(0, len(all_phrases_with_context), batch_size)
    ]

    for batch_index, batch in enumerate(phrases_batches):
        # Prepariamo un JSON con le frasi da valutare (semplificate)
        simplified_batch = [
            {"phrase_id": p["phrase_id"], "real_phrase": p["real_phrase"]}
            for p in batch
        ]
        prompt_batch = json.dumps(simplified_batch, ensure_ascii=False)

        prompt = (
            f"Stai valutando quanto queste frasi sono rilevanti per il tema '{theme}'.\n"
            f"Assegna un punteggio 'relevance_to_theme' da 1 a 10 a ciascuna.\n\n"
            f"Restituisci un JSON valido del tipo:\n"
            f'{{"scored_phrases": ['
            f'{{"phrase_id": "...", "relevance_to_theme": 8.5, "justification": "..."}}, ...]}}\n\n'
            f"Ecco il batch {batch_index+1}/{len(phrases_batches)} da valutare:\n"
            f"{prompt_batch}"
        )

        response = AIModelsService().call_llm(prompt, llm_type=LLMType.CHEAP)

        try:
            result = json.loads(response)
            scored_batch = result.get("scored_phrases", [])

            # Reintegriamo punteggi di rilevanza
            for item in scored_batch:
                pid = item.get("phrase_id")
                for original in batch:
                    if original["phrase_id"] == pid:
                        original["relevance_to_theme"] = item.get("relevance_to_theme", 0.0)
                        original["relevance_justification"] = item.get("justification", "")
                        break

        except json.JSONDecodeError as e:
            logger.error(f"Errore di parsing JSON per relevanza batch {batch_index+1}: {e}")

    return all_phrases_with_context


def filter_for_irony_and_relevance(vector_results, irony_level=IRONY_LEVEL, relevance_level=RELEVANCE_LEVEL):
    cached_filtered = load_json_file(FILTERED_FILE)
    if cached_filtered:
        return cached_filtered

    all_phrases_with_context = []
    phrase_id_counter = 0
    phrase_id_to_metadata = {}

    # Collect all phrases from vector store results
    for topic, topic_results in vector_results.items():
        for entry in topic_results:
            if entry["results"]:
                phrase = entry["phrase"]
                for result in entry["results"]:
                    pid = f"phrase_{phrase_id_counter}"
                    phrase_id_counter += 1
                    context_entry = {
                        "phrase_id": pid,
                        "original_query": phrase,
                        "real_phrase": result["page_content"],
                        "previous_sentence": result.get("previous_sentence"),
                        "next_sentence": result.get("next_sentence"),
                        "topic": topic,
                        "metadata": result["metadata"]
                    }
                    all_phrases_with_context.append(context_entry)

    # Evaluate irony and relevance
    all_phrases_with_context = score_irony(all_phrases_with_context, THEME)
    all_phrases_with_context = score_relevance(all_phrases_with_context, THEME)


    # log how many phrases are above irony score
    irony_above = sum(1 for ph in all_phrases_with_context if ph.get("irony_score", 0) >= irony_level)
    total_phrases = len(all_phrases_with_context)
    logger.info(f"Irony threshold analysis: {irony_above}/{total_phrases} phrases ({irony_above/total_phrases:.1%}) "
                f"meet or exceed irony threshold ({irony_level}+)")

    # log how many phrases are above relevance score
    relevance_above = sum(1 for ph in all_phrases_with_context if ph.get("relevance_to_theme", 0) >= relevance_level)
    logger.info(f"Relevance threshold analysis: {relevance_above}/{total_phrases} phrases ({relevance_above/total_phrases:.1%}) "
                f"meet or exceed relevance threshold ({relevance_level}+)")

    # Optional: Log overlap between criteria
    both_above = sum(1 for ph in all_phrases_with_context 
                    if ph.get("irony_score", 0) >= irony_level 
                    and ph.get("relevance_to_theme", 0) >= relevance_level)
    logger.info(f"Phrases meeting both thresholds: {both_above} ({both_above/total_phrases:.1%})")


    # Filter phrases clearly meeting EITHER irony OR relevance thresholds
    filtered_phrases = [
        ph for ph in all_phrases_with_context
        if ph.get("irony_score", 0) >= irony_level or ph.get("relevance_to_theme", 0) >= relevance_level
    ]

    # If total phrases after filtering are fewer than expected, 
    # fill up the list based on combined scores to ensure enough phrases.
    desired_total = TOP_IRONY_LIMIT + TOP_RELEVANCE_LIMIT
    if len(filtered_phrases) < desired_total:
        remaining_needed = desired_total - len(filtered_phrases)
        remaining_phrases = [
            ph for ph in all_phrases_with_context
            if ph not in filtered_phrases
        ]
        remaining_phrases.sort(
            key=lambda x: (x.get("irony_score", 0) + x.get("relevance_to_theme", 0)),
            reverse=True
        )
        filtered_phrases.extend(remaining_phrases[:remaining_needed])

    # Finally, sort and limit to exactly desired_total phrases
    filtered_phrases.sort(key=lambda x: (x.get("irony_score", 0), x.get("relevance_to_theme", 0)), reverse=True)
    final_filtered = filtered_phrases[:desired_total]

    # Save to file
    save_json_file(FILTERED_FILE, final_filtered)

    logger.info(f"Final phrases selected: {len(final_filtered)}")
    return final_filtered

def algorithmic_narrative_batches(filtered_phrases):
    sorted_by_irony = sorted(filtered_phrases, key=lambda x: x.get("irony_score", 0))
    total_phrases = len(sorted_by_irony)

    # Determine batch sizes (flexible percentage-based)
    intro_size = max(3, int(0.1 * total_phrases))
    climax_size = max(3, int(0.15 * total_phrases))
    conclusion_size = max(3, int(0.1 * total_phrases))

    used_phrase_ids = set()

    # Introduction (low irony, high relevance)
    intro_candidates = sorted(sorted_by_irony, key=lambda x: (x.get("irony_score", 0), -x.get("relevance_to_theme", 0)))
    intro_batch = []
    for phrase in intro_candidates:
        if len(intro_batch) >= intro_size:
            break
        if phrase["phrase_id"] not in used_phrase_ids:
            intro_batch.append(phrase)
            used_phrase_ids.add(phrase["phrase_id"])

    # Climax (highest irony)
    climax_candidates = sorted(sorted_by_irony, key=lambda x: x.get("irony_score", 0), reverse=True)
    climax_batch = []
    for phrase in climax_candidates:
        if len(climax_batch) >= climax_size:
            break
        if phrase["phrase_id"] not in used_phrase_ids:
            climax_batch.append(phrase)
            used_phrase_ids.add(phrase["phrase_id"])

    # Conclusion (medium irony around median)
    mid_start = int(total_phrases * 0.4)
    mid_end = int(total_phrases * 0.7)
    conclusion_candidates = sorted_by_irony[mid_start:mid_end]
    conclusion_batch = []
    for phrase in conclusion_candidates:
        if len(conclusion_batch) >= conclusion_size:
            break
        if phrase["phrase_id"] not in used_phrase_ids:
            conclusion_batch.append(phrase)
            used_phrase_ids.add(phrase["phrase_id"])

    # Remaining phrases for build-up
    remaining_phrases = [p for p in sorted_by_irony if p["phrase_id"] not in used_phrase_ids]
    midpoint = len(remaining_phrases) // 2
    build_up_batch_1 = remaining_phrases[:midpoint]
    build_up_batch_2 = remaining_phrases[midpoint:]

    batches = [
        {"name": "introduzione", "phrases": intro_batch},
        {"name": "build_up_1", "phrases": build_up_batch_1},
        {"name": "build_up_2", "phrases": build_up_batch_2},
        {"name": "climax", "phrases": climax_batch},
        {"name": "conclusione", "phrases": conclusion_batch}
    ]

    save_json_file(BATCHES_FILE, batches)
    return batches


def order_batch_phrases(batch, batch_index, total_batches):
    phrases = batch["phrases"]
    batch_name = batch["name"]

    phrase_id_map = {}
    simplified_phrases = []

    for p_idx, phrase in enumerate(phrases):
        phrase_id = phrase.get("phrase_id", f"B{batch_index}_P{p_idx}")
        phrase["phrase_id"] = phrase_id
        phrase_id_map[phrase_id] = phrase

        simplified_phrase = {
            "phrase_id": phrase_id,
            "real_phrase": phrase["real_phrase"],
            "irony_score": phrase.get("irony_score", 0),
            "relevance_to_theme": phrase.get("relevance_to_theme", 0)
        }
        simplified_phrases.append(simplified_phrase)

    phrases_json = json.dumps(simplified_phrases, ensure_ascii=False)

    narrative_purpose = {
        "introduzione": "Introdurre il tema, ironia lieve.",
        "build_up_1": "Costruire tensione ironica graduale.",
        "build_up_2": "Intensificare ironia preparando al climax.",
        "climax": "Massimo livello di ironia e assurdità.",
        "conclusione": "Finale ironico o riflessivo."
    }

    narrative_description = narrative_purpose.get(batch_name, "Sequenza ironica coerente.")

    prompt = (
        f"Stai ordinando la sezione '{batch_name}' per un video SATIRICO sul tema '{THEME}'.\n"
        f"Scopo narrativo della sezione: {narrative_description}\n\n"
        f"Frasi disponibili (TOTALE {len(simplified_phrases)} frasi):\n{phrases_json}\n\n"
        f"Restituisci ESCLUSIVAMENTE un JSON valido con TUTTE E {len(simplified_phrases)} LE FRASI incluse, senza omissioni:\n"
        f'{{"ordered_phrases": [{{"phrase_id": "...", "order": 1, "transition_note": "..."}}, ...]}}'
    )

    response = AIModelsService().call_llm(prompt, llm_type=LLMType.CHEAP)

    ordered_phrases = []

    try:
        ordered_raw = json.loads(response)
        ordered_ids_returned = set()

        for item in ordered_raw.get("ordered_phrases", []):
            phrase_id = item.get("phrase_id")
            order = item.get("order")
            transition_note = item.get("transition_note", "")

            if phrase_id in phrase_id_map:
                original_phrase = phrase_id_map[phrase_id]
                ordered_phrase = original_phrase.copy()
                ordered_phrase["batch_order"] = order
                ordered_phrase["batch_name"] = batch_name
                ordered_phrase["transition_note"] = transition_note
                ordered_phrases.append(ordered_phrase)
                ordered_ids_returned.add(phrase_id)

        # Include any phrases missing from LLM response
        missing_phrase_ids = set(phrase_id_map.keys()) - ordered_ids_returned
        if missing_phrase_ids:
            logger.warning(f"Phrases omitted by LLM in batch '{batch_name}': {missing_phrase_ids}")
            '''next_order = max([p["batch_order"] for p in ordered_phrases], default=0) + 1
            for phrase_id in missing_phrase_ids:
                original_phrase = phrase_id_map[phrase_id]
                ordered_phrases.append({
                    **original_phrase,
                    "batch_order": next_order,
                    "batch_name": batch_name,
                    "transition_note": "Automatic reinclusion due to LLM omission."
                })
                next_order += 1'''

        # Ensure correct sorting
        ordered_phrases.sort(key=lambda x: x["batch_order"])
        return ordered_phrases

    except json.JSONDecodeError as e:
        logger.error(f"Errore JSON batch '{batch_name}': {e}")
        # Explicit fallback ensuring no loss
        ordered_phrases = []
        for i, phrase in enumerate(phrases, start=1):
            phrase["batch_order"] = i
            phrase["batch_name"] = batch_name
            phrase["transition_note"] = "Fallback, ordinamento originale mantenuto."
            ordered_phrases.append(phrase)
        return ordered_phrases

def assemble_final_ordering(ordered_batches):
    """Assembla tutti i batch ordinati in una sequenza finale coerente."""
    all_phrases = []
    global_order = 1
    
    for batch in ordered_batches:
        for phrase in batch:
            phrase["global_order"] = global_order
            global_order += 1
            
            # Assicurati che i metadati esistano
            metadata = phrase.get("metadata")
            if metadata is None:
                logger.warning(f"Metadati mancanti per la frase {phrase.get('phrase_id')}")
                metadata = {}
            
            # Prepara il formato finale richiesto
            final_phrase = {
                "matched_phrase": phrase["real_phrase"],
                "order": phrase["global_order"],
                "batch_name": phrase.get("batch_name", ""),
                "transition_note": phrase.get("transition_note", ""),
                "irony_score": phrase.get("irony_score", 0),
                "tags": phrase.get("tags", []),
                "source": f"{metadata.get('video_id', '')}/{metadata.get('sentence_number', '')}",
                "metadata": metadata,
                "previous_sentence": phrase.get("previous_sentence", ""),
                "next_sentence": phrase.get("next_sentence", "")
            }
            all_phrases.append(final_phrase)
    
    return {"ordered_phrases": all_phrases}

def validate_narrative_flow(ordered_phrases):
    """Verifica e ottimizza il flusso narrativo del video ironico."""
    final_order = ordered_phrases["ordered_phrases"]


    
    # Prima metà e seconda metà (per verificare la distribuzione dell'ironia)
    midpoint = len(final_order) // 2
    first_half = final_order[:midpoint]
    second_half = final_order[midpoint:]
    
    # Calcola il punteggio medio di ironia per ogni metà
    first_half_score = sum(p.get("irony_score", 0) for p in first_half) / max(1, len(first_half))
    second_half_score = sum(p.get("irony_score", 0) for p in second_half) / max(1, len(second_half))
    
    # Se la prima metà è più ironica della seconda, potremmo voler riorganizzare
    if first_half_score > second_half_score + 1.5:
        logger.warning(f"Possibile squilibrio nell'ironia: prima metà ({first_half_score:.1f}) > seconda metà ({second_half_score:.1f})")
        
        # Alternativa: riordina alcune frasi per distribuire meglio l'ironia
        # (implementazione omessa per brevità)
    
    # Verifica la presenza di troppe frasi simili consecutive
    for i in range(1, len(final_order)):
        curr = final_order[i]
        prev = final_order[i-1]
        
        # Se due frasi consecutive hanno gli stessi tag, potrebbero essere troppo simili
        curr_tags = set(curr.get("tags", []))
        prev_tags = set(prev.get("tags", []))
        
        if len(curr_tags.intersection(prev_tags)) > 1 and "#contrasto" not in curr_tags and "#contrasto" not in prev_tags:
            # Aggiungi una nota per il montatore
            curr["montaggio_note"] = "Considerare l'inserimento di una pausa o transizione visiva qui per evitare monotonia"
    

    unique_phrases = {}
    for p in ordered_phrases["ordered_phrases"]:
        key = p["metadata"].get("doc_id") or p["matched_phrase"]
        if key not in unique_phrases:
            unique_phrases[key] = p
    ordered_phrases["ordered_phrases"] = list(unique_phrases.values())
    return ordered_phrases

def main():
    # Fase 1: Ottenere o generare i topics
    topics = get_topics(THEME)
    logger.info(f"Topics ottenuti: {len(topics)}")

    # Fase 2: Ottenere o generare le frasi per ciascun topic
    phrases_link_topics = {topic: get_phrases_for_topic(topic) for topic in topics}
    total_phrases = sum(len(phrases) for phrases in phrases_link_topics.values())
    logger.info(f"Frasi generate: {total_phrases}")

    # Fase 3: Inizializzazione del Vector Store (ChromaDB)
    ai_service = AIModelsService()
    embedding_model = ai_service.get_embedding_model()
    chroma_db = Chroma(
        collection_name=VECTOR_STORE_SETTINGS["collection_name"],
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embedding_model
    )

    # Fase 4: Ricerca vettoriale per trovare frasi reali
    vector_results = load_json_file(VECTORS_FILE)
    seen_doc_ids = set()

    if vector_results is not None:
        for topic, entries in vector_results.items():
            for entry in entries:
                for result in entry["results"]:
                    doc_id = result["metadata"].get("doc_id")
                    if doc_id:
                        seen_doc_ids.add(doc_id)
    else:
        vector_results = {}

        for topic, phrases in phrases_link_topics.items():
            topic_results = []

            for phrase in phrases:
                
                results = search_vector_store(
                    query=phrase,
                    vector_store=chroma_db,
                    k=8,
                    min_similarity=MIN_SIMILARITY_THRESHOLD,
                    excluded_doc_ids=seen_doc_ids  # Escludi doc_id già visti
                )

                if results:
                    # Aggiorna doc_id visti
                    for res in results:
                        doc_id = res["metadata"].get("doc_id")
                        if doc_id:
                            seen_doc_ids.add(doc_id)

                    topic_results.append({
                        "phrase": phrase,
                        "results": results
                    })
                    logger.info(f"Aggiunte  {len(results)} sentences per frase {phrase}'")

            if topic_results:
                vector_results[topic] = topic_results

        save_json_file(VECTORS_FILE, vector_results)
        logger.info(f"Risultati della ricerca vettoriale salvati in '{VECTORS_FILE}'")
    
    # Fase 5: Filtraggio per potenziale ironico
    filtered_phrases = filter_for_irony_and_relevance(vector_results, IRONY_LEVEL)
    logger.info(f"Frasi filtrate per ironia: {len(filtered_phrases)}")
    
    # REPLACE existing tagging and batching steps with this:
    # Fase 6 (new): Algorithmically creating narrative batches
    phrase_batches = algorithmic_narrative_batches(filtered_phrases)
    logger.info(f"Narrative batches created: {len(phrase_batches)}")

    # Fase 7 (unchanged): Ordering phrases within batches
    ordered_batches = []
    for i, batch in enumerate(phrase_batches):
        logger.info(f"Ordering batch {i+1}/{len(phrase_batches)}: {batch['name']} ({len(batch['phrases'])} phrases)")
        ordered_batch = order_batch_phrases(batch, i, len(phrase_batches))
        ordered_batches.append(ordered_batch)
    
    # Fase 9: Assemblaggio dell'ordine finale
    final_ordering = assemble_final_ordering(ordered_batches)
    logger.info(f"Frasi ordinate: {len(final_ordering['ordered_phrases'])}")
    
    # Fase 10: Validazione e ottimizzazione del flusso narrativo
    validated_ordering = validate_narrative_flow(final_ordering)
    
    # Salvataggio del risultato finale
    save_json_file(ORDERED_FILE, validated_ordering)
    logger.info(f"Script video ironico ottimizzato e salvato in '{ORDERED_FILE}'")

if __name__ == "__main__":
    main()