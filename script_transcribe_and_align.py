import json
import os
import time
import logging
import nltk
import re
import torch
import whisperx
from nltk.tokenize import sent_tokenize
from rapidfuzz import fuzz

# ▼▼▼ Custom modules from your code ▼▼▼
from src.config.settings import AUDIO_DIR, MODEL_SETTINGS
from src.ai_models import AIModelsService, LLMType
from punctuators.models import PunctCapSegModelONNX

import asyncio

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nltk.download("punkt")

# Key parameters
DATASET_PATH = "data/dataset_pulito.jsonl"
OUTPUT_PATH = "data/matched_sentences.jsonl"
PROCESSED_VIDEOS_PATH = "data/processed_videos.json"

TRANSCRIPTS_SEGMENTS_DIR = "cache/transcripts_segments/"
SENTENCE_CON_ERRORE_PATH = "data/sentences_con_errore.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

# Initialize models at global scope
whisper_model = None
align_model = None
align_metadata = None
punctuation_model = None

def initialize_models():
    global whisper_model, align_model, align_metadata, punctuation_model
    logger.info(f"Initializing models on device: {device}")

    # Load WhisperX model
    whisper_model = whisperx.load_model(
        MODEL_SETTINGS["whisper_model"],
        device,
        compute_type=compute_type,
        language="it"
    )
    
    # Load alignment model
    align_model, align_metadata = whisperx.load_align_model(
        language_code="it",
        device=device
    )
    
    # Load punctuation model
    punctuation_model = PunctCapSegModelONNX.from_pretrained(
        "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
    )

def iter_videos(dataset_path: str):
    """Generator yielding videos one by one from JSON Lines file"""
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON line: {e}")
                continue

def load_processed_videos():
    """Load list of already processed videos"""
    if not os.path.exists(PROCESSED_VIDEOS_PATH):
        return set()
    
    try:
        with open(PROCESSED_VIDEOS_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception as e:
        logger.warning(f"Couldn't load {PROCESSED_VIDEOS_PATH}: {e}")
        return set()

def save_processed_video(video_id: str):
    """Save processed video ID to tracking file"""
    processed = load_processed_videos()
    processed.add(video_id)
    
    try:
        with open(PROCESSED_VIDEOS_PATH, "w", encoding="utf-8") as f:
            json.dump(list(processed), f, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving processed videos: {e}")

def save_matched_sentences(video_id: str, video_data: dict):
    """Append results for a single video to JSON Lines output"""
    output_line = json.dumps(
        {"video_id": video_id, **video_data},
        ensure_ascii=False
    )
    
    try:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(output_line + "\n")
    except Exception as e:
        logger.error(f"Error saving results for {video_id}: {e}")

def normalize_text(text: str) -> str:
    """
    Normalizes the text according to specified rules.
    """
    text = text.lower()
    
    # Remove dots in acronyms (e.g., s.p.a. → spa)
    text = re.sub(r'\b([a-z])(?:\.([a-z]))+\b', lambda m: m.group(0).replace('.', ''), text)
    
    # Replace commas/dots NOT between numbers with a space
    text = re.sub(r'(?<!\d)[.,]|[.,](?!\d)', ' ', text)
    
    # Remove trailing commas/dots at the end of the text
    text = re.sub(r'[.,]+$', '', text)
    
    # Remove unwanted punctuation (keep ` and apostrophes)
    text = re.sub(r"[^\w\s`',.]", " ", text)
    
    # Collapse multiple spaces to one
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_compound_words(word_list):
    """
    Identifica parole concatenate con trattini o punti e le divide in parole separate,
    mantenendo i timestamp originali divisi proporzionalmente.
    """
    new_word_list = []
    new_index = 0  # Nuovo indice per le parole divise
    for word_data in word_list:
        w = word_data["word"]
        if "-" in w or "." in w:
            sub_words = re.split(r"[-.]" , w)
            num_sub_words = len(sub_words)
            duration = word_data["end"] - word_data["start"]
            sub_duration = duration / num_sub_words if num_sub_words > 0 else duration
            start_t = word_data["start"]
            for sub_word in sub_words:
                new_word_list.append({
                    "word": sub_word.strip(),
                    "start": start_t,
                    "end": start_t + sub_duration,
                    "score": word_data["score"],
                    "index": new_index,  # Assegniamo un nuovo indice
                    "original_index": word_data.get("index")  # Manteniamo traccia dell'indice originale
                })
                start_t += sub_duration
                new_index += 1
        else:
            # Manteniamo l'indice originale ma aggiorniamo anche il nuovo indice
            word_data["original_index"] = word_data.get("index")
            word_data["index"] = new_index
            new_word_list.append(word_data)
            new_index += 1
    return new_word_list

def download_audio(url: str, output_path: str) -> bool:
    """
    Scarica il file audio dal URL specificato usando yt_dlp.
    Ritorna True se il download ha avuto successo, False altrimenti.
    """
    if yt_dlp is None:
        logger.error("Il modulo yt_dlp non è installato. Impossibile scaricare l'audio.")
        return False

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        logger.info(f"Audio scaricato e salvato in: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Errore durante il download dell'audio: {e}")
        return False

def get_aligned_words(video: dict) -> list:
    """
    Gestisce la trascrizione e l'allineamento dell'audio di un video,
    restituendo una lista di parole con i relativi timestamp.
    """
    video_id = video["video_id"]
    url = video["url"]

    audio_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")
    transcript_path = os.path.join(TRANSCRIPTS_SEGMENTS_DIR, f"{video_id}.json")

    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPTS_SEGMENTS_DIR, exist_ok=True)

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        logger.info(f"Audio non trovato per il video {video_id}. Scaricamento in corso...")
        if not download_audio(url, audio_path):
            logger.error(f"Errore nel download dell'audio per il video {video_id}.")
            return []

    # Caricamento o creazione dell'allineamento
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        if "alignment" in saved_data:
            aligned_result = saved_data["alignment"]
        else:
            audio = whisperx.load_audio(audio_path)
            aligned_result = whisperx.align(
                saved_data["transcription"]["segments"],
                align_model,
                align_metadata,
                audio,
                device
            )
            saved_data["alignment"] = aligned_result
            with open(transcript_path, "w", encoding="utf-8") as fw:
                json.dump(saved_data, fw, ensure_ascii=False, indent=4)
    else:
        audio = whisperx.load_audio(audio_path)
        transcription_result = whisper_model.transcribe(audio, batch_size=16)
        aligned_result = whisperx.align(
            transcription_result["segments"],
            align_model,
            align_metadata,
            audio,
            device
        )
        saved_data = {
            "transcription": transcription_result,
            "alignment": aligned_result
        }
        with open(transcript_path, "w", encoding="utf-8") as fw:
            json.dump(saved_data, fw, ensure_ascii=False, indent=4)

    
    if 'audio' in locals():
        del audio

    # Estraiamo l'elenco di parole
    words = []
    word_index = 0  # Indice globale per le parole
    for segment in aligned_result.get("segments", []):
        seg_words = segment.get("words", [])
        for w in seg_words:
            words.append({
                "word": w.get("word", "").strip(),
                "start": float(w.get("start", -1)),
                "end": float(w.get("end", -1)),
                "score": float(w.get("score", 0)),
                "index": word_index  # Aggiungiamo l'indice della parola
            })
            word_index += 1

    # Segmentazione parole col trattino/punto
    words = split_compound_words(words)

    if 'aligned_result' in locals():
        del aligned_result

    return words

def call_llm_for_correction(sentence: str, words: list) -> list:
    """
    Chiede a un Large Language Model di fornire una correzione o
    segmentazione coerente tra la frase `sentence` e la lista di parole `words`.
    
    Parametri
    ---------
    sentence : str
        Frase target (punteggiata) generata dal modello di punteggiatura.
    words : list of dict
        Lista di dizionari, ciascuno con:
            {
              "word": <string>,
              "start": <float>,
              "end": <float>,
              "index": <int>,
              "original_index": <int> (opzionale)
            }
    
    Ritorna
    -------
    corrected_words : list of dict
        Struttura simile a `words`, ma con eventuale splitting o merging
        che renda coerente la somma delle "word" con la frase `sentence`.
        Il LLM deve restituire i timestamp rivisitati in caso di split
        (proporzionalmente, o con la logica desiderata).
    """

    # Esempio di prompt (molto semplificato). Devi personalizzarlo
    # in modo da istruire il modello su come *deve* restituire il risultato
    # (ad es. in JSON con un campo "aligned_words" o simile).
    prompt = f"""
Sei un assistente specializzato in correzione di testo e timestamp.
Abbiamo una frase target (punteggiata): 
    "{sentence}"

Abbiamo un elenco di parole con start/end time e indici:
    {json.dumps(words, ensure_ascii=False, indent=2)}

Alcune parole potrebbero dover essere divise o unite per eguagliare esattamente la frase target.

Restituisci un JSON che contiene una lista di elementi:
  [
    {{
      "word": <string>,
      "start": <float>,
      "end": <float>,
      "index": <int>,
      "original_index": <int> (se disponibile)
    }},
    ...
  ]

dove la concatenazione delle parole corrisponda esattamente al testo della frase target (spazi esclusi o trattati correttamente). 
Se una parola deve essere spezzata, ripartisci i timestamp in modo proporzionale e restituisci le nuove parole con i relativi timestamp.
Stessa cosa, se due parole devono essere unite, fai la fusione e aggiorna i timestamp di conseguenza.
È FONDAMENTALE che tu mantenga gli indici corretti per ogni parola, in modo che possiamo tracciare quali parole sono state utilizzate.
Non aggiungere altro output al di fuori del JSON!
Se non riesci a correggere la frase, restituisci una lista vuota.
    """
    
    max_retries = 3
    retry_count = 0

    llm_type = LLMType.CHEAP
    
    while retry_count < max_retries:
        try:
            # Chiamata al LLM
            response = AIModelsService().call_llm(prompt, llm_type)

            # Parsing della risposta
            corrected_data = json.loads(response.strip())
            
            if corrected_data and len(corrected_data) > 0:
                logger.info(f"✅ LLM per la frase '{sentence}' ha restituito: {corrected_data}")
                return corrected_data
            else:
                logger.warning(f"⚠️ Risposta LLM malformata (tentativo {retry_count + 1}/{max_retries}): {response}")
                retry_count += 1
                llm_type = LLMType.INTELLIGENT
                if retry_count >= max_retries:
                    logger.error(f"⚠️ Tutti i tentativi falliti per la frase: '{sentence}'")
                    return {}
        
        except json.JSONDecodeError as e:
            retry_count += 1
            logger.warning(f"❌ Errore nel parsing JSON (tentativo {retry_count}/{max_retries}): {e}")
            if retry_count >= max_retries:
                logger.error(f"❌ Tutti i tentativi di parsing JSON falliti per la frase: '{sentence}'")
                return {}
            
        
        except Exception as e:
            logger.error(f"❌ Errore nella chiamata al LLM: {e}")
            return {}
            
    # Non dovremmo mai arrivare qui, ma per sicurezza
    return {}

# -----------------------------------------
# FUNZIONE PER RISEGMENTARE LE FRASI E MANTENERE I TIMING
# -----------------------------------------
def resegment_with_punctuation(words, punctuation_model) -> list:
    text_rebuilt = " ".join(w["word"] for w in words if w["word"]).strip()
    if not text_rebuilt:
        return []

    # Get normalized version for matching
    normalized_text = normalize_text(text_rebuilt)
    sentences_punctuated = punctuation_model.infer([normalized_text], apply_sbd=True)[0]
    
    results = []
    word_ptr = 0  # Pointer to track position in original words list
    total_words = len(words)
    last_used_index = -1  # Track the last index used to ensure sequential alignment

    for sentence_idx, sentence in enumerate(sentences_punctuated):
        
        with_error = False
        
        normalized_sentence = normalize_text(sentence)
        
        accum = []
        accum_normalized = ""
        start_time = None
        end_time = None
        
        # Ensure word_ptr is at least last_used_index + 1
        word_ptr = max(word_ptr, last_used_index + 1)

        if word_ptr >= total_words:
            break  # Stop if we've used all words
        
        initial_word_ptr = word_ptr  # Remember where we started for this sentence

        previous_word = words[initial_word_ptr - 1] if initial_word_ptr > 0 else None
        previous_normalized = normalize_text(previous_word["word"]) if previous_word else None

        while word_ptr < total_words:
            current_word = words[word_ptr]
            current_normalized = normalize_text(current_word["word"])
            

            # Calculate potential new accumulated text
            potential_text = accum_normalized + " " + current_normalized if accum_normalized else current_normalized
            potential_text_with_previous_word = previous_normalized + " " + potential_text if previous_normalized else None
            potential_text_without_first_word = " ".join(accum_normalized.split()[1:] + [current_normalized]) if accum_normalized else None



            potential_text = normalize_text(potential_text)
            potential_text_with_previous_word = normalize_text(potential_text_with_previous_word) if potential_text_with_previous_word else None
            potential_text_without_first_word = normalize_text(potential_text_without_first_word) if potential_text_without_first_word else None

            # Check similarity with target sentence
            similarity = fuzz.ratio(potential_text, normalized_sentence)


            similarity_con_parola_prima = None
            similarity_senza_prima_parola = None

            if potential_text_with_previous_word is not None:
                similarity_con_parola_prima = fuzz.ratio(potential_text_with_previous_word, normalized_sentence)
            if potential_text_without_first_word is not None:
                similarity_senza_prima_parola = fuzz.ratio(potential_text_without_first_word, normalized_sentence)

            LOOKAHEAD = 3

            if similarity > 98:  # High confidence match
                accum.append(current_word)
                if similarity_con_parola_prima is not None and similarity_con_parola_prima > similarity:
                    start_time = previous_word["start"] 
                elif similarity_senza_prima_parola is not None and similarity_senza_prima_parola > similarity:
                    start_time = accum[1]["start"]
                else:
                    start_time = accum[0]["start"]

                end_time = current_word["end"]
                last_used_index = word_ptr  # Update last used index
                word_ptr += 1
                break
            elif similarity_con_parola_prima is not None and similarity_con_parola_prima > 98:
                accum.append(current_word)
                start_time = previous_word["start"] 
                end_time = current_word["end"]
                last_used_index = word_ptr  # Update last used index
                word_ptr += 1
                break
            elif similarity_senza_prima_parola is not None and similarity_senza_prima_parola > 98:
                accum.append(current_word)
                start_time = accum[1]["start"]
                end_time = current_word["end"]
                last_used_index = word_ptr  # Update last used index
                word_ptr += 1
                break

            elif len(normalize_text(potential_text)) <= len(normalized_sentence) + LOOKAHEAD:
                accum.append(current_word)
                accum_normalized = potential_text
                word_ptr += 1
            else:
                # Prepare context for LLM correction
                context_words = []

                # Add up to 5 words before current sentence
                before_context_start = max(0, initial_word_ptr - 5)
                context_words.extend(words[before_context_start:initial_word_ptr])

                # Add accumulated words
                context_words.extend(accum)

                # Add up to 5 words after the current position
                after_context_end = min(total_words, word_ptr + 5)
                context_words.extend(words[word_ptr:after_context_end])

                # Call LLM for correction
                correction = call_llm_for_correction(normalized_sentence, context_words)

                if correction:
                    # Update accum with corrected words
                    accum = correction
                    start_time = correction[0]["start"]
                    end_time = correction[-1]["end"]

                    # Determine last used index from LLM correction
                    last_used_index = max((w["index"] for w in correction if "index" in w), default=last_used_index)

                    word_ptr = last_used_index + 1  # Move to the next word after correction
                    break
                else:
                    # Fallback: use current accumulation
                    start_time = accum[0]["start"] if accum else 0
                    end_time = accum[-1]["end"] if accum else 0
                    #last_used_index = max(last_used_index, word_ptr - 1) 
                    last_used_index = word_ptr - 1 - LOOKAHEAD
                    with_error = True
                    break

        if accum and with_error == False:
            results.append({
                "sentence": sentence,
                "sentence_number": sentence_idx + 1,  # 1-based indexing
                "start_time": start_time,
                "end_time": end_time,
                "words": accum
            })
        elif with_error == True:
            results.append({
                "sentence": sentence,
                "sentence_number": sentence_idx + 1,  # 1-based indexing
                "start_time": start_time,
                "end_time": end_time,
                "words": accum,
                "with_error": with_error
            })
        else:
            pass

    return results




async def process_video(video: dict):
    """Process a single video and return its results"""

    # Add this at start of processing
    torch.backends.cudnn.benchmark = True  # Better GPU memory utilization

    video_id = video["video_id"]
    logger.info(f"Processing video {video_id}")
    
    try:
        # Get aligned words
        words = get_aligned_words(video)
        if not words:
            logger.warning(f"No words aligned for {video_id}")
            return None

        # Segment sentences
        sentences_info = resegment_with_punctuation(words, punctuation_model)
        if not sentences_info:
            return None

        # Build result structure
        return {
            "url": video.get("url"),
            "genre": video.get("genre"),
            "channel_name": video.get("channel_name"),
            "sentences": [{
                "sentence_number": s["sentence_number"],
                "sentence": s["sentence"],
                "start_time": s["start_time"],
                "end_time": s["end_time"],
                "duration": s["end_time"] - s["start_time"],
                "words": s["words"]
            } for s in sentences_info]
        }
        
    except Exception as e:
        logger.error(f"Error processing {video_id}: {e}")
        return None

    finally:
        # Expanded cleanup
        variables_to_delete = ['words', 'sentences_info', 'audio', 'transcription_result', 'aligned_result']
        for var in variables_to_delete:
            if var in locals():
                del var
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Additional NVIDIA IPC cleanup
        import gc
        gc.collect()

async def main():
    initialize_models()
    processed_videos = load_processed_videos()
    
    # Create empty output file if needed
    if not os.path.exists(OUTPUT_PATH):
        open(OUTPUT_PATH, "w").close()

    # Process videos in streaming fashion
    total_processed = 0
    start_time = time.time()
    
    for video_idx, video in enumerate(iter_videos(DATASET_PATH), start=1):
        video_id = video.get("video_id")
        if not video_id:
            logger.warning(f"Missing video ID in record {video_idx}")
            continue
            
        if video_id in processed_videos:
            continue

        # Process video
        video_data = await process_video(video)
        if not video_data:
            continue

        # Save results
        save_matched_sentences(video_id, video_data)
        save_processed_video(video_id)
        total_processed += 1
        
        # Periodic status update
        if video_idx % 10 == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Processed {video_idx} videos ({total_processed} new) "
                f"at {elapsed:.2f}s total runtime"
            )

    logger.info(f"Processing complete. Total videos processed: {total_processed}")

if __name__ == "__main__":
    asyncio.run(main())