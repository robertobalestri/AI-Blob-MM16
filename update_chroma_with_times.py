import json
import logging
import chromadb
from src.config.settings import VECTOR_STORE_DIR, VECTOR_STORE_SETTINGS, LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

# Percorso al file JSON intermedio
MERGED_SENTENCES_PATH = "data/merged_sentences.jsonl"

def update_metadata():
    # Inizializza ChromaDB
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_or_create_collection(name=VECTOR_STORE_SETTINGS["collection_name"])

    updated_count = 0

    with open(MERGED_SENTENCES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())

                # Il nostro doc_id è anche l'ID effettivo nel vector store
                doc_id = f"{entry['video_id']}_{entry.get('sentence_number')}"

                # Recupera il documento esistente da ChromaDB
                existing_doc = collection.get(ids=[doc_id], include=["documents", "embeddings", "metadatas"])

                # Se il documento non è trovato, salta
                if not existing_doc or len(existing_doc["ids"]) == 0:
                    logger.warning(f"Documento non trovato in ChromaDB: doc_id={doc_id}")
                    continue

                # Recupera il contenuto originale e i metadati
                original_text = existing_doc["documents"][0]  # Testo originale
                current_metadata = existing_doc["metadatas"][0]  # Metadati esistenti
                original_embedding = existing_doc["embeddings"][0]  # Embedding originale

                # Aggiorna i metadati con start_time, end_time e duration
                start_time = entry.get("start_time")
                end_time = entry.get("end_time")
                duration = entry.get("duration")
                sentence_number = entry.get("sentence_number")

                current_metadata["start_time"] = start_time
                current_metadata["end_time"] = end_time
                current_metadata["duration"] = duration
                current_metadata["sentence_number"] = sentence_number

                # Aggiorniamo il documento in ChromaDB con il metodo update()
                collection.update(
                    ids=[doc_id],
                    embeddings=[original_embedding],  # Manteniamo lo stesso embedding
                    metadatas=[current_metadata],  # Metadati aggiornati
                    documents=[original_text]  # Testo originale
                )

                # **Verifica che l'aggiornamento sia stato applicato correttamente**
                #updated_doc = collection.get(ids=[doc_id])

                # Estrai i nuovi metadati per la verifica
                #new_metadata = updated_doc["metadatas"][0]
                #new_metadata.pop("words", None)
                #logger.info(f"Documento aggiornato con successo: {doc_id}")

                #logger.info(f"Nuovi metadati: {new_metadata}")

                updated_count += 1
                if updated_count % 50 == 0:
                    logger.info(f"Aggiornati {updated_count} documenti finora...")

            except json.JSONDecodeError as e:
                logger.error(f"Errore di decodifica JSON: {e} in riga: {line.strip()}")
            except KeyError as e:
                logger.error(f"Chiave mancante nel JSON: {e} in riga: {line.strip()}")

    logger.info(f"Aggiornamento completato. {updated_count} documenti aggiornati in ChromaDB.")

if __name__ == "__main__":
    update_metadata()
