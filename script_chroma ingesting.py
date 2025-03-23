import json
import logging
from uuid import uuid4

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.ai_models import AIModelsService
from src.config.settings import VECTOR_STORE_DIR, VECTOR_STORE_SETTINGS, LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)

# Percorso al file JSON intermedio
MERGED_SENTENCES_PATH = "data/merged_sentences.jsonl"
BATCH_SIZE = 50  # Numero di documenti da processare per batch

def main():
    # Inizializza il servizio AI per gli embedding
    ai_service = AIModelsService()
    embedding_model = ai_service.get_embedding_model()

    # Inizializza ChromaDB con parametri aggiuntivi per HNSW
    chroma_db = Chroma(
        collection_name=VECTOR_STORE_SETTINGS["collection_name"],
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embedding_model,
        collection_metadata={
            "hnsw:space": "cosine",
            "hnsw:search_ef": 200,
            "hnsw:M": 30
        }
    )

    documents = []
    ids = []
    processed_count = 0

    # Set per tenere traccia dei doc_id già elaborati durante l'esecuzione corrente
    seen_doc_ids = set()

    with open(MERGED_SENTENCES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())

                # Estrazione dei dati necessari
                sentence = entry["sentence"]
                video_id = entry["video_id"]
                sentence_number = entry.get("sentence_number", None)
                genre = entry.get("genre", "unknown")
                url = entry.get("url", "")
                channel_name = entry.get("channel_name", "unknown")
                start_time = entry.get("start_time", None)
                end_time = entry.get("end_time", None)

                if sentence_number is None:
                    logger.warning(f"Sentence number mancante per video {video_id}, salto il documento.")
                    continue

                # Creazione dell'ID univoco per il documento utilizzando video_id e sentence_number
                doc_id = f"{video_id}_{sentence_number}"

                # Verifica se il doc_id è già stato elaborato nel batch corrente
                if doc_id in seen_doc_ids:
                    logger.info(f"Documento duplicato nel batch corrente, saltato: doc_id={doc_id}")
                    continue

                # Verifica se il documento esiste già in ChromaDB tramite il doc_id nei metadata
                existing_docs = chroma_db.get(ids=[doc_id])
                if existing_docs and len(existing_docs["ids"]) > 0:
                    logger.info(f"Documento già presente in ChromaDB, saltato: doc_id={doc_id}")
                    seen_doc_ids.add(doc_id)
                    continue

                # Creazione del documento includendo il doc_id nei metadata
                document = Document(
                    page_content=sentence,
                    metadata={
                        "doc_id": doc_id,
                        "video_id": video_id,
                        "sentence_number": sentence_number,
                        "genre": genre,
                        "url": url,
                        "channel_name": channel_name,
                        "start_time": start_time,
                        "end_time": end_time,
                        "words": json.dumps(entry.get("words", []), ensure_ascii=False)
                    }
                )
                documents.append(document)
                ids.append(doc_id)
                seen_doc_ids.add(doc_id)

                # Batch processing: inserisce i documenti in Chroma ogni BATCH_SIZE elementi
                if len(documents) >= BATCH_SIZE:
                    logger.info(f"Inserimento batch di {len(documents)} documenti (ultimo doc_id={doc_id}) in ChromaDB...")
                    chroma_db.add_documents(documents=documents, ids=ids)
                    processed_count += len(documents)
                    logger.info(f"Inseriti {processed_count} documenti in totale in ChromaDB...")
                    documents = []
                    ids = []

            except json.JSONDecodeError as e:
                logger.error(f"Errore di decodifica JSON: {e} in riga: {line.strip()}")
            except KeyError as e:
                logger.error(f"Chiave mancante nel JSON: {e} in riga: {line.strip()}")

    # Inserimento degli eventuali documenti residui
    if documents:
        chroma_db.add_documents(documents=documents, ids=ids)
        processed_count += len(documents)
        logger.info(f"Inseriti {processed_count} documenti in totale in ChromaDB.")

    logger.info("Inserimento completato. I documenti sono stati salvati in ChromaDB.")

if __name__ == "__main__":
    main()
