import logging
import chromadb
from src.config.settings import VECTOR_STORE_DIR, VECTOR_STORE_SETTINGS, LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

# Definiamo i metadati obbligatori da verificare
REQUIRED_METADATA = ["start_time", "end_time", "duration", "sentence_number"]

def check_metadata():
    # Inizializza ChromaDB
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    collection = client.get_or_create_collection(name=VECTOR_STORE_SETTINGS["collection_name"])

    # Recupera tutti i documenti dal database
    all_docs = collection.get(include=["metadatas"])

    total_docs = len(all_docs["ids"])
    missing_metadata_count = 0
    missing_details = []

    for idx, doc_id in enumerate(all_docs["ids"]):
        metadata = all_docs["metadatas"][idx]

        # Controlla se tutti i campi obbligatori sono presenti
        missing_fields = [field for field in REQUIRED_METADATA if field not in metadata]

        if missing_fields:
            missing_metadata_count += 1
            missing_details.append({"doc_id": doc_id, "missing_fields": missing_fields})

            logger.warning(f"Documento {doc_id} mancante di {missing_fields}")

    logger.info(f"Totale documenti controllati: {total_docs}")
    logger.info(f"Documenti con metadati mancanti: {missing_metadata_count}")

    # Se ci sono documenti con metadati mancanti, stampiamo i dettagli
    if missing_metadata_count > 0:
        logger.info("Dettagli dei documenti con metadati mancanti:")
        for entry in missing_details[:10]:  # Mostriamo solo i primi 10 per non sovraccaricare i log
            logger.info(entry)

    logger.info("Verifica completata.")

if __name__ == "__main__":
    check_metadata()
