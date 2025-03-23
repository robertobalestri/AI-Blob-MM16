from langchain.schema import Document
from langchain_chroma import Chroma
import json
import logging
from src.ai_models import AIModelsService
from src.config.settings import VECTOR_STORE_DIR, VECTOR_STORE_SETTINGS, LOG_LEVEL

LOG_LEVEL = logging.INFO
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def verifica_duplicati():
    try:
        ai_service = AIModelsService()
        # Inizializza l'embedding model (assicurati che sia lo stesso usato per creare il DB)
        embedding_model = ai_service.get_embedding_model()

        # Inizializza ChromaDB
        chroma_db = Chroma(
            collection_name=VECTOR_STORE_SETTINGS["collection_name"],
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embedding_model
        )

        # Ottieni tutti gli id della collezione
        all_ids = chroma_db.get()['ids']

        # Ottieni tutti i documenti con i loro metadati
        all_data = chroma_db.get(ids=all_ids, include=['metadatas', 'documents'])

        seen_combinations = set()
        duplicates = []
        duplicated_combinations = []
        unique_video_ids = set()

        for i in range(len(all_data['ids'])):
            sentence = all_data['documents'][i]
            video_id = all_data['metadatas'][i].get('video_id')
            doc_id = all_data['metadatas'][i].get('doc_id')
            sentence_number = all_data['metadatas'][i].get('sentence_number')
            current_id = all_data['ids'][i]

            if video_id is not None:
                unique_video_ids.add(video_id)  # Aggiunge il video_id all'insieme dei video unici
                combination = (video_id, doc_id)
                if combination in seen_combinations:
                    duplicates.append({
                        "id": current_id,
                        "sentence": sentence,
                        "video_id": video_id,
                        "sentence_number": sentence_number,
                        "doc_id": doc_id
                    })
                    duplicated_combinations.append(combination)
                else:
                    seen_combinations.add(combination)

        total_unique_video_ids = len(unique_video_ids)

        if duplicates:
            print("Trovati i seguenti duplicati (sentence, video_id):")
            for i in range(len(duplicated_combinations)):
                duplicate = duplicates[i]
                combination = duplicated_combinations[i]
                print(f"* ID: {duplicate['id']}, Sentence: '{duplicate['sentence']}', Video ID: {duplicate['video_id']}")
                print(f"Combination: '{combination[0]}', {combination[1]}")

            print("Found duplicates: ", len(duplicates), " ", len(duplicated_combinations), 
                  " su un totale di ", len(all_ids), 
                  " | Numero totale di video_id unici: ", total_unique_video_ids)
        else:
            print("Nessuna riga duplicata trovata per sentence e video_id su un totale di ", len(all_ids),
                  " | Numero totale di video_id unici: ", total_unique_video_ids)

    except Exception as e:
        logger.error(f"Errore durante la verifica dei duplicati in ChromaDB: {e}")

if __name__ == "__main__":
    verifica_duplicati()
