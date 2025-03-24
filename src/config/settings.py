"""Application settings and configuration."""
from typing import Dict, Any
from pathlib import Path
#from . import environment as env
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path, override=True)

# Base Paths
BASE_DIR = Path(__file__).parent.parent.parent
CACHE_DIR = BASE_DIR / "cache"
AUDIO_DIR = CACHE_DIR / "audio"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
FONT_FILE = BASE_DIR / "src" / "config" / "fonts" / "Times New Roman.ttf"

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


#Mode
DATASET_TO_BE_PROCESSED_BOOL = os.getenv("DATASET_TO_BE_PROCESSED_BOOL", "false").lower() in ("1", "true", "yes")



# Model Settings
MODEL_SETTINGS: Dict[str, Any] = {
    "whisper_model": os.getenv("WHISPER_MODEL"),
    "device": os.getenv("DEVICE"),
    "compute_type": os.getenv("COMPUTE_TYPE"),
    "batch_size": os.getenv("BATCH_SIZE"),
    "num_workers": os.getenv("NUM_WORKERS")
}

# Video processing settings
VIDEO_SETTINGS = {
    "data_video_dir": str(BASE_DIR / "data" / "videos"),
    "max_concurrent_downloads": 2,
    "fade_audio_duration": 0.3,
    "target_width": 1280,
    "target_height": 720,
    "video_bitrate": "2000k",
    "audio_bitrate": "192k",
}

# Vector Store Settings
VECTOR_STORE_SETTINGS: Dict[str, Any] = {
    "collection_name": os.getenv('VECTOR_STORE_COLLECTION', 'video_segments'),
    "persist_directory": str(BASE_DIR / "data" / "vector_store"),
    "cohere_api_key": os.getenv("AZURE_COHERE_EMBEDDING_API_KEY"),
    "cohere_api_endpoint": os.getenv("AZURE_COHERE_EMBEDDING_API_ENDPOINT"),
    "cohere_api_embedding_model": os.getenv('AZURE_COHERE_EMBEDDING_MODEL_NAME')
}

# Azure OpenAI Settings
AZURE_OPENAI_SETTINGS: Dict[str, Any] = {
    "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    "api_endpoint": os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
    "intelligent_deployment": os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME_INTELLIGENT"),
    "intelligent_model": os.getenv("AZURE_OPENAI_LLM_MODEL_NAME_INTELLIGENT"),
    "cheap_deployment": os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT_NAME_CHEAP"),
    "cheap_model": os.getenv("AZURE_OPENAI_LLM_MODEL_NAME_CHEAP")
}

# Logging Settings
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": LOG_LEVEL
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(BASE_DIR / "logs" / "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
            "level": LOG_LEVEL
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": LOG_LEVEL
    }
}

# Ensure all required directories exist
for setting_dict in [VIDEO_SETTINGS]:
    for key, value in setting_dict.items():
        if isinstance(value, str) and key.endswith('_dir'):
            Path(value).mkdir(parents=True, exist_ok=True)

# New GPU settings
GPU_CLEANUP_AGGRESSIVE: bool = os.getenv("GPU_CLEANUP_AGGRESSIVE", "false").lower() in ("1", "true", "yes") 



THEME =  os.getenv("THEME")
SEED = os.getenv("SEED")
TOP_IRONY_LIMIT = int(os.getenv("TOP_IRONY_LIMIT"))
TOP_RELEVANCE_LIMIT = int(os.getenv("TOP_RELEVANCE_LIMIT"))