
import re

def sanitize_filename_cv(name: str) -> str: # Renamed to avoid conflict if this script is merged or utils are shared
    """Sanitizza una stringa per renderla un nome di file valido."""
    name = name.lower()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^a-z0-9_\-]', '', name)
    return name[:50]