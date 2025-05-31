"""
GUI module for the AI Blob video generation system.

This module provides web-based interfaces for interactive clip selection
and video montage creation.
"""

from .gui_service import GUIService, SelectionState, ClipCandidate
from .components import (
    render_clip_card, 
    render_progress_bar, 
    render_theme_input
)

__all__ = [
    'GUIService',
    'SelectionState', 
    'ClipCandidate',
    'render_clip_card',
    'render_progress_bar',
    'render_theme_input'
]
