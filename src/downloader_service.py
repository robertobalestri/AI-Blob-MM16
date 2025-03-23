import logging
import os
import asyncio
import yt_dlp
from typing import Optional, Tuple
from pathlib import Path
import ffmpeg

from src.config.settings import VIDEO_SETTINGS, CACHE_DIR

logger = logging.getLogger(__name__)

class DownloaderService:
    """Service per il download e il ritaglio dei video (precut)."""
    
    def __init__(self):
        self.output_dir = CACHE_DIR
        self.temp_dir = os.path.join(self.output_dir, "temp")
        self.download_semaphore = asyncio.Semaphore(VIDEO_SETTINGS.get("max_concurrent_downloads", 2))

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _extract_video_id(url: str) -> str:
        """Estrae l'ID del video da una URL di YouTube."""
        return url.split('watch?v=')[-1].split('&')[0]

    async def download_segment(
        self,
        url: str,
        start_time: float,
        end_time: float,
        output_path: str
    ) -> Tuple[bool, float, float]:
        """Scarica un segmento del video usando i buffer globali."""
        try:
            video_id = self._extract_video_id(url)
            actual_start = start_time
            actual_end = end_time
            duration = actual_end - actual_start

            async with self.download_semaphore:
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: ydl.extract_info(url, download=False)
                    )
                    formats = info.get('formats', [])
                    if not formats:
                        logger.error("No formats available for video")
                        return False, 0, 0

                    # Trova il formato migliore
                    video_format = max(
                        (f for f in formats if f.get('vcodec') != 'none' and f.get('acodec') != 'none' and f.get('ext') == 'mp4'),
                        key=lambda f: f.get('tbr', 0),
                        default=None
                    )

                    if not video_format:
                        logger.error("No suitable mp4 format found")
                        return False, 0, 0

                    # Scarica direttamente il segmento desiderato (con margini)
                    stream = ffmpeg.input(video_format['url'], ss=actual_start, t=duration)
                    stream = ffmpeg.output(
                        stream,
                        output_path,
                        vcodec='libx264',
                        acodec='aac',
                        crf=23,
                        preset='veryfast',
                        audio_bitrate='192k',
                        movflags='+faststart',
                        fflags='+genpts',
                        avoid_negative_ts='make_zero',
                        **{'y': None, 'loglevel': 'error'}
                    )
                    await asyncio.get_event_loop().run_in_executor(None, stream.run)

                    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                        return True, actual_start, actual_end
                    else:
                        return False, 0, 0

        except Exception as e:
            logger.error(f"Error downloading segment: {e}")
            return False, 0, 0

    async def get_or_create_precut(
        self,
        url: str,
        start_time: float,
        end_time: float,
        sentence_number: int
    ) -> Optional[str]:
        """Ritorna il path al precut usando i buffer globali."""
        video_id = self._extract_video_id(url)
        output_path = os.path.join(self.temp_dir, f"{video_id}_{sentence_number}_precut.mp4")

        # Rimuovi il parametro buffer_time dalla chiamata
        success, _, _ = await self.download_segment(
            url=url,
            start_time=start_time,
            end_time=end_time,
            output_path=output_path
        )
        return output_path if success else None

    async def clear_cache(self):
        """Remove all temporary files"""
        try:
            for f in Path(self.temp_dir).glob("*"):
                f.unlink()
            logger.info(f"Cleared cache at: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
