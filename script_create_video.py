import json
import os
import asyncio
import logging
import re
import torch
import ffmpeg
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from src.ai_models import AIModelsService, LLMType

# ▼▼▼ Moduli del tuo progetto ▼▼▼
from src.config.settings import THEME, SEED, AUDIO_DIR, MODEL_SETTINGS, VIDEO_SETTINGS, FONT_FILE
from src.downloader_service import DownloaderService

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




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

# ----------------------------------------------------------------------
# COSTANTI DI PATH (adatta come preferisci)
# ----------------------------------------------------------------------
OUTPUT_DIR = f"output/{sanitize_filename(THEME)}_{SEED}"
ORDERED_FILE = os.path.join(OUTPUT_DIR, "ordered_sentences.json")
CLIPS_DIR = os.path.join(OUTPUT_DIR, "clips")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "final_montage.mp4")

DATA_VIDEO_DIR = VIDEO_SETTINGS.get("data_video_dir", None)

Path(CLIPS_DIR).mkdir(parents=True, exist_ok=True)

TITLE_SCREEN_PATH = os.path.join(OUTPUT_DIR, "title_screen.mp4")
# ----------------------------------------------------------------------
# MODELLI WHISPERX (CARICAMENTO E FUNZIONE DI TRASCRIZIONE)
# ----------------------------------------------------------------------

previous_extra_time = 0.15
later_extra_time = 0.3





# ----------------------------------------------------------------------
# VIDEO ASSEMBLER
# ----------------------------------------------------------------------
class VideoAssembler:
    """Class to handle downloading and assembling video clips."""
    
    def __init__(self, downloader: DownloaderService):
        self.downloader = downloader
        self.ordered_sentences = []
        self.clip_paths = []
        

    def _build_llm_prompt(self, previous_words, current_words, next_words, theme, transition_note, previous_decided_text) -> str:
        return f"""
                Hai ricevuto tre frasi consecutive, ciascuna già suddivisa in parole con timestamp.

                CONTESTO PRECEDENTE (ultime due frasi selezionate): {previous_decided_text}

                Il tuo compito è:
                1. Scegliere una sequenza CONTINUA di parole che abbia senso come frase autonoma.
                2. La frase può iniziare nella frase precedente e finire in quella successiva.
                3. Non spezzare frasi o selezionare parole isolate. Mantieni continuità e coerenza.
                4. Non deve essere lunga più di circa 10 secondi.
                5. Deve avere senso!

                TEMA DEL VIDEO: {theme}

                NOTA DI TRANSIZIONE: {transition_note}

                Considerazioni importanti:
                - Il tema del video è '{theme}'. Seleziona parole che siano coerenti con questo tema.
                - La nota di transizione fornisce indicazioni sul tono e lo stile della frase: '{transition_note}'
                - Cerca di selezionare parole che rispettino il tono indicato nella nota di transizione.
                - Mantieni coerenza con il contesto precedente (ultime due frasi selezionate): {previous_decided_text}

                Restituisci SOLO un JSON nel seguente formato assicurandoti che i timestamp delle parole siano corretti:

                {{
                "reason": "Motivo sintetico della scelta",
                "selected_words": [
                    {{"word": "esempio", "start": 12.3, "end": 12.6}}
                    ...
                ]
                }}

                Dati disponibili:

                {json.dumps({
                        "previous": previous_words,
                        "current": current_words,
                        "next": next_words
                    }, ensure_ascii=False, indent=2)}
                """
        
    async def load_ordered_sentences(self, mock_call = True) -> None:
        """Load the ordered sentences from JSON file e aggiorna start/end via LLM."""
        try:
            with open(ORDERED_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.ordered_sentences = data.get('ordered_phrases', [])
                logger.info(f"Loaded {len(self.ordered_sentences)} ordered sentences")

            # Applichiamo il refining con l'LLM su ogni frase
            refined_count = 0
            for idx, sentence in enumerate(self.ordered_sentences):

                if mock_call:
                    logger.info(f"Skipping Mock LLM call for sentence {sentence.get('order', 0)}")
                    result = None
                    refined_count += 1
                else:
                    result = self._refine_start_end_with_llm(sentence, idx)  # Pass index here



                    if result:
                        refined_start, refined_end = result
                        sentence["metadata"]["start_time"] = refined_start
                        sentence["metadata"]["end_time"] = refined_end
                        refined_count += 1
                    else:
                        logger.warning("LLM non ha restituito un segmento valido. Uso timestamp originali.")

            logger.info(f"Refined {refined_count} sentences using LLM")

        except Exception as e:
            logger.error(f"Error loading or refining ordered sentences: {e}")
            raise

    def _refine_start_end_with_llm(self, sentence: Dict, idx: int) -> Optional[Tuple[float, float]]:

        # Collect words from the previous two decided sentences
        previous_decided_words = []
        for prev_sentence in self.ordered_sentences[max(0, idx - 2):idx]:  # Only last two sentences
            selected = prev_sentence.get("llm_selected_words", [])
            words = [w["word"] for w in selected]
            previous_decided_words.extend(words)
        previous_decided_text = " ".join(previous_decided_words)

        current_words = json.loads(sentence["metadata"]["words"])
        previous_words = json.loads((sentence.get("previous_sentence") or {}).get("metadata", {}).get("words", "[]"))
        next_words = json.loads((sentence.get("next_sentence") or {}).get("metadata", {}).get("words", "[]"))
        theme = sentence.get("batch_name", "")
        transition_note = sentence.get("transition_note", "")

        try:


            # Include previous_decided_text in the prompt
            prompt = self._build_llm_prompt(
                previous_words, 
                current_words, 
                next_words, 
                theme, 
                transition_note,
                previous_decided_text  # Pass only the last two sentences' words
            )

            # Call the LLM and process the response
            logger.info("Calling LLM for sentence refinement...")
            response = AIModelsService().call_llm(prompt, llm_type=LLMType.INTELLIGENT)
            data = json.loads(response)

            selected = data.get("selected_words")
            reason = data.get("reason", "")

            sentence_words_list = ""
            for word in selected:
                sentence_words_list = [w["word"] for w in selected]
            
            logger.info(f"Selected words: {sentence_words_list}")

            if not selected:
                logger.warning("LLM did not return selected words")
                return None

            # Save the selected words and reason in the original sentence
            sentence["llm_selected_words"] = selected
            sentence["llm_reason"] = reason

            refined_start = selected[0]['start'] - previous_extra_time
            refined_end = selected[-1]['end'] + later_extra_time

            return max(0, refined_start), refined_end

        except Exception as e:
            logger.error(f"Errore nella selezione LLM: {e}")
            return None
    
    async def download_all_clips(self) -> None:
        """Download all video clips based on ordered sentences."""
        tasks = []
        for idx, sentence in enumerate(self.ordered_sentences):
            metadata = sentence.get('metadata', {})
            source_info = sentence.get('source', '').split('/')
            
            if len(source_info) != 2:
                logger.warning(f"Invalid source format for sentence {idx+1}: {sentence.get('source')}")
                continue

            video_id, _ = source_info
            order_number = sentence.get("order", idx+1)

            clip_metadata = {
                'source': f"https://www.youtube.com/watch?v={video_id}",
                'start_time': max(0, metadata.get('start_time', 0) - previous_extra_time),
                'end_time': metadata.get('end_time', 0) + later_extra_time,
                'search_text': sentence.get('matched_phrase', ''),
                'order_number': order_number,
                'sentence_number': metadata.get('sentence_number', 0)
            }
            
            output_path = os.path.join(CLIPS_DIR, f"clip_{order_number:03d}.mp4")
            
            tasks.append(self.download_clip(clip_metadata, output_path, order_number))
        
        await asyncio.gather(*tasks)
        
        # Make sure clip_paths is properly populated and sorted
        logger.info(f"Downloaded {len(self.clip_paths)} clips out of {len(self.ordered_sentences)} sentences")
        self.clip_paths.sort(key=lambda x: x[0])
    
    async def download_clip(self, metadata: Dict, output_path: str, order: int) -> None:
        """Download a single video clip."""
        try:
            logger.info(
                f"Downloading clip {order} of {len(self.ordered_sentences)}: "
                f"{metadata['search_text'][:30]}..."
            )
            
            success = await self.process_clip(metadata, output_path)
            
            if success:
                self.clip_paths.append((order, output_path))
                logger.info(f"Successfully downloaded clip {order}")
            else:
                logger.error(f"Failed to download clip {order}")
        except Exception as e:
            logger.error(f"Error downloading clip {order}: {e}")
    
    async def process_clip(self, metadata: Dict, output_path: str) -> bool:
        """
        Usa direttamente i file precut esistenti invece di scaricare nuovamente i clip.
        """
        try:
            temp_cut_path = await self.downloader.get_or_create_precut(
                url=metadata['source'],
                start_time=metadata.get("start_time", 0),
                end_time=metadata.get("end_time", 0),
                sentence_number=metadata['sentence_number']
            )

            if not temp_cut_path:
                logger.error(f"Impossibile ottenere il precut per: {metadata['source']}")
                return False
                
            logger.info(f"Using existing precut file: {temp_cut_path}")
            
            # Copia direttamente il file precut nella directory di output
            # Aggiungi fade audio
            fade_output_path = output_path.replace(".mp4", "_fade.mp4")
            
            # Copia il file con ffmpeg per assicurarsi che sia compatibile
            input_stream = ffmpeg.input(temp_cut_path)
            
            # Mantieni lo stesso formato video ma assicurati che sia 720p e 30fps
            v_stream = (
                input_stream.video
                .filter('setpts', 'PTS-STARTPTS')
                .filter('scale', 1280, 720)
                .filter('fps', fps=30)
            )

            a_stream = (
                input_stream.audio
                .filter('asetpts', 'PTS-STARTPTS')
                .filter('aresample', **{'async': '1', 'first_pts': '0'})
            )
            
            # Output con parametri di codifica
            stream = ffmpeg.output(
                v_stream,
                a_stream,
                output_path,
                acodec='aac',
                vcodec='libx264',
                preset='veryfast',
                crf=23,
                audio_bitrate='192k',
                movflags='+faststart',
                fflags='+genpts',
                avoid_negative_ts='make_zero',
                **{'y': None, 'loglevel': 'error'}  # Add this
            )
            
            await self._run_ffmpeg(stream)
            
            # Aggiungi fade audio
            fade_success = await self._add_audio_fade(output_path, fade_output_path)
            if fade_success:
                os.replace(fade_output_path, output_path)
            
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
        except Exception as e:
            logger.error(f"Errore in process_clip (frase {metadata.get('search_text', '')}): {e}")
            return False

    async def _run_ffmpeg(self, stream):
        """Run ffmpeg asynchronously with error checking."""
        args = stream.get_args()
        
        logger.debug(f"Running FFmpeg command: {' '.join(args)}")
        
        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y", 
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            if process.returncode != 0:
                stderr_text = stderr.decode('utf-8', errors='replace')
                logger.error(f"FFmpeg error: {stderr_text}")
                raise RuntimeError(f"FFmpeg error: {stderr_text}")
            return True
        except asyncio.TimeoutError:
            process.kill()
            logger.error("FFmpeg process timed out after 300 seconds.")
            raise RuntimeError("FFmpeg timeout")

    async def _add_audio_fade(self, input_path: str, output_path: str) -> bool:
        """
        Esegue fade in/out dell'audio in maniera accurata.
        Ritorna True se il file risultante è valido, False in caso di errore.
        """
        try:
            duration = float(ffmpeg.probe(input_path)['format']['duration'])
            input_stream = ffmpeg.input(input_path)
            
            # Video invariato
            v_stream = input_stream.video
            
            # Fade audio
            a_stream = (
                input_stream.audio
                .filter('afade', t='in', start_time=0, duration=previous_extra_time)
                .filter(
                    'afade',
                    t='out',
                    start_time=duration - later_extra_time,
                    duration=later_extra_time
                )
            )
            
            # Output con parametri di codifica audio
            stream = ffmpeg.output(
                v_stream,
                a_stream,
                output_path,
                vcodec='copy',  # Non ricodifica il video
                acodec='aac',   # Ricodifica solo l'audio (con fades)
                audio_bitrate='192k',
                **{'y': None, 'loglevel': 'error'}  # Add this
            )
            
            await self._run_ffmpeg(stream)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception as e:
            logger.error(f"Fade audio fallito: {e}")
            return False



    async def preprocess_intro_video(self, input_path: str, output_path: str) -> bool:
        """
        Ricodifica la sigla in modo da uniformarla ai clip (720p, 30fps, AAC audio).
        """
        try:
            if not os.path.exists(input_path):
                logger.error(f"Intro video non trovato: {input_path}")
                return False

            input_stream = ffmpeg.input(input_path)

            v_stream = (
                input_stream.video
                .filter('scale', 1280, 720)
                .filter('fps', fps=30)
                .filter('setpts', 'PTS-STARTPTS')
            )

            a_stream = (
                input_stream.audio
                .filter('asetpts', 'PTS-STARTPTS')
                .filter('aresample', **{'async': '1', 'first_pts': '0'})
            )

            stream = ffmpeg.output(
                v_stream,
                a_stream,
                output_path,
                acodec='aac',
                vcodec='libx264',
                crf=23,
                preset='veryfast',
                audio_bitrate='192k',
                movflags='+faststart',
                fflags='+genpts',
                avoid_negative_ts='make_zero',
                **{'y': None, 'loglevel': 'error'}  # Add this
            )

            await self._run_ffmpeg(stream)
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        except Exception as e:
            logger.error(f"Errore nella ricodifica della sigla: {e}")
            return False

    async def create_title_screen(self) -> Optional[str]:
        """Generate a title screen video with specified text elements."""
        try:
            duration = 5  # seconds
            output_path = TITLE_SCREEN_PATH

            # Calculate text positions
            theme_fontsize = 48
            detail_fontsize = 24
            line_height = 40
            start_y = 720 * 0.4  # Start at 40% of screen height

            # Create input stream
            input_stream = ffmpeg.input(
                f"color=black:s=1280x720:d={duration}", 
                f='lavfi'
            )

            # Build video stream with chained drawtext filters
            v_stream = input_stream.video
            v_stream = v_stream.drawtext(
                text=THEME,
                fontfile=FONT_FILE,
                fontcolor='white',
                fontsize=theme_fontsize,
                x='(w-text_w)/2',
                y=start_y
            )
            v_stream = v_stream.drawtext(
                text='AI! Blob',
                fontfile=FONT_FILE,
                fontcolor='white',
                fontsize=detail_fontsize,
                x='(w-text_w)/2',
                y=start_y + theme_fontsize + 20
            )
            v_stream = v_stream.drawtext(
                text='Semantic Cataloging for the Retrieval and Recontextualization of Italian Television Archives',
                fontfile=FONT_FILE,
                fontcolor='white',
                fontsize=detail_fontsize,
                x='(w-text_w)/2',
                y=start_y + theme_fontsize + 20 + line_height
            )
            v_stream = v_stream.drawtext(
                text='Roberto Balestri - Ph. D. Student, Dipartimento delle Arti, Università di Bologna',
                fontfile=FONT_FILE,
                fontcolor='white',
                fontsize=detail_fontsize,
                x='(w-text_w)/2',
                y=start_y + theme_fontsize + 20 + line_height * 2
            )
            v_stream = v_stream.drawtext(
                text='Media Mutations 16 - 2025 - Unlocking Television Archives in the Digital Era',
                fontfile=FONT_FILE,
                fontcolor='white',
                fontsize=detail_fontsize,
                x='(w-text_w)/2',
                y=start_y + theme_fontsize + 20 + line_height * 3
            )

            # Create audio stream
            a_stream = ffmpeg.input('anullsrc', f='lavfi', t=duration).audio

            # Create output stream
            stream = ffmpeg.output(
                v_stream,
                a_stream,
                output_path,
                vcodec='libx264',
                acodec='aac',
                preset='veryfast',
                crf=18,
                **{'y': None, 'loglevel': 'error'}
            )

            await self._run_ffmpeg(stream)
            return output_path if os.path.exists(output_path) else None

        except Exception as e:
            logger.error(f"Error generating title screen: {e}")
            return None

    
    async def create_final_montage(self) -> None:
        """Combine all clips into a final montage with intro video, applying compression and loudness normalization."""
        try:
            sorted_clips = sorted(self.clip_paths, key=lambda x: x[0])
            clip_paths = [path for _, path in sorted_clips]

            if not clip_paths:
                logger.error("No clips available to create montage. Check if clips were downloaded successfully.")
                return

            # Validate all clip paths
            valid_clip_paths = [p for p in clip_paths if os.path.exists(p) and os.path.getsize(p) > 0]
            if not valid_clip_paths:
                logger.error("No valid clip files found. Cannot create montage.")
                return

            logger.info(f"Creating montage with {len(valid_clip_paths)} valid clips")

            # Handle intro video processing
            intro_original = os.path.join(DATA_VIDEO_DIR, "sigla_ai_blob.mp4")
            intro_processed = os.path.join(DATA_VIDEO_DIR, "intro_processed.mp4")

            if os.path.exists(intro_processed):
                logger.info(f"Using existing intro: {intro_processed}")
            elif os.path.exists(intro_original):
                logger.info(f"Processing intro: {intro_original} -> {intro_processed}")
                success = await self.preprocess_intro_video(intro_original, intro_processed)
                if not success:
                    logger.warning("Intro processing failed")
                    intro_processed = None
            else:
                logger.warning("No intro video found")
                intro_processed = None

            # Create concat list file
            concat_file = os.path.join(OUTPUT_DIR, "concat_list.txt")

            # Handle title screen
            title_screen_path = await self.create_title_screen()
            if not title_screen_path:
                logger.warning("Title screen generation failed")

            # Create concat list file
            with open(concat_file, 'w', encoding='utf-8') as f:
                if intro_processed:
                    f.write(f"file '{os.path.abspath(intro_processed)}'\n")
                if title_screen_path:
                    f.write(f"file '{os.path.abspath(title_screen_path)}'\n")
                for path in valid_clip_paths:
                    f.write(f"file '{os.path.abspath(path)}'\n")

            logger.info(f"Created concat file with {len(valid_clip_paths) + (1 if intro_processed else 0)} entries")

            # Input FFmpeg from concat list
            input_stream = ffmpeg.input(concat_file, format='concat', safe=0)

            # Video stream unchanged
            v_stream = input_stream.video

            # Audio stream with compressor and loudness normalization
            a_stream = (
                input_stream.audio
                .filter('acompressor', level_in=1.0, threshold=0.1, ratio=9, attack=20, release=250)
                .filter('loudnorm', i=-16, tp=-1.5, lra=11)
            )

            # Final output stream
            stream = ffmpeg.output(
                v_stream,
                a_stream,
                FINAL_OUTPUT,
                vcodec='libx264',
                acodec='aac',
                crf=23,
                preset='veryfast',
                audio_bitrate='192k',
                g=60,
                movflags='+faststart',
                fflags='+genpts',
                avoid_negative_ts='make_zero',
                **{'y': None, 'loglevel': 'info'}  # instead of 'error'
            )

            await self._run_ffmpeg(stream)
            logger.info(f"Final montage created at: {FINAL_OUTPUT}")

        except Exception as e:
            logger.error(f"Error creating final montage: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    async def run(self) -> None:
        """Run the entire assembly process."""
        try:
            await self.load_ordered_sentences()
            logger.info(f"Loaded {len(self.ordered_sentences)} sentences for processing")
            
            await self.download_all_clips()
            
            # Check if we have clips before proceeding
            if not self.clip_paths:
                logger.error("No clips were downloaded. Cannot create final montage.")
                return
                
            logger.info(f"Successfully downloaded {len(self.clip_paths)} clips. Creating final montage...")
            await self.create_final_montage()
            logger.info("Video assembly completed successfully!")
        except Exception as e:
            logger.error(f"Error in video assembly process: {e}")
            import traceback
            logger.error(traceback.format_exc())


# ------------------------------------------------
# FUNZIONE PRINCIPALE
# ------------------------------------------------
async def main():

    try:
        downloader = DownloaderService()
        await downloader.clear_cache()  # Add this
        assembler = VideoAssembler(downloader = downloader)
        await assembler.run()
    except Exception as e:
        logger.error(f"Error creating video: {e}")

    #finally:
    #    for f in Path(CLIPS_DIR).glob("*"):
    #        f.unlink()
    #        logger.info(f"Cleared clips at: {str(CLIPS_DIR)}")
    


if __name__ == "__main__":
    asyncio.run(main())
