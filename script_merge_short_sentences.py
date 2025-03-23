import json
import logging
import re
from typing import List, Dict, Any

from src.config.settings import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.ERROR)

MATCHED_SENTENCES_PATH = "data/matched_sentences.jsonl"
MERGED_SENTENCES_PATH = "data/merged_sentences.jsonl"

MIN_BEAT_LENGTH = 3.0  # Minimum duration in seconds for a sentence
MAX_SENTENCE_DISTANCE = 8.0

def ends_with_strong_punctuation(sentence: str) -> bool:
    """Checks if a sentence ends with strong punctuation."""
    return bool(re.search(r'(\.{3}|[.!?])$', sentence.strip()))

def clean_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes empty words from the list."""
    return [word for word in words if word.get("word", "").strip()]

def interpolate_word_times(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Interpolates missing start and end times (-1) using surrounding words."""
    valid_indices = [i for i, w in enumerate(words) if w["start"] != -1 and w["end"] != -1]
    if not valid_indices:
        return words  # No valid times to interpolate

    first_valid, last_valid = valid_indices[0], valid_indices[-1]

    # Forward fill before first valid word
    for i in range(first_valid):
        words[i]["start"] = words[first_valid]["start"] - (first_valid - i) * 0.1
        words[i]["end"] = words[first_valid]["start"] - (first_valid - i) * 0.05

    # Backward fill after last valid word
    for i in range(last_valid + 1, len(words)):
        words[i]["start"] = words[last_valid]["end"] + (i - last_valid) * 0.1
        words[i]["end"] = words[last_valid]["end"] + (i - last_valid) * 0.15

    # Interpolate missing times
    for i in range(len(words)):
        if words[i]["start"] == -1 or words[i]["end"] == -1:
            prev_idx = max([j for j in valid_indices if j < i], default=None)
            next_idx = min([j for j in valid_indices if j > i], default=None)

            if prev_idx is not None and next_idx is not None:
                words[i]["start"] = words[prev_idx]["end"] + ((words[next_idx]["start"] - words[prev_idx]["end"]) / (next_idx - prev_idx)) * (i - prev_idx)
                words[i]["end"] = words[i]["start"] + 0.1  # Small estimated duration

            elif prev_idx is not None:
                words[i]["start"] = words[prev_idx]["end"] + 0.1
                words[i]["end"] = words[i]["start"] + 0.1

            elif next_idx is not None:
                words[i]["start"] = words[next_idx]["start"] - 0.1
                words[i]["end"] = words[i]["start"] + 0.1

    return words

def update_sentence_timestamps(sentences: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
    """Updates start_time, end_time, and duration for all sentences with sanity checks."""
    last_valid_end_time = 0  # Track last known valid end time
    updated_sentences = []

    for i, sentence in enumerate(sentences):

        if debug and i == 207:
            pass  # Debugging breakpoint

        # Backup original timestamps
        backup_start_time = sentence["start_time"]
        backup_end_time = sentence["end_time"]
        backup_duration = sentence["duration"]

        sentence["words"] = clean_words(interpolate_word_times(sentence.get("words", [])))

        # If the sentence has only invalid timestamps, merge it with the previous one
        if not any(word["start"] != -1 and word["end"] != -1 for word in sentence["words"]):
            if updated_sentences:
                prev_sentence = updated_sentences[-1]
                estimated_start = prev_sentence["end_time"] + 0.3  # Start slightly after the previous sentence

                if len(sentence["words"]) > 0:
                    # Re‐timestamp words at 0.3s increments
                    for j, word in enumerate(sentence["words"]):
                        word["start"] = estimated_start + j * 0.3
                        word["end"] = word["start"] + 0.3
                    # Merge current sentence’s text and words into the previous sentence
                    prev_sentence["sentence"] += " " + sentence["sentence"]
                    prev_sentence["end_time"] = sentence["words"][-1]["end"]
                    prev_sentence["duration"] = prev_sentence["end_time"] - prev_sentence["start_time"]
                    prev_sentence["words"].extend(sentence["words"])
                    continue  # Skip adding this sentence separately

                else:
                    # **Fix: Convert entire sentence into a single "word" with backup timestamps**
                    sentence["words"] = [{
                        "word": sentence["sentence"],
                        "start": backup_start_time,
                        "end": backup_end_time,
                        "score": 0.0,  # Score unknown
                        "index": 0,
                        "original_index": 0
                    }]
                    sentence["start_time"] = backup_start_time
                    sentence["end_time"] = backup_end_time
                    sentence["duration"] = backup_duration

        #
        # HERE IS THE MAIN ADJUSTMENT:
        # Ensure word timestamps are in proper sequence, handling multiple consecutive invalid words.
        #
        for idx in range(1, len(sentence["words"])):
            prev_word = sentence["words"][idx - 1]
            curr_word = sentence["words"][idx]

            # If current word's start is before the previous word's end, fix it
            if curr_word["start"] < prev_word["end"]:
                # If it's not the last word, see if we can interpolate from the next word
                if idx < len(sentence["words"]) - 1:
                    next_word = sentence["words"][idx + 1]
                    # Check if we can do a "nice" interpolation
                    if next_word["start"] > prev_word["end"]:
                        # Interpolate: place curr_word roughly 1/3 into the gap between prev_word and next_word
                        segment_duration = (next_word["start"] - prev_word["end"]) / 3
                        curr_word["start"] = prev_word["end"] + segment_duration
                        curr_word["end"] = curr_word["start"] + segment_duration
                    else:
                        # Fallback shift approach: +0.3 beyond prev_word
                        curr_word["start"] = prev_word["end"] + 0.3
                        curr_word["end"] = curr_word["start"] + 0.3
                else:
                    # If it's the LAST word in the list, just shift it by 0.3
                    curr_word["start"] = prev_word["end"] + 0.3
                    curr_word["end"] = curr_word["start"] + 0.3

        # Ensure sentence-level timestamps are valid
        if sentence["words"]:
            sentence["start_time"] = sentence["words"][0]["start"]
            # If last word's end is still -1, fix it with a small default increment
            if sentence["words"][-1]["end"] == -1:
                if len(sentence["words"]) > 1:
                    sentence["words"][-1]["end"] = sentence["words"][-2]["end"] + 0.1
                else:
                    sentence["words"][-1]["end"] = sentence["words"][0]["start"] + 0.1
            sentence["end_time"] = sentence["words"][-1]["end"]
        else:
            # In case there are no words after cleaning/interpolation
            sentence["start_time"] = last_valid_end_time
            sentence["end_time"] = last_valid_end_time + 1  # Default 1 second duration

        sentence["duration"] = sentence["end_time"] - sentence["start_time"]

        last_valid_end_time = sentence["end_time"]
        updated_sentences.append(sentence)

        if debug:
            print(f"Processed sentence {i + 1}: {sentence['sentence']}")
            print(f"Start time: {sentence['start_time']}, End time: {sentence['end_time']}, Duration: {sentence['duration']}")
            print("----------------------------------------------")
            if i == 1:
                pass  # Additional debug

    return updated_sentences

def merge_sentences(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges short sentences with the closest neighbor."""
    if not sentences:
        return []

    #sentences = update_sentence_timestamps(sentences)  # Fix timestamps first
    result = []
    index_counter = 1
    i = 0

    while i < len(sentences):
        current = sentences[i]

        # Verifica se la frase NON termina con punteggiatura forte
        needs_merge = not ends_with_strong_punctuation(current["sentence"])

        # If the sentence is too short, merge it with the closest neighbor
        if needs_merge:
            next_sentence = sentences[i + 1] if i + 1 < len(sentences) else None
            if next_sentence:
                merge_target = next_sentence
                    
                # Merge into the chosen target
                merge_target["sentence"] = current["sentence"] + " " + merge_target["sentence"]
                merge_target["start_time"] = current["start_time"]
                merge_target["duration"] = merge_target["end_time"] - merge_target["start_time"]
                merge_target["words"] = current["words"] + merge_target["words"]
                merge_target["sentence_number"] = index_counter
                result.append(merge_target)

                i += 1  # Skip next sentence since it's merged

        elif current["duration"] < MIN_BEAT_LENGTH:
            prev_sentence = result[-1] if result else None
            next_sentence = sentences[i + 1] if i + 1 < len(sentences) else None

            if prev_sentence and next_sentence:
                prev_gap = abs(prev_sentence["end_time"] - current["start_time"])
                next_gap = abs(current["end_time"] - next_sentence["start_time"])

                merge_target = prev_sentence if prev_gap <= next_gap else next_sentence
                if merge_target == next_sentence:
                    i += 1  # Skip next sentence since it's merged

            elif prev_sentence:
                prev_gap = abs(prev_sentence["end_time"] - current["start_time"])

                if prev_gap <= MAX_SENTENCE_DISTANCE:
                    merge_target = prev_sentence
                    # Merge into the chosen target
                    merge_target["sentence"] += " " + current["sentence"]
                    merge_target["end_time"] = current["end_time"]
                    merge_target["duration"] = merge_target["end_time"] - merge_target["start_time"]
                    merge_target["words"].extend(current["words"])

            elif next_sentence:
                next_gap = abs(current["end_time"] - next_sentence["start_time"])

                if next_gap <= MAX_SENTENCE_DISTANCE:
                    merge_target = next_sentence
                    
                    # Merge into the chosen target
                    merge_target["sentence"] = current["sentence"] + " " + merge_target["sentence"]
                    merge_target["start_time"] = current["start_time"]
                    merge_target["duration"] = merge_target["end_time"] - merge_target["start_time"]
                    merge_target["words"] = current["words"] + merge_target["words"]
                    merge_target["sentence_number"] = index_counter
                    result.append(merge_target)

                    i += 1  # Skip next sentence since it's merged

            else:
                # If no neighbors, just add the current sentence
                result.append(current)
                
        else:
            current["sentence_number"] = index_counter
            index_counter += 1
            result.append(current)

        i += 1

    return result

def main():
    all_merged_sentences = []

    with open(MATCHED_SENTENCES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                video_data = json.loads(line.strip())
                video_id = video_data.get("video_id")

                


                sentences = video_data.get("sentences", [])

                logger.info(f"Found {len(sentences)} sentences for video ID: {video_id}")


                if video_id == "geF_lC3Nahk":
                    sentences = update_sentence_timestamps(sentences, debug = True)
                else:
                    sentences = update_sentence_timestamps(sentences)

                for idx, sentence in enumerate(sentences):

                    # Print merged sentences
                    if sentence["duration"] < 0:
                        previous_text = sentences[idx - 1]["sentence"] if idx > 0 else "None"
                        print(f"{video_id}, {sentence['sentence_number']} - {sentence['sentence']} - {sentence['duration']:.2f}s - {previous_text}")


                # Step 2: Merge short sentences
                sentences = merge_sentences(sentences)

                #DOUBLE PASS
                sentences = merge_sentences(sentences)

                for idx, sentence in enumerate(sentences):
                    sentence["video_id"] = video_id
                    sentence["genre"] = video_data.get("genre")
                    sentence["url"] = video_data.get("url")
                    sentence["channel_name"] = video_data.get("channel_name")

                    # Remove "index" and "original_index" from words
                    for word in sentence.get("words", []):
                        word.pop("index", None)
                        word.pop("original_index", None)
                        word.pop("score", None)

                    all_merged_sentences.append(sentence)

                    # Print merged sentences
                    if sentence["duration"] < MIN_BEAT_LENGTH:
                        previous_text = sentences[idx - 1]["sentence"] if idx > 0 else "None"
                        print(f"{video_id}, {sentence['sentence_number']} - {sentence['sentence']} - {sentence['duration']:.2f}s - {previous_text}")

                # print the average duration of sentences
                total_duration = sum(sentence["duration"] for sentence in sentences)
                average_duration = total_duration / len(sentences)
                print(f"Average duration: {average_duration:.2f}s")



                pass
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e} in line: {line.strip()}")
            except Exception as e:
                logger.error(f"Error processing line: {e} with line: {line.strip()}")

    # Step 3: Save the merged sentences to a new file
    logger.info(f"Saving {len(all_merged_sentences)} merged sentences to {MERGED_SENTENCES_PATH}")

    with open(MERGED_SENTENCES_PATH, "w", encoding="utf-8") as outfile:
        for sentence in all_merged_sentences:
            json.dump(sentence, outfile, ensure_ascii=False)
            outfile.write("\n")

if __name__ == "__main__":
    main()