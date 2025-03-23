import json
import fasttext

# Load FastText language detection model
MODEL_PATH = "lid.176.bin"
model = fasttext.load_model(MODEL_PATH)

# File paths
input_file_path = "video_transcripts.transcripts_full.json"
output_file_path = "data/dataset_pulito.jsonl"  # Changed to JSON Lines format

def detect_language(text):
    prediction = model.predict(text, k=1)
    return prediction[0][0].replace("__label__", "")

def process_and_save_videos():
    # Initialize counters
    total_count = 0
    valid_count = 0
    
    with open(input_file_path, "r", encoding="utf-8") as infile, \
         open(output_file_path, "w", encoding="utf-8") as outfile:
        
        data = json.load(infile)
        print(f"Loaded {len(data)} videos")
        
        for video in data:
            total_count += 1
            text = video.get("text", "").strip()
            
            # Skip non-Italian content
            if not text or detect_language(text) != "it":
                continue
            
            # Extract only essential metadata
            metadata = video.get("metadata", {})
            video_info = {
                "video_id": video.get("video_id"),
                "url": video.get("url"),
                "duration": metadata.get("duration"),
                "genre": metadata.get("genre"),
                "channel_name": metadata.get("channel_name")
            }
            
            # Write directly as JSON Lines
            outfile.write(json.dumps(video_info, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"\nProcessing complete:")
    print(f" - Total videos analyzed: {total_count}")
    print(f" - Italian videos saved: {valid_count}")
    print(f" - JSON Lines dataset created at: {output_file_path}")

if __name__ == "__main__":
    process_and_save_videos()