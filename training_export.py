"""
training_export.py — Converts forensic JSON files into training-ready JSONL format.
Strips Whisper diagnostic bloat, adds training-critical metadata fields,
and outputs streaming-compatible JSONL for HuggingFace datasets.

Can be run standalone to batch-convert existing files, or called from harvester.py
to emit training records in real-time during ingestion.
"""

import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = "astrologer_data_hybrid"
OUTPUT_JSONL = "astrologer_training.jsonl"

# Quality gate constants
MIN_WORD_COUNT = 30
MIN_UWR = 0.35


def calculate_uwr(text):
    if not text: return 0.0
    words = [w.strip(".,?!;:-") for w in text.split() if w.strip(".,?!;:-")]
    if not words: return 0.0
    return len(set(words)) / len(words)


def extract_channel_name(channel_url):
    """Extract clean channel identifier from URL."""
    if not channel_url: return "unknown"
    match = re.search(r'@([^/]+)', channel_url)
    return match.group(1) if match else "unknown"


def convert_single_file(filepath):
    """Convert a single forensic JSON into a training-ready dict.
    Returns None if the file fails quality gates.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    metadata = data.get("metadata", {})
    full_text = data.get("full_text", "")
    segments = data.get("segments", [])

    if not full_text:
        return None

    # Quality Gates
    words = [w.strip(".,?!;:-") for w in full_text.split() if w.strip(".,?!;:-")]
    word_count = len(words)
    if word_count < MIN_WORD_COUNT:
        return None

    global_uwr = calculate_uwr(full_text)
    if global_uwr < MIN_UWR:
        return None

    # Phonetic Gluing check
    glued_hits = len(re.findall(r'(.)\1{4,}', full_text))

    # Compute duration from last segment
    duration_s = 0
    if segments:
        duration_s = int(segments[-1].get("end", 0))

    # Strip segments to training-essential fields only
    clean_segments = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        clean_segments.append({
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "text": text
        })

    # Build training record
    record = {
        "id": metadata.get("video_id", ""),
        "source": extract_channel_name(metadata.get("channel", "")),
        "language": "ta",
        "domain": "vedic_astrology",
        "title": metadata.get("title", ""),
        "duration_s": duration_s,
        "text": full_text,
        "segments": clean_segments,
        "word_count": word_count,
        "quality": {
            "global_uwr": round(global_uwr, 3),
            "glued_hits": glued_hits
        }
    }

    return record


def emit_training_record(data_dict, video_id, channel_url, title, video_duration, jsonl_path=OUTPUT_JSONL):
    """Called directly from harvester.py after a successful transcription.
    Appends a single training-ready JSONL line to the output file.
    
    Args:
        data_dict: The full harvester data dict with 'full_text' and 'segments'.
        video_id: YouTube video ID.
        channel_url: Channel URL string.
        title: Video title string.
        video_duration: Duration in seconds (from yt-dlp metadata).
        jsonl_path: Path to output JSONL file.
    """
    full_text = data_dict.get("full_text", "")
    segments = data_dict.get("segments", [])

    if not full_text:
        return

    words = [w.strip(".,?!;:-") for w in full_text.split() if w.strip(".,?!;:-")]
    word_count = len(words)
    if word_count < MIN_WORD_COUNT:
        return

    global_uwr = calculate_uwr(full_text)
    if global_uwr < MIN_UWR:
        return

    glued_hits = len(re.findall(r'(.)\1{4,}', full_text))

    clean_segments = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        clean_segments.append({
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2),
            "text": text
        })

    record = {
        "id": video_id,
        "source": extract_channel_name(channel_url),
        "language": "ta",
        "domain": "vedic_astrology",
        "title": title,
        "duration_s": int(video_duration) if video_duration else 0,
        "text": full_text,
        "segments": clean_segments,
        "word_count": word_count,
        "quality": {
            "global_uwr": round(global_uwr, 3),
            "glued_hits": glued_hits
        }
    }

    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def batch_convert():
    """Standalone batch converter: scans all forensic JSONs and emits training JSONL."""
    target_files = []
    for root, dirs, files in os.walk(DB_PATH):
        if "review" in root:
            continue
        for f in files:
            if f.endswith(".json") and not f.startswith("REJECTED_"):
                target_files.append(os.path.join(root, f))

    print(f"[Training Export] Scanning {len(target_files)} forensic JSONs...")

    records = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(convert_single_file, fp): fp for fp in target_files}
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)
            else:
                skipped += 1

    # Write JSONL
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    # Stats
    total_words = sum(r["word_count"] for r in records)
    total_segments = sum(len(r["segments"]) for r in records)
    total_duration_hrs = sum(r["duration_s"] for r in records) / 3600

    raw_size = sum(os.path.getsize(fp) for fp in target_files) / (1024 * 1024)
    clean_size = os.path.getsize(OUTPUT_JSONL) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f" 📦 TRAINING EXPORT COMPLETE")
    print(f"{'='*60}")
    print(f" Records exported:    {len(records)}")
    print(f" Records skipped:     {skipped} (failed quality gates)")
    print(f" Total words:         {total_words:,}")
    print(f" Total segments:      {total_segments:,}")
    print(f" Total audio hours:   {total_duration_hrs:.1f}h")
    print(f" Raw forensic size:   {raw_size:.1f} MB")
    print(f" Clean JSONL size:    {clean_size:.1f} MB")
    print(f" Compression ratio:   {raw_size / max(clean_size, 0.01):.1f}x")
    print(f" Output: {OUTPUT_JSONL}")


if __name__ == "__main__":
    batch_convert()
