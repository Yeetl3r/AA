import os
# [CRITICAL] Route HuggingFace Cache to External Drive BEFORE any imports
# HF_HUB_CACHE is the highest-priority variable for snapshot_download blob storage
os.environ["HF_HOME"] = "/Volumes/Storage Drive/AA/hf_cache"
os.environ["HF_HUB_CACHE"] = "/Volumes/Storage Drive/AA/hf_cache/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/Volumes/Storage Drive/AA/hf_cache/hub"

import yt_dlp
import mlx_whisper
import json
import re
import time
import datetime
import gc
import random
import sys
import psutil
import argparse
import threading
import subprocess
import fcntl

import numpy as np
import mlx.core as mx
import math
import transcribe_engine
import training_export
import manifest_manager
import validator_v2

# --- CONFIGURATION ---
ALL_CHANNELS = [
    "https://www.youtube.com/@adityagurujiastrologerchennai/videos",
    "https://www.youtube.com/@adityagurujiastrologerchennai/streams",
    "https://www.youtube.com/@shrimahalakshmi-premium5868/videos",
    "https://www.youtube.com/@shrimahalakshmi-premium5868/streams",
    "https://www.youtube.com/@SriMahalakshmiJothidam/videos",
    "https://www.youtube.com/@SriMahalakshmiJothidam/streams",
    "https://www.youtube.com/@AstroSriramJI/videos",
    "https://www.youtube.com/@AstroSriramJI/streams"
]

OUTPUT_FOLDER = "astrologer_data_hybrid"
ERROR_LOG_FILE = "harvest_errors.log"
HEARTBEAT_LOG_FILE = "heartbeat.log"
LOCAL_MODEL_PATH = "/Volumes/Storage Drive/AA/mlx_models/large-v3"
MAX_VIDEOS_PER_TAB = None 
LOCK_FILE = "/tmp/harvester_pipeline.lock"

# --- ROBUSTNESS FUNCTIONS ---

def get_thermal_level():
    """Returns thermal pressure level: 0=nominal, 1=moderate, 2=heavy, 3=critical.
    Falls back to pmset CPU_Speed_Limit parsing if sysctl is restricted."""
    try:
        output = subprocess.check_output(["sysctl", "-n", "kern.thermal_pressure"], stderr=subprocess.DEVNULL).decode().strip()
        return int(output)
    except Exception:
        pass
    try:
        output = subprocess.check_output(["pmset", "-g", "therm"], stderr=subprocess.DEVNULL).decode()
        limit_match = re.search(r"CPU_Speed_Limit\s+=\s+(\d+)", output)
        if limit_match:
            limit = int(limit_match.group(1))
            if limit < 70: return 3
            if limit < 85: return 2
            if limit < 95: return 1
    except Exception:
        pass
    return 0

def cooldown_if_needed():
    """Reactive thermal cooldown tuned for fanless M4 Air.
    Triggers at MODERATE (level 1) since there's no fan for active recovery.
    Uses exponential backoff (20s → 40s → 80s → 160s) with shorter intervals."""
    def log_thermal(level, duration):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[THERMAL GUARD] | {timestamp} | Level: {level} | Sleeping for {duration}s\n"
        with open(HEARTBEAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg)

    backoff = 20
    while True:
        level = get_thermal_level()
        if level < 1:  # React to MODERATE (1), HEAVY (2), CRITICAL (3) — fanless M4 Air
            return
        sleep_time = min(backoff, 160)
        print(f"  🚨 Thermal Level {level}. Cooling down {sleep_time}s...")
        log_thermal(level, sleep_time)
        time.sleep(sleep_time)
        backoff *= 2  # Exponential backoff: 20 → 40 → 80 → 160


def acquire_pipeline_lock():
    """Acquire an exclusive file lock to prevent concurrent runs with semantic_formatter.
    Returns the lock file descriptor (must be kept open for the duration of the process)."""
    try:
        lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(f"{os.getpid()}\n")
        lock_fd.flush()
        return lock_fd
    except (IOError, OSError):
        print("❌ Another pipeline process (harvester or semantic_formatter) is already running.")
        print("   If this is wrong, remove /tmp/harvester_pipeline.lock and retry.")
        sys.exit(1)

def cleanup_stale_data():
    """Proactively remove incomplete (.tmp) or corrupted JSON files on startup.
    Ensures that interrupted runs start from scratch for those specific videos."""
    raw_dir = os.path.join(OUTPUT_FOLDER, "raw_queue")
    if not os.path.exists(raw_dir):
        return
    
    removed_count = 0
    for f in os.listdir(raw_dir):
        path = os.path.join(raw_dir, f)
        # 1. Always delete .tmp files
        if f.endswith(".tmp"):
            try:
                os.remove(path)
                removed_count += 1
            except: pass
        # 2. Delete empty or corrupted .json files
        elif f.endswith(".json"):
            try:
                if os.path.getsize(path) == 0:
                    os.remove(path)
                    removed_count += 1
                    continue
                with open(path, 'r') as jf:
                    json.load(jf)
            except (json.JSONDecodeError, OSError):
                try:
                    os.remove(path)
                    removed_count += 1
                except: pass
    
    if removed_count > 0:
        print(f"  🧹 Startup Cleanup: Removed {removed_count} incomplete/stale files.")

def verify_model_architecture():
    config_path = os.path.join(LOCAL_MODEL_PATH, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            num_layers = config.get("n_text_layer", 0)
            if num_layers == 32:
                print(f"  ✅ Model Architecture Verified: 32-Layer Large-v3 detected (External Drive).")
            elif num_layers == 4:
                print("  ✅ Model Architecture Verified: 4-Layer Turbo detected.")
            else:
                print(f"  ⚠️ Unknown layer count: {num_layers}")
        except Exception as e:
            print(f"  ⚠️ Could not verify config.json: {e}")
    else:
        print(f"  ⚠️ config.json not found in {LOCAL_MODEL_PATH}. Skipping architecture check.")

def ssd_pulse():
    pulse_file = os.path.join(OUTPUT_FOLDER, ".heartbeat")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    while True:
        try:
            with open(pulse_file, "w") as f:
                f.write(str(time.time()))
        except Exception:
            pass
        time.sleep(60)

def log_overall(status, video_id, extra=""):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{status}] | {timestamp} | {video_id}"
    if extra:
        msg += f" | {extra}"
    with open(HEARTBEAT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def log_error(video_id, url, error_msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ID: {video_id} | URL: {url} | Error: {error_msg}\n")

# [Deprecated in Phase 2: Moved to manifest_manager.py]
# def get_existing_video_ids(folder): ...

def clean_channel_name(url):
    match = re.search(r'@([^/]+)', url)
    if match:
        name = match.group(1)
        return "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip()
    return "Unknown_Channel"

def safe_filename(title, video_id, channel_url):
    folder_path = os.path.join(OUTPUT_FOLDER, "raw_queue")
    os.makedirs(folder_path, exist_ok=True)
    clean_title = "".join([c for c in title if c.isalnum() or c in (' ', '-', '_')]).strip()
    clean_title = clean_title[:100]
    return os.path.join(folder_path, f"{clean_title}_{video_id}.json")

# [Moved denoise_loops and get_whisper_transcription to transcribe_engine.py]

# --- MAIN LOOP ---

# --- SHARED VIDEO PROCESSING ---

def process_single_video(vid_id, url, title, channel_url, video_duration, params, manifest_mgr):
    """Unified video processing: transcribe → denoise → quality check → write JSON + JSONL.
    Returns True on success, False on failure, None on skip (MEMBERS_ONLY)."""
    
    # --- TRANSCRIPTION PASS (v2.2 Self-Correcting) ---
    max_passes = 2
    for pass_idx in range(max_passes):
        vid_start = time.time()
        
        # Adjust params for retry pass if we detected a failure previously
        current_params = params.copy()
        if pass_idx > 0:
            print(f"    [Retry Pass] Attempting loop-breaking configuration (temp=0.2, no-condition)...")
            current_params["temperature"] = (0.2,)
            current_params["condition_on_previous_text"] = False
            
        transcribe_res = transcribe_engine.transcribe_video(url, vid_id, current_params)
        processing_time = time.time() - vid_start
        
        if not transcribe_res:
            print(" -> ⚠️ FAILED (No result).")
            return False
        if transcribe_res == "MEMBERS_ONLY":
            return None
            
        # --- EXTRACT RESULTS (v2.3 Omega-Repair aware) ---
        segments = transcribe_res.get('segments', [])
        # The engine now returns 'text' (corrected) and 'raw_text' (ASR direct output)
        final_text = transcribe_res.get('text', '')
        raw_text = transcribe_res.get('raw_text', '')
        sentry_status = transcribe_res.get('sentry_status', 'UNKNOWN')
        
        # Apply word-level deduplication to segments for backward compatibility
        for seg in segments:
            seg['text'] = transcribe_engine.denoise_loops(seg.get('text', ''))
        
        # --- IN-LINE AUDIT (The "Quality Gate") ---
        data_to_audit = {
            "metadata": {"title": title, "video_id": vid_id},
            "full_text": final_text,
            "raw_text": raw_text,
            "sentry_status": sentry_status,
            "segments": segments
        }
        category, metrics = validator_v2.validate_transcription(data_to_audit)
        
        if category == "SUCCESS" or pass_idx == max_passes - 1:
            if category != "SUCCESS":
                print(f" -> ⚠️ Final Category: {category} (Retry failed to resolve completely).")
            break
        else:
            print(f" -> 🔄 Audit FAILURE: {category}. Triggering in-line retry...")

    rtf_str = ""
    if processing_time > 0 and video_duration:
        rtf = video_duration / processing_time
        rtf_str = f"RTF: {rtf:.2f}x"

    # --- WRITE OUTPUT ---
    data = {
        "metadata": {
            "video_id": vid_id, 
            "title": title, 
            "channel": channel_url,
            "duration": video_duration,
            "timestamp": datetime.datetime.now().isoformat()
        }, 
        "full_text": final_text,
        "raw_text": raw_text,
        "sentry_status": sentry_status,
        "segments": segments,
        "audit": {
            "category": category,
            "metrics": metrics
        }
    }
    
    final_path = safe_filename(title, vid_id, channel_url)
    target_dir = os.path.dirname(final_path)
    os.makedirs(target_dir, exist_ok=True)
    temp_path = final_path + ".tmp"
    
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.rename(temp_path, final_path)
        
        # Update Manifest (Atomic)
        manifest_mgr.update_entry(vid_id, {
            "filepath": final_path,
            "title": title,
            "category": category,
            "duration": video_duration,
            "sentry_status": sentry_status
        })
        
        print(f" -> ✅ Done! {rtf_str} | Audit: {category}")
        log_overall("SUCCESS", vid_id, f"{rtf_str} | {category}")
        
        # Emit training record (only for SUCCESS cats)
        if category == "SUCCESS":
            try:
                training_export.emit_training_record(data, vid_id, channel_url, title, video_duration)
            except Exception as te:
                print(f"    [Training Export] Warning: {te}")
        return True
    except Exception as e:
        print(f" -> ❌ Disk Error: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        log_overall("FAIL", vid_id, str(e))
        return False


def main():
    print("\n=== 🛠️ INTERPRETER & PROCESS HYGIENE ===")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version.split()[0]}")
    verify_model_architecture()
    
    # Acquire exclusive pipeline lock (prevents UMA collision with semantic_formatter)
    lock_fd = acquire_pipeline_lock()
    print("  🔒 Pipeline lock acquired.")
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    # Proactively clean up incomplete files from previous interrupted runs
    cleanup_stale_data()
        
    # Start Heartbeat Thread
    heartbeat_thread = threading.Thread(target=ssd_pulse, daemon=True)
    heartbeat_thread.start()

    # --- CLI ARGS ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Run ALL channels in a single stream")
    parser.add_argument("--queue", type=str, help="Skip channel scan and run specific JSON list of Video IDs")
    args = parser.parse_args()

    if args.queue and os.path.exists(args.queue):
        with open(args.queue, 'r') as f:
            queue_ids = json.load(f)
        print(f"\n=== 🚜 HARVESTER OMEGA: RE-HARVEST QUEUE ({len(queue_ids)} targets) ===")
        my_channels = []
    else:
        my_channels = ALL_CHANNELS[:]
        print(f"\n=== 🚜 HARVESTER UNIFIED (All {len(ALL_CHANNELS)} Channel Tabs) ===")
        queue_ids = []

    mm = manifest_manager.ManifestManager()
    downloaded_ids = mm.get_existing_ids()
    print(f"  📂 Index Loaded: {len(downloaded_ids)} videos known.")
    
    # Randomize the channel list so we don't pound one channel continuously (helps avoid YouTube temp limits)
    shuffled_channels = my_channels[:]
    random.shuffle(shuffled_channels)
    
    processed_files_count = 0
    consecutive_escapes = 0
    
    for channel_url in shuffled_channels:
        print(f"\nScanning: {channel_url}")
        time.sleep(random.uniform(2, 5)) 

        ydl_opts = {
            'extract_flat': True, 
            'quiet': True, 
            'ignoreerrors': True, 
            'playlistend': MAX_VIDEOS_PER_TAB,
            'socket_timeout': 30,
            'retries': 20
        }

        try:
            # type: ignore
            with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
                result = ydl.extract_info(channel_url, download=False)
                
            if not result or 'entries' not in result:
                print("  -> Channel skipped.")
                continue

            videos = result['entries']
            total_in_tab = len(videos)
            print(f"  -> Found {total_in_tab} videos.")
            
            start_time = time.time()
            processed_count = 0 

            for i, entry in enumerate(videos):
                if not entry: continue

                vid_id = entry.get('id')
                title = entry.get('title', 'Unknown Title')
                url = entry.get('url', f"https://www.youtube.com/watch?v={vid_id}")

                if vid_id in downloaded_ids:
                    continue
                
                processed_count += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_count if processed_count > 0 else 0
                remaining_items = total_in_tab - (i + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * remaining_items)))

                # Reactive thermal cooldown (only triggers on HEAVY+ pressure)
                cooldown_if_needed()

                video_duration = entry.get('duration', 0)
                
                # Endurance V2.3: Shorts Contamination Bypass
                if video_duration > 0 and video_duration < 90:
                    print(f"[{i+1}/{total_in_tab}] {title[:30]}... ⏭️ SKIPPED (Shorts < 90s)")
                    downloaded_ids.add(vid_id)
                    log_overall("SKIP_SHORT", vid_id, str(video_duration))
                    continue

                print(f"[{i+1}/{total_in_tab}] {title[:30]}...")
                print(f"  ETA: {eta} | Transcribing...", end="", flush=True)

                params = {
                    "condition_on_previous_text": False, 
                    "no_speech_threshold": 0.6, 
                    "use_vad": True,
                    "title": title,
                    "video_id": vid_id,
                    "model_path": LOCAL_MODEL_PATH
                }
                
                process_single_video(vid_id, url, title, channel_url, video_duration, params, mm)

                # GPU Metal buffer release + Python GC (no sync — APFS atomic rename is sufficient)
                mx.metal.clear_cache()
                gc.collect()
                time.sleep(random.uniform(1, 3))
                
                # Reactive deep flush: only pause when thermally stressed
                processed_files_count += 1
                if processed_files_count % 50 == 0:
                    level = get_thermal_level()
                    if level >= 1:
                        flush_time = 45 if level == 1 else 120
                        print(f"\n  ❄️ [Endurance Sweep] 50 videos done. Thermal level {level}, cooling {flush_time}s...")
                        mx.metal.clear_cache()
                        gc.collect()
                        time.sleep(flush_time)
                        print(f"  ❄️ Flush Complete. Resuming...\n")
                    else:
                        print(f"\n  ✅ [Endurance Check] 50 videos done. Thermals nominal — continuing.\n")

        except Exception as e:
            print(f"  [Channel Error] {e}")

    # Process Queue if present
    if queue_ids:
        print("\n=== EXECUTING RE-HARVEST QUEUE ===")
        total_in_tab = len(queue_ids)
        start_time = time.time()
        for i, vid_id in enumerate(queue_ids):
            url = f"https://www.youtube.com/watch?v={vid_id}"
            
            # Fetch metadata quickly
            with yt_dlp.YoutubeDL({'quiet': True, 'ignoreerrors': True}) as ydl:
                entry = ydl.extract_info(url, download=False)
            if not entry: continue
            
            title = entry.get('title', 'Unknown Title')
            channel_url = entry.get('channel_url', 'Unknown_Channel')
            video_duration = entry.get('duration', 0)
            
            processed_count = i + 1
            elapsed = time.time() - start_time
            avg_time = elapsed / processed_count if processed_count > 0 else 0
            remaining_items = total_in_tab - (i + 1)
            eta = str(datetime.timedelta(seconds=int(avg_time * remaining_items)))
            
            cooldown_if_needed()
            
            print(f"[RE-HARVEST {i+1}/{total_in_tab}] {title[:30]}...")
            print(f"  ETA: {eta} | Transcribing...", end="", flush=True)
            
            params = {
                "condition_on_previous_text": False, 
                "no_speech_threshold": 0.6, 
                "use_vad": True, 
                "title": title,
                "video_id": vid_id,
                "model_path": LOCAL_MODEL_PATH
            }
            
            process_single_video(vid_id, url, title, channel_url, video_duration, params, mm)

            # GPU Metal buffer release (no sync — APFS atomic rename is sufficient)
            if video_duration and video_duration > 2700:
                mx.metal.clear_cache()
            gc.collect()

    print("\n=== HARVEST COMPLETE ===")

if __name__ == "__main__":
    main()