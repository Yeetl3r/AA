import os
import json
import time
import re
import gc
import sys
import fcntl
import mlx.core as mx
from mlx_lm import load, generate
from subprocess import check_output
from config_glossary import get_flat_glossary

LOCK_FILE = "/tmp/harvester_pipeline.lock"

# --- INFRASTRUCTURE CONFIG ---
os.environ["HF_HOME"] = "/Volumes/Storage Drive/AA/hf_cache"
RAW_QUEUE = "/Volumes/Storage Drive/AA/astrologer_data_hybrid/raw_queue/"
COMPLETED_DIR = "/Volumes/Storage Drive/AA/astrologer_data_hybrid/completed/"
MODEL_PATH = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

# Initialize Directories
os.makedirs(RAW_QUEUE, exist_ok=True)
os.makedirs(COMPLETED_DIR, exist_ok=True)

# --- LINGUISTIC SECURITY AUDITOR PROMPT ---
GLOSSARY_STR = get_flat_glossary()
SYSTEM_PROMPT = f"""
REFERENCE TERMINOLOGY: {GLOSSARY_STR}

TASK: You are a Linguistic Security Auditor. Analyze the following Tamil segments. 
Compare the acoustic text against the REFERENCE TERMINOLOGY. If you find a phonetic 
approximation (e.g., 'நவாவம்சம்'), replace it with the corrected term from the glossary 
(e.g., 'நவாம்சம்'). 

OUTPUT ONLY a JSON list of corrections formatted exactly as:
```json
[
  {{"id": 0, "original": "acoustic_text", "corrected": "corrected_text"}},
  {{"id": 1, "original": "acoustic_text", "corrected": "corrected_text"}}
]
```

STRICT FIDELITY: 
1. If a segment is a repetitive hallucination or gibberish, set "corrected": "[REJECTED_LOOP]".
2. Do not paraphrase valid speech.
3. Do not add conversational filler or explanations.
"""

def get_thermal_level():
    """Returns the macOS thermal pressure level (0=nominal, >0=throttled)."""
    try:
        return int(check_output(["sysctl", "-n", "kern.thermal_pressure"]).strip())
    except:
        return 0

def acquire_pipeline_lock():
    """Acquire exclusive file lock to prevent concurrent runs with harvester.
    Returns the lock file descriptor (must be kept open for the duration)."""
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

def process_batch():
    # Acquire exclusive pipeline lock (prevents UMA collision with harvester)
    lock_fd = acquire_pipeline_lock()
    print("  🔒 Pipeline lock acquired.")
    
    print("🧠 Initializing Semantic Formatter (Mistral-7B)...")
    model, tokenizer = load(MODEL_PATH)
    
    while True:
        # Thermal Firewall: Only pause on HEAVY+ pressure (level >= 2)
        # Level 1 (moderate) is normal M4 behavior under sustained load
        thermal = get_thermal_level()
        if thermal >= 2:
            cooldown = 120 if thermal == 2 else 300
            print(f"🔥 Thermal Pressure Level {thermal}. Cooldown {cooldown}s...")
            time.sleep(cooldown)
            continue

        files = [f for f in os.listdir(RAW_QUEUE) if f.endswith('.json')]
        if not files:
            print("💤 Queue empty. Sleeping 60s...")
            time.sleep(60)
            continue

        for file_name in files[:5]: # Process in mini-batches of 5
            file_path = os.path.join(RAW_QUEUE, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"🧐 Auditing: {file_name}")
                segments = data.get('segments', [])
                
                # Chunking logic (Sliding Window of 10 segments to avoid context overflow)
                for i in range(0, len(segments), 10):
                    chunk = segments[i:i+10]
                    prompt_text = "\n".join([f"ID {s['id']}: {s['text']}" for s in chunk if 'text' in s and 'id' in s])
                    if not prompt_text:
                        continue
                        
                    full_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\nINPUT SEGMENTS:\n{prompt_text} [/INST]"
                    
                    response = generate(model, tokenizer, prompt=full_prompt, max_tokens=1000)
                    
                    # Extraction & Mutation
                    try:
                        # Find all code blocks that might contain json
                        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
                        if json_match:
                            corrections = json.loads(json_match.group())
                            for corr in corrections:
                                if 'id' not in corr or 'corrected' not in corr:
                                    continue
                                seg_id = corr['id']
                                # Segment-Indexed Mutator
                                for s in segments:
                                    if s.get('id') == seg_id:
                                        s['text'] = corr['corrected']
                    except Exception as e:
                        print(f"⚠️ Regex/JSON Fail on chunk {i}: {e}")

                # Recompile Full Text for RAG consistency
                data['full_text'] = " ".join([s['text'] for s in segments if s.get('text') != "[REJECTED_LOOP]"])
                # Also drop segments that are completely rejected loop? We'll leave them with target [REJECTED_LOOP] string for now
                data['metadata']['omega_certified'] = True
                
                # Final Channel Direction
                channel_name = data['metadata'].get('channel', 'Unknown_Channel').split('@')[-1].split('/')[0] if '@' in data['metadata'].get('channel', '') else 'Unknown_Channel'
                dest_folder = os.path.join(COMPLETED_DIR, channel_name)
                os.makedirs(dest_folder, exist_ok=True)
                
                with open(os.path.join(dest_folder, file_name), 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                
                os.remove(file_path)
                print(f"✅ Golden Transcript Finalized: {file_name}")

            except Exception as e:
                print(f"❌ Error processing {file_name}: {e}")
            
            # Prevent RAM fragmentation
            mx.metal.clear_cache()
            gc.collect()

if __name__ == "__main__":
    process_batch()
