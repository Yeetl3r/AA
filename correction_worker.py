import os
import json
import time
import re
import datetime
import fcntl
import sys
import gc
import urllib.request
from typing import Optional, List, Dict, Any

# Local project imports
import manifest_manager
import validator_v2
import training_export
import transcribe_engine # For terminology anchoring and OllamaSentry
from config_keys import GEMINI_API_KEYS

# SDK Imports
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    Cerebras = None

# --- CONFIGURATION ---
OUTPUT_FOLDER = "astrologer_data_hybrid"
CORRECTION_QUEUE = os.path.join(OUTPUT_FOLDER, "correction_queue")
CORRECTION_DONE = os.path.join(OUTPUT_FOLDER, "correction_queue_done")
COMPLETED_DIR = os.path.join(OUTPUT_FOLDER, "completed")
LOCK_FILE = "/tmp/correction_worker.lock"
HEARTBEAT_FILE = "correction_worker_heartbeat.log"

os.makedirs(CORRECTION_QUEUE, exist_ok=True)
os.makedirs(CORRECTION_DONE, exist_ok=True)
os.makedirs(COMPLETED_DIR, exist_ok=True)

# --- PROMPT ---
CORRECTION_PROMPT = """Role: Expert Tamil Linguist & Astrology Editor.
Task: Perform 'Life Correction' on a raw Whisper ASR transcript.
Context: Video Title: {title}

Rules:
1. CRITICAL: Identify and remove 'phonetic loops' (repetitive syllables or phrases caused by ASR hallucinations).
2. DO NOT change the meaning or the speaker's core intent.
3. Correct grammatical errors and ensure the technical astrology terms flow naturally.
4. If a word is repeated 5+ times consecutively, collapse it to a single instance or remove if it's a hallucination.
5. If the text is repeating nonsense or consists entirely of a hallucination loop, return ONLY the word [DISCARD].
6. Return ONLY the corrected Tamil text. No explanations, no formatting.

Raw Transcript:
{text}"""

class CorrectionWorker:
    def __init__(self):
        self.mm = manifest_manager.ManifestManager()
        self.gemini_keys = GEMINI_API_KEYS
        self.current_gemini_key_idx = 0
        
        self.groq_key = os.environ.get("GROQ_API_KEY")
        self.cerebras_key = os.environ.get("CEREBRAS_API_KEY")
        
        # Initialize clients lazily if keys are present
        self._init_clients()
        
        # Local Ollama Sentry
        self.ollama = transcribe_engine.OllamaSentry()

    def _init_clients(self):
        # Allow API key from env or config_keys
        self.google_key = os.environ.get("GOOGLE_API_KEY") or (self.gemini_keys[self.current_gemini_key_idx] if self.gemini_keys else None)
        
        if genai and self.google_key:
            genai.configure(api_key=self.google_key)
            # Using gemini-1.5-flash-latest for stability
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        else:
            self.gemini_model = None
            
        if Groq and self.groq_key:
            self.groq_client = Groq(api_key=self.groq_key)
        else:
            self.groq_client = None
            
        if Cerebras and self.cerebras_key:
            self.cerebras_client = Cerebras(api_key=self.cerebras_key)
        else:
            self.cerebras_client = None

    def _rotate_gemini_key(self):
        if len(self.gemini_keys) > 1:
            self.current_gemini_key_idx = (self.current_gemini_key_idx + 1) % len(self.gemini_keys)
            genai.configure(api_key=self.gemini_keys[self.current_gemini_key_idx])
            print(f"  🔄 Rotated to Gemini Key #{self.current_gemini_key_idx}")

    def call_gemini(self, prompt: str) -> Optional[str]:
        if not self.gemini_model: return None
        try:
            response = self.gemini_model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"  ⚠️ Gemini Error: {e}")
            if "429" in str(e):
                self._rotate_gemini_key()
        return None

    def call_groq(self, prompt: str, model: str = "llama-3.3-70b-versatile") -> Optional[str]:
        if not self.groq_client: return None
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.1,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️ Groq ({model}) Error: {e}")
        return None

    def call_cerebras(self, prompt: str) -> Optional[str]:
        if not self.cerebras_client: return None
        try:
            response = self.cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3.1-70b",
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️ Cerebras Error: {e}")
        return None

    def run_cascade(self, text: str, title: str) -> str:
        prompt = CORRECTION_PROMPT.format(title=title, text=text)
        
        # 1. Gemini 1.5 Flash
        print("    -> Trying Gemini 1.5 Flash...", end="", flush=True)
        res = self.call_gemini(prompt)
        if res: 
            print(" ✅")
            return res
            
        # 2. Groq Llama 3.3 70b
        print(" ❌\n    -> Trying Groq Llama 3.3 70b...", end="", flush=True)
        res = self.call_groq(prompt, "llama-3.3-70b-versatile")
        if res:
            print(" ✅")
            return res
            
        # 3. Cerebras Llama 3.1 70b
        print(" ❌\n    -> Trying Cerebras Llama 3.1 70b...", end="", flush=True)
        res = self.call_cerebras(prompt)
        if res:
            print(" ✅")
            return res
            
        # 4. Groq Llama 3.1 8b
        print(" ❌\n    -> Trying Groq Llama 3.1 8b...", end="", flush=True)
        res = self.call_groq(prompt, "llama-3.1-8b-instant")
        if res:
            print(" ✅")
            return res
            
        # 5. Ollama Qwen 2.5 3b (Last Resort)
        print(" ❌\n    -> Trying Local Ollama (Last Resort)...", end="", flush=True)
        res = self.ollama.correct_transcript(text, title)
        if res:
            print(" ✅")
            return res
            
        print(" ❌ (All providers failed)")
        return text

    def process_file(self, filename: str):
        filepath = os.path.join(CORRECTION_QUEUE, filename)
        if not os.path.exists(filepath): return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")
            return

        vid_id = data.get("metadata", {}).get("video_id")
        title = data.get("metadata", {}).get("title", "")
        raw_text = data.get("raw_text", data.get("full_text", ""))
        
        print(f"🧐 Processing: {title[:40]} ({vid_id})")
        
        # Run Correction Cascade
        corrected_text = self.run_cascade(raw_text, title)
        
        if corrected_text == "[DISCARD]":
            print(f"  🛑 LLM signaled [DISCARD]. Marking as FAIL.")
            self.mm.update_entry(vid_id, {"category": "FAIL_LLM_DISCARD", "sentry_status": "DISCARDED"})
            os.rename(filepath, os.path.join(CORRECTION_DONE, f"REJECTED_{filename}"))
            return

        # Final Terminology Anchoring
        corrected_text = transcribe_engine.anchor_terminology(corrected_text)
        
        # Update data dict
        data["full_text"] = corrected_text
        data["metadata"]["correction_timestamp"] = datetime.datetime.now().isoformat()
        data["metadata"]["corrected_by_worker"] = True
        
        # Validate result
        category, metrics = validator_v2.validate_transcription(data)
        data["metadata"]["category"] = category
        data["metadata"]["metrics"] = metrics
        
        # Target Path
        channel_name = transcribe_engine.clean_channel_name(data["metadata"].get("channel", "Unknown"))
        dest_folder = os.path.join(COMPLETED_DIR, channel_name)
        os.makedirs(dest_folder, exist_ok=True)
        final_path = os.path.join(dest_folder, filename)
        
        # Save Finalized JSON
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            
        # Update Manifest
        self.mm.update_entry(vid_id, {
            "category": category,
            "sentry_status": "SUCCESS" if category == "SUCCESS" else "REVIEW_REQUIRED",
            "global_uwr": metrics.get("global_uwr"),
            "corrected": True
        })
        
        # Emit Training Record if Success
        if category == "SUCCESS":
            training_export.emit_training_record(
                data, vid_id, 
                data["metadata"].get("channel", ""), 
                title, 
                data["metadata"].get("duration", 0)
            )
            print(f"  ✅ SUCCESS! Exported to training set.")
        else:
            print(f"  ⚠️ Final Category: {category}. Review required.")
            
        # Move queue file to done
        os.rename(filepath, os.path.join(CORRECTION_DONE, filename))

    def run(self):
        print("\n=== 🦾 ZENITH-OMEGA CORRECTION WORKER ACTIVE ===")
        print(f"Queue: {CORRECTION_QUEUE}")
        
        while True:
            # Heartbeat
            with open(HEARTBEAT_FILE, "w") as f:
                f.write(str(time.time()))
                
            files = [f for f in os.listdir(CORRECTION_QUEUE) if f.endswith(".json") and not f.startswith(".")]
            if not files:
                print(".", end="", flush=True)
                time.sleep(30)
                continue
            
            print(f"\nFound {len(files)} files in queue.")
            for f in files:
                self.process_file(f)
                # Small pause to avoid hitting API rate limits too hard
                time.sleep(2)
                # GC and Metal clear if we used Ollama
                mx_clear()
                gc.collect()

def mx_clear():
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except:
        pass

def acquire_lock():
    try:
        lock_fd = open(LOCK_FILE, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.write(f"{os.getpid()}\n")
        lock_fd.flush()
        return lock_fd
    except (IOError, OSError):
        print("❌ Another correction_worker is already running.")
        sys.exit(1)

if __name__ == "__main__":
    # Optional: ensure we are using the venv
    lock_fd = acquire_lock()
    worker = CorrectionWorker()
    try:
        worker.run()
    except KeyboardInterrupt:
        print("\n👋 Worker stopping.")
