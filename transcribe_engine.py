"""
transcribe_engine.py — Zenith-Omega Shared Transcription Core
Extracted from harvester.py to allow both the main harvester and the 
Zenith-Omega multi-agent pipeline to call transcription with different parameters.
"""

import os
# Route Heavy HF Downloads strictly to External Drive
os.environ["HF_HOME"] = "/Volumes/Storage Drive/AA/hf_cache"
os.environ["HF_HUB_CACHE"] = "/Volumes/Storage Drive/AA/hf_cache/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/Volumes/Storage Drive/AA/hf_cache/hub"

import re
import sys
import struct
import subprocess
import numpy as np
import mlx_whisper

# --- GLOBAL VAD INITIALIZATION (lightweight webrtcvad — no PyTorch) ---
try:
    import webrtcvad
    _vad = webrtcvad.Vad(2)  # Aggressiveness: 0 (least) to 3 (most)
    print("[Engine] WebRTC-VAD Active (no PyTorch overhead).")
except ImportError:
    print("[Engine] WARNING: webrtcvad not installed. Run: pip install webrtcvad-wheels")
    print("[Engine] Falling back to full-audio mode (no VAD).")
    _vad = None

import google.generativeai as genai
try:
    from config_keys import GEMINI_API_KEYS
except ImportError:
    GEMINI_API_KEYS = []

class GeminiSentry:
    """Manages Gemini API for 'Life Correction' of transcripts with key rotation."""
    def __init__(self):
        self.keys = GEMINI_API_KEYS
        self.current_key_idx = 0
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        if not self.keys:
            print("[Sentry] WARNING: No Gemini API keys found in config_keys.py")
            return
        key = self.keys[self.current_key_idx]
        genai.configure(api_key=key)
        # Use Gemini 2.0 Flash for superior Tamil reasoning and speed
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        print(f"[Sentry] Initialized with Key {self.current_key_idx + 1}")

    def rotate_key(self):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.keys)
        self._initialize_model()

    def correct_transcript(self, text, title="", retry_count=0):
        """Perform 'Life Correction' on raw transcript using Gemini with recursive retry logic."""
        if not self.model or not text or retry_count >= len(self.keys):
            return text
            
        prompt = f"""
        Role: Expert Tamil Linguist & Astrology Editor.
        Task: Perform 'Life Correction' on a raw Whisper ASR transcript.
        Context: Video Title: {title}
        
        Rules:
        1. CRITICAL: Identify and remove 'phonetic loops' (repetitive syllables or phrases caused by ASR hallucinations).
        2. DO NOT change the meaning or the speaker's core intent.
        3. Correct grammatical errors and ensure the technical astrology terms flow naturally.
        4. If a word is repeated 10+ times (e.g. 'நன்றி நன்றி...'), collapse it to a single instance or remove if it's a hallucination.
        5. Return ONLY the corrected Tamil text.
        
        Raw Transcript:
        {text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            print(f"[Sentry] API Error: {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"[Sentry] Quota reached for Key {self.current_key_idx + 1}. Rotating...")
                self.rotate_key()
                # Recursive retry with the next key
                return self.correct_transcript(text, title, retry_count + 1)
            return None # Other errors halt for resumability logic
        return text

_sentry = GeminiSentry()


LOCAL_MODEL_PATH = "/Volumes/Storage Drive/AA/mlx_models/large-v3"
GOLDEN_GLOSSARY = "சுபத்துவ பரிவர்த்தனை, லக்னம், திசை, புத்தி, நவாம்சம், ராகு தோஷம், சுக்கிரன், குரு, சனி, கேது"

# --- DENOISE ---

def denoise_loops(text):
    """Remove phonetic loops and character-level gluing from Whisper output."""
    if not text: return text
    
    # 1. Advanced Structural Syllable Degluing via Regex
    text = re.sub(r'(.{2,20}?)\1{2,}', r'\1', text)
    
    # 2. Word-Level Exact Matches
    words = text.split()
    if not words: return text
    deduped = [words[0]]
    consecutive_count = 1
    
    for i in range(1, len(words)):
        w_prev = words[i-1].strip(".,?!;")
        w_curr = words[i].strip(".,?!;")
        
        if w_curr == w_prev:
            consecutive_count += 1
            if consecutive_count <= 2:
                deduped.append(words[i])
        else:
            consecutive_count = 1
            deduped.append(words[i])
            
    return " ".join(deduped)

# --- CORE TRANSCRIPTION ---

def _resolve_tool(name):
    """Find the full path for a CLI tool, checking venv bin and common locations."""
    import shutil
    # 1. Check if it's in the same venv as the running Python
    venv_bin = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.isfile(venv_bin):
        return venv_bin
    # 2. Check system PATH
    found = shutil.which(name)
    if found:
        return found
    # 3. Check common Homebrew / system locations
    for prefix in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]:
        candidate = os.path.join(prefix, name)
        if os.path.isfile(candidate):
            return candidate
    return name  # Fallback to bare name

_YTDLP_PATH = _resolve_tool("yt-dlp")
_FFMPEG_PATH = _resolve_tool("ffmpeg")

def download_audio(url):
    """Download audio from YouTube and return raw float32 numpy array at 16kHz.
    Returns: numpy array, or string sentinel ("MEMBERS_ONLY", "HTTP_429"), or None on failure.
    """
    yt_cmd = [
        _YTDLP_PATH, 
        '-f', 'bestaudio/best', 
        '--quiet', '--no-warnings',
        '--abort-on-error',
        '-o', '-', 
        url
    ]
    ffmpeg_cmd = [
        _FFMPEG_PATH,
        '-i', 'pipe:0',
        '-f', 'f32le',
        '-ac', '1',
        '-ar', '16000',
        'pipe:1'
    ]
    
    yt_proc = None
    ffmpeg_proc = None
    try:
        yt_proc = subprocess.Popen(yt_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=yt_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        
        # Release yt-dlp's stdout FD from this process (ffmpeg owns it now)
        if yt_proc.stdout is not None:
            yt_proc.stdout.close()
            
        audio_bytes, _ = ffmpeg_proc.communicate()
        yt_proc.wait()
        
        # Read and close stderr to prevent FD leak
        yt_err = b""
        if yt_proc.stderr:
            yt_err = yt_proc.stderr.read()
            yt_proc.stderr.close()
        
        if yt_proc.returncode != 0:
            if b"members-only" in yt_err.lower():
                return "MEMBERS_ONLY"
            if b"http error 429" in yt_err.lower() or b"too many requests" in yt_err.lower():
                return "HTTP_429"
            return None
            
        if ffmpeg_proc.returncode != 0 or not audio_bytes:
            return None
            
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        if len(audio_np) == 0:
            return None
            
        # Normalize to [-1.0, 1.0]
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val
            
        return audio_np
        
    except Exception as e:
        print(f"    [Download Error] {e}")
        return None
    finally:
        # Defensive FD cleanup for long-running processes
        for proc in [yt_proc, ffmpeg_proc]:
            if proc is None:
                continue
            for pipe in [proc.stdout, proc.stderr, proc.stdin]:
                if pipe:
                    try: pipe.close()
                    except: pass


def apply_vad(audio_np, sample_rate=16000):
    """Detect speech chunks using WebRTC VAD (lightweight, no PyTorch).
    Returns a list of timestamp dicts (e.g., [{'start': 304, 'end': 12000}]), or None if no speech.
    """
    if _vad is None:
        # Fallback: treat entire audio as a single speech chunk
        return [{'start': 0, 'end': len(audio_np)}]
    
    # WebRTC-VAD requires 16-bit PCM; convert from float32
    pcm_16 = (audio_np * 32767).astype(np.int16)
    frame_duration_ms = 30  # WebRTC supports 10, 20, 30ms frames
    frame_size = int(sample_rate * frame_duration_ms / 1000)  # 480 samples per frame
    
    # Scan all frames for speech/non-speech
    is_speech = []
    for i in range(0, len(pcm_16) - frame_size + 1, frame_size):
        frame_bytes = pcm_16[i:i + frame_size].tobytes()
        try:
            is_speech.append(_vad.is_speech(frame_bytes, sample_rate))
        except Exception:
            is_speech.append(False)
    
    if not any(is_speech):
        return None
    
    # Merge consecutive speech frames into chunks with min_silence_duration of 500ms
    min_silence_frames = int(500 / frame_duration_ms)  # ~17 frames of silence to split
    chunks = []
    in_speech = False
    start_frame = 0
    silence_count = 0
    
    for idx, speech in enumerate(is_speech):
        if speech:
            if not in_speech:
                start_frame = idx
                in_speech = True
            silence_count = 0
        else:
            if in_speech:
                silence_count += 1
                if silence_count >= min_silence_frames:
                    end_frame = idx - silence_count + 1
                    chunks.append({
                        'start': start_frame * frame_size,
                        'end': end_frame * frame_size
                    })
                    in_speech = False
                    silence_count = 0
    
    # Flush final chunk
    if in_speech:
        chunks.append({
            'start': start_frame * frame_size,
            'end': len(audio_np)
        })
    
    return chunks if chunks else None


try:
    import config_glossary
    GLOSSARY_DICT = config_glossary.GOLDEN_GLOSSARY
except ImportError:
    GLOSSARY_DICT = {}

def get_dynamic_prompt(title, fallback_glossary=""):
    """Filter glossary terms based on title to keep prompt under Whisper context limits."""
    if not GLOSSARY_DICT or not title:
        return fallback_glossary
    
    matched_terms = []
    title_lower = title.lower()
    
    # Simple word-match logic
    for category, terms in GLOSSARY_DICT.items():
        # If title matches category keywords, include all terms in that category
        if any(cat_key in title_lower for cat_key in category.split('_')):
            matched_terms.extend(terms)
        else:
            # Otherwise just check individual terms
            for term in terms:
                # Extract Tamil or English part for matching
                clean_term = re.sub(r'\(.*?\)', '', term).strip()
                if clean_term in title or (len(clean_term) > 3 and clean_term in title_lower):
                    matched_terms.append(term)
    
    if not matched_terms:
        # Fallback to a few core terms if no matches
        return "சுபத்துவம், லக்னம், நவகிரகம், தசா புக்தி"
        
    return ", ".join(list(set(matched_terms))[:30]) # Cap at 30 terms for reliability

def transcribe_audio(audio_np, params=None):
    """Run mlx-whisper on a pre-processed numpy audio array.
    
    params dict can override:
      - condition_on_previous_text (bool, default True)
      - temperature (tuple, default (0.0,))
      - initial_prompt (str, default None)
      - no_speech_threshold (float, default 0.8)
      - model_path (str, default LOCAL_MODEL_PATH)
    """
    if params is None:
        params = {}
    
    condition_on_previous_text = params.get("condition_on_previous_text", True)
    temperature = params.get("temperature", (0.0,))
    initial_prompt = params.get("initial_prompt", None)
    no_speech_threshold = params.get("no_speech_threshold", 0.8)
    model_path = params.get("model_path", LOCAL_MODEL_PATH)
    beam_size = params.get("beam_size", 1) # Set to 1 as mlx-whisper doesn't support beam search yet
    if beam_size > 1:
        print(f"  ⚠️ Warning: beam_size > 1 requested, but mlx-whisper only supports greedy decoding. Forcing beam_size=1.")
        beam_size = 1
    kwargs = {
        "path_or_hf_repo": model_path,
        "language": "ta",
        "task": "transcribe",
        "temperature": temperature,
        "compression_ratio_threshold": 2.2, # Lowered from 3.8 to break loops earlier
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": condition_on_previous_text,
        "verbose": False,
    }
    
    # Logical prompt assembly
    title = params.get("title", "")
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    elif title:
        dynamic_glossary = get_dynamic_prompt(title)
        kwargs["initial_prompt"] = f"{title}. {dynamic_glossary}"
    
    result = mlx_whisper.transcribe(audio_np, **kwargs)
    return result


def transcribe_video(url, video_id, params=None):
    """Full pipeline: download -> VAD -> batch-transcribe.
    
    Concatenates VAD speech chunks into a single audio buffer and transcribes 
    in one pass to avoid redundant encoder overhead per-chunk.
    
    params dict can include all transcribe_audio params plus:
      - use_vad (bool, default True)
      
    Returns: transcription result dict, sentinel string, or None.
    """
    if params is None:
        params = {}
    
    use_vad = params.get("use_vad", True)
    
    # Step 1: Download
    audio = download_audio(url)
    if audio is None or isinstance(audio, str):
        return audio  # Pass sentinel or None through
    
    # Step 2: VAD Chunking
    vad_chunks = [{'start': 0, 'end': len(audio)}]
    if use_vad:
        vad_chunks = apply_vad(audio)
        if not vad_chunks:
            print("    [VAD] Zero human speech detected. Aborting MLX inference.")
            return None
    
    # Step 3: Concatenate speech chunks with 400ms padding, transcribe in single pass
    try:
        pad_frames = int(16000 * 0.4)  # 400ms contextual padding
        
        # Build a merged audio buffer from all speech chunks
        merged_audio_parts = []
        chunk_offsets = []  # Track where each original chunk starts in merged buffer
        merged_pos = 0
        
        for chunk in vad_chunks:
            vad_start = chunk['start']
            vad_end = chunk['end']
            
            slice_start = max(0, vad_start - pad_frames)
            slice_end = min(len(audio), vad_end + pad_frames)
            chunk_audio = audio[slice_start:slice_end]
            
            chunk_offsets.append({
                'original_offset': slice_start / 16000.0,
                'merged_start': merged_pos / 16000.0,
                'length': len(chunk_audio)
            })
            
            merged_audio_parts.append(chunk_audio)
            merged_pos += len(chunk_audio)
        
        merged_audio = np.concatenate(merged_audio_parts) if merged_audio_parts else audio
        
        # Single-pass transcription on merged audio (one encoder pass instead of N)
        result = transcribe_audio(merged_audio, params)
        
        # Re-map segment timestamps back to original audio timeline
        all_segments = []
        for seg in result.get('segments', []):
            merged_time = seg['start']
            # Find which chunk this segment belongs to and remap
            for ci, co in enumerate(chunk_offsets):
                chunk_end_time = co['merged_start'] + (co['length'] / 16000.0)
                if merged_time < chunk_end_time or ci == len(chunk_offsets) - 1:
                    # Offset: subtract merged_start, add original_offset
                    time_delta = co['original_offset'] - co['merged_start']
                    seg['start'] = round(seg['start'] + time_delta, 3)
                    seg['end'] = round(seg['end'] + time_delta, 3)
                    break
            all_segments.append(seg)
        
        # Stage 2: Gemini 'Life Correction'
        raw_text = result.get('text', '').strip()
        if not raw_text:
            return None
            
        print(f"    [Omega-Repair] Running Gemini Life Correction...")
        corrected_text = _sentry.correct_transcript(raw_text, title=params.get("title", ""))
        
        return {
            'text': corrected_text if corrected_text else raw_text,
            'raw_text': raw_text, # Keep raw for debugging/resumability
            'segments': all_segments,
            'sentry_status': "CORRECTED" if corrected_text else "RAW_ONLY"
        }
    except Exception as e:
        print(f"    [Transcribe Error] {e}")
        return None
