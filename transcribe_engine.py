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

import json
import urllib.request

# --- OLLAMA SENTRY: Local LLM Transcript Correction (No API Limits) ---

class OllamaSentry:
    """Local LLM transcript correction via Ollama (qwen2.5:3b).
    Replaces GeminiSentry — no API rate limits, GPU-accelerated on Metal."""
    
    CORRECTION_PROMPT = """Role: Expert Tamil Linguist & Astrology Editor.
Task: Perform 'Life Correction' on a raw Whisper ASR transcript.
Context: Video Title: {title}

Rules:
1. CRITICAL: Identify and remove 'phonetic loops' (repetitive syllables or phrases caused by ASR hallucinations).
2. DO NOT change the meaning or the speaker's core intent.
3. Correct grammatical errors and ensure the technical astrology terms flow naturally.
4. If a word is repeated 5+ times consecutively, collapse it to a single instance or remove if it's a hallucination.
5. Return ONLY the corrected Tamil text. No explanations, no formatting.

Raw Transcript:
{text}"""

    def __init__(self, model="qwen2.5:3b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._available = self._verify_model()
    
    def _verify_model(self):
        """Check if Ollama is running and the model is available."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                models = [m.get("name", "") for m in data.get("models", [])]
                # Match model name (ollama uses name:tag format)
                found = any(self.model in m for m in models)
                if found:
                    print(f"[OllamaSentry] ✅ Model '{self.model}' ready on Metal GPU.")
                else:
                    available = ", ".join(models[:5]) if models else "none"
                    print(f"[OllamaSentry] ⚠️ Model '{self.model}' not found. Available: {available}")
                    print(f"[OllamaSentry] Run: ollama pull {self.model}")
                return found
        except Exception as e:
            print(f"[OllamaSentry] ⚠️ Ollama not reachable ({e}). Correction disabled.")
            print(f"[OllamaSentry] Run: ollama serve")
            return False
    
    def correct_transcript(self, text, title=""):
        """Correct Tamil transcript using local Ollama model.
        Returns corrected text, or original text if Ollama unavailable."""
        if not self._available or not text:
            return text
        
        # Truncate extremely long texts to prevent Ollama OOM
        words = text.split()
        if len(words) > 2000:
            text_to_correct = " ".join(words[:2000])
            suffix = " ".join(words[2000:])
        else:
            text_to_correct = text
            suffix = ""
        
        prompt = self.CORRECTION_PROMPT.format(title=title, text=text_to_correct)
        
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,      # Low temp for faithful correction
                "num_predict": 4096,      # Max output tokens
                "num_ctx": 8192,          # Context window
            }
        }).encode("utf-8")
        
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                corrected = result.get("response", "").strip()
                if corrected and len(corrected) > 20:
                    # Reattach any truncated suffix
                    if suffix:
                        corrected = corrected + " " + suffix
                    return corrected
                return text  # Fallback to original if correction too short
        except Exception as e:
            print(f"[OllamaSentry] Correction failed: {e}")
            return text  # Graceful fallback — never block pipeline

_sentry = OllamaSentry()


LOCAL_MODEL_PATH = "/Volumes/Storage Drive/AA/mlx_models/large-v3"
TAMIL_MEDIUM_PATH = "/Volumes/Storage Drive/AA/model_tamil_medium"
GOLDEN_GLOSSARY = "சுபத்துவ பரிவர்த்தனை, லக்னம், திசை, புத்தி, நவாம்சம், ராகு தோஷம், சுக்கிரன், குரு, சனி, கேது"

# --- DENOISE ---

# Common Tamil suffix variations for normalization during dedup
_TAMIL_SUFFIX_PAIRS = [
    ("பற்றிய", "பற்றி"), ("என்ற", "என்று"), ("ஆன", "ஆக"),
    ("இல்", "இல்லை"), ("உள்ள", "உள்ளது"), ("செய்த", "செய்து"),
    ("வரும்", "வர"), ("இருக்கும்", "இருக்கு"), ("பண்ண", "பண்ணு"),
]

def _normalize_tamil(word):
    """Normalize common Tamil suffix variations for dedup comparison."""
    stripped = word.strip(".,?!;:-")
    for long_form, short_form in _TAMIL_SUFFIX_PAIRS:
        if stripped.endswith(long_form):
            return stripped[:-len(long_form)] + short_form
    return stripped

def denoise_loops(text):
    """Multi-level loop removal pipeline for Tamil Whisper output.
    
    Stage 1: Structural syllable degluing (regex — catches character-level repetition)
    Stage 2: Phrase-level n-gram dedup (catches 3-8 word phrase loops)
    Stage 3: Word-level exact matches (max 2 consecutive identical words)
    Stage 4: Tamil suffix normalization dedup (catches near-duplicate words)
    """
    if not text: return text
    
    # Stage 1: Structural Syllable Degluing — catch character/syllable gluing
    text = re.sub(r'(.{2,20}?)\1{2,}', r'\1', text)
    
    # Stage 2: Phrase-Level N-gram Dedup — catch 3-8 word phrase loops
    words = text.split()
    if len(words) < 6:
        # Too short for phrase analysis, skip to word-level
        pass
    else:
        for n in range(8, 2, -1):  # Check longest phrases first
            if len(words) < n * 2:
                continue
            i = 0
            cleaned = []
            while i < len(words):
                if i + n * 2 <= len(words):
                    phrase = words[i:i+n]
                    next_phrase = words[i+n:i+n*2]
                    # Check if phrase repeats
                    if phrase == next_phrase:
                        # Skip duplicates — scan forward past all repeats
                        cleaned.extend(phrase)
                        i += n
                        while i + n <= len(words) and words[i:i+n] == phrase:
                            i += n
                    else:
                        cleaned.append(words[i])
                        i += 1
                else:
                    cleaned.append(words[i])
                    i += 1
            words = cleaned
    
    if not words: return text
    
    # Stage 3: Word-Level Exact Match Dedup (allow max 2 consecutive identical words)
    deduped = [words[0]]
    consecutive_count = 1
    
    for i in range(1, len(words)):
        w_prev = words[i-1].strip(".,?!;:-")
        w_curr = words[i].strip(".,?!;:-")
        
        if w_curr == w_prev:
            consecutive_count += 1
            if consecutive_count <= 2:
                deduped.append(words[i])
        else:
            consecutive_count = 1
            deduped.append(words[i])
    
    # Stage 4: Tamil Suffix Normalization Dedup
    # Catches near-duplicates like "பற்றி பற்றிய பற்றி" → "பற்றி பற்றிய"
    final = [deduped[0]]
    for i in range(1, len(deduped)):
        norm_prev = _normalize_tamil(deduped[i-1])
        norm_curr = _normalize_tamil(deduped[i])
        if norm_curr != norm_prev or norm_curr == "":
            final.append(deduped[i])
        # If normalized forms match, skip (it's a suffix-variant repeat)
    
    return " ".join(final)

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


def fetch_youtube_captions(video_id, preferred_langs=("ta", "ta-IN")):
    """Fetch YouTube captions via yt-dlp for initial_prompt anchoring.
    
    Priority: manual Tamil (ta) → auto-generated Tamil (ta-IN) → None.
    Non-Tamil captions are explicitly rejected to avoid poisoning Whisper's Tamil decoder.
    
    Returns: First ~200 words of caption text, or None if no Tamil captions available.
    """
    import tempfile
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        for lang in preferred_langs:
            # Try manual subs first, then auto-subs
            for sub_flag in (['--write-subs'], ['--write-auto-subs']):
                cmd = [
                    _YTDLP_PATH,
                    *sub_flag,
                    '--sub-lang', lang,
                    '--sub-format', 'vtt/srt/best',
                    '--skip-download',
                    '--quiet', '--no-warnings',
                    '-o', os.path.join(tmpdir, '%(id)s'),
                    url
                ]
                try:
                    subprocess.run(cmd, capture_output=True, timeout=30)
                    
                    # Look for downloaded subtitle files
                    for fname in os.listdir(tmpdir):
                        if fname.endswith(('.vtt', '.srt')):
                            fpath = os.path.join(tmpdir, fname)
                            with open(fpath, 'r', encoding='utf-8') as f:
                                raw = f.read()
                            
                            # Parse VTT/SRT: strip timestamps, headers, blank lines
                            lines = []
                            for line in raw.split('\n'):
                                line = line.strip()
                                # Skip VTT header, timestamps, and sequence numbers
                                if not line or line.startswith('WEBVTT') or line.startswith('NOTE'):
                                    continue
                                if '-->' in line:  # Timestamp line
                                    continue
                                if line.isdigit():  # SRT sequence number
                                    continue
                                # Strip HTML tags
                                clean = re.sub(r'<[^>]+>', '', line)
                                if clean:
                                    lines.append(clean)
                            
                            if lines:
                                full_text = ' '.join(lines)
                                # Return first 200 words as anchor
                                words = full_text.split()[:200]
                                caption_text = ' '.join(words)
                                print(f"    [Captions] Fetched {len(words)} words ({lang}, {'manual' if '--write-subs' in sub_flag else 'auto'})")
                                return caption_text
                except subprocess.TimeoutExpired:
                    continue
                except Exception:
                    continue
    
    return None


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
      - condition_on_previous_text (bool, default False) — disabled to prevent loop cascades
      - temperature (tuple, default (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)) — multi-temp fallback
      - initial_prompt (str, default None) — YouTube captions or glossary terms
      - no_speech_threshold (float, default 0.6) — tightened for M4
      - model_path (str, default LOCAL_MODEL_PATH)
      - video_id (str) — used for caption fetching
    """
    if params is None:
        params = {}
    
    condition_on_previous_text = params.get("condition_on_previous_text", False)
    temperature = params.get("temperature", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    initial_prompt = params.get("initial_prompt", None)
    no_speech_threshold = params.get("no_speech_threshold", 0.6)
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
        "compression_ratio_threshold": 1.8, # Tight threshold to reject looping segments early
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": condition_on_previous_text,
        "verbose": False,
    }
    
    # Logical prompt assembly — Priority: explicit > captions > glossary
    title = params.get("title", "")
    video_id = params.get("video_id", "")
    
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    else:
        # Phase 2: Try YouTube Tamil captions first
        caption_text = None
        if video_id:
            caption_text = fetch_youtube_captions(video_id)
        
        if caption_text:
            # Anchor Whisper with real Tamil text from captions
            kwargs["initial_prompt"] = caption_text
        elif title:
            # Fallback to glossary-based prompt
            dynamic_glossary = get_dynamic_prompt(title)
            kwargs["initial_prompt"] = f"{title}. {dynamic_glossary}"
    
    result = mlx_whisper.transcribe(audio_np, **kwargs)
    return result


def _compute_uwr(text):
    """Compute Unique Word Rate — ratio of unique words to total words.
    Low UWR (< 0.3) indicates heavy repetition / loop contamination."""
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def transcribe_video(url, video_id, params=None):
    """Full pipeline: download -> VAD -> batch-transcribe -> fallback -> OllamaSentry.
    
    Concatenates VAD speech chunks into a single audio buffer and transcribes 
    in one pass to avoid redundant encoder overhead per-chunk.
    
    Phase 3: If UWR < 0.3 on large-v3 output, automatically retries with 
    Tamil-medium model as a domain-specific fallback.
    
    params dict can include all transcribe_audio params plus:
      - use_vad (bool, default True)
      - video_id (str) — for caption fetching
      
    Returns: transcription result dict, sentinel string, or None.
    """
    if params is None:
        params = {}
    
    use_vad = params.get("use_vad", True)
    
    # Ensure video_id is passed through to transcribe_audio for caption fetching
    if video_id and "video_id" not in params:
        params["video_id"] = video_id
    
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
        raw_text = result.get('text', '').strip()
        
        # Phase 3: Tamil-Medium Fallback on low UWR
        model_used = params.get("model_path", LOCAL_MODEL_PATH)
        uwr = _compute_uwr(raw_text) if raw_text else 0.0
        
        if raw_text and uwr < 0.3 and model_used == LOCAL_MODEL_PATH:
            # large-v3 produced heavily repetitive output — try Tamil-medium
            if os.path.isdir(TAMIL_MEDIUM_PATH):
                print(f"    [Fallback] UWR={uwr:.2f} < 0.3 — retrying with Tamil-medium model...")
                fallback_params = dict(params)
                fallback_params["model_path"] = TAMIL_MEDIUM_PATH
                result_fallback = transcribe_audio(merged_audio, fallback_params)
                fallback_text = result_fallback.get('text', '').strip()
                fallback_uwr = _compute_uwr(fallback_text) if fallback_text else 0.0
                
                if fallback_text and fallback_uwr > uwr:
                    print(f"    [Fallback] Tamil-medium UWR={fallback_uwr:.2f} > {uwr:.2f}. Using fallback.")
                    result = result_fallback
                    raw_text = fallback_text
                    model_used = TAMIL_MEDIUM_PATH
                else:
                    print(f"    [Fallback] Tamil-medium didn't improve (UWR={fallback_uwr:.2f}). Keeping large-v3.")
            else:
                print(f"    [Fallback] Tamil-medium model not found at {TAMIL_MEDIUM_PATH}. Skipping.")
        
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
        
        # Stage 2: Ollama 'Life Correction' (local, no API limits)
        if not raw_text:
            return None
            
        print(f"    [OllamaSentry] Running Life Correction...")
        corrected_text = _sentry.correct_transcript(raw_text, title=params.get("title", ""))
        
        return {
            'text': corrected_text if corrected_text else raw_text,
            'raw_text': raw_text, # Keep raw for debugging/resumability
            'segments': all_segments,
            'model_used': os.path.basename(model_used),
            'uwr': round(uwr, 3),
            'sentry_status': "CORRECTED" if corrected_text and corrected_text != raw_text else "RAW_ONLY"
        }
    except Exception as e:
        print(f"    [Transcribe Error] {e}")
        return None
