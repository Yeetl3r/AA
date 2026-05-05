"""
caption_only_harvester.py — Targeted YouTube Caption Extraction Utility
Part of the AA (Astrology ASR) Pipeline.

PURPOSE
-------
Sweeps all target channels and extracts Tamil captions for every video that has them.
Saves structured JSON (with segments + timestamps) for later cleanup and training.
Videos without captions remain PENDING in the manifest for harvester.py to transcribe.

OUTPUT FORMAT
-------------
Each successful caption is saved as:
  astrologer_data_hybrid/captions/<vid_id>.json

Schema:
  {
    "metadata": {
      "video_id", "title", "url", "channel",
      "duration_s", "sub_type", "lang",
      "word_count", "coverage_pct", "tamil_ratio",
      "timestamp"
    },
    "full_text": "...",         ← deduplicated, HTML-cleaned plain text
    "segments": [               ← timestamps preserved for ASR alignment
      {"start": 0.0, "end": 4.2, "text": "..."},
      ...
    ]
  }

MANIFEST STATUS
---------------
  CAPTION_RAW      → captions saved, pending cleanup pass
  (no entry)       → no captions found, stays PENDING for harvester.py

IMPORTANT: Do NOT use SUCCESS here. harvester.py uses SUCCESS as its skip signal.
Using SUCCESS would permanently block Whisper transcription of low-quality caption videos.

KNOWN ISSUES IN SOURCE CAPTIONS (handled by this script)
---------------------------------------------------------
1. VTT rolling stutter     → deduplicated before saving
2. HTML entity leaks       → html.unescape() applied
3. Inline player tags      → stripped with regex
4. Coverage dropout        → coverage gate (min 40% of expected word count)
5. Non-Tamil captions      → Tamil script ratio gate (min 30%)
6. Loopy auto-captions     → unique word ratio gate (min 25%)

Issues handled by the CLEANUP pass (not this script):
  - Meta-talk / sponsor reads
  - Phonetic vocabulary corruption
  - Punctuation restoration
  - Domain vocabulary density check

RUN ORDER
---------
1. python3 caption_only_harvester.py    ← this script (bulk caption sweep)
2. python3 caption_cleanup.py           ← cleanup/validation pass (build later)
3. python3 harvester.py                 ← Whisper transcription for remaining videos
4. python3 correction_worker.py         ← async LLM correction
"""

import os
import re
import sys
import json
import argparse
import time
import html
import random
import datetime
import tempfile
import shutil
import subprocess
import unicodedata
import yt_dlp

# ── PIPELINE IMPORTS ──────────────────────────────────────────────────────────
from manifest_manager import ManifestManager

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
COOKIES_FILE = os.path.join(os.path.dirname(__file__), "cookies.txt")

def get_cookie_opts():
    """Return cookie config for yt-dlp. Prefers cookies.txt over browser."""
    if os.path.exists(COOKIES_FILE):
        return {'cookiefile': COOKIES_FILE}
    return {'cookiesfrombrowser': ('safari',)}  # fallback

ALL_CHANNELS = [
    "https://www.youtube.com/@adityagurujiastrologerchennai/videos",
    "https://www.youtube.com/@adityagurujiastrologerchennai/streams",
    "https://www.youtube.com/@shrimahalakshmi-premium5868/videos",
    "https://www.youtube.com/@shrimahalakshmi-premium5868/streams",
    "https://www.youtube.com/@SriMahalakshmiJothidam/videos",
    "https://www.youtube.com/@SriMahalakshmiJothidam/streams",
    "https://www.youtube.com/@AstroSriramJI/videos",
    "https://www.youtube.com/@AstroSriramJI/streams",
]

OUTPUT_FOLDER  = "astrologer_data_hybrid"
CAPTION_FOLDER = os.path.join(OUTPUT_FOLDER, "captions")
MISSING_LOG    = os.path.join(OUTPUT_FOLDER, "missing_captions.log")
HEARTBEAT_LOG  = os.path.join(OUTPUT_FOLDER, "caption_harvester_heartbeat.log")
STATS_LOG      = os.path.join(OUTPUT_FOLDER, "caption_harvest_stats.json")

os.makedirs(CAPTION_FOLDER, exist_ok=True)

# Quality gate thresholds — tuned from sample analysis of 5 astrology channel videos
COVERAGE_THRESHOLD   = 0.40   # min fraction of expected word count (based on duration)
TAMIL_RATIO_THRESHOLD = 0.30  # min fraction of alphabetic chars that are Tamil Unicode
UNIQUE_RATIO_THRESHOLD = 0.08 # min unique word ratio (catches loopy auto-captions)
# NOTE: Tamil astrology lectures naturally score 0.10-0.25 because domain terms
# like சனி, ராகு, லக்னம் repeat hundreds of times. 0.10 was incorrectly rejecting
# borderline real content. True VTT loops score below 0.07 — 0.08 is the safe floor.
MIN_WORD_COUNT        = 100   # hard minimum regardless of duration
TAMIL_WPM             = 110   # approximate Tamil speaking rate (words per minute)

TAMIL_UNICODE_RANGE = (0x0B80, 0x0BFF)

# ── THERMAL MANAGEMENT (M4 Air tuned — exponential backoff, reacts at level 1) ──

def get_thermal_level():
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "kern.thermal_pressure"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return int(out)
    except Exception:
        return 0

def cooldown_if_needed():
    """
    Exponential backoff thermal guard tuned for MacBook Air M4 (fanless).
    Reacts at level 1 (not level 2) — fanless machines cannot buffer heat.
    """
    backoff = 20
    while True:
        level = get_thermal_level()
        if level == 0:
            return
        sleep_time = min(backoff, 180)
        print(f"  🌡️  Thermal Level {level}. Cooling {sleep_time}s...")
        log_event("THERMAL", "system", f"Level {level}, sleep {sleep_time}s")
        time.sleep(sleep_time)
        backoff *= 2

# ── LOGGING ───────────────────────────────────────────────────────────────────

def log_event(status, vid_id, extra=""):
    os.makedirs(os.path.dirname(HEARTBEAT_LOG), exist_ok=True)
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{status}] | {ts} | {vid_id} | {extra}\n"
    with open(HEARTBEAT_LOG, "a", encoding="utf-8") as f:
        f.write(msg)

def log_missing(vid_id, title, reason="NO_CAPTIONS"):
    os.makedirs(os.path.dirname(MISSING_LOG), exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(MISSING_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] [{reason}] {vid_id} | {title}\n")

# ── VTT PARSING AND CLEANING ──────────────────────────────────────────────────

def ts_to_seconds(ts_str):
    """Convert VTT timestamp (HH:MM:SS.mmm) to float seconds."""
    ts_str = ts_str.replace(',', '.')
    parts = ts_str.split(':')
    try:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0

def parse_vtt_to_segments(vtt_text):
    """
    Parse raw VTT text into a list of segment dicts with start, end, text.
    Handles: WEBVTT headers, timestamp lines, inline player tags, HTML entities.
    Does NOT deduplicate yet — dedup is a separate pass.
    """
    segments = []
    blocks   = re.split(r'\n{2,}', vtt_text.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue

        # Skip header/metadata blocks
        if any(kw in lines[0] for kw in ('WEBVTT', 'Kind:', 'Language:', 'NOTE')):
            continue

        # Find the timestamp line
        time_line  = None
        text_lines = []
        for line in lines:
            if '-->' in line:
                time_line = line
            elif time_line is not None and line.strip():
                text_lines.append(line.strip())

        if not time_line or not text_lines:
            continue

        # Parse timestamps
        match = re.match(
            r'(\d{2}:\d{2}:\d{2}[\.,]\d+)\s+-->\s+(\d{2}:\d{2}:\d{2}[\.,]\d+)',
            time_line
        )
        if not match:
            continue

        start = ts_to_seconds(match.group(1))
        end   = ts_to_seconds(match.group(2))

        # Clean each text line
        raw_text = ' '.join(text_lines)

        # 1. Decode HTML entities (&gt;&gt; → >>, &amp; → &, etc.)
        raw_text = html.unescape(raw_text)

        # 2. Remove inline VTT player tags (<00:00:00.000>, <c>, </c>, <b>, etc.)
        raw_text = re.sub(r'<[^>]+>', '', raw_text)

        # 3. Remove speaker tags (>> text, [SPEAKER]:)
        raw_text = re.sub(r'^>+\s*', '', raw_text)
        raw_text = re.sub(r'^\[[^\]]+\]:\s*', '', raw_text)

        # 4. Collapse whitespace
        raw_text = re.sub(r'\s+', ' ', raw_text).strip()

        if raw_text:
            segments.append({
                'start': round(start, 3),
                'end':   round(end,   3),
                'text':  raw_text
            })

    return segments

def deduplicate_segments(segments):
    """
    Remove VTT rolling stutter: consecutive segments with identical or near-identical text.

    The rolling window overlap in YouTube VTT causes every sentence to appear 2-4 times
    across consecutive blocks. This pass collapses each unique sentence to one occurrence
    while preserving the timestamp of its FIRST appearance.

    Strategy:
      - Normalise text (lowercase, strip punctuation) for comparison
      - Keep a segment only if its normalised text hasn't been seen in the last 5 segments
      - Timestamp from first occurrence is preserved
    """
    if not segments:
        return segments

    def normalise(text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    deduped  = []
    seen_window = []  # normalised text of last N segments

    for seg in segments:
        norm = normalise(seg['text'])
        if not norm:
            continue

        # Check if this text appeared in recent window
        if norm not in seen_window:
            deduped.append(seg)
            seen_window.append(norm)
            if len(seen_window) > 8:  # sliding window of 8
                seen_window.pop(0)

    return deduped

def segments_to_full_text(segments):
    """Join deduplicated segments into a single clean string."""
    return ' '.join(s['text'] for s in segments if s.get('text', '').strip())

# ── QUALITY GATES ─────────────────────────────────────────────────────────────

def compute_tamil_ratio(text):
    """Fraction of alphabetic characters that fall in Tamil Unicode block."""
    alpha = [c for c in text if unicodedata.category(c).startswith('L')]
    if not alpha:
        return 0.0
    tamil = sum(1 for c in alpha if TAMIL_UNICODE_RANGE[0] <= ord(c) <= TAMIL_UNICODE_RANGE[1])
    return tamil / len(alpha)

def validate_caption_quality(segments, duration_s, vid_id):
    """
    Multi-gate quality check on deduplicated segments.

    Returns: (passed: bool, reason: str, stats: dict)

    Gates (in order of elimination power):
      1. Minimum word count     — catches completely empty/sparse captions
      2. Coverage gate          — catches the 85%-dropout failure mode
      3. Tamil ratio gate       — catches English/Hindi auto-captions
      4. Unique word ratio gate — catches loopy auto-captions
    """
    full_text = segments_to_full_text(segments)
    words     = full_text.split()
    word_count = len(words)

    stats = {
        'word_count':   word_count,
        'segment_count': len(segments),
        'duration_s':   round(duration_s, 1),
        'tamil_ratio':  round(compute_tamil_ratio(full_text), 3),
        'unique_ratio': round(len(set(words)) / max(word_count, 1), 3),
        'coverage_pct': 0.0,
    }

    # Gate 1: Absolute minimum
    if word_count < MIN_WORD_COUNT:
        return False, "TOO_SHORT", stats

    # Gate 2: Coverage (compare to expected word count from duration)
    expected_words = (duration_s / 60.0) * TAMIL_WPM
    coverage = word_count / max(expected_words, 1)
    stats['coverage_pct'] = round(coverage * 100, 1)
    stats['expected_words'] = round(expected_words)

    if coverage < COVERAGE_THRESHOLD:
        return False, f"LOW_COVERAGE_{stats['coverage_pct']}pct", stats

    # Gate 3: Tamil script ratio
    if stats['tamil_ratio'] < TAMIL_RATIO_THRESHOLD:
        return False, f"LOW_TAMIL_{stats['tamil_ratio']:.2f}", stats

    # Gate 4: Unique word ratio (loopy captions)
    if stats['unique_ratio'] < UNIQUE_RATIO_THRESHOLD:
        return False, f"LOOPY_{stats['unique_ratio']:.2f}", stats

    return True, "PASS", stats

# ── SESSION-LEVEL 429 STATE ───────────────────────────────────────────────────
# Once YouTube starts rate-limiting subtitle requests, all subsequent requests
# in the session fail too. Track this at module level and back off aggressively.

_429_consecutive = 0          # consecutive 429 errors this session
_429_last_hit_ts  = 0.0       # epoch timestamp of last 429
_429_backoff_base = 120       # initial sleep on first 429: 2 minutes
_429_backoff_max  = 1800      # cap at 30 minutes

def handle_429_backoff():
    """
    Exponential backoff when YouTube rate-limits subtitle requests.
    Called once per 429 hit. If running under restart_harvester.sh,
    exits after 3 consecutive 429s so the shell loop can restart
    with a fresh session (much faster than sleeping through backoff).
    """
    global _429_consecutive, _429_last_hit_ts
    _429_consecutive += 1
    _429_last_hit_ts  = time.time()

    # Early exit: let the shell restart loop handle recovery
    if _429_consecutive >= 3:
        print(f"\n  ⛔ HTTP 429 (#{_429_consecutive}). "
              f"Session rate-limited — exiting for shell restart.", flush=True)
        log_event("HTTP_429_EXIT", "system",
                  f"consecutive={_429_consecutive} — exiting for restart")
        sys.exit(42)  # non-zero exit so shell loop knows it was a 429 bail

    sleep_time = min(_429_backoff_base * (2 ** (_429_consecutive - 1)), _429_backoff_max)
    print(f"\n  ⛔ HTTP 429 (#{_429_consecutive}). YouTube rate limit hit. "
          f"Sleeping {sleep_time//60}m {sleep_time%60}s before retry...", flush=True)
    log_event("HTTP_429_BACKOFF", "system",
              f"consecutive={_429_consecutive} sleep={sleep_time}s")
    time.sleep(sleep_time)

def maybe_decay_429_state():
    """
    Time-based decay. Called before every video attempt.
    If it's been >20 minutes since the last 429, halve the counter.
    This prevents the counter from staying high permanently just because
    videos don't have captions (which is the majority case).
    """
    global _429_consecutive, _429_last_hit_ts
    if _429_consecutive == 0:
        return
    elapsed = time.time() - _429_last_hit_ts
    if elapsed > 1200:   # 20 minutes since last actual 429
        old = _429_consecutive
        _429_consecutive = max(0, _429_consecutive // 2)
        if _429_consecutive == 0:
            print(f"  ✓ 429 counter decayed to 0 after {elapsed/60:.0f}m "
                  f"(was {old}).")
        else:
            print(f"  ↓ 429 counter decayed: {old} → {_429_consecutive}")
        _429_last_hit_ts = time.time()  # reset decay timer

def reset_429_state():
    """Reset 429 counter after a successful subtitle download."""
    global _429_consecutive, _429_last_hit_ts
    if _429_consecutive > 0:
        print(f"  ✓ 429 state cleared after {_429_consecutive} consecutive errors.")
    _429_consecutive  = 0
    _429_last_hit_ts  = 0.0

# ── CAPTION DOWNLOAD ──────────────────────────────────────────────────────────

def download_captions(url, vid_id):
    """
    Download Tamil captions for a single video via yt-dlp.

    Returns: (vtt_text, sub_type, lang, skip_reason)
      - On success:              (str, str, str, None)
      - No captions available:   (None, None, None, "NO_CAPTIONS")
      - YouTube rate limited:    (None, None, None, "RATE_LIMITED")
      - Livestream/no VTT:       (None, None, None, "FORMAT_UNAVAILABLE")
    """
    def _attempt_download(url, vid_id):
        """Single download attempt. Returns (vtt_text, sub_type, lang, reason)."""
        tmp_dir    = tempfile.mkdtemp()
        tmp_prefix = os.path.join(tmp_dir, vid_id)

        ydl_opts = {
            'writesubtitles':    True,
            'writeautomaticsub': True,
            'subtitleslangs':    ['ta', 'ta-IN'],
            'subtitlesformat':   'vtt',
            'skip_download':     True,
            'quiet':             True,
            'no_warnings':       True,
            'outtmpl':           tmp_prefix,
            'socket_timeout':    30,
            'retries':           3,
            'ignoreerrors':      True,
            'format':            'best',
            'allow_unplayable_formats': True,
            'ignore_no_formats_error':  True,
            **get_cookie_opts(),
        }

        try:
            import io
            from contextlib import redirect_stderr
            stderr_capture = io.StringIO()
            with redirect_stderr(stderr_capture):
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

            stderr_output = stderr_capture.getvalue().lower()
            if '429' in stderr_output or 'too many requests' in stderr_output:
                return None, None, None, "RATE_LIMITED"

            candidates = [
                (f"{tmp_prefix}.ta.vtt",    "manual", "ta"),
                (f"{tmp_prefix}.ta-IN.vtt", "auto",   "ta-IN"),
            ]
            for filepath, sub_type, lang in candidates:
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if len(content.strip()) > 50:
                        return content, sub_type, lang, None   # success

            return None, None, None, "NO_CAPTIONS"

        except Exception as e:
            err_str = str(e).lower()
            if '429' in err_str or 'too many requests' in err_str:
                return None, None, None, "RATE_LIMITED"
            if 'requested format is not available' in err_str:
                return None, None, None, "FORMAT_UNAVAILABLE"
            return None, None, None, "NO_CAPTIONS"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── First attempt ─────────────────────────────────────────────────────────
    vtt_text, sub_type, lang, reason = _attempt_download(url, vid_id)

    if reason == "RATE_LIMITED":
        handle_429_backoff()
        # Retry once after backoff
        vtt_text, sub_type, lang, reason = _attempt_download(url, vid_id)
        if reason == "RATE_LIMITED":
            log_event("HTTP_429_SKIP", vid_id, "429 persists after backoff")
            return None, None, None, "RATE_LIMITED"

    if vtt_text:
        reset_429_state()

    return vtt_text, sub_type, lang, reason

# ── OUTPUT WRITER ─────────────────────────────────────────────────────────────

def save_caption_record(vid_id, title, url, channel, segments, full_text, stats,
                        sub_type, lang):
    """
    Write caption record as structured JSON with segments preserved.
    Uses atomic write (tmp → rename) to prevent partial files.

    Saves to: astrologer_data_hybrid/captions/<vid_id>.json
    """
    record = {
        "metadata": {
            "video_id":     vid_id,
            "title":        title,
            "url":          url,
            "channel":      channel,
            "sub_type":     sub_type,
            "lang":         lang,
            "duration_s":   stats.get('duration_s', 0),
            "word_count":   stats.get('word_count', 0),
            "expected_words": stats.get('expected_words', 0),
            "coverage_pct": stats.get('coverage_pct', 0),
            "tamil_ratio":  stats.get('tamil_ratio', 0),
            "unique_ratio": stats.get('unique_ratio', 0),
            "segment_count": stats.get('segment_count', 0),
            "quality_gate": "PASS",
            "cleanup_status": "PENDING",   # caption_cleanup.py will update this
            "timestamp":    datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "full_text": full_text,
        "segments":  segments,
    }

    out_path = os.path.join(CAPTION_FOLDER, f"{vid_id}.json")
    tmp_path = out_path + ".tmp"

    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    os.rename(tmp_path, out_path)

    return out_path

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    mm = ManifestManager()

    # ── Load manifest into memory once — O(1) skip checks ────────────────────
    print("  Loading manifest into memory...", end=" ", flush=True)
    DONE_CATEGORIES = {
        'SUCCESS', 'CAPTION_RAW', 'CAPTION_COMPLETE',
        'MEMBERS_ONLY', 'HTTP_429', 'NO_CAPTION_PERMANENT', 'FORMAT_UNAVAILABLE',
        'NO_CAPTION_WHISPER_PENDING'
    }
    done_ids = set()

    # Strategy 1: get_existing_ids() — fastest, returns a set directly
    if hasattr(mm, 'get_existing_ids'):
        try:
            existing = mm.get_existing_ids()
            if existing:
                # get_existing_ids() may return ALL ids regardless of category.
                # Cross-check with get_manifest() to filter by DONE_CATEGORIES.
                if hasattr(mm, 'get_manifest'):
                    manifest_dict = mm.get_manifest() or {}
                    done_ids = {
                        vid for vid, entry in manifest_dict.items()
                        if entry.get('category', '') in DONE_CATEGORIES
                    }
                else:
                    # No category info available — treat all existing as done
                    done_ids = set(existing)
        except Exception:
            pass

    # Strategy 2: get_manifest() returns full dict — filter by category
    if not done_ids and hasattr(mm, 'get_manifest'):
        try:
            manifest_dict = mm.get_manifest() or {}
            done_ids = {
                vid for vid, entry in manifest_dict.items()
                if entry.get('category', '') in DONE_CATEGORIES
            }
        except Exception:
            pass

    # Strategy 3: Read mm.path (the actual manifest file path) directly
    if not done_ids:
        manifest_file = getattr(mm, 'path', None)
        if manifest_file and os.path.exists(manifest_file):
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                # Handle both JSON dict and JSONL formats
                if content.startswith('{'):
                    data = json.loads(content)
                    done_ids = {
                        vid for vid, entry in data.items()
                        if isinstance(entry, dict)
                        and entry.get('category', '') in DONE_CATEGORIES
                    }
                else:
                    for line in content.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            cat = entry.get('category', '')
                            if cat in DONE_CATEGORIES:
                                vid = entry.get('video_id') or entry.get('id')
                                if vid:
                                    done_ids.add(vid)
                        except Exception:
                            pass
            except Exception:
                pass

    # Strategy 4: Scan captions folder — any .json file = already harvested
    if not done_ids:
        try:
            caption_files = os.listdir(CAPTION_FOLDER)
            done_ids = {
                f.replace('.json', '') for f in caption_files
                if f.endswith('.json') and not f.endswith('.tmp')
            }
        except Exception:
            pass

    print(f"{len(done_ids)} already done — will skip instantly.")

    print("=" * 64)
    print("  AA Caption-Only Harvester")
    print("  Saves CAPTION_RAW to manifest (not SUCCESS).")
    print("  Run caption_cleanup.py after this to validate content.")
    print("=" * 64)

    # ── Argument Parsing ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="AA Caption Only Harvester")
    parser.add_argument('--channel', action='append', help='Specific channel(s) to process (substring match)')
    parser.add_argument('--limit', type=int, help='Limit videos per channel')
    args = parser.parse_args()

    shuffled_channels = ALL_CHANNELS[:]
    
    if args.channel:
        # Filter channels that match any of the provided substrings
        shuffled_channels = [
            c for c in shuffled_channels 
            if any(target.lower() in c.lower() for target in args.channel)
        ]
        if not shuffled_channels:
            print(f"Error: No channels found matching: {args.channel}")
            return

    random.shuffle(shuffled_channels)

    session_stats = {
        'total_scanned':  0,
        'already_done':   0,
        'caption_pass':   0,
        'gate_fail':      {},
        'no_captions':    0,
        'rate_limited':   0,   # 429 skips — not permanent, retried next session
        'format_unavail': 0,   # livestream/premiere — permanent, no VTT possible
        'errors':         0,
    }

    for channel_url in shuffled_channels:
        print(f"\n── Scanning: {channel_url.split('@')[1].split('/')[0]}")

        try:
            ydl_opts = {
                'extract_flat': True,
                'quiet':        True,
                'ignoreerrors': True,
                'socket_timeout': 30,
                'retries':      10,
                **get_cookie_opts(),
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(channel_url, download=False)

            if not result or 'entries' not in result:
                print("  → Channel empty or unavailable.")
                continue

            videos = [e for e in result['entries'] if e]
            if args.limit:
                videos = videos[:args.limit]
            print(f"  → {len(videos)} videos found.")

            _skip_batch = 0   # count consecutive skips before printing a summary

            for i, entry in enumerate(videos):
                vid_id   = entry.get('id')
                title    = entry.get('title', 'Unknown')
                duration = entry.get('duration') or 0
                url      = f"https://www.youtube.com/watch?v={vid_id}"

                if not vid_id:
                    continue

                session_stats['total_scanned'] += 1

                # ── O(1) skip check against in-memory set ────────────────────
                if vid_id in done_ids:
                    session_stats['already_done'] += 1
                    _skip_batch += 1
                    # Print a skip summary every 50 skips instead of per-video
                    if _skip_batch % 50 == 0:
                        print(f"  ... skipped {_skip_batch} already-done videos so far")
                    continue

                # Print any accumulated skip batch before showing next active video
                if _skip_batch > 0:
                    print(f"  ↷ Skipped {_skip_batch} already-done videos")
                    _skip_batch = 0

                progress = f"[{i+1}/{len(videos)}]"
                print(f"  {progress} {vid_id} | {title[:45]}...", end=" ", flush=True)

                # ── Rate limit decay + thermal check ─────────────────────────
                maybe_decay_429_state()
                cooldown_if_needed()

                # ── Download captions ────────────────────────────────────────
                try:
                    vtt_text, sub_type, lang, skip_reason = download_captions(url, vid_id)
                except Exception as e:
                    print(f"ERROR: {str(e)[:60]}")
                    log_event("ERROR", vid_id, str(e))
                    session_stats['errors'] += 1
                    time.sleep(random.uniform(1.0, 2.0))
                    continue

                # ── Handle each result type with a distinct label ─────────────
                if skip_reason == "RATE_LIMITED":
                    # 429 — video stays PENDING, will be retried next session
                    print(f"⛔ RATE LIMITED — skipping (stays PENDING)")
                    session_stats['rate_limited'] += 1
                    time.sleep(random.uniform(1.5, 3.0))
                    continue

                if skip_reason == "FORMAT_UNAVAILABLE":
                    # Livestream/premiere — subtitles structurally impossible
                    print(f"🎞  FORMAT UNAVAILABLE (livestream/premiere) — marking permanent")
                    log_missing(vid_id, title, "FORMAT_UNAVAILABLE")
                    log_event("FORMAT_UNAVAILABLE", vid_id, "Permanent — no VTT for livestream")
                    # Mark in manifest so we never try this video again
                    mm.update_entry(vid_id, {
                        "title":    title,
                        "category": "FORMAT_UNAVAILABLE",
                        "duration": duration,
                    })
                    done_ids.add(vid_id)   # update in-memory set
                    session_stats['format_unavail'] += 1
                    time.sleep(random.uniform(0.5, 1.0))
                    continue

                if not vtt_text:
                    # Genuine "no Tamil captions exist for this video"
                    print(f"— no Tamil captions")
                    log_missing(vid_id, title, "NO_CAPTIONS")
                    session_stats['no_captions'] += 1
                    # Update manifest so we don't check this video again in caption_only_harvester
                    # It will stay pending for harvester.py which looks for different DONE categories
                    mm.update_entry(vid_id, {
                        "title": title,
                        "category": "NO_CAPTION_WHISPER_PENDING",
                        "duration": duration,
                    })
                    done_ids.add(vid_id)
                    time.sleep(random.uniform(1.5, 3.0))
                    continue

                # ── Parse VTT → segments ─────────────────────────────────────
                raw_segments   = parse_vtt_to_segments(vtt_text)
                clean_segments = deduplicate_segments(raw_segments)
                full_text      = segments_to_full_text(clean_segments)

                # ── Quality gates ────────────────────────────────────────────
                passed, reason, stats = validate_caption_quality(
                    clean_segments, duration, vid_id
                )

                if not passed:
                    print(f"FAIL:{reason}")
                    log_missing(vid_id, title, reason)
                    log_event("GATE_FAIL", vid_id,
                              f"{reason} | {stats.get('word_count')}w "
                              f"{stats.get('coverage_pct')}%cov "
                              f"{stats.get('tamil_ratio', 0):.2f}ta")
                    session_stats['gate_fail'][reason] = \
                        session_stats['gate_fail'].get(reason, 0) + 1
                    # Update manifest so caption_only_harvester skips it next time,
                    # but harvester.py still processes it.
                    mm.update_entry(vid_id, {
                        "title": title,
                        "category": "NO_CAPTION_WHISPER_PENDING",
                        "duration": duration,
                        "sentry_status": f"GATE_FAIL_{reason}",
                    })
                    done_ids.add(vid_id)
                    time.sleep(random.uniform(1.5, 3.0))
                    continue

                # ── Save to disk (atomic) ────────────────────────────────────
                stats['duration_s'] = duration
                out_path = save_caption_record(
                    vid_id, title, url, channel_url,
                    clean_segments, full_text, stats,
                    sub_type, lang
                )

                # ── Update manifest as CAPTION_RAW (NOT SUCCESS) ─────────────
                mm.update_entry(vid_id, {
                    "title":        title,
                    "category":     "CAPTION_RAW",
                    "sentry_status": "PENDING_CLEANUP",
                    "filepath":     out_path,
                    "duration":     duration,
                    "sub_type":     sub_type,
                    "lang":         lang,
                    "coverage_pct": stats['coverage_pct'],
                    "word_count":   stats['word_count'],
                })
                done_ids.add(vid_id)   # keep in-memory set in sync

                print(
                    f"✓ {stats['word_count']}w "
                    f"{stats['coverage_pct']}%cov "
                    f"{stats['tamil_ratio']*100:.0f}%ta "
                    f"({sub_type}/{lang})"
                )
                log_event("CAPTION_RAW", vid_id,
                          f"{sub_type}/{lang} | {stats['word_count']}w | "
                          f"{stats['coverage_pct']}%cov")
                session_stats['caption_pass'] += 1

                time.sleep(random.uniform(1.5, 3.5))

            # Print any remaining skip batch at end of channel
            if _skip_batch > 0:
                print(f"  ↷ Skipped {_skip_batch} already-done videos")

        except Exception as e:
            print(f"\n  [Channel Error] {e}")
            log_event("CHANNEL_ERROR", channel_url, str(e))

    # ── Session summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  CAPTION HARVEST COMPLETE")
    print(f"  Scanned        : {session_stats['total_scanned']}")
    print(f"  ↷ Already done : {session_stats['already_done']}  (skipped instantly)")
    print(f"  ✓ Captions     : {session_stats['caption_pass']}")
    print(f"  — No captions  : {session_stats['no_captions']}  (genuine — no Tamil subs exist)")
    print(f"  🎞  Format N/A  : {session_stats['format_unavail']}  (livestream/premiere — permanent)")
    print(f"  ⛔ Rate limited : {session_stats['rate_limited']}  (429 — retry next session)")
    print(f"  Gate failures  : {sum(session_stats['gate_fail'].values())}")
    for reason, count in sorted(session_stats['gate_fail'].items(), key=lambda x: -x[1]):
        print(f"    {reason:35s}: {count}")
    print(f"  Errors         : {session_stats['errors']}")
    total_attempted = (session_stats['caption_pass'] +
                       session_stats['no_captions'] +
                       sum(session_stats['gate_fail'].values()))
    if total_attempted:
        yield_pct = session_stats['caption_pass'] / total_attempted * 100
        print(f"\n  Caption yield : {yield_pct:.1f}% of videos attempted")

    remaining = (session_stats['total_scanned'] - session_stats['already_done']
                 - session_stats['caption_pass'] - sum(session_stats['gate_fail'].values()))
    print(f"  Remaining for harvester.py: ~{max(remaining, 0)} videos")
    print("=" * 64)
    print(f"\n  Next step: python3 caption_cleanup.py")
    print(f"  (cleanup validates domain vocabulary, restores punctuation,")
    print(f"   filters meta-talk, promotes CAPTION_RAW → CAPTION_COMPLETE)")

    # Save stats to disk for later reference
    stats_path = STATS_LOG
    tmp_stats  = stats_path + ".tmp"
    with open(tmp_stats, 'w', encoding='utf-8') as f:
        json.dump({
            **session_stats,
            'run_timestamp': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        }, f, indent=2)
    os.rename(tmp_stats, stats_path)


if __name__ == "__main__":
    main()
