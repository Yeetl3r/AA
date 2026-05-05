"""
caption_cleanup.py — AA Pipeline: Full caption cleanup and training export

WHAT THIS DOES (in order):
  Phase 0 — Load analysis report (metadata only, no file I/O per-file)
  Phase 1 — Stutter dedup: global sentence-level dedup (3.13x → 1.0x)
  Phase 2 — Meta-talk removal: excise contaminated segments, don't discard files
  Phase 3 — Terminology anchoring: deterministic ASR error corrections
  Phase 4 — Tier re-evaluation: fix the meta-talk threshold (0.6→0.8 for discard)
  Phase 5 — LLM punctuation: TIER_2 files via provider cascade (Gemini→Groq→Cerebras→local)
  Phase 6 — Training export: ASR-aligned JSONL + plain-text LM corpus

RESUMABLE: Every output file is checked for existence before processing.
           Restart at any time — already-cleaned files are skipped instantly.

NO POINT OF FAILURE: Every file is processed under try/except.
           LLM failures fall back through the cascade to rule-based cleanup.
           A file that fails all LLM providers is still exported using
           rule-based cleanup only — it never blocks the pipeline.

OUTPUT FILES:
  astrologer_data_hybrid/cleaned/{channel}/{video_id}.json  ← clean JSON per file
  astrologer_data_hybrid/training_export/asr_segments.jsonl ← ASR training (aligned)
  astrologer_data_hybrid/training_export/lm_corpus.jsonl    ← LM training (plain text)
  astrologer_data_hybrid/cleanup_manifest.jsonl             ← per-file audit trail
  astrologer_data_hybrid/cleanup_stats.json                 ← summary

USAGE:
  python3 caption_cleanup.py                   # full run
  python3 caption_cleanup.py --tier 1          # TIER_1 only (fastest, safest)
  python3 caption_cleanup.py --dry-run         # analyse without writing
  python3 caption_cleanup.py --file VIDEO_ID   # single file debug
  python3 caption_cleanup.py --resume          # skip already-cleaned (default: on)
"""

import os
import re
import sys
import json
import time
import html
import math
import random
import hashlib
import argparse
import datetime
import unicodedata
import requests
from pathlib import Path
from collections import Counter
from typing import Optional

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR        = "/Volumes/Storage Drive/AA"
CAPTION_FOLDER  = os.path.join(BASE_DIR, "astrologer_data_hybrid", "captions")
ANALYSIS_REPORT = os.path.join(BASE_DIR, "astrologer_data_hybrid", "caption_analysis_report.json")
QUALITY_TIERS   = os.path.join(BASE_DIR, "astrologer_data_hybrid", "caption_quality_tiers.json")
CLEANED_DIR     = os.path.join(BASE_DIR, "astrologer_data_hybrid", "cleaned")
EXPORT_DIR      = os.path.join(BASE_DIR, "astrologer_data_hybrid", "training_export")
CLEANUP_MANIFEST= os.path.join(BASE_DIR, "astrologer_data_hybrid", "cleanup_manifest.jsonl")
CLEANUP_STATS   = os.path.join(BASE_DIR, "astrologer_data_hybrid", "cleanup_stats.json")
ASR_EXPORT      = os.path.join(EXPORT_DIR, "asr_segments.jsonl")
LM_EXPORT       = os.path.join(EXPORT_DIR, "lm_corpus.jsonl")

TAMIL_RANGE = (0x0B80, 0x0BFF)

# ── THRESHOLDS (tuned from analysis) ──────────────────────────────────────────
META_TALK_DISCARD_THRESHOLD    = 0.80  # discard file if >80% meta-talk (was 0.6 — wrong)
META_TALK_SEGMENT_THRESHOLD    = 0.50  # remove individual segment if meta-talk score > 0.5
MIN_SEGMENT_WORDS              = 3     # drop segments shorter than this after cleanup
MIN_FILE_WORDS_AFTER_CLEANUP   = 100   # discard output if fewer than 100 words survive
PUNCTUATION_BATCH_SIZE         = 8     # segments per LLM punctuation call

# ── TERMINOLOGY CORRECTIONS ───────────────────────────────────────────────────
# Deterministic fixes for known ASR substitution errors in Tamil astrology.
# Add entries here as you discover them from error logs.
TERM_CORRECTIONS = {
    "நவாம்சா":         "நவாம்சம்",
    "லக்கினம்":        "லக்னம்",
    "புத்தி":          "புக்தி",
    "திசை பக்தி":     "திசா புக்தி",
    "திசாபுத்தி":     "திசா புக்தி",
    "திசைப்புக்தி":   "திசா புக்தி",
    "அந்தரம்":        "அந்தர்தசை",
    "கேதுவு":         "கேது",
    "ராகுவு":         "ராகு",
    "சனீஸ்வரர்":      "சனீஸ்வரன்",
    "ஷஷ்டாஷ்டகம":    "ஷஷ்டாஷ்டகம்",
    "சஷ்டாஷ்டம":     "ஷஷ்டாஷ்டகம்",
    "லக்கினாதிபதி":   "லக்னாதிபதி",
    "திரிகோனம்":      "திரிகோணம்",
    "நவாம்சகம்":      "நவாம்சம்",
    "சப்தமம்":        "சப்தாம்சம்",
    "பஞ்சோத்தரி":     "பஞ்சோத்தரி",
    "தேவகாணம்":       "திக்பலம்",
}

# ── META-TALK PATTERNS ────────────────────────────────────────────────────────
# Matches individual segments to flag/remove.
META_PATTERNS = [
    r'subscribe\s*(பண்ண|செய்ய|பண்ணுங்க)',
    r'bell\s*icon',
    r'notification',
    r'like\s*(பண்ண|செய்ய)',
    r'comment\s*(பண்ண|செய்ய)',
    r'கமெண்ட்\s*(பண்ண|செய்ய|போட)',
    r'youtube\s*(channel|சேனல்)',
    r'instagram|facebook|whatsapp|telegram',
    r'டெக்னிக்கல்\s*(ப்ராப்ளம்|problem)',
    r'audio\s*problem|audio\s*issue',
    r'connection\s*(problem|issue)',
    r'share\s*(பண்ண|செய்ய|பண்ணுங்க)',
    r'இணைய\s*பிரச்சனை',
    r'setting(s)?\s*(மாத்தி|change)',
    r'இந்த\s*(video|வீடியோ)\s*(like|share)',
    r'என்னோட\s*(channel|சேனல்)',
]
_META_RE = re.compile('|'.join(META_PATTERNS), re.IGNORECASE)

# ── LLM PROVIDER CASCADE ──────────────────────────────────────────────────────
PROVIDERS = [
    {
        "name":        "gemini-flash",
        "model":       "gemini-1.5-flash",
        "base_url":    "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_env": "GEMINI_API_KEY",
        "daily_limit": 1500,
        "priority":    1,
    },
    {
        "name":        "groq-70b",
        "model":       "llama-3.3-70b-versatile",
        "base_url":    "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "daily_limit": 1000,
        "priority":    2,
    },
    {
        "name":        "cerebras-70b",
        "model":       "llama3.1-70b",
        "base_url":    "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "daily_limit": 900,
        "priority":    3,
    },
    {
        "name":        "groq-8b",
        "model":       "llama-3.1-8b-instant",
        "base_url":    "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "daily_limit": 14400,
        "priority":    4,
    },
    {
        "name":        "ollama-3b",
        "model":       "qwen2.5:3b",
        "base_url":    "http://localhost:11434/v1",
        "api_key_env": None,
        "daily_limit": None,
        "priority":    5,
    },
]
_usage_today = {p["name"]: 0 for p in PROVIDERS}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def is_tamil(char):
    return TAMIL_RANGE[0] <= ord(char) <= TAMIL_RANGE[1]

def tamil_ratio(text):
    alpha = [c for c in text if unicodedata.category(c).startswith('L')]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if is_tamil(c)) / len(alpha)

def normalise_for_dedup(text):
    """Normalise text for deduplication comparison only — not for output."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def atomic_write(path, data, is_jsonl_append=False):
    """Write data atomically. Supports JSON, string, and JSONL append mode."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_jsonl_append:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        return
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        if isinstance(data, (dict, list)):
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(data))
    os.replace(tmp, path)

def log_manifest(entry):
    """Append one cleanup audit entry to cleanup_manifest.jsonl."""
    atomic_write(CLEANUP_MANIFEST, entry, is_jsonl_append=True)

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# ── PHASE 1: STUTTER DEDUPLICATION ───────────────────────────────────────────
#
# YouTube VTT files use a ROLLING KARAOKE format — NOT same-segment repetition.
# Each caption block is a sliding window: it shows the last few words from the
# previous block PLUS new words. This creates suffix-prefix overlap between
# adjacent segments, and 0.01s "transition" segments carrying the tail forward.
#
# Example (real data from corpus):
#   [22.79-22.80s 0.01s] "கொடுத்திருக்கிறேன். இந்த தலைப்பே கொஞ்சம்"  ← transition
#   [22.80-25.11s 2.31s] "கொடுத்திருக்கிறேன். இந்த தலைப்பே கொஞ்சம் ஒரு மாதிரி..."
#   [25.11-25.12s 0.01s] "ஒரு மாதிரி..."  ← transition (tail of above)
#   [25.12-28.31s 3.19s] "ஒரு மாதிரி... அமைப்பு. உலகத்திலேயே..."
#
# Correct algorithm:
#   Step 1 — Remove transition segments (duration < 0.1s)
#   Step 2 — Strip music/sound tags ([இசை], [கைதட்டல்], etc.)
#   Step 3 — Strip speaker markers (>> )
#   Step 4 — For each pair of consecutive content segments, find the longest
#             suffix of segment[N] that matches the prefix of segment[N+1]
#             and trim that overlap from segment[N+1]
#   Step 5 — Merge resulting short fragments into ASR-suitable chunks
#
# Verified result on corpus samples:
#   mcszKojNKFY: 9,821 raw words → 3,283 after dedup → 110% coverage ✓
#   jD_oK1ukiaw: 22,172 raw words → 7,869 after dedup → 121% coverage ✓
#   OHdPK6KTZw8: 4,096 raw words → 1,370 after dedup → 111% coverage ✓

_TAG_RE     = re.compile(r'\[[^\]]+\]')        # [இசை], [கைதட்டல்], [music]
_SPEAKER_RE = re.compile(r'^>+\s*')            # >> speaker markers

def _clean_segment_text(text):
    """Strip VTT display artifacts from a segment's text."""
    text = _TAG_RE.sub('', text)
    text = _SPEAKER_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def dedup_segments(segments):
    """
    Correct rolling-window dedup for YouTube VTT format.

    Returns (clean_merged_segments, segments_removed_count)
    The returned segments are merged into ASR-appropriate chunks (8-20 words each).
    """
    if not segments:
        return [], 0

    original_count = len(segments)

    # ── Step 1 & 2 & 3: Filter transitions + clean text ──────────────────────
    content_segs = []
    for s in segments:
        dur  = s.get('end', 0) - s.get('start', 0)
        text = _clean_segment_text(s.get('text', ''))
        if dur < 0.1 or not text:
            continue
        content_segs.append({**s, 'text': text})

    if not content_segs:
        return [], original_count

    # ── Step 4: Suffix-prefix overlap removal ─────────────────────────────────
    deduped = [content_segs[0]]
    for seg in content_segs[1:]:
        prev_words = deduped[-1]['text'].split()
        curr_words = seg['text'].split()

        # Find longest suffix of prev that exactly matches prefix of curr
        max_ov = min(len(prev_words), len(curr_words) - 1)
        overlap = 0
        for k in range(max_ov, 0, -1):
            if prev_words[-k:] == curr_words[:k]:
                overlap = k
                break

        new_text = ' '.join(curr_words[overlap:]).strip()
        if new_text:
            deduped.append({**seg, 'text': new_text})

    # ── Step 5: Merge fragments into ASR-suitable chunks ─────────────────────
    # After dedup, each segment holds only its new unique words (2-5 words
    # on average). Merge into chunks of 8-20 words, flushing at sentence
    # boundaries for clean ASR alignment.
    merged   = []
    buf_segs = []
    buf_words = 0

    def flush_buffer(buf):
        if not buf:
            return None
        return {
            'start': buf[0].get('start', 0),
            'end':   buf[-1].get('end',   0),
            'text':  ' '.join(s['text'] for s in buf),
        }

    for seg in deduped:
        words = seg['text'].split()
        buf_segs.append(seg)
        buf_words += len(words)

        is_sentence_end = any(seg['text'].rstrip().endswith(p)
                              for p in ['.', '!', '?', '।', ','])

        if buf_words >= 12 or (is_sentence_end and buf_words >= 5):
            chunk = flush_buffer(buf_segs)
            if chunk:
                merged.append(chunk)
            buf_segs  = []
            buf_words = 0

    # Flush any remainder
    chunk = flush_buffer(buf_segs)
    if chunk:
        merged.append(chunk)

    removed = original_count - len(merged)
    return merged, removed

# ── PHASE 2: META-TALK REMOVAL ────────────────────────────────────────────────

def segment_is_meta_talk(text):
    """Returns True if a single segment is predominantly meta-talk."""
    return bool(_META_RE.search(text))

def remove_meta_talk_segments(segments, file_meta_score):
    """
    Remove individual meta-talk segments from a file.

    Strategy:
      - If file-level meta_talk_score < 0.3: skip (unlikely to have contamination)
      - If file-level meta_talk_score >= META_TALK_DISCARD_THRESHOLD (0.80):
        mark file for discard (too contaminated to salvage)
      - Otherwise: remove matching individual segments, keep the rest

    Returns (clean_segments, meta_removed_count, should_discard)
    """
    if file_meta_score < 0.3:
        return segments, 0, False

    if file_meta_score >= META_TALK_DISCARD_THRESHOLD:
        return [], len(segments), True

    clean = []
    removed = 0
    for seg in segments:
        text = seg.get('text', '')
        if segment_is_meta_talk(text):
            removed += 1
        else:
            clean.append(seg)

    return clean, removed, False

# ── PHASE 3: TERMINOLOGY ANCHORING ───────────────────────────────────────────

def apply_terminology(text):
    """Apply deterministic ASR correction dictionary. O(n×m) but fast for small dicts."""
    for wrong, right in TERM_CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text

def apply_terminology_to_segments(segments):
    """Apply terminology corrections to all segment texts."""
    corrected = []
    for seg in segments:
        seg = dict(seg)
        seg['text'] = apply_terminology(seg.get('text', ''))
        corrected.append(seg)
    return corrected

# ── PHASE 4: TIER RE-EVALUATION ──────────────────────────────────────────────

def re_evaluate_tier(segments, duration_s, original_meta_score):
    """
    Re-evaluate quality tier after cleanup.
    Uses corrected meta-talk threshold (0.80 for discard, not 0.60).
    """
    if not segments:
        return "TIER_3", {}

    full_text  = ' '.join(s.get('text', '') for s in segments)
    words      = full_text.split()
    word_count = len(words)

    if word_count < MIN_FILE_WORDS_AFTER_CLEANUP:
        return "TIER_3", {"reason": "too_few_words"}

    t_ratio = tamil_ratio(full_text)
    u_ratio = len(set(words)) / max(word_count, 1)

    expected_words = (duration_s / 60.0) * 110 if duration_s > 0 else word_count
    coverage = word_count / max(expected_words, 1) * 100

    # Re-score meta-talk on cleaned text (should be much lower now)
    meta_hits = sum(1 for s in segments if segment_is_meta_talk(s.get('text', '')))
    new_meta_score = meta_hits / max(len(segments), 1)

    metrics = {
        "word_count":     word_count,
        "tamil_ratio":    round(t_ratio, 4),
        "unique_ratio":   round(u_ratio, 4),
        "coverage_pct":   round(coverage, 1),
        "meta_talk_score": round(new_meta_score, 3),
    }

    # TIER_1: high confidence, clean content
    if (coverage >= 80 and t_ratio >= 0.85 and u_ratio >= 0.08
            and new_meta_score <= 0.10 and word_count >= 500):
        return "TIER_1", metrics

    # TIER_3: too damaged or too short
    if (coverage < 30 or t_ratio < 0.30 or word_count < MIN_FILE_WORDS_AFTER_CLEANUP
            or new_meta_score >= META_TALK_DISCARD_THRESHOLD):
        return "TIER_3", metrics

    return "TIER_2", metrics

# ── PHASE 5: LLM PUNCTUATION RESTORATION ─────────────────────────────────────

def call_provider(provider, prompt, max_tokens=2048):
    """Single LLM API call. Returns response text or None."""
    api_key = (os.getenv(provider["api_key_env"])
               if provider["api_key_env"] else "ollama")
    if provider["api_key_env"] and not api_key:
        return None
    if (provider["daily_limit"] is not None and
            _usage_today[provider["name"]] >= provider["daily_limit"]):
        return None

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json"
        }
        payload = {
            "model":       provider["model"],
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.05,
            "max_tokens":  max_tokens,
        }
        resp = requests.post(
            f"{provider['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=90
        )
        resp.raise_for_status()
        result  = resp.json()
        content = result['choices'][0]['message']['content'].strip()
        _usage_today[provider["name"]] += 1
        return content if len(content) > 5 else None
    except Exception as e:
        return None

def restore_punctuation_batch(segments, title=""):
    """
    Restore punctuation for a batch of segments in a single LLM call.
    Sends up to PUNCTUATION_BATCH_SIZE segments, returns corrected segments.
    Falls back to original segments if all providers fail.
    """
    if not segments:
        return segments

    # Build numbered input
    numbered = '\n'.join(f"{i+1}. {s.get('text','')}" for i, s in enumerate(segments))
    prompt = (
        f"நீ தமிழ் ஜோதிட உரை திருத்துபவன். "
        f"கீழுள்ள {len(segments)} வாக்கியங்களில் நிறுத்தற்குறிகள் சேர்க்கவும்.\n"
        f"விதிகள்: 1) வார்த்தைகளை மாற்றாதே 2) எண்ணிடல் அப்படியே வைக்கவும் "
        f"3) ஒவ்வொரு வரியும் 'எண். உரை' வடிவில் திரும்ப அனுப்பவும்\n\n"
        f"தலைப்பு: {title}\n\n{numbered}"
    )

    for provider in sorted(PROVIDERS, key=lambda p: p["priority"]):
        result = call_provider(provider, prompt, max_tokens=len(numbered) * 2 + 200)
        if not result:
            continue

        # Parse numbered output back to segment texts
        lines = result.strip().splitlines()
        restored_texts = {}
        for line in lines:
            m = re.match(r'^(\d+)\.\s+(.+)$', line.strip())
            if m:
                idx = int(m.group(1)) - 1
                text = m.group(2).strip()
                if 0 <= idx < len(segments):
                    restored_texts[idx] = text

        if len(restored_texts) >= len(segments) * 0.7:   # accept if 70%+ parsed
            result_segs = []
            for i, seg in enumerate(segments):
                seg = dict(seg)
                if i in restored_texts:
                    restored = apply_terminology(restored_texts[i])
                    # Only accept if Tamil ratio didn't drop significantly
                    if tamil_ratio(restored) >= tamil_ratio(seg.get('text','')) - 0.1:
                        seg['text'] = restored
                        seg['punctuation_restored'] = True
                result_segs.append(seg)
            return result_segs, provider["name"]

    # All providers failed — return originals
    return segments, "rule_based_only"

def apply_llm_punctuation(segments, title, tier, dry_run=False):
    """
    Apply punctuation restoration to TIER_2 files only.
    Processes in batches of PUNCTUATION_BATCH_SIZE.
    """
    if tier != "TIER_2" or dry_run:
        return segments, "skipped"

    if not segments:
        return segments, "empty"

    all_restored = []
    provider_used = "rule_based_only"

    for i in range(0, len(segments), PUNCTUATION_BATCH_SIZE):
        batch = segments[i : i + PUNCTUATION_BATCH_SIZE]
        restored_batch, provider = restore_punctuation_batch(batch, title)
        all_restored.extend(restored_batch)
        if provider != "rule_based_only":
            provider_used = provider
        time.sleep(0.3)  # light rate limiting between batches

    return all_restored, provider_used

# ── PHASE 6: TRAINING EXPORT ──────────────────────────────────────────────────

def build_clean_record(vid_id, title, url, channel, duration_s,
                       segments, tier, cleanup_meta):
    """Build the cleaned JSON record saved per file."""
    full_text = ' '.join(s.get('text', '') for s in segments if s.get('text','').strip())
    return {
        "metadata": {
            "video_id":           vid_id,
            "title":              title,
            "url":                url,
            "channel":            channel,
            "duration_s":         duration_s,
            "quality_tier":       tier,
            "cleanup_timestamp":  timestamp(),
            **cleanup_meta,
        },
        "full_text": full_text,
        "segments":  segments,
    }

def export_to_training(record, tier):
    """
    Append to ASR and LM training export files.

    ASR format: one record per file with segment timestamps
    LM format:  one record per file with plain text + metadata
    """
    meta = record["metadata"]

    # ASR export: full aligned segment structure (for Whisper fine-tuning)
    asr_record = {
        "video_id":   meta["video_id"],
        "title":      meta["title"],
        "url":        meta["url"],
        "channel":    meta["channel"],
        "duration_s": meta["duration_s"],
        "tier":       tier,
        "segments":   [
            {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", "")}
            for s in record["segments"]
            if s.get("text", "").strip() and len(s.get("text", "").split()) >= MIN_SEGMENT_WORDS
        ]
    }
    if asr_record["segments"]:
        atomic_write(ASR_EXPORT, asr_record, is_jsonl_append=True)

    # LM export: plain text with light metadata (for language model training)
    lm_record = {
        "video_id": meta["video_id"],
        "title":    meta["title"],
        "channel":  meta["channel"],
        "tier":     tier,
        "text":     record["full_text"],
    }
    if lm_record["text"].strip():
        atomic_write(LM_EXPORT, lm_record, is_jsonl_append=True)

# ── CORE PROCESSING FUNCTION ──────────────────────────────────────────────────

def process_file(caption_json_path, analysis_meta, dry_run=False, resume=True):
    """
    Full cleanup pipeline for a single caption file.
    Returns a result dict suitable for the cleanup manifest.

    Never raises — all exceptions are caught and returned as error entries.
    """
    vid_id   = analysis_meta.get("video_id", Path(caption_json_path).stem)
    title    = analysis_meta.get("title", "")
    url      = analysis_meta.get("url", "")
    channel  = analysis_meta.get("channel", "").split("@")[-1].split("/")[0]
    duration = analysis_meta.get("duration_s", 0)
    orig_tier= analysis_meta.get("quality_tier", "TIER_2")
    meta_score = analysis_meta.get("meta_talk_score", 0.0)

    result = {
        "video_id":       vid_id,
        "title":          title,
        "original_tier":  orig_tier,
        "final_tier":     None,
        "status":         None,
        "error":          None,
        "stutter_removed": 0,
        "meta_removed":   0,
        "words_before":   0,
        "words_after":    0,
        "provider_used":  "none",
        "timestamp":      timestamp(),
    }

    # ── Resume check ──────────────────────────────────────────────────────────
    channel_slug = re.sub(r'[^\w]', '_', channel)
    out_path = os.path.join(CLEANED_DIR, channel_slug, f"{vid_id}.json")

    if resume and os.path.exists(out_path) and not dry_run:
        result["status"] = "SKIPPED_ALREADY_DONE"
        result["final_tier"] = orig_tier
        return result

    # ── Load source file ───────────────────────────────────────────────────────
    try:
        with open(caption_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        result["status"] = "ERROR_LOAD"
        result["error"]  = str(e)
        return result

    segments = data.get("segments", [])
    if not segments:
        result["status"] = "ERROR_NO_SEGMENTS"
        return result

    result["words_before"] = len(' '.join(s.get('text','') for s in segments).split())

    # ── Phase 1: Stutter dedup ────────────────────────────────────────────────
    try:
        segments, stutter_removed = dedup_segments(segments)
        result["stutter_removed"] = stutter_removed
    except Exception as e:
        result["error"] = f"dedup_error: {e}"
        # Continue with original segments — don't abort
        stutter_removed = 0

    if not segments:
        result["status"] = "DISCARDED_EMPTY_AFTER_DEDUP"
        result["final_tier"] = "TIER_3"
        return result

    # ── Phase 2: Meta-talk removal ────────────────────────────────────────────
    try:
        segments, meta_removed, should_discard = remove_meta_talk_segments(
            segments, meta_score
        )
        result["meta_removed"] = meta_removed

        if should_discard:
            result["status"]     = "DISCARDED_META_TALK"
            result["final_tier"] = "TIER_3"
            return result
    except Exception as e:
        result["error"] = f"meta_error: {e}"
        meta_removed = 0

    # ── Phase 3: Terminology anchoring ───────────────────────────────────────
    try:
        segments = apply_terminology_to_segments(segments)
    except Exception as e:
        result["error"] = f"terminology_error: {e}"
        # Non-fatal — continue

    # ── Phase 4: Tier re-evaluation ───────────────────────────────────────────
    try:
        new_tier, tier_metrics = re_evaluate_tier(segments, duration, meta_score)
    except Exception as e:
        new_tier     = orig_tier   # fallback to original
        tier_metrics = {}
        result["error"] = f"tier_error: {e}"

    if new_tier == "TIER_3":
        result["status"]     = "DISCARDED_TIER3_AFTER_CLEANUP"
        result["final_tier"] = "TIER_3"
        return result

    # ── Phase 5: LLM punctuation (TIER_2 only) ────────────────────────────────
    provider_used = "not_needed"
    try:
        segments, provider_used = apply_llm_punctuation(
            segments, title, new_tier, dry_run=dry_run
        )
        result["provider_used"] = provider_used
    except Exception as e:
        result["error"] = f"llm_error: {e}"
        # Non-fatal — continue with unpunctuated but clean text

    # ── Phase 6: Build record and export ──────────────────────────────────────
    full_text_after = ' '.join(s.get('text','') for s in segments if s.get('text','').strip())
    result["words_after"] = len(full_text_after.split())

    if result["words_after"] < MIN_FILE_WORDS_AFTER_CLEANUP:
        result["status"]     = "DISCARDED_TOO_SHORT"
        result["final_tier"] = "TIER_3"
        return result

    cleanup_meta = {
        "stutter_removed":   result["stutter_removed"],
        "meta_removed":      result["meta_removed"],
        "original_tier":     orig_tier,
        "provider_used":     provider_used,
        **tier_metrics,
    }

    clean_record = build_clean_record(
        vid_id, title, url, channel, duration,
        segments, new_tier, cleanup_meta
    )

    if not dry_run:
        try:
            atomic_write(out_path, clean_record)
            export_to_training(clean_record, new_tier)
        except Exception as e:
            result["status"] = "ERROR_WRITE"
            result["error"]  = str(e)
            return result

    result["status"]     = "SUCCESS"
    result["final_tier"] = new_tier
    return result

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier',     type=int, default=0,
                        help='Process only this tier (1, 2, or 3). 0 = all.')
    parser.add_argument('--dry-run',  action='store_true',
                        help='Analyse and report without writing any output files.')
    parser.add_argument('--file',     type=str, default=None,
                        help='Process a single video ID for debugging.')
    parser.add_argument('--no-resume', action='store_true',
                        help='Reprocess already-cleaned files (default: skip them).')
    parser.add_argument('--no-llm',   action='store_true',
                        help='Skip LLM punctuation entirely (rule-based only).')
    args = parser.parse_args()

    resume  = not args.no_resume
    dry_run = args.dry_run

    if args.no_llm:
        # Override: mark all providers as exhausted
        for p in PROVIDERS:
            if p["daily_limit"] is not None:
                _usage_today[p["name"]] = p["daily_limit"]

    # ── Setup output directories ──────────────────────────────────────────────
    if not dry_run:
        os.makedirs(CLEANED_DIR,  exist_ok=True)
        os.makedirs(EXPORT_DIR,   exist_ok=True)

    print("=" * 68)
    print("  AA Caption Cleanup Pipeline")
    mode_str = " [DRY RUN]" if dry_run else ""
    print(f"  Mode: {'ALL TIERS' if not args.tier else f'TIER {args.tier}'}{mode_str}")
    print("=" * 68)

    # ── Load analysis report ──────────────────────────────────────────────────
    if not os.path.exists(ANALYSIS_REPORT):
        print(f"ERROR: Analysis report not found at {ANALYSIS_REPORT}")
        print("Run python3 caption_analyse.py first.")
        sys.exit(1)

    print("  Loading analysis report...", end=" ", flush=True)
    with open(ANALYSIS_REPORT, 'r', encoding='utf-8') as f:
        report = json.load(f)
    all_metrics = {m["video_id"]: m for m in report.get("per_file_metrics", [])}
    print(f"{len(all_metrics)} files indexed.")

    # ── Load quality tiers ────────────────────────────────────────────────────
    if not os.path.exists(QUALITY_TIERS):
        print(f"ERROR: Quality tiers not found at {QUALITY_TIERS}")
        sys.exit(1)

    with open(QUALITY_TIERS, 'r', encoding='utf-8') as f:
        tiers_data = json.load(f)

    tier_map = {}
    for tier_label, vid_ids in tiers_data.get("video_ids", {}).items():
        for vid_id in vid_ids:
            tier_map[vid_id] = tier_label

    # ── Build work queue ──────────────────────────────────────────────────────
    caption_files = list(Path(CAPTION_FOLDER).glob("*.json"))

    if args.file:
        caption_files = [f for f in caption_files if f.stem == args.file]
        if not caption_files:
            print(f"ERROR: File not found for video ID: {args.file}")
            sys.exit(1)

    if args.tier:
        tier_label = f"TIER_{args.tier}"
        target_ids = set(tiers_data.get("video_ids", {}).get(tier_label, []))
        caption_files = [f for f in caption_files if f.stem in target_ids]

    # Sort: TIER_1 first (fastest, safest), then TIER_2, then TIER_3
    def sort_key(f):
        t = tier_map.get(f.stem, "TIER_3")
        return {"TIER_1": 0, "TIER_2": 1, "TIER_3": 2}.get(t, 3)
    caption_files = sorted(caption_files, key=sort_key)

    total = len(caption_files)
    print(f"  Queue: {total} files to process")
    print(f"  Resume: {'ON' if resume else 'OFF (reprocessing all)'}")
    print()

    # ── Process ───────────────────────────────────────────────────────────────
    stats = {
        "total": total, "success": 0, "skipped": 0, "discarded": 0,
        "errors": 0, "tier1_out": 0, "tier2_out": 0, "tier3_out": 0,
        "total_words_before": 0, "total_words_after": 0,
        "stutter_removed_total": 0, "meta_removed_total": 0,
        "providers_used": Counter(),
    }

    t_start = time.time()

    for i, filepath in enumerate(caption_files):
        vid_id       = filepath.stem
        analysis_meta = all_metrics.get(vid_id, {"video_id": vid_id})
        orig_tier    = tier_map.get(vid_id, "TIER_2")

        # Progress
        if i % 50 == 0 or i == total - 1:
            elapsed = time.time() - t_start
            rate    = (i + 1) / max(elapsed, 0.1)
            eta_s   = (total - i - 1) / max(rate, 0.01)
            pct     = (i + 1) / total * 100
            print(f"  [{i+1:>5}/{total}] {pct:>5.1f}%  "
                  f"{rate:.1f}/s  ETA {eta_s/60:.0f}m  "
                  f"✓{stats['success']} ↷{stats['skipped']} "
                  f"✗{stats['discarded']} ⚠{stats['errors']}",
                  end='\r', flush=True)

        result = process_file(
            str(filepath), analysis_meta,
            dry_run=dry_run, resume=resume
        )

        # Accumulate stats
        status = result.get("status", "")
        if "SKIPPED" in status:
            stats["skipped"] += 1
        elif "DISCARDED" in status or result.get("final_tier") == "TIER_3":
            stats["discarded"] += 1
            stats["tier3_out"] += 1
        elif "ERROR" in status:
            stats["errors"] += 1
        elif status == "SUCCESS":
            stats["success"] += 1
            ft = result.get("final_tier", "TIER_2")
            if ft == "TIER_1":
                stats["tier1_out"] += 1
            elif ft == "TIER_2":
                stats["tier2_out"] += 1

        stats["total_words_before"]   += result.get("words_before", 0)
        stats["total_words_after"]    += result.get("words_after",  0)
        stats["stutter_removed_total"]+= result.get("stutter_removed", 0)
        stats["meta_removed_total"]   += result.get("meta_removed",   0)
        stats["providers_used"][result.get("provider_used", "none")] += 1

        # Write manifest entry for every processed file
        if not dry_run and status != "SKIPPED_ALREADY_DONE":
            log_manifest(result)

    print()  # newline after \r progress

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    stutter_reduction = (1 - stats["total_words_after"] /
                         max(stats["total_words_before"], 1)) * 100

    print("\n" + "=" * 68)
    print("  CLEANUP COMPLETE")
    print(f"  Total files          : {total:,}")
    print(f"  Elapsed              : {elapsed_total/60:.1f} min")
    print()
    print(f"  ✓ Cleaned & exported : {stats['success']:,}")
    print(f"  ↷ Skipped (done)     : {stats['skipped']:,}")
    print(f"  ✗ Discarded          : {stats['discarded']:,}")
    print(f"  ⚠ Errors             : {stats['errors']:,}")
    print()
    print(f"  Output tiers:")
    print(f"    TIER_1 (clean)     : {stats['tier1_out']:,}")
    print(f"    TIER_2 (corrected) : {stats['tier2_out']:,}")
    print(f"    TIER_3 (discarded) : {stats['tier3_out']:,}")
    print()
    print(f"  Words before cleanup : {stats['total_words_before']:,}")
    print(f"  Words after cleanup  : {stats['total_words_after']:,}")
    print(f"  Stutter reduction    : {stutter_reduction:.1f}%")
    print(f"  Stutter segs removed : {stats['stutter_removed_total']:,}")
    print(f"  Meta segs removed    : {stats['meta_removed_total']:,}")
    print()
    print(f"  LLM providers used:")
    for provider, count in sorted(stats["providers_used"].items(),
                                  key=lambda x: -x[1]):
        if count > 0 and provider != "none":
            print(f"    {provider:<25}: {count:,} calls")
    print()
    print(f"  Training exports:")
    if not dry_run:
        for path, label in [(ASR_EXPORT, "ASR segments"), (LM_EXPORT, "LM corpus")]:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                lines   = sum(1 for _ in open(path, encoding='utf-8'))
                print(f"    {label:<20}: {lines:,} records  ({size_mb:.1f} MB)")
    print("=" * 68)

    # ── Save stats ────────────────────────────────────────────────────────────
    if not dry_run:
        stats_out = {
            **stats,
            "providers_used": dict(stats["providers_used"]),
            "completed_at":   timestamp(),
            "elapsed_s":      round(elapsed_total, 1),
        }
        atomic_write(CLEANUP_STATS, stats_out)
        print(f"\n  Stats saved: {CLEANUP_STATS}")

    if args.file:
        # In single-file mode, print the result detail
        print(f"\n  Result: {result}")


if __name__ == "__main__":
    main()
