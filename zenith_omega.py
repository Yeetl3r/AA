"""
zenith_omega.py — Multi-Agent Self-Healing Transcription Pipeline
=================================================================
Three specialized agents in a recursive loop:
  1. PROSECUTOR: CoT structural audit via Gemini (flags failures with reasons)
  2. ENGINEER:   Mutates transcription parameters based on failure diagnosis
  3. JUDGE:      Semantic delta analysis, produces "Golden Transcript"
"""

import os
import sys
import json
import math
import re
import gc
import time
import random
import datetime
import urllib.request
import argparse

import numpy as np

# Lazy-import heavy ML modules only when Engineer needs them
_engine = None

def get_engine():
    """Lazy-load transcribe_engine to avoid loading MLX/torch when only auditing."""
    global _engine
    if _engine is None:
        import transcribe_engine
        _engine = transcribe_engine
    return _engine

# --- CONFIGURATION ---
BASE_DIR = "astrologer_data_hybrid"
OMEGA_LOG = os.path.join(BASE_DIR, "omega_audit.log")
QUARANTINE_DIR = os.path.join(BASE_DIR, "review", "OMEGA_QUARANTINE")

# =====================================================================
#  UTILITY FUNCTIONS 
# =====================================================================

def get_api_key():
    """Pull a random API key from the environment pool."""
    keys_str = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", ""))
    keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    if not keys:
        print("❌ FATAL: No GEMINI_API_KEY or GEMINI_API_KEYS set in environment.")
        sys.exit(1)
    return random.choice(keys)


def call_gemini(prompt, expect_json=False):
    """Send a prompt to Gemini 2.5 Flash and return the response text."""
    api_key = get_api_key()
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    
    if expect_json:
        payload["generationConfig"] = {
            "responseMimeType": "application/json"
        }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(endpoint, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            res_json = json.loads(response.read())
            reply = res_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
            return reply
    except Exception as e:
        return f"API_ERROR: {str(e)}"


def log_omega(vid_id, phase, verdict, detail=""):
    """Append to the omega audit log."""
    ts = datetime.datetime.now().isoformat()
    with open(OMEGA_LOG, "a", encoding="utf-8") as f:
        f.write(f"{ts} | {vid_id} | {phase} | {verdict} | {detail}\n")


def scan_production_db(limit=None):
    """Walk the production DB and yield (filepath, data) for eligible JSON files."""
    count = 0
    for root, dirs, files in os.walk(BASE_DIR):
        # Skip review folders
        if "review" in root:
            continue
        for fname in files:
            if not fname.endswith(".json"):
                continue
            filepath = os.path.join(root, fname)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Skip already certified files
                if data.get("metadata", {}).get("omega_certified"):
                    continue
                yield filepath, data
                count += 1
                if limit and count >= limit:
                    return
            except Exception:
                continue


# =====================================================================
#  AGENT 1: THE PROSECUTOR 
# =====================================================================

PROSECUTOR_PROMPT_TEMPLATE = """
SYSTEM ROLE: You are a forensic Tamil speech-to-text auditor. Think step by step.

TASK: Perform a structural integrity audit on this Tamil transcription from an astrology lecture.

ANALYSIS STEPS:
1. Read the first 100 words. Do they form coherent Tamil sentences about astrology?
2. Read the middle section (words 100-250). Does the topic flow naturally from the opening?
3. Read the final 100 words. Do they maintain semantic coherence, or do they decay into repetition?
4. Check for character-level gluing (e.g., same character repeated 5+ times like 'லில்லிலில்லில').
5. Check for phonetic loops (same 3-5 word phrase repeated 5+ times consecutively).

CLASSIFICATION RULES:
- "SILENCE_LOOPING": Extended passages of the same word/phrase repeating with no semantic content
- "LOW_CONFIDENCE": Text is real Tamil but contains suspicious fragments that don't connect logically  
- "STRUCTURAL_DECAY": Text starts coherent but degrades into gibberish in the second half
- "PHONETIC_GLUING": Character-level repetition artifacts (not word-level)
- "NONE": The text is a legitimate, coherent Tamil astrology discourse

OUTPUT FORMAT (respond with valid JSON only, no markdown):
{{
  "certainty_score": <float 0.0 to 1.0>,
  "verdict": "CERTIFIED" or "FLAGGED",
  "failure_reason": "NONE" or "SILENCE_LOOPING" or "LOW_CONFIDENCE" or "STRUCTURAL_DECAY" or "PHONETIC_GLUING",
  "clean_cut_index": <word index where decay begins, or -1 if clean>,
  "reasoning": "<1-2 sentence chain-of-thought explanation>"
}}

TEXT TO AUDIT:
{text_sample}
"""


def prosecutor_audit(data):
    """Run the Prosecutor agent on a transcript. Returns structured verdict dict."""
    full_text = data.get("full_text", "")
    words = full_text.split()
    
    # Send a representative sample: first 150 + middle 150 + last 150 words 
    total = len(words)
    if total <= 450:
        sample = full_text
    else:
        head = " ".join(words[:150])
        mid_start = max(0, total // 2 - 75)
        mid = " ".join(words[mid_start:mid_start + 150])
        tail = " ".join(words[-150:])
        sample = f"[OPENING]\n{head}\n\n[MIDDLE]\n{mid}\n\n[ENDING]\n{tail}"
    
    prompt = PROSECUTOR_PROMPT_TEMPLATE.format(text_sample=sample)
    raw_reply = call_gemini(prompt, expect_json=True)
    
    if raw_reply.startswith("API_ERROR"):
        return {"certainty_score": 0.5, "verdict": "FLAGGED", "failure_reason": "API_ERROR", 
                "clean_cut_index": -1, "reasoning": raw_reply}
    
    try:
        # Try to parse JSON from the response
        verdict = json.loads(raw_reply)
        # Validate required fields
        if "certainty_score" not in verdict or "verdict" not in verdict:
            raise ValueError("Missing required fields")
        return verdict
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to extract JSON from markdown code blocks
        json_match = re.search(r'\{[\s\S]*\}', raw_reply)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # Complete fallback
        return {"certainty_score": 0.5, "verdict": "FLAGGED", "failure_reason": "PARSE_ERROR",
                "clean_cut_index": -1, "reasoning": f"Could not parse: {raw_reply[:200]}"}


# =====================================================================
#  AGENT 2: THE ENGINEER
# =====================================================================

def engineer_suggest_params(failure_reason, data):
    """Based on the Prosecutor's diagnosis, suggest mutated transcription parameters."""
    title = data.get("metadata", {}).get("title", "")
    
    if failure_reason == "SILENCE_LOOPING":
        return {
            "use_vad": True,
            "condition_on_previous_text": False,
            "temperature": (0.0,),
            "strategy": "VAD + context-free"
        }
    
    elif failure_reason == "LOW_CONFIDENCE":
        # Build a dynamic initial prompt from the video title to prime Whisper's weights
        initial_prompt = (
            f"இது ஒரு ஜோதிடம் பற்றிய தமிழ் விரிவுரை. "
            f"தலைப்பு: {title}. "
            f"கிரகங்கள், ராசி, நட்சத்திரம், தசா புக்தி பற்றிய விளக்கம்."
        )
        return {
            "use_vad": True,
            "condition_on_previous_text": True,
            "temperature": (0.0,),
            "initial_prompt": initial_prompt,
            "strategy": "Title-primed context"
        }
    
    elif failure_reason == "STRUCTURAL_DECAY":
        return {
            "use_vad": True,
            "condition_on_previous_text": False,
            "temperature": (0.2,),
            "strategy": "Thermal variance + context-free"
        }
    
    elif failure_reason == "PHONETIC_GLUING":
        return {
            "use_vad": True,
            "condition_on_previous_text": False,
            "temperature": (0.0, 0.2, 0.4),
            "strategy": "Multi-temp fallback sweep"
        }
    
    else:
        # Generic retry
        return {
            "use_vad": True,
            "condition_on_previous_text": False,
            "temperature": (0.0,),
            "strategy": "Generic context-free retry"
        }


def engineer_retry(data, params):
    """Re-transcribe a video using mutated parameters. Returns new full_text or None."""
    engine = get_engine()
    
    vid_id = data.get("metadata", {}).get("video_id")
    if not vid_id:
        return None
    
    url = f"https://www.youtube.com/watch?v={vid_id}"
    
    print(f"    🔧 [Engineer] Strategy: {params.get('strategy', 'unknown')}")
    print(f"       Params: temp={params.get('temperature')}, vad={params.get('use_vad')}, copt={params.get('condition_on_previous_text')}")
    
    result = engine.transcribe_video(url, vid_id, params)
    
    if result is None or isinstance(result, str):
        return None
    
    raw_text = result.get("text", "")
    final_text = engine.denoise_loops(raw_text)
    segments = result.get("segments", [])
    for seg in segments:
        seg["text"] = engine.denoise_loops(seg.get("text", ""))
        
    return {"full_text": final_text, "segments": segments}


# =====================================================================
#  AGENT 3: THE JUDGE
# =====================================================================

JUDGE_PROMPT_TEMPLATE = """
SYSTEM ROLE: Semantic Consensus Judge for Tamil transcription quality.

You are given two transcriptions (A and B) of the same Tamil astrology video.
Both were generated by Whisper but with different parameters.

TASK:
1. Compare both versions semantically (not character-by-character).
2. If they convey the same meaning with ≥90% overlap → verdict: "STABLE"
3. If they diverge significantly → identify which version has better quality in each section.
4. When merging, keep the version that reads as more natural, coherent Tamil.

IMPORTANT: The "golden_text" field must contain the COMPLETE final transcript text.
If verdict is "STABLE", set golden_text to Candidate A's text.
If verdict is "MERGED", combine the best sections from both.

OUTPUT FORMAT (respond with valid JSON only, no markdown):
{{
  "verdict": "STABLE" or "MERGED",
  "similarity_pct": <float 0-100>,
  "golden_text": "<the final complete transcript text>",
  "merge_notes": "<explanation of what was kept from A vs B>"
}}

CANDIDATE A (Original):
{candidate_a}

CANDIDATE B (Re-transcribed):
{candidate_b}
"""


def judge_compare(original_text, candidate_b_text):
    """Compare two transcription candidates and return consensus."""
    # Truncate to avoid token limits — send first 600 words of each
    a_words = original_text.split()
    b_words = candidate_b_text.split()
    
    a_sample = " ".join(a_words[:600])
    b_sample = " ".join(b_words[:600])
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(candidate_a=a_sample, candidate_b=b_sample)
    raw_reply = call_gemini(prompt, expect_json=True)
    
    if raw_reply.startswith("API_ERROR"):
        return {"verdict": "STABLE", "similarity_pct": 0, "golden_text": original_text,
                "merge_notes": f"API error, keeping original: {raw_reply}"}
    
    try:
        result = json.loads(raw_reply)
        if "verdict" not in result:
            raise ValueError("Missing verdict")
        # If STABLE and golden_text is missing/truncated, use original
        if result.get("verdict") == "STABLE" and not result.get("golden_text"):
            result["golden_text"] = original_text
        return result
    except (json.JSONDecodeError, ValueError):
        json_match = re.search(r'\{[\s\S]*\}', raw_reply)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"verdict": "STABLE", "similarity_pct": 0, "golden_text": original_text,
                "merge_notes": f"Parse error, keeping original: {raw_reply[:200]}"}


# =====================================================================
#  ORCHESTRATOR: THE MAIN LOOP
# =====================================================================

def stamp_certified(filepath, data, merge_notes=""):
    """Write the omega_certified stamp into the JSON file."""
    data["metadata"]["omega_certified"] = True
    data["metadata"]["omega_timestamp"] = datetime.datetime.now().isoformat()
    if merge_notes:
        data["metadata"]["omega_merge_notes"] = merge_notes
    
    temp_path = filepath + ".omega_tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.rename(temp_path, filepath)


def quarantine(filepath, data, reason):
    """Move a file to the OMEGA_QUARANTINE folder."""
    os.makedirs(QUARANTINE_DIR, exist_ok=True)
    dest = os.path.join(QUARANTINE_DIR, os.path.basename(filepath))
    data["metadata"]["omega_quarantine_reason"] = reason
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.rename(filepath, dest)
    print(f"    🔒 Quarantined -> {os.path.basename(dest)}")


def run_omega(limit=None, skip_retranscribe=False):
    """Execute the full Zenith-Omega sweep."""
    
    print("\n" + "="*70)
    print("  ⚡ ZENITH-OMEGA: Multi-Agent Self-Healing Pipeline")
    print("="*70)
    
    stats = {"scanned": 0, "certified": 0, "flagged": 0, "retried": 0, 
             "merged": 0, "quarantined": 0, "skipped": 0}
    
    consecutive_api_fails = 0
    
    for filepath, data in scan_production_db(limit=limit):
        vid_id = data.get("metadata", {}).get("video_id", "???")
        title = data.get("metadata", {}).get("title", "???")[:40]
        stats["scanned"] += 1
        
        print(f"\n[{stats['scanned']}] {title}... ({vid_id})")
        
        # ── Phase I: PROSECUTOR ──
        print(f"    🔴 [Prosecutor] Auditing...", end="", flush=True)
        verdict = prosecutor_audit(data)
        score = verdict.get("certainty_score", 0)
        reason = verdict.get("failure_reason", "UNKNOWN")
        reasoning = verdict.get("reasoning", "")
        
        print(f" Score: {score:.2f} | {verdict.get('verdict', '?')} | {reason}")
        if reasoning:
            print(f"       └─ {reasoning}")
        
        log_omega(vid_id, "PROSECUTOR", verdict.get("verdict", "?"), f"score={score} reason={reason}")
        
        # Check for API errors
        if reason == "API_ERROR" or reason == "PARSE_ERROR":
            consecutive_api_fails += 1
            stats["skipped"] += 1
            if consecutive_api_fails >= 5:
                print("\n🚨 FATAL: 5 consecutive API failures. Aborting Omega sweep.")
                break
            continue
        else:
            consecutive_api_fails = 0
        
        # Auto-certify high-confidence files
        if score >= 0.85 and verdict.get("verdict") == "CERTIFIED":
            stamp_certified(filepath, data)
            stats["certified"] += 1
            print(f"    ✅ Certified (score {score:.2f})")
            continue
        
        # ── Phase II: ENGINEER ──
        stats["flagged"] += 1
        
        if skip_retranscribe:
            print(f"    ⏭️  Skipping re-transcription (audit-only mode)")
            log_omega(vid_id, "ENGINEER", "SKIPPED", reason)
            continue
        
        params = engineer_suggest_params(reason, data)
        print(f"    🔧 [Engineer] Mutating parameters...")
        
        candidate_b = engineer_retry(data, params)
        stats["retried"] += 1
        
        if candidate_b is None:
            print(f"    ⚠️  Engineer retry failed. Quarantining.")
            quarantine(filepath, data, f"ENGINEER_RETRY_FAILED: {reason}")
            stats["quarantined"] += 1
            log_omega(vid_id, "ENGINEER", "RETRY_FAILED", reason)
            continue
        
        log_omega(vid_id, "ENGINEER", "RETRY_SUCCESS", params.get("strategy", ""))
        
        # ── Phase III: JUDGE ──
        original_text = data.get("full_text", "")
        new_text = candidate_b.get("full_text", "")
        
        print(f"    ⚖️  [Judge] Cross-examining A vs B...", end="", flush=True)
        consensus = judge_compare(original_text, new_text)
        judge_verdict = consensus.get("verdict", "STABLE")
        similarity = consensus.get("similarity_pct", 0)
        
        print(f" {judge_verdict} (similarity: {similarity}%)")
        log_omega(vid_id, "JUDGE", judge_verdict, f"sim={similarity}%")
        
        if judge_verdict == "STABLE":
            # Original is fine — stamp it
            stamp_certified(filepath, data, f"Judge confirmed stable ({similarity}% similar)")
            stats["certified"] += 1
            print(f"    ✅ Certified (original confirmed stable)")
        else:
            # Merge: replace text with golden version
            golden_text = consensus.get("golden_text", "")
            if golden_text and len(golden_text) > 50:
                data["full_text"] = golden_text
                # Keep original segments but update text provenance
                data["metadata"]["omega_original_text_len"] = len(original_text)
                data["metadata"]["omega_golden_text_len"] = len(golden_text)
                stamp_certified(filepath, data, consensus.get("merge_notes", "Merged by Judge"))
                stats["merged"] += 1
                print(f"    🔀 Golden Transcript written ({len(golden_text)} chars)")
            else:
                # Golden text too short or missing — keep original
                stamp_certified(filepath, data, "Judge merge produced insufficient text, keeping original")
                stats["certified"] += 1
                print(f"    ✅ Certified (merge insufficient, keeping original)")
        
        # Memory management between files
        gc.collect()
        time.sleep(1)  # Rate-limit API calls
    
    # ── FINAL REPORT ──
    print("\n" + "="*70)
    print("  📊 ZENITH-OMEGA SWEEP COMPLETE")
    print("="*70)
    print(f"  Scanned:      {stats['scanned']}")
    print(f"  Certified:    {stats['certified']}")
    print(f"  Flagged:      {stats['flagged']}")
    print(f"  Re-tried:     {stats['retried']}")
    print(f"  Merged:       {stats['merged']}")
    print(f"  Quarantined:  {stats['quarantined']}")
    print(f"  Skipped:      {stats['skipped']}")
    print(f"\n  Log: {OMEGA_LOG}")
    print("="*70)


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Zenith-Omega Multi-Agent QA Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Max files to process (for testing)")
    parser.add_argument("--audit-only", action="store_true", help="Run Prosecutor only, skip re-transcription")
    args = parser.parse_args()
    
    run_omega(limit=args.limit, skip_retranscribe=args.audit_only)


if __name__ == "__main__":
    main()
