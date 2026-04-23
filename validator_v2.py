import math
import re

def calculate_uwr(text):
    if not text: return 0.0
    words = text.split()
    if not words: return 0.0
    words = [w.strip(".,?!;:-") for w in words if w.strip(".,?!;:-")]
    if not words: return 0.0
    return len(set(words)) / len(words)

def sliding_uwr(words, window_size=100):
    if len(words) < window_size:
        return calculate_uwr(" ".join(words))
    min_uwr = 1.0
    for i in range(len(words) - window_size + 1):
        window_words = words[i:i+window_size]
        window_uwr = calculate_uwr(" ".join(window_words))
        if window_uwr < min_uwr:
            min_uwr = window_uwr
    return min_uwr

def check_prompt_contamination(words, title, window_size=50):
    if not title or len(words) < window_size: return False
    title_words = set([w.strip(".,?!;:-") for w in title.split() if w.strip(".,?!;:-")])
    if not title_words: return False
    for i in range(len(words) - window_size + 1):
        window_words = set([w.strip(".,?!;:-") for w in words[i:i+window_size] if w.strip(".,?!;:-")])
        if not window_words: continue
        intersection = title_words.intersection(window_words)
        union = title_words.union(window_words)
        jaccard = len(intersection) / len(union) if union else 0
        if jaccard > 0.8: return True
    return False

def validate_transcription(data):
    """Returns (category, metrics_dict). 
    Category is one of: SUCCESS, GHOST, SYLLABLE_TRAP, PROMPT_CONTAMINATED, SANDWICH_LOOP, TERMINAL_LOOP.
    """
    full_text = data.get("full_text", "")
    segments = data.get("segments", [])
    title = data.get("metadata", {}).get("title", "")
    
    # NaN Check
    has_nan = False
    for seg in segments:
        lgp = seg.get("avg_logprob")
        if lgp is not None:
            try:
                if math.isnan(float(lgp)):
                    has_nan = True
                    break
            except: pass

    words = full_text.split()
    global_uwr = calculate_uwr(full_text)
    
    # Tail UWR (Whisper context collapse)
    tail_words = words[-50:] if len(words) >= 50 else words
    tail_uwr = calculate_uwr(" ".join(tail_words)) if len(tail_words) > 0 else 0.0
    
    # CPS (Characters Per Second)
    final_time = segments[-1].get("end", 0) if segments else 0
    cps = len(full_text) / final_time if final_time > 0 else 0.0
    
    # Glued syllables
    glued_matches = re.findall(r'(.)\1{4,}', full_text)
    glued_count = len(glued_matches)
    
    # Sliding window
    min_sliding_uwr = sliding_uwr(words, window_size=100)
    
    # Jaccard title match
    prompt_contamination = check_prompt_contamination(words, title, window_size=50)
    
    category = "SUCCESS"
    if has_nan or len(full_text) < 10:
        category = "GHOST"
    elif glued_count >= 3:
        category = "SYLLABLE_TRAP"
    elif prompt_contamination:
        category = "PROMPT_CONTAMINATED"
    elif min_sliding_uwr < 0.25:
        category = "SANDWICH_LOOP"
    elif global_uwr > 0.4 and tail_uwr < 0.2:
        category = "TERMINAL_LOOP"
    elif global_uwr < 0.40 or cps > 12.0:
        category = "TERMINAL_LOOP"
        
    return category, {
        "global_uwr": global_uwr,
        "tail_uwr": tail_uwr,
        "cps": cps,
        "glued_count": glued_count,
        "min_sliding_uwr": min_sliding_uwr,
        "prompt_contamination": prompt_contamination
    }
