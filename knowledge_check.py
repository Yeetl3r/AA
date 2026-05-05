"""
knowledge_check.py — Tamil Astrology Knowledge Base Chat
Part of the AA Pipeline.

Uses cleaned caption files as a knowledge database.
Retrieves relevant segments via embedding similarity, then answers via local Ollama model.

HARDWARE RECOMMENDATION (M4 Air 16GB, no other apps):
  IDE + Python running      → use qwen2.5:7b   (4.5GB model, safe headroom)
  Terminal + Python only    → use qwen2.5:14b  (9GB model, ~6.5GB left for embeddings)

  Embedding model (always): nomic-embed-text (274MB, runs alongside either LLM)

SETUP (run once):
  ollama pull nomic-embed-text
  ollama pull qwen2.5:7b       # or qwen2.5:14b if terminal-only

USAGE:
  python3 knowledge_check.py                        # interactive chat
  python3 knowledge_check.py --model qwen2.5:14b   # use 14b model
  python3 knowledge_check.py --rebuild              # force rebuild index
  python3 knowledge_check.py --query "ராகு தசை பலன்கள்"  # single query
  python3 knowledge_check.py --stats               # show corpus stats only
"""

import os
import re
import sys
import json
import time
import struct
import hashlib
import argparse
import unicodedata
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
BASE_DIR       = "/Volumes/Storage Drive/AA"
DATA_SOURCE    = os.path.join(BASE_DIR, "astrologer_data_hybrid", "cleaned")
INDEX_DIR      = os.path.join(BASE_DIR, "astrologer_data_hybrid", "knowledge_index")
INDEX_DIR      = os.path.join(BASE_DIR, "astrologer_data_hybrid", "knowledge_index")

OLLAMA_BASE    = "http://localhost:11434"
EMBED_MODEL    = "nomic-embed-text"
DEFAULT_LLM    = "qwen2.5:7b"

# RAG parameters (tuned for Tamil astrology)
TOP_K           = 8      # segments to retrieve per query
MIN_SCORE       = 0.30   # minimum cosine similarity to include
MAX_CONTEXT_WORDS = 600  # cap context sent to LLM
CHUNK_WORDS     = 60     # target words per indexed chunk

TAMIL_RANGE = (0x0B80, 0x0BFF)

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Tamil astrology expert. Answer ONLY using the text provided below.

Strict rules:
1. Always answer in Tamil, even if the question is in English.
2. Use facts from the provided text. If the text is partially relevant, provide the best possible answer based on it.
3. If the answer is absolutely not in the provided text, respond exactly: "இந்த தகவல் என்னிடம் இல்லை"
4. Keep answers concise and clear. Do NOT generate follow-up questions.
5. Do NOT repeat the question or add any preamble before your answer.
6. Use correct astrological Tamil terminology."""

# ── HELPERS ───────────────────────────────────────────────────────────────────

def tamil_ratio(text):
    alpha = [c for c in text if unicodedata.category(c).startswith('L')]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if TAMIL_RANGE[0] <= ord(c) <= TAMIL_RANGE[1]) / len(alpha)

def check_ollama():
    """Verify Ollama is running and required models are available."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = {m['name'].split(':')[0] for m in r.json().get('models', [])}
        return models
    except Exception:
        return None

def pull_model_if_needed(model_name, available):
    """Pull a model if not already downloaded."""
    base = model_name.split(':')[0]
    if base not in available:
        print(f"  Pulling {model_name}... (this may take a few minutes)")
        try:
            r = requests.post(
                f"{OLLAMA_BASE}/api/pull",
                json={"name": model_name, "stream": False},
                timeout=300
            )
            return r.status_code == 200
        except Exception as e:
            print(f"  Pull failed: {e}")
            return False
    return True

def cosine_similarity(a, b):
    """Cosine similarity between two numpy vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# ── CORPUS LOADING ─────────────────────────────────────────────────────────────

def load_corpus():
    """
    Load all caption files into a flat list of chunks.
    Priority: cleaned/ directory first, then captions/ as fallback.

    Returns list of dicts:
      {text, video_id, title, channel, start, end, source}
    """
    chunks = []

    # Use specified cleaned data source
    source_dir = DATA_SOURCE

    def _load_dir(directory, source_label):
        loaded = 0
        for json_path in Path(directory).rglob("*.json"):
            if json_path.name.endswith('.tmp'):
                continue
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                meta     = data.get('metadata', {})
                segments = data.get('segments', [])
                vid_id   = meta.get('video_id', json_path.stem)
                title    = meta.get('title', '')
                channel  = meta.get('channel', '').split('@')[-1].split('/')[0]
                full_text = data.get('full_text', '')

                # If no segments, chunk full_text directly
                if not segments and full_text:
                    words = full_text.split()
                    for i in range(0, len(words), CHUNK_WORDS):
                        chunk_text = ' '.join(words[i:i+CHUNK_WORDS])
                        if len(chunk_text.split()) >= 5:
                            chunks.append({
                                'text':     chunk_text,
                                'video_id': vid_id,
                                'title':    title,
                                'channel':  channel,
                                'start':    0.0,
                                'end':      0.0,
                                'source':   source_label,
                                'url':      meta.get('url', ''),
                            })
                    loaded += 1
                    continue

                # Use segments — merge into CHUNK_WORDS chunks
                buf = []
                buf_words = 0
                buf_start = None

                def flush():
                    if not buf:
                        return
                    chunk_text = ' '.join(s.get('text','') for s in buf).strip()
                    if len(chunk_text.split()) >= 5 and tamil_ratio(chunk_text) > 0.3:
                        chunks.append({
                            'text':     chunk_text,
                            'video_id': vid_id,
                            'title':    title,
                            'channel':  channel,
                            'start':    buf[0].get('start', 0.0),
                            'end':      buf[-1].get('end',   0.0),
                            'url':      meta.get('url', ''),
                            'source':   source_label,
                        })

                for seg in segments:
                    text = seg.get('text', '').strip()
                    if not text:
                        continue
                    wc = len(text.split())
                    buf.append(seg)
                    buf_words += wc
                    if buf_words >= CHUNK_WORDS:
                        flush()
                        buf = []
                        buf_words = 0
                flush()
                loaded += 1
            except Exception as e:
                pass   # skip corrupt files silently
        return loaded

    count = _load_dir(source_dir, 'cleaned')
    return chunks, count

# ── EMBEDDING INDEX ────────────────────────────────────────────────────────────

INDEX_VECTORS_FILE = os.path.join(INDEX_DIR, "vectors.npy")
INDEX_META_FILE    = os.path.join(INDEX_DIR, "meta.json")
INDEX_HASH_FILE    = os.path.join(INDEX_DIR, "corpus_hash.txt")

def corpus_hash(chunks):
    """Fingerprint the corpus to detect if rebuild is needed."""
    sample = ''.join(c['text'][:50] for c in chunks[:100])
    return hashlib.md5((sample + str(len(chunks))).encode()).hexdigest()

def embed_text(text, model=EMBED_MODEL):
    """Get embedding vector from Ollama."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30
        )
        return np.array(r.json()['embedding'], dtype=np.float32)
    except Exception:
        return None

def build_index(chunks, embed_model=EMBED_MODEL):
    """
    Embed all chunks and save index to disk.
    Uses multi-threading to speed up Ollama embedding calls.
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    print(f"\n  Building knowledge index ({len(chunks):,} chunks)...")
    print(f"  Embedding model: {embed_model}")
    
    # M4 can easily handle 10-20 parallel requests to Ollama
    MAX_WORKERS = 10 
    print(f"  Parallel workers: {MAX_WORKERS}")

    vectors_map = {} # index -> vector
    meta    = []
    t_start = time.time()

    def process_chunk(idx, chunk):
        vec = embed_text(chunk['text'], embed_model)
        return idx, vec

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_chunk, i, c): i for i, c in enumerate(chunks)}
        
        done_count = 0
        for future in as_completed(futures):
            idx, vec = future.result()
            if vec is not None:
                vectors_map[idx] = vec
            
            done_count += 1
            if done_count % 100 == 0 or done_count == len(chunks):
                elapsed = time.time() - t_start
                rate    = done_count / max(elapsed, 0.1)
                eta_s   = (len(chunks) - done_count) / max(rate, 0.01)
                print(f"  [{done_count:>6}/{len(chunks)}]  {rate:.1f}/s  ETA {eta_s/60:.1f}m",
                      end='\r', flush=True)

    # Reassemble in order
    sorted_vectors = []
    final_meta = []
    for i in range(len(chunks)):
        if i in vectors_map:
            sorted_vectors.append(vectors_map[i])
            # Keep meta but ensure text is included
            m = {k: v for k, v in chunks[i].items() if k != 'text'}
            m['text'] = chunks[i]['text']
            final_meta.append(m)

    print(f"\n  Embedded {len(sorted_vectors):,} chunks ({len(chunks) - len(sorted_vectors)} failed)")

    if not sorted_vectors:
        print("  ERROR: No embeddings produced. Is Ollama running?")
        return False

    np.save(INDEX_VECTORS_FILE, np.stack(sorted_vectors))
    with open(INDEX_META_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_meta, f, ensure_ascii=False)
    with open(INDEX_HASH_FILE, 'w') as f:
        f.write(corpus_hash(chunks))

    print(f"  Index saved: {len(sorted_vectors):,} vectors")
    return True

def load_index():
    """Load pre-built index from disk. Returns (vectors, meta) or (None, None)."""
    if not (os.path.exists(INDEX_VECTORS_FILE) and os.path.exists(INDEX_META_FILE)):
        return None, None
    try:
        vectors = np.load(INDEX_VECTORS_FILE)
        with open(INDEX_META_FILE, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        return vectors, meta
    except Exception as e:
        print(f"  Index load failed: {e}")
        return None, None

# ── RETRIEVAL ──────────────────────────────────────────────────────────────────

def retrieve(query, vectors, meta, embed_model=EMBED_MODEL, top_k=TOP_K):
    """
    Hybrid retrieval combining vector similarity (Ollama/Nomic) 
    with keyword boosting for technical Tamil terms.
    """
    # 1. Vector Search
    q_vec = embed_text(query, embed_model)
    if q_vec is None:
        return []

    q_norm  = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    v_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    vector_scores = v_norms @ q_norm

    # 2. Keyword Boosting
    # Simple tokenization for Tamil/English
    keywords = re.findall(r'\w+', query.lower())
    # Filter common short words if any, but keep technical terms
    keywords = [k for k in keywords if len(k) > 1]
    
    hybrid_scores = []
    for i, v_score in enumerate(vector_scores):
        chunk = meta[i]
        text_lower  = chunk['text'].lower()
        title_lower = chunk['title'].lower()
        
        kw_boost = 0.0
        for kw in keywords:
            # Title matches are very strong signals
            if kw in title_lower:
                kw_boost += 0.15
            # Text matches are helpful
            if kw in text_lower:
                kw_boost += 0.05
        
        # Combine: Vector score + Keyword boost (capped)
        # Vector score is usually 0.4 - 0.8. We allow boost up to 0.4.
        final_score = float(v_score) + min(kw_boost, 0.4)
        hybrid_scores.append(final_score)

    # 3. Sort and Filter
    hybrid_scores = np.array(hybrid_scores)
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k * 3]
    
    results = []
    seen_texts = set()

    for idx in top_indices:
        score = float(hybrid_scores[idx])
        if score < MIN_SCORE:
            continue

        chunk = meta[idx]
        # Dedup by text similarity
        text_key = chunk['text'][:60]
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        results.append((score, chunk))
        if len(results) >= top_k:
            break

    return results

def build_context(results):
    """
    Build context string from retrieved chunks.
    Groups by video, orders by timestamp, caps at MAX_CONTEXT_WORDS.
    """
    # Group by video
    by_video = {}
    for score, chunk in results:
        vid = chunk['video_id']
        if vid not in by_video:
            by_video[vid] = {'title': chunk['title'], 'chunks': []}
        by_video[vid]['chunks'].append((chunk.get('start', 0), chunk['text'], score))

    context_parts = []
    total_words   = 0

    for vid_id, data in by_video.items():
        if total_words >= MAX_CONTEXT_WORDS:
            break
        # Sort chunks by timestamp
        sorted_chunks = sorted(data['chunks'], key=lambda x: x[0])
        title = data['title'][:60]
        video_text = ' '.join(text for _, text, _ in sorted_chunks)

        words = video_text.split()
        remaining = MAX_CONTEXT_WORDS - total_words
        if remaining <= 0:
            break

        trimmed = ' '.join(words[:remaining])
        context_parts.append(f"[{title}]\n{trimmed}")
        total_words += len(words[:remaining])

    return '\n\n'.join(context_parts)

# ── LLM CALL ──────────────────────────────────────────────────────────────────

def ask_llm(query, context, llm_model=DEFAULT_LLM):
    """
    Send query + retrieved context to local LLM.
    Streams response token by token.
    Returns full response string.
    """
    user_message = f"Provided text:\n\n{context}\n\nQuestion: {query}"

    payload = {
        "model":    llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "stream":      True,
        "options": {
            "temperature":    0.3,
            "num_predict":    350,
            "num_ctx":        6144,
            "repeat_penalty": 1.1,
        }
    }

    try:
        full_response = []
        with requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            stream=True,
            timeout=120
        ) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = chunk.get('message', {}).get('content', '')
                    if token:
                        print(token, end='', flush=True)
                        full_response.append(token)
                    if chunk.get('done'):
                        break
                except Exception:
                    continue
        print()   # newline after streamed response
        return ''.join(full_response)
    except Exception as e:
        return f"LLM error: {e}"

# ── CHAT LOOP ──────────────────────────────────────────────────────────────────

def format_sources(results):
    """Format retrieved sources for display."""
    lines = []
    seen  = set()
    for score, chunk in results:
        vid = chunk['video_id']
        if vid in seen:
            continue
        seen.add(vid)
        title   = chunk['title'][:55]
        channel = chunk['channel'][:25]
        ts      = f"{int(chunk.get('start',0)//60)}:{int(chunk.get('start',0)%60):02d}"
        url     = chunk.get('url', f"https://youtube.com/watch?v={vid}")
        if chunk.get('start', 0) > 0:
            url += f"&t={int(chunk.get('start',0))}s"
        lines.append(f"  [{score:.2f}] {title} ({channel}) @ {ts}")
        lines.append(f"         {url}")
    return '\n'.join(lines)

def interactive_chat(vectors, meta, llm_model, embed_model):
    """Run the interactive chat loop."""
    print(f"\n{'='*64}")
    print(f"  Tamil Astrology Knowledge Check")
    print(f"  LLM: {llm_model}  |  Corpus: {len(meta):,} chunks")
    print(f"  Type your question in Tamil or English.")
    print(f"  Commands: /sources (toggle), /stats, /quit")
    print(f"{'='*64}\n")

    show_sources = True
    history = []

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Exiting. Goodbye!")
            break

        if not query:
            continue

        # Commands
        if query.lower() in ('/quit', '/exit', 'quit', 'exit'):
            print("  Goodbye!")
            break
        if query.lower() == '/sources':
            show_sources = not show_sources
            print(f"  Sources display: {'ON' if show_sources else 'OFF'}")
            continue
        if query.lower() == '/stats':
            channels = {}
            for m in meta:
                ch = m.get('channel', 'unknown')
                channels[ch] = channels.get(ch, 0) + 1
            print(f"  Corpus: {len(meta):,} chunks")
            for ch, n in sorted(channels.items(), key=lambda x: -x[1]):
                print(f"    {ch}: {n:,} chunks")
            continue
        if query.lower() == '/clear':
            history = []
            print("  History cleared.")
            continue

        # Retrieve
        t0      = time.time()
        results = retrieve(query, vectors, meta, embed_model, TOP_K)
        t_ret   = time.time() - t0

        if not results:
            print("\n  Astrologer: No relevant information found in the knowledge base.\n")
            continue

        context = build_context(results)

        # Show sources before answer
        if show_sources:
            print(f"\n  📚 Sources (retrieved in {t_ret:.1f}s):")
            print(format_sources(results))
            print()

        # Generate answer
        print(f"  Astrologer: ", end='', flush=True)
        t0       = time.time()
        response = ask_llm(query, context, llm_model)
        t_llm    = time.time() - t0

        if show_sources:
            print(f"  [{t_llm:.1f}s]\n")
        else:
            print()

        history.append({"q": query, "a": response[:200]})

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str, default=DEFAULT_LLM,
                        help=f'Ollama LLM model (default: {DEFAULT_LLM})')
    parser.add_argument('--embed',   type=str, default=EMBED_MODEL,
                        help=f'Ollama embedding model (default: {EMBED_MODEL})')
    parser.add_argument('--rebuild', action='store_true',
                        help='Force rebuild the embedding index')
    parser.add_argument('--query',   type=str, default=None,
                        help='Single query mode (non-interactive)')
    parser.add_argument('--stats',   action='store_true',
                        help='Show corpus stats and exit')
    parser.add_argument('--topk',    type=int, default=TOP_K,
                        help=f'Number of chunks to retrieve (default: {TOP_K})')
    args = parser.parse_args()

    # ── Check Ollama ──────────────────────────────────────────────────────────
    print("  Checking Ollama...", end=" ", flush=True)
    available = check_ollama()
    if available is None:
        print("NOT RUNNING")
        print("\n  Start Ollama first:")
        print("    ollama serve")
        sys.exit(1)
    print(f"OK ({len(available)} models available)")

    # Pull required models
    for model in [args.embed, args.model]:
        if not pull_model_if_needed(model, available):
            print(f"\n  Could not pull {model}. Run manually:")
            print(f"    ollama pull {model}")
            sys.exit(1)

    # ── Load corpus ───────────────────────────────────────────────────────────
    print("  Loading corpus...", end=" ", flush=True)
    chunks, file_count = load_corpus()
    if not chunks:
        print("NO FILES FOUND")
        print(f"\n  No caption files found in:")
        print(f"    {CLEANED_DIR}")
        print(f"    {CAPTION_DIR}")
        print("  Run caption_cleanup.py or caption_only_harvester.py first.")
        sys.exit(1)
    print(f"{len(chunks):,} chunks from {file_count:,} files")

    if args.stats:
        channels = {}
        for c in chunks:
            ch = c.get('channel', 'unknown')
            channels[ch] = channels.get(ch, 0) + 1
        total_words = sum(len(c['text'].split()) for c in chunks)
        print(f"\n  Corpus statistics:")
        print(f"    Total chunks : {len(chunks):,}")
        print(f"    Total words  : {total_words:,}")
        print(f"    Files loaded : {file_count:,}")
        for ch, n in sorted(channels.items(), key=lambda x: -x[1]):
            print(f"    {ch:<35}: {n:,} chunks")
        return

    # ── Build or load index ───────────────────────────────────────────────────
    vectors, meta = None, None
    need_rebuild  = args.rebuild

    if not need_rebuild and os.path.exists(INDEX_HASH_FILE):
        with open(INDEX_HASH_FILE) as f:
            saved_hash = f.read().strip()
        if saved_hash != corpus_hash(chunks):
            print("  Corpus changed — index needs rebuild.")
            need_rebuild = True

    if not need_rebuild:
        print("  Loading index...", end=" ", flush=True)
        vectors, meta = load_index()
        if vectors is None:
            need_rebuild = True
            print("not found")
        else:
            print(f"{len(vectors):,} vectors loaded")

    if need_rebuild:
        ok = build_index(chunks, args.embed)
        if not ok:
            sys.exit(1)
        vectors, meta = load_index()
        if vectors is None:
            print("  ERROR: Index build succeeded but load failed.")
            sys.exit(1)

    # ── Single query mode ─────────────────────────────────────────────────────
    if args.query:
        print(f"\n  Query: {args.query}")
        results = retrieve(args.query, vectors, meta, args.embed, args.topk)
        if not results:
            print("  No relevant content found.")
            return
        print(f"\n  Sources:")
        print(format_sources(results))
        context = build_context(results)
        print(f"\n  Answer:")
        ask_llm(args.query, context, args.model)
        return

    # ── Interactive chat ──────────────────────────────────────────────────────
    interactive_chat(vectors, meta, args.model, args.embed)


if __name__ == "__main__":
    main()
