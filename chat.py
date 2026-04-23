import chromadb
from chromadb.utils import embedding_functions
import ollama
import sys
import re

# --- CONFIGURATION ---
DB_PATH = "./astrology_db"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_MODEL = "qwen2.5:14b" 

# --- THE "BRAIN" (SYSTEM PROMPT) ---
SYSTEM_INSTRUCTION = """
You are an expert Vedic Astrologer with a deep, human-like understanding of planetary nuances. 
You are NOT a robot. You do not summarize text; you **apply astrological logic** to the user's life/question.

### 1. YOUR AUTHORITY & BELIEF
- Astrology is **factual and absolute**. Never use phrases like "Astrology is not proven."
- Speak with authority. Use direct language ("This WILL happen" or "This indicates...").

### 2. HIERARCHY OF TRUTH (The "Goldmine" Logic)
You have access to 'Aditya Guruji' (Primary) and 'Other Channels' (Secondary).
- **Rule A (Synthesis & Expansion):** The other channels are a goldmine of nuances. You are FREE to use their rules, tips, and details to expand your answer, provided they do not contradict Guruji.
- **Rule B (Conflict Resolution):** ONLY if there is a direct contradiction between Aditya Guruji and another channel, **Aditya Guruji's rule is final**.
- **Rule C (Standalone):** If Aditya Guruji does not cover a specific topic, treat the information from the other channels as **absolute truth**. Do not discard it.
- **Rule D (Timeline):** If a source contradicts itself over time, the **Newer Video** (e.g., 2026) overrides the Older one.

### 3. LANGUAGE & TERMINOLOGY (The "Tanglish" Rule)
- Explain the logic in **English**.
- **MANDATORY:** You MUST use the original Tamil terms for these technical concepts:
  - **Subathuvam** (Beneficence)
  - **Papathuvam** (Maleficence)
  - **Sookshma** (Subtle/Micro)
  - **Lagnam** (Ascendant), **Rasi** (Moon Sign)
  - **Dasha/Bukthi** (Periods)
  - **Aatchi/Ucham/Neecham** (Exaltation/Debilitation)
  - **Digbala** (Directional Strength)

### 4. MISSING DATA PROTOCOL
- If the provided Context is completely empty:
  - You MAY answer using your own general knowledge.
  - **BUT** you must start the answer with this exact tag: **"[⚠️ General Knowledge - Data Not in Videos]"**

### 5. THINKING PROCESS
- Do not just quote the text. **Think.**
- Example: If Guruji gives the main rule and another channel adds a "Sookshma" detail, combine them: "Guruji states that Guru in 8th is weak, but as per the Sookshma rules from [Other Source], if he is with Moon, it becomes valid."

Context from Database:
{context}
"""

def extract_year(title):
    """Helper to find year in video title for sorting."""
    match = re.search(r'202[0-9]', title)
    return int(match.group()) if match else 0

def main():
    print(f"Loading Knowledge Base ({DB_PATH})...")
    
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME, 
        device="cpu" 
    )
    
    try:
        # type: ignore - Pylance strictness fix
        collection = client.get_collection(
            name="tamil_astrology_rules", 
            embedding_function=embedding_func 
        )
    except Exception as e:
        print(f"Error: Database not found. ({e})")
        sys.exit()
    
    print("\n" + "="*60)
    print(f"🌌 Sovereign AI Astrologer ({LLM_MODEL}) is Online.")
    print("   - Strategy: Synthesis Mode (Guruji Baseline + Nuances)")
    print("   - Language: Hybrid (English + Tamil Technical Terms)")
    print("="*60 + "\n")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit']:
            break
            
        print("Consulting the stars...", end="\r")
        
        # 1. Search (Fetch TOP 7 chunks for maximum context synthesis)
        results = collection.query(
            query_texts=[user_query],
            n_results=7 
        )
        
        # 2. Build Context & Sort by Priority
        context_text = ""
        sources = []
        
        # Safe check for documents
        if results.get('documents') and results.get('metadatas'):
            docs = results['documents'][0] # type: ignore
            metas = results['metadatas'][0] # type: ignore
            
            if docs and metas:
                for i, doc in enumerate(docs):
                    # Force conversion to string to satisfy Pylance
                    raw_title = metas[i].get('source_title', 'Unknown Source')
                    title = str(raw_title) 
                    
                    # Check for Aditya Guruji explicitly
                    is_guru = "aditya" in title.lower() or "guruji" in title.lower()
                    source_label = "🌟 ADITYA GURUJI" if is_guru else "ℹ️ Other Source"
                    
                    context_text += f"\n[Source {i+1} | {source_label} | Title: {title}]:\n{doc}\n"
                    
                    source_entry = f"{title}"
                    if source_entry not in sources:
                        sources.append(source_entry)
        
        # 3. Prompting
        prompt = f"User Question: {user_query}\n\n" + SYSTEM_INSTRUCTION.format(context=context_text)
        
        print(f"Found {len(sources)} relevant texts. Synthesizing predictions...\n")

        # 4. Generate
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            stream=True
        )
        
        print("Astrologer: ", end="")
        full_response = ""
        for chunk in stream:
            text = chunk['message']['content']
            print(text, end="", flush=True)
            full_response += text
            
        print("\n\n" + "-"*30)
        print("Sources Analyzed:")
        for s in sources:
            print(f"- {s}")
        print("-" * 30)

if __name__ == "__main__":
    main()