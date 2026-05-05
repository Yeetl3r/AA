import streamlit as st
import os
import sys
import time
import json
import requests
import re
from datetime import datetime
import numpy as np

# Import core RAG logic from knowledge_check
# Ensure the parent directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import knowledge_check as rag

# Set Page Config for a premium look
st.set_page_config(
    page_title="AA Astrology AI | Self-Evolving Virtual Astrologer",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PREMIUM CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
        color: #f8fafc;
    }
    
    /* Glassmorphism sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -1.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Chat styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        backdrop-filter: blur(8px);
    }
    
    .stChatMessage[data-testid="stChatMessageUser"] {
        background: rgba(99, 102, 241, 0.1) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
    }

    /* Sources expander */
    .stExpander {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(to right, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #818cf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE MANAGEMENT ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
    st.session_state.meta = None
    st.session_state.chunks = None

# --- INITIALIZATION ---
@st.cache_resource
def load_rag_engine():
    """Initial load of the corpus and embedding index."""
    chunks, count = rag.load_corpus()
    vectors, meta = rag.load_index()
    if vectors is None:
        # If no index, we must build it (could take time)
        st.info("Building knowledge index for the first time. This may take a few minutes...")
        rag.build_index(chunks)
        vectors, meta = rag.load_index()
    return chunks, vectors, meta

# --- SIDEBAR UI ---
with st.sidebar:
    st.markdown("<h1 style='font-size: 1.5rem; margin-bottom: 0;'>🔮 AA Astrology</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.8rem;'>Self-Evolving Virtual Astrologer</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("Config")
    llm_model = st.selectbox("LLM Brain", ["qwen2.5:7b", "qwen2.5:14b", "llama3.1:8b"], index=0)
    
    st.markdown("---")
    st.subheader("Knowledge Stats")
    
    # Load engine once
    if st.session_state.chunks is None:
        with st.spinner("Syncing knowledge..."):
            chunks, vectors, meta = load_rag_engine()
            st.session_state.chunks = chunks
            st.session_state.vectors = vectors
            st.session_state.meta = meta
            
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks", f"{len(st.session_state.chunks):,}")
    with col2:
        num_vids = len(set(c['video_id'] for c in st.session_state.chunks))
        st.metric("Sources", f"{num_vids:,}")
        
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN PAGE UI ---
st.markdown("<h1 class='main-header'>Self-Evolving Virtual Astrologer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Expert consultation powered by 1,600+ Tamil Astrology master transcripts</p>", unsafe_allow_html=True)

# Display Chat History
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 Sources for this answer"):
                for s_idx, s in enumerate(message["sources"]):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"**{s['title']}**")
                        st.markdown(f"<span style='color: #94a3b8; font-size: 0.8rem;'>Channel: {s['channel']} | Match: {s['score']:.2f}</span>", unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"[Watch on YouTube]({s['url']})")
                    if s_idx < len(message["sources"]) - 1:
                        st.divider()

# Chat Input Area
if prompt := st.chat_input("கேள்வி கேளுங்கள்... (e.g., சந்திரிகா யோகம் என்றால் என்ன?)"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Consulting knowledge base..."):
            # 1. Hybrid Retrieval
            t_start = time.time()
            results = rag.retrieve(prompt, st.session_state.vectors, st.session_state.meta, rag.EMBED_MODEL)
            t_ret = time.time() - t_start
            
            if not results:
                response = "இந்த தகவல் என்னிடம் இல்லை. (No relevant information found in the master transcripts.)"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                context = rag.build_context(results)
                
                # 2. LLM Generation with Streaming
                response_placeholder = st.empty()
                full_response = ""
                
                user_msg = f"Provided text:\n\n{context}\n\nQuestion: {prompt}"
                payload = {
                    "model": llm_model,
                    "messages": [
                        {"role": "system", "content": rag.SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": True,
                    "options": {
                        "temperature": 0.3,
                        "num_ctx": 8192
                    }
                }
                
                try:
                    r = requests.post(f"{rag.OLLAMA_BASE}/api/chat", json=payload, stream=True, timeout=120)
                    for line in r.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if 'message' in chunk and 'content' in chunk['message']:
                                token = chunk['message']['content']
                                full_response += token
                                response_placeholder.markdown(full_response + "▌")
                            if chunk.get('done'):
                                break
                    
                    response_placeholder.markdown(full_response)
                    
                    # 3. Format sources for the UI
                    sources = []
                    seen_vids = set()
                    for score, chunk in results:
                        vid = chunk['video_id']
                        if vid not in seen_vids:
                            seen_vids.add(vid)
                            sources.append({
                                'title': chunk['title'],
                                'channel': chunk['channel'],
                                'url': chunk.get('url', f"https://youtube.com/watch?v={vid}"),
                                'score': score
                            })
                    
                    # Add result to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
                    
                    # Force rerun to show the expander properly
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Generation Error: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>Built with ❤️ for Tamil Astrology community | M4 Optimized RAG Engine</p>", unsafe_allow_html=True)
