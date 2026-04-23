import os
import json
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
# Use the folder where hybrid_harvester saved the data
DATA_FOLDER = "astrologer_data_hybrid" 

# Where to save the database (On your external drive)
DB_PATH = "./astrology_db"            

# The Best Multilingual Model for Tamil (Downloads once ~2GB)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" 

import manifest_manager

def main():
    # 1. Initialize the Vector Database
    print(f"Initializing ChromaDB at {DB_PATH}...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    # Setup the Embedding Function (The "Translator" from Text to Numbers)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device="mps"  # Correct: Use M4 GPU via Metal Performance Shaders
    )
    
    # Create (or get) the collection
    collection = chroma_client.get_or_create_collection(
        name="tamil_astrology_rules",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"} # Cosine similarity is best for text matching
    )

    # 2. Setup the Text Splitter
    # We want chunks that are long enough to contain a full rule, 
    # but short enough to be specific.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Approx 200-300 Tamil words
        chunk_overlap=200,  # Keep context between chunks
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # 3. Process Files — Use Manifest for O(1) file discovery
    mm = manifest_manager.ManifestManager()
    manifest = mm.get_manifest()
    
    # Filter for SUCCESS results and existing files
    target_entries = [data for vid_id, data in manifest.items() if data.get("category") == "SUCCESS"]
    
    print(f"Found {len(target_entries)} verified videos in manifest.")
    
    if not target_entries:
        print("No successful videos found in manifest! Run harvester first.")
        return

    # --- BATCHED INGESTION ---
    batch_ids = []
    batch_docs = []
    batch_metas = []
    batch_size = 50   # M4 with 16GB handles batch size 50 of e5-large well
    total_chunk_count = 0
    
    for entry in target_entries:
        filepath = entry.get("filepath")
        if not filepath or not os.path.exists(filepath):
            continue
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            meta = data.get("metadata", {})
            video_id = meta.get("video_id", "")
            title = meta.get("title", "Unknown")
            url = f"https://www.youtube.com/watch?v={video_id}"
            full_text = data.get("full_text", "")
            
            if not full_text: continue
            
            # Split text into chunks
            chunks = text_splitter.create_documents([full_text])
            
            for i, chunk in enumerate(chunks):
                batch_ids.append(f"{video_id}_chunk_{i}")
                batch_docs.append(chunk.page_content)
                batch_metas.append({
                    "source_title": title,
                    "source_url": url,
                    "video_id": video_id,
                    "chunk_index": i
                })

                # Flush batch to GPU
                if len(batch_ids) >= batch_size:
                    collection.upsert(
                        ids=batch_ids,
                        documents=batch_docs,
                        metadatas=batch_metas
                    )
                    total_chunk_count += len(batch_ids)
                    print(f"  -> Upserted batch. Total chunks: {total_chunk_count}...", end="\r")
                    batch_ids, batch_docs, batch_metas = [], [], []

        except Exception as e:
            print(f"\nError processing {os.path.basename(filepath)}: {e}")

    # Final Flush
    if batch_ids:
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
        total_chunk_count += len(batch_ids)

    print(f"\nSuccess! Database built with {total_chunk_count} chunks.")
    print(f"Saved to: {os.path.abspath(DB_PATH)}")

if __name__ == "__main__":
    main()