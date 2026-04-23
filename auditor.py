import os
import json
import shutil
import urllib.request
from datetime import datetime

# Zenith v2.1 Paths
BASE_DIR = "astrologer_data_hybrid"
REVIEW_DIR = os.path.join(BASE_DIR, "review/LOOP_HARVEST")
OUTPUT_DIR = BASE_DIR
AUDIT_LOG = os.path.join(BASE_DIR, "audit_history.log")

def log_audit(vid_id, verdict, reason=""):
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} | {vid_id} | {verdict} | {reason}\n")

def check_gemini(api_key, text_sample):
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    # Forensic-Hardened Prompt
    prompt = (
        "SYSTEM ROLE: Data Integrity Auditor for Tamil Speech-to-Text.\n"
        "TASK: Identify structural 'phonetic loops' or 'hallucination cascades' in Whisper-generated Tamil.\n\n"
        "INDICATORS OF FAILURE:\n"
        "1. Extreme repetition of phonemes (e.g., 'இறும் இ இறும் இ' or 'பார் பார் பார்').\n"
        "2. Nonsensical character gluing (e.g., 'புதனலில்லிலில்லில').\n"
        "3. Semantic death: 50+ words with no syntactic variation.\n\n"
        "INSTRUCTIONS: Analyze the text below. If it is a real discourse by an astrologer, reply 'VALID'. "
        "If it is a hallucinated loop, reply 'HALLUCINATION'. "
        "Reply ONLY with one of those two words.\n\n"
        f"TEXT SAMPLE:\n{text_sample}"
    )
    
    payload = {"contents": [{"parts":[{"text": prompt}]}]}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(endpoint, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            res_json = json.loads(response.read())
            reply = res_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip().upper()
            return "VALID" if "VALID" in reply else "HALLUCINATION"
    except Exception as e:
        return f"API_ERROR: {str(e)}"

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY missing.")
        return
        
    if not os.path.exists(REVIEW_DIR):
        print("✅ LOOP_HARVEST is empty. No work for the Oracle.")
        return
        
    files = [f for f in os.listdir(REVIEW_DIR) if f.endswith(".json") and not f.startswith("REJECTED_")]
    print(f"--- 🔮 Oracle Bridge: Auditing {len(files)} files ---")
    
    for filename in files:
        filepath = os.path.join(REVIEW_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        vid_id = data.get("metadata", {}).get("video_id", "Unknown")
        sample = " ".join(data.get("full_text", "").split()[:400])
        
        print(f"Auditing {vid_id}... ", end="", flush=True)
        verdict = check_gemini(api_key, sample)
        print(f"[{verdict}]")
        
        log_audit(vid_id, verdict)

        if verdict == "VALID":
            # Success: Move to main DB
            channel = data.get("metadata", {}).get("channel_name", "General_Archive")
            target_out = os.path.join(OUTPUT_DIR, channel)
            os.makedirs(target_out, exist_ok=True)
            
            # Stamp the JSON with Audit Approval
            data["metadata"]["audit_verified"] = True
            data["metadata"]["audit_timestamp"] = datetime.now().isoformat()
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
                
            shutil.move(filepath, os.path.join(target_out, filename))
        else:
            # Failure: Quarantine
            os.rename(filepath, os.path.join(REVIEW_DIR, f"REJECTED_{filename}"))

if __name__ == "__main__":
    main()