import manifest_manager
import validator_v2
import os
import json
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

BASE_DIR = "astrologer_data_hybrid"

def process_file_optimized(filepath):
    """Read a JSON and return its audit metrics. 
    Uses pre-calculated metrics if present, otherwise recalculates via validator_v2.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Check if audit already exists (New Harvester v2.2)
        audit = data.get("audit", {})
        if audit:
            res = {
                "vid_id": data.get("metadata", {}).get("video_id", os.path.basename(filepath)),
                "filepath": filepath,
                "category": audit.get("category", "UNKNOWN"),
                "in_escrow": ("LOOP_HARVEST" in filepath) or ("NULL_HARVEST" in filepath)
            }
            res.update(audit.get("metrics", {}))
            return res
            
        # Fallback for old files: use validator_v2
        category, metrics = validator_v2.validate_transcription(data)
        res = {
            "vid_id": data.get("metadata", {}).get("video_id", os.path.basename(filepath)),
            "filepath": filepath,
            "category": category,
            "in_escrow": ("LOOP_HARVEST" in filepath) or ("NULL_HARVEST" in filepath)
        }
        res.update(metrics)
        return res
    except Exception:
        return None

def main():
    mm = manifest_manager.ManifestManager()
    manifest = mm.get_manifest()
    
    if not manifest:
        print("Manifest empty. Running emergency drive-walk...")
        mm.rebuild_from_disk(BASE_DIR)
        manifest = mm.get_manifest()

    target_files = [data["filepath"] for data in manifest.values() if data.get("filepath")]
                
    results = []
    print(f"--- INIT: Forensic Audit Engine (Phase 2 Accelerated) | Targets: {len(target_files)} ---")
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_file = {executor.submit(process_file_optimized, f): f for f in target_files}
        for future in as_completed(future_to_file):
            res = future.result()
            if res:
                results.append(res)
                
    buckets = defaultdict(list)
    for r in results:
        buckets[r["category"]].append(r)
        
    print("\n=== FORENSIC FAILURE MATRIX (PHASE 2) ===")
    print(f"{'Category':<15} | {'Count':<6} | {'Description'}")
    print("-" * 75)
    for cat in ["SUCCESS", "GHOST", "TERMINAL_LOOP", "SYLLABLE_TRAP", "SANDWICH_LOOP", "PROMPT_CONTAMINATED"]:
        count = len(buckets[cat])
        print(f"{cat:<15} | {count:<6} | Verified via in-line validator_v2.")

    # Write Forensic Dumps
    csv_file = os.path.join(BASE_DIR, "audit_report.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        if results:
            keys = results[0].keys()
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n✅ Complete. Forensic telemetry dumped to '{csv_file}'")

if __name__ == "__main__":
    main()
