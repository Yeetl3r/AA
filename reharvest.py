import os
import subprocess
import shutil
import glob
import json

TARGET_DIR = "astrologer_data_hybrid"
BACKUP_DIR = "astrologer_data_hybrid/backup"
QUEUE_FILE = "reharvest_queue.json"

def main():
    print("=== GREAT RECOVERY ORCHESTRATOR ===")
    
    if not os.path.exists(QUEUE_FILE):
        print(f"❌ Error: {QUEUE_FILE} not found. Ensure it was generated securely.")
        return
        
    with open(QUEUE_FILE, "r") as f:
        queue_ids = json.load(f)
        
    print(f"Loaded {len(queue_ids)} contaminated IDs from the queue.")
    
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    # Move existing contaminated data to backup
    moved_count = 0
    for root, dirnames, filenames in os.walk(TARGET_DIR):
        if "backup" in root or "review" in root: continue
        for filename in filenames:
            if filename.endswith(".json"):
                for vid_id in queue_ids:
                    if vid_id in filename:
                        src = os.path.join(root, filename)
                        dst = os.path.join(BACKUP_DIR, filename)
                        shutil.move(src, dst)
                        moved_count += 1
                        print(f"  [MOVED] {filename} -> backup/")
                        
    print(f"\nSecured {moved_count} legacy files into backup.\n")
    
    # Hand off to Harvester OMEGA Queue
    print("Handing off to Bulletproof Harvester v2.0...")
    
    cmd = [
        "caffeinate", "-d", "-i", 
        "./venv_314/bin/python", "harvester.py", 
        "--queue", QUEUE_FILE
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
