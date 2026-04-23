import os
import json
import fcntl
import time

MANIFEST_PATH = "astrologer_data_hybrid/manifest.json"

class ManifestManager:
    def __init__(self, path=MANIFEST_PATH):
        self.path = path
        # Ensure directory exists but don't create manifest yet
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def _get_lock(self, mode='r'):
        """Acquire an exclusive lock on the manifest file."""
        return open(self.path, 'a+' if mode == 'w' else 'r')

    def update_entry(self, video_id, data):
        """Add or update a video entry in the manifest atomically."""
        lock_fd = open(self.path, 'a+')
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            
            # Read existing
            lock_fd.seek(0)
            content = lock_fd.read()
            manifest = json.loads(content) if content else {}
            
            # Update
            manifest[video_id] = {
                "timestamp": time.time(),
                "filepath": data.get("filepath", ""),
                "title": data.get("title", "Unknown"),
                "category": data.get("category", "UNKNOWN"),
                "duration": data.get("duration", 0),
                "in_escrow": data.get("in_escrow", False),
                "sentry_status": data.get("sentry_status", "UNKNOWN")
            }
            
            # Write back
            lock_fd.seek(0)
            lock_fd.truncate()
            json.dump(manifest, lock_fd, indent=2)
            lock_fd.flush()
            os.fsync(lock_fd.fileno())
            
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def get_manifest(self):
        """Read the full manifest safely."""
        if not os.path.exists(self.path):
            return {}
            
        with open(self.path, 'r') as f:
            try:
                fcntl.flock(f, fcntl.LOCK_SH)
                content = f.read()
                return json.loads(content) if content else {}
            except Exception:
                return {}
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def get_existing_ids(self):
        """Returns a set of all video IDs in the manifest."""
        return set(self.get_manifest().keys())

    def rebuild_from_disk(self, root_dir="astrologer_data_hybrid"):
        """Emergency utility to rebuild the manifest by walking the disk."""
        print(f"Index Recovery: Scanning {root_dir}...")
        new_manifest = {}
        for root, _, files in os.walk(root_dir):
            if "backup" in root or "review" in root: continue
            for f in files:
                if f.endswith(".json") and f != "manifest.json" and not f.startswith("REJECTED_"):
                    path = os.path.join(root, f)
                    try:
                        # Extract minimal ID from filename or content
                        # Assuming filename is title_ID.json
                        vid_id = f.rsplit('_', 1)[1].replace('.json', '')
                        with open(path, 'r') as jf:
                            data = json.load(jf)
                            meta = data.get("metadata", {})
                            new_manifest[vid_id] = {
                                "timestamp": os.path.getmtime(path),
                                "filepath": path,
                                "title": meta.get("title", "Unknown"),
                                "category": data.get("category", "SUCCESS"), # Assume success on rebuild
                                "duration": meta.get("duration", 0),
                                "in_escrow": False
                            }
                    except:
                        continue
        
        # Atomic write
        with open(self.path, 'w') as f:
            json.dump(new_manifest, f, indent=2)
        print(f"Index Recovery Complete: {len(new_manifest)} entries.")

if __name__ == "__main__":
    # If run directly, rebuild the manifest
    mgr = ManifestManager()
    mgr.rebuild_from_disk()
