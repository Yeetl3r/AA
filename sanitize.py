import os
import csv
import shutil

BASE_DIR = "astrologer_data_hybrid"
REPORT_CSV = os.path.join(BASE_DIR, "audit_report.csv")
NULL_HARVEST = os.path.join(BASE_DIR, "review", "NULL_HARVEST")
LOOP_HARVEST = os.path.join(BASE_DIR, "review", "LOOP_HARVEST")

def sanitize_db():
    if not os.path.exists(REPORT_CSV):
        print("❌ Cannot find audit_report.csv. Run forensic_audit.py first.")
        return

    os.makedirs(NULL_HARVEST, exist_ok=True)
    os.makedirs(LOOP_HARVEST, exist_ok=True)

    print("=== DATA SANITIZER v2.2 ===")
    moved_count = 0

    with open(REPORT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row['filepath']
            category = row['category']
            in_escrow = row['in_escrow'] == 'True'

            # We only care about files sitting in the active production database
            if not in_escrow and category != "SUCCESS":
                target_folder = NULL_HARVEST if category == "GHOST" else LOOP_HARVEST
                
                # Check if file still exists in the origin
                if os.path.exists(filepath):
                    filename = os.path.basename(filepath)
                    # We forcefully flag it as REJECTED so the auditor doesn't loop
                    dest_name = f"REJECTED_{filename}" if category != "GHOST" else filename
                    dest_path = os.path.join(target_folder, dest_name)
                    
                    shutil.move(filepath, dest_path)
                    print(f"🚜 PURGED [{category}]: {filename} -> {os.path.basename(target_folder)}")
                    moved_count += 1
                else:
                    print(f"⚠️ File missing (already purged?): {filepath}")

    print(f"\n✅ Clean Summary: Successfully purged {moved_count} contaminated files from production scope.")

if __name__ == "__main__":
    sanitize_db()
