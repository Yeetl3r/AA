#!/usr/bin/env python3
"""
monitor.py — Harvester Process Watchdog & Health Dashboard
==========================================================
Monitors the harvester.py process for crashes, memory leaks, thermal 
throttling, and disk health. Optionally auto-restarts on crash.

Usage:
    # Monitor only (no auto-restart):
    python monitor.py

    # Monitor + auto-restart on crash (max 5 restarts):
    python monitor.py --restart

    # Custom restart limit:
    python monitor.py --restart --max-restarts 10
"""

import os
import sys
import time
import signal
import subprocess
import datetime
import argparse
import re

# --- CONFIGURATION ---
HARVESTER_SCRIPT = "harvester.py"
PYTHON_BIN = "./venv_314/bin/python"
MONITOR_LOG = "monitor.log"
HEARTBEAT_FILE = "astrologer_data_hybrid/.heartbeat"
HEARTBEAT_LOG = "heartbeat.log"
POLL_INTERVAL = 15  # seconds between health checks


def log(msg, level="INFO"):
    """Log to both console and monitor.log."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line)
    with open(MONITOR_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_system_memory():
    """Return (used_gb, total_gb, percent) for system RAM."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.used / (1024**3), mem.total / (1024**3), mem.percent
    except ImportError:
        # Fallback to vm_stat on macOS
        try:
            out = subprocess.check_output(["vm_stat"], stderr=subprocess.DEVNULL).decode()
            page_size = 16384  # Apple Silicon default
            free = int(re.search(r"Pages free:\s+(\d+)", out).group(1)) * page_size
            active = int(re.search(r"Pages active:\s+(\d+)", out).group(1)) * page_size
            inactive = int(re.search(r"Pages inactive:\s+(\d+)", out).group(1)) * page_size
            wired = int(re.search(r"Pages wired down:\s+(\d+)", out).group(1)) * page_size
            total = free + active + inactive + wired
            used = active + wired
            pct = (used / total) * 100 if total > 0 else 0
            return used / (1024**3), total / (1024**3), pct
        except Exception:
            return 0, 0, 0


def get_thermal_state():
    """Check macOS thermal pressure. Returns (level_str, is_throttled)."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "kern.thermal_pressure"], stderr=subprocess.DEVNULL
        ).decode().strip()
        level = int(out)
        labels = {0: "NOMINAL", 1: "MODERATE", 2: "HEAVY", 3: "CRITICAL"}
        return labels.get(level, f"UNKNOWN({level})"), level >= 2
    except Exception:
        pass

    try:
        out = subprocess.check_output(
            ["pmset", "-g", "therm"], stderr=subprocess.DEVNULL
        ).decode()
        match = re.search(r"CPU_Speed_Limit\s+=\s+(\d+)", out)
        if match:
            limit = int(match.group(1))
            if limit < 70:
                return f"THROTTLED ({limit}%)", True
            elif limit < 90:
                return f"WARM ({limit}%)", False
            else:
                return f"NOMINAL ({limit}%)", False
    except Exception:
        pass

    return "UNKNOWN", False


def get_disk_free(path="."):
    """Return free disk space in GB."""
    try:
        stat = os.statvfs(path)
        return (stat.f_bavail * stat.f_frsize) / (1024**3)
    except Exception:
        return -1


def find_harvester_process():
    """Find the running harvester.py process. Returns (pid, rss_mb, cpu_pct) or None."""
    try:
        import psutil
        for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info", "cpu_percent"]):
            try:
                cmdline = p.info.get("cmdline") or []
                if any(HARVESTER_SCRIPT in arg for arg in cmdline):
                    mem = p.info.get("memory_info")
                    rss_mb = mem.rss / (1024**2) if mem else 0
                    cpu = p.cpu_percent(interval=0.5)
                    return p.info["pid"], rss_mb, cpu
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        # Fallback: use pgrep
        try:
            out = subprocess.check_output(
                ["pgrep", "-f", HARVESTER_SCRIPT], stderr=subprocess.DEVNULL
            ).decode().strip()
            if out:
                pid = int(out.split("\n")[0])
                # Get RSS via ps
                ps_out = subprocess.check_output(
                    ["ps", "-o", "rss=,pcpu=", "-p", str(pid)], stderr=subprocess.DEVNULL
                ).decode().strip()
                parts = ps_out.split()
                rss_mb = int(parts[0]) / 1024 if parts else 0
                cpu = float(parts[1]) if len(parts) > 1 else 0
                return pid, rss_mb, cpu
        except Exception:
            pass
    return None


def count_completed_videos():
    """Count SUCCESS entries in heartbeat.log."""
    try:
        if not os.path.exists(HEARTBEAT_LOG):
            return 0
        with open(HEARTBEAT_LOG, "r") as f:
            return sum(1 for line in f if "[SUCCESS]" in line)
    except Exception:
        return 0


def check_heartbeat_stale():
    """Check if the SSD heartbeat file is stale (>120s old)."""
    try:
        if os.path.exists(HEARTBEAT_FILE):
            with open(HEARTBEAT_FILE, "r") as f:
                last_ts = float(f.read().strip())
            age = time.time() - last_ts
            return age, age > 120
        return -1, False
    except Exception:
        return -1, False


def print_dashboard(proc_info, completed, thermal, mem, disk_free, heartbeat_age):
    """Print a compact health dashboard."""
    thermal_label, is_throttled = thermal
    mem_used, mem_total, mem_pct = mem

    # Header
    print("\n" + "=" * 62)
    print("  📡 HARVESTER HEALTH MONITOR")
    print("=" * 62)

    # Process status
    if proc_info:
        pid, rss_mb, cpu = proc_info
        status_icon = "🟢" if rss_mb < 6000 else "🟡" if rss_mb < 10000 else "🔴"
        print(f"  Process:    {status_icon} RUNNING (PID {pid})")
        print(f"  RSS Memory: {rss_mb:.0f} MB ({rss_mb/1024:.1f} GB)")
        print(f"  CPU Usage:  {cpu:.1f}%")
    else:
        print(f"  Process:    🔴 NOT RUNNING")

    # Progress
    print(f"  Completed:  {completed} videos")

    # System health
    thermal_icon = "🔴" if is_throttled else "🟢"
    print(f"  Thermal:    {thermal_icon} {thermal_label}")
    print(f"  System RAM: {mem_used:.1f}/{mem_total:.1f} GB ({mem_pct:.0f}%)")
    print(f"  Disk Free:  {disk_free:.1f} GB")

    # Heartbeat
    if heartbeat_age >= 0:
        hb_icon = "🟢" if heartbeat_age < 120 else "🟡" if heartbeat_age < 300 else "🔴"
        print(f"  Heartbeat:  {hb_icon} {heartbeat_age:.0f}s ago")

    print("=" * 62)


def launch_harvester(gemini_keys):
    """Launch harvester.py as a subprocess. Returns Popen object."""
    env = os.environ.copy()
    env["GEMINI_API_KEYS"] = gemini_keys
    
    log("🚀 Launching harvester.py --all")
    proc = subprocess.Popen(
        ["caffeinate", "-d", "-i", PYTHON_BIN, HARVESTER_SCRIPT, "--all"],
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log(f"   PID: {proc.pid}")
    return proc


def main():
    parser = argparse.ArgumentParser(description="Harvester Process Watchdog")
    parser.add_argument("--restart", action="store_true",
                        help="Auto-restart harvester on crash")
    parser.add_argument("--max-restarts", type=int, default=5,
                        help="Maximum auto-restarts before giving up (default: 5)")
    args = parser.parse_args()

    log("=" * 50)
    log("📡 Monitor starting")
    log(f"   Auto-restart: {'ON' if args.restart else 'OFF'}")
    if args.restart:
        log(f"   Max restarts: {args.max_restarts}")

    # Check for Gemini keys (needed if auto-restart is on)
    gemini_keys = os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", ""))
    if args.restart and not gemini_keys:
        log("⚠️  GEMINI_API_KEYS not set. Auto-restart will launch without Oracle keys.", "WARN")

    restart_count = 0
    managed_proc = None  # Only set if we launched it ourselves

    try:
        while True:
            proc_info = find_harvester_process()
            completed = count_completed_videos()
            thermal = get_thermal_state()
            mem = get_system_memory()
            disk_free = get_disk_free("/Volumes/Storage Drive/AA")
            hb_age, hb_stale = check_heartbeat_stale()

            print_dashboard(proc_info, completed, thermal, mem, disk_free, hb_age)

            # --- ALERTS ---
            _, is_throttled = thermal
            mem_used, mem_total, mem_pct = mem

            if is_throttled:
                log("🌡️  THERMAL THROTTLE DETECTED", "WARN")

            if mem_pct > 90:
                log(f"🧠 CRITICAL: System memory at {mem_pct:.0f}%", "CRIT")

            if disk_free >= 0 and disk_free < 5:
                log(f"💾 CRITICAL: Only {disk_free:.1f} GB disk space remaining", "CRIT")

            if proc_info:
                _, rss_mb, _ = proc_info
                if rss_mb > 12000:
                    log(f"🧠 WARNING: Harvester RSS at {rss_mb:.0f} MB — potential memory leak", "WARN")

            # --- CRASH DETECTION & AUTO-RESTART ---
            if not proc_info:
                if args.restart:
                    if restart_count >= args.max_restarts:
                        log(f"🚨 Max restarts ({args.max_restarts}) reached. Giving up.", "CRIT")
                        break

                    restart_count += 1
                    log(f"💀 Harvester not running! Restarting ({restart_count}/{args.max_restarts})...", "WARN")
                    time.sleep(10)  # Brief cooldown before restart
                    managed_proc = launch_harvester(gemini_keys)
                else:
                    log("💀 Harvester is NOT running. Use --restart to enable auto-restart.", "WARN")

            # Reset restart counter if process has been stable for a while
            if proc_info and restart_count > 0:
                # Process is alive — it recovered
                pass  # Keep counter for lifetime tracking

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        log("\n🛑 Monitor stopped by user")
        if managed_proc and managed_proc.poll() is None:
            log("   (Harvester is still running in background)")


if __name__ == "__main__":
    main()
