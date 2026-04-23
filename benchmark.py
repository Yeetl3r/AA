"""
benchmark.py — AA Pipeline Model Throughput Benchmark
MacBook Air M4 16GB | MLX Backend

Tests two things:
  1. Raw model throughput (silence) — isolates pure inference speed
  2. Real audio throughput (noise) — approximates actual harvest conditions

Run from your AA project root:
    python3 benchmark.py

Paste the output block into your conversation with Claude.
"""

import os
import time
import json
import platform
import subprocess
import numpy as np

# ── CONFIGURE MODEL PATH ──────────────────────────────────────────────────────
LARGE_V3_PATH     = "/Volumes/Storage Drive/AA/mlx_models/large-v3"
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 16000
FRAMES_SIZE   = 160        # 10ms per Whisper frame
TEST_DURATION = 30         # seconds of audio per test

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_metal_memory_mb():
    """Read current Metal GPU memory allocation via powermetrics (best-effort)."""
    try:
        out = subprocess.check_output(
            ["sudo", "-n", "powermetrics", "-n", "1", "-i", "100",
             "--samplers", "gpu_power"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode()
        for line in out.splitlines():
            if "GPU" in line and "MB" in line:
                return line.strip()
    except Exception:
        pass
    return "unavailable (run with sudo for Metal memory stats)"


def get_mlx_memory_mb():
    """Get MLX active memory in MB."""
    try:
        import mlx.core as mx
        mem = mx.get_active_memory()   # bytes
        return round(mem / 1024 / 1024, 1)
    except Exception:
        return "unavailable"


def check_model_format(path):
    """
    Detect whether model weights are native MLX or converted HuggingFace.
    Returns a dict with format info.
    """
    info = {"path": path, "exists": os.path.exists(path), "format": "unknown", "files": []}
    if not info["exists"]:
        return info

    files = os.listdir(path)
    info["files"] = sorted(files)

    config_path = os.path.join(path, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        info["config_keys"] = list(cfg.keys())[:10]
        info["model_type"] = cfg.get("model_type", "unknown")

        # Native MLX whisper configs have these keys
        mlx_keys = {"n_mels", "n_audio_ctx", "n_text_layer", "n_text_head"}
        # HuggingFace configs have these keys
        hf_keys   = {"architectures", "_name_or_path", "torch_dtype", "transformers_version"}

        if mlx_keys & set(cfg.keys()):
            info["format"] = "NATIVE MLX ✅"
        elif hf_keys & set(cfg.keys()):
            info["format"] = "HUGGINGFACE (needs conversion) ⚠️"
        else:
            info["format"] = "UNKNOWN — inspect config.json manually"

    # Weight file detection
    if any(f.endswith(".npz") for f in files):
        info["weights"] = "npz (MLX native)"
    elif "weights.safetensors" in files or "model.safetensors" in files:
        info["weights"] = "safetensors (HuggingFace or converted)"
    elif any(f.endswith(".bin") for f in files):
        info["weights"] = "pytorch .bin (unconverted HuggingFace)"
    else:
        info["weights"] = "unknown"

    return info


def run_benchmark(model_path, model_label, audio_np):
    """
    Run a single model benchmark. Returns a result dict.
    """
    import mlx.core as mx
    import mlx_whisper
    import gc

    print(f"\n  Loading {model_label}...")

    # Warm up Metal (first call always slower due to shader compilation)
    warmup_audio = np.zeros(5 * SAMPLE_RATE, dtype=np.float32)
    try:
        mlx_whisper.transcribe(
            warmup_audio,
            path_or_hf_repo=model_path,
            language="ta",
            verbose=False
        )
    except Exception as e:
        return {"model": model_label, "error": f"Warmup failed: {e}"}

    mx.clear_cache()
    gc.collect()
    time.sleep(1)

    mem_before = get_mlx_memory_mb()

    # ── PASS 1: Silence (pure inference speed, no tokenizer pressure) ──
    print(f"  Pass 1/2: silence test ({TEST_DURATION}s)...", end=" ", flush=True)
    silence = np.zeros(TEST_DURATION * SAMPLE_RATE, dtype=np.float32)
    t0 = time.perf_counter()
    try:
        r1 = mlx_whisper.transcribe(
            silence,
            path_or_hf_repo=model_path,
            language="ta",
            task="transcribe",
            temperature=(0.0,),
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=1.8,
            verbose=False
        )
        silence_elapsed = time.perf_counter() - t0
        silence_fps = (TEST_DURATION * SAMPLE_RATE) / silence_elapsed / FRAMES_SIZE
        silence_rtf = TEST_DURATION / silence_elapsed
        print(f"{silence_fps:.0f} fps ({silence_rtf:.1f}x realtime)")
    except Exception as e:
        return {"model": model_label, "error": f"Silence pass failed: {e}"}

    mx.clear_cache()
    gc.collect()
    time.sleep(2)

    # ── PASS 2: Real-world audio (pink noise — approximates speech energy) ──
    print(f"  Pass 2/2: noise test ({TEST_DURATION}s)...", end=" ", flush=True)
    rng = np.random.default_rng(42)
    # Pink noise: better approximation of speech frequency profile than white noise
    white = rng.standard_normal(TEST_DURATION * SAMPLE_RATE).astype(np.float32)
    freqs = np.fft.rfftfreq(len(white))
    freqs[0] = 1e-10  # avoid divide-by-zero at DC
    pink_filter = 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(np.fft.rfft(white) * pink_filter).astype(np.float32)
    pink = pink / np.max(np.abs(pink)) * 0.3  # normalize to 30% amplitude

    t0 = time.perf_counter()
    try:
        r2 = mlx_whisper.transcribe(
            pink,
            path_or_hf_repo=model_path,
            language="ta",
            task="transcribe",
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=1.8,
            verbose=False
        )
        noise_elapsed = time.perf_counter() - t0
        noise_fps     = (TEST_DURATION * SAMPLE_RATE) / noise_elapsed / FRAMES_SIZE
        noise_rtf     = TEST_DURATION / noise_elapsed
        print(f"{noise_fps:.0f} fps ({noise_rtf:.1f}x realtime)")
    except Exception as e:
        noise_fps = noise_rtf = noise_elapsed = None
        print(f"FAILED: {e}")

    mem_after = get_mlx_memory_mb()

    # ── PASS 3: With our new parameters (temperature tuple — approximates real pipeline) ──
    print(f"  Pass 3/3: pipeline-params test (temperature fallback tuple)...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        r3 = mlx_whisper.transcribe(
            audio_np,
            path_or_hf_repo=model_path,
            language="ta",
            task="transcribe",
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            compression_ratio_threshold=1.8,
            initial_prompt=(
                "இது ஒரு தமிழ் ஜோதிட வீடியோ. "
                "இதில் லக்னம், நவாம்சம், திசை, புக்தி போன்ற சொற்கள் உள்ளன."
            ),
            verbose=False
        )
        pipeline_elapsed = time.perf_counter() - t0
        pipeline_fps     = (TEST_DURATION * SAMPLE_RATE) / pipeline_elapsed / FRAMES_SIZE
        pipeline_rtf     = TEST_DURATION / pipeline_elapsed
        print(f"{pipeline_fps:.0f} fps ({pipeline_rtf:.1f}x realtime)")
    except Exception as e:
        pipeline_fps = pipeline_rtf = None
        print(f"FAILED: {e}")

    mx.clear_cache()
    gc.collect()

    return {
        "model":            model_label,
        "path":             model_path,
        "mlx_mem_before_mb": mem_before,
        "mlx_mem_after_mb":  mem_after,
        "silence": {
            "fps": round(silence_fps, 1),
            "rtf": round(silence_rtf, 1),
            "elapsed_s": round(silence_elapsed, 2),
        },
        "noise": {
            "fps": round(noise_fps, 1) if noise_fps else None,
            "rtf": round(noise_rtf, 1) if noise_rtf else None,
        },
        "pipeline_params": {
            "fps": round(pipeline_fps, 1) if pipeline_fps else None,
            "rtf": round(pipeline_rtf, 1) if pipeline_rtf else None,
        },
    }


def estimate_harvest_time(rtf, n_videos=7000, avg_duration_min=35):
    """Calculate total harvest time at a given realtime factor."""
    total_audio_min  = n_videos * avg_duration_min
    total_process_min = total_audio_min / rtf
    days  = total_process_min / 60 / 24
    hours = (total_process_min / 60) % 24
    return {
        "total_audio_hours": round(total_audio_min / 60, 0),
        "processing_days":   round(days, 1),
        "processing_hours":  round(hours, 1),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import mlx.core as mx

    print("=" * 62)
    print("  AA Pipeline — Model Throughput Benchmark")
    print(f"  {platform.machine()} | macOS {platform.mac_ver()[0]}")
    print("=" * 62)

    # ── Model format inspection ──
    print("\n── Model Format Check ──────────────────────────────────────")
    for label, path in [("whisper-large-v3",     LARGE_V3_PATH)]:
        info = check_model_format(path)
        print(f"\n  {label}")
        print(f"    Path exists : {info['exists']}")
        if info["exists"]:
            print(f"    Format      : {info['format']}")
            print(f"    Weights     : {info.get('weights', 'unknown')}")
            print(f"    Files       : {', '.join(info['files'][:8])}"
                  + (" ..." if len(info["files"]) > 8 else ""))

    # ── mlx-community availability check ──
    print("\n── mlx-community Native Tamil Models (HuggingFace check) ───")
    print("  These are pre-converted MLX models — no conversion needed.")
    print("  If your tamil-medium is slow, these are the alternative:")
    mlx_community_models = [
        "mlx-community/whisper-large-v3-mlx",
        "mlx-community/whisper-medium-mlx",
        "mlx-community/whisper-small-mlx",
    ]
    for m in mlx_community_models:
        print(f"    • {m}")
    print("  Check https://huggingface.co/mlx-community for Tamil fine-tunes.")

    # ── Shared test audio ──
    print(f"\n── Generating test audio ({TEST_DURATION}s) ──────────────────────────")
    rng       = np.random.default_rng(42)
    test_audio = rng.standard_normal(TEST_DURATION * SAMPLE_RATE).astype(np.float32)
    test_audio /= np.max(np.abs(test_audio)) * 3  # low amplitude
    print("  Done.")

    # ── Run benchmarks ──
    results = []

    # Models to test check removed for abandoned tamil-medium
    models_to_test = []

    if os.path.exists(LARGE_V3_PATH):
        models_to_test.append((LARGE_V3_PATH, "whisper-large-v3"))
    else:
        print(f"\n  ⚠️  Large-v3 not found at {LARGE_V3_PATH} — skipping.")

    for path, label in models_to_test:
        print(f"\n── Benchmarking: {label} ─────────────────────────────────")
        res = run_benchmark(path, label, test_audio)
        results.append(res)

    # ── Results table ──
    print("\n")
    print("=" * 62)
    print("  BENCHMARK RESULTS — paste this block to Claude")
    print("=" * 62)

    for r in results:
        if "error" in r:
            print(f"\n  {r['model']}: ERROR — {r['error']}")
            continue

        print(f"\n  ┌─ {r['model']}")
        print(f"  │  Silence test  : {r['silence']['fps']} fps  "
              f"({r['silence']['rtf']}x realtime)")
        print(f"  │  Noise test    : {r['noise']['fps']} fps  "
              f"({r['noise']['rtf']}x realtime)" if r['noise']['fps']
              else "  │  Noise test    : FAILED")
        print(f"  │  Pipeline test : {r['pipeline_params']['fps']} fps  "
              f"({r['pipeline_params']['rtf']}x realtime)" if r['pipeline_params']['fps']
              else "  │  Pipeline test : FAILED")
        print(f"  │  MLX mem delta : {r['mlx_mem_before_mb']} → {r['mlx_mem_after_mb']} MB")

        # Harvest time estimate using pipeline RTF (most realistic)
        rtf = r['pipeline_params']['rtf'] or r['noise']['rtf'] or r['silence']['rtf']
        if rtf:
            est = estimate_harvest_time(rtf)
            print(f"  │")
            print(f"  │  ── Harvest estimate (7,000 videos × 35 min) ──")
            print(f"  │  Total audio    : {est['total_audio_hours']:.0f} hours")
            print(f"  │  Processing time: {est['processing_days']} days "
                  f"({est['processing_hours']:.0f} hrs remainder)")
        print(f"  └{'─' * 50}")

    # ── Decision guide ──
    print("""
── What to tell Claude ──────────────────────────────────────

  Copy the BENCHMARK RESULTS block above and paste it.

  Also answer:
    a) Did the model format check say NATIVE MLX or HUGGINGFACE?
    b) Is large-v3 throughput within expectations?
    c) What's the pipeline-params RTF for each model?

  Claude will then recommend:
    • Path A  — use mlx-community native Tamil model (if conversion is the bottleneck)
    • Path B  — stay on large-v3 with parameter fixes (if it's faster and quality is ok)
    • Path C  — use YouTube captions as primary dataset (reduces video count dramatically)
    • Path D  — cloud ASR for bulk job (if local speed is fundamentally too slow)
""")


if __name__ == "__main__":
    main()
