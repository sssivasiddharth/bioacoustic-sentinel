# bioacoustic_monitor.py
# Bioacoustic Sentinel — Field Sensor
# Captures live audio, classifies bird species using BirdNet,
# and sends real-time alerts for endangered species over a local socket.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import csv
import socket
import tempfile
import wave
from datetime import datetime

import librosa
import numpy as np
import pyaudio
from birdnetlib.analyzer import Analyzer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SILENCE_THRESHOLD = 2.5          # RMS threshold below which a chunk is treated as silence
BG_RMS = 2.5                     # initial background RMS estimate
BG_ALPHA = 0.05                  # smoothing factor for adaptive background update
SPEECH_OFFSET = 1.5              # offset above background for speech threshold
RECORD_SECONDS = 4               # duration of each recording chunk (seconds)

ENDANGERED_SPECIES = ["Malabar whistling thrush","Nilgiri marten"]
ALERT_CONFIDENCE = 0.7           # minimum confidence to trigger an alert

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def is_silence(filename="chunk.wav"):
    """Return True if the RMS energy of the WAV file is below the silence threshold."""
    wf = wave.open(filename, 'rb')
    frames = wf.readframes(wf.getnframes())
    wf.close()
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    return np.sqrt(np.mean(samples ** 2)) < SILENCE_THRESHOLD


def record_chunk(filename="chunk.wav"):
    """Record RECORD_SECONDS of mono audio at 16 kHz and save as a WAV file."""
    CHUNK, RATE = 1024, 16000
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK)
              for _ in range(int(RATE / CHUNK * RECORD_SECONDS))]
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def update_background_and_get_speech_threshold(filename="chunk.wav"):
    """Update the adaptive background RMS estimate and return the current speech threshold."""
    global BG_RMS
    wf = wave.open(filename, 'rb')
    frames = wf.readframes(wf.getnframes())
    wf.close()
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    current_rms = np.sqrt(np.mean(samples ** 2))
    if current_rms < BG_RMS * 1.5:
        BG_RMS = BG_RMS * (1 - BG_ALPHA) + current_rms * BG_ALPHA
    return BG_RMS + SPEECH_OFFSET


def trim_silence(filename, speech_threshold=4.0, frame_ms=20, padding_ms=50):
    """Trim leading and trailing silence from a WAV file.

    Returns the path to a temporary trimmed WAV file, or None if no active
    frames are detected.
    """
    wf = wave.open(filename, 'rb')
    rate = wf.getframerate()
    frames = wf.readframes(wf.getnframes())
    wf.close()
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)

    frame_len = int(rate * frame_ms / 1000)
    pad_len = int(rate * padding_ms / 1000)
    n_frames = len(samples) // frame_len
    rms_values = [np.sqrt(np.mean(samples[i * frame_len:(i + 1) * frame_len] ** 2))
                  for i in range(n_frames)]
    active = np.where(np.array(rms_values) > speech_threshold)[0]

    if len(active) == 0:
        return None

    start = max(0, active[0] * frame_len - pad_len)
    end = min(len(samples), (active[-1] + 1) * frame_len + pad_len)

    out_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    with wave.open(out_wav.name, 'wb') as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)
        wf_out.setframerate(rate)
        wf_out.writeframes(samples[start:end].astype(np.int16).tobytes())
    return out_wav.name


# ---------------------------------------------------------------------------
# BirdNet classification
# ---------------------------------------------------------------------------

def classify_chunk(filename):
    """Classify a WAV file with BirdNet and return {species: confidence}."""
    audio, sr = librosa.load(filename, sr=48000, mono=True)
    scores = birdnet_model.predict(audio, sensitivity=1.0)[0]  # first (only) row
    best_idx = np.argmax(scores)
    best_conf = scores[best_idx]

    # Resolve species label list
    if hasattr(birdnet_model, 'species_list'):
        species_list = birdnet_model.species_list
    elif hasattr(birdnet_model, 'labels'):
        species_list = birdnet_model.labels
    else:
        return {f"species_{best_idx}": float(best_conf)}

    best_species = (species_list[best_idx]
                    if best_idx < len(species_list)
                    else f"species_{best_idx}")
    return {best_species: float(best_conf)}


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

def alert(species, confidence):
    """Fire an alert: print to console, log to file, and send over local socket."""
    timestamp = datetime.now().isoformat()
    msg = f"🚨 ALERT at {timestamp}: {species} detected ({confidence:.2%})"
    print(msg)
    with open("alerts.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 5000))
            s.sendall(msg.encode())
    except Exception:
        pass  # Control room may not be running; that's okay


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------

def main():
    birdnet_model = Analyzer()

    log_filename = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    logfile = open(log_filename, "a", newline="")
    writer = csv.writer(logfile)
    writer.writerow(["cycle", "model", "species", "confidence"])

    cycle = 0
    print("🟢 Bioacoustic monitor running. Press Ctrl+C to stop.\n")

    try:
        while True:
            cycle += 1
            try:
                print(f"[Cycle {cycle}] Recording chunk...", flush=True)
                record_chunk()

                print(f"[Cycle {cycle}] Checking silence...", flush=True)
                if is_silence():
                    print("🔇  [silence] — skipping")
                    continue

                print(f"[Cycle {cycle}] Computing threshold...", flush=True)
                thresh = update_background_and_get_speech_threshold()

                print(f"[Cycle {cycle}] Trimming silence...", flush=True)
                trimmed = trim_silence("chunk.wav", thresh)
                if trimmed is None:
                    print("🔇  [trimmed silence] — skipping")
                    continue

                print(f"[Cycle {cycle}] Classifying...", flush=True)
                predictions = classify_chunk(trimmed)

                detected = False
                for species, conf in predictions.items():
                    if species in ENDANGERED_SPECIES and conf >= ALERT_CONFIDENCE:
                        alert(species, conf)
                        detected = True
                    else:
                        print(f"   {species}: {conf:.2%} (not endangered)")

                writer.writerow([cycle, "birdnet",
                                 species if detected else "none",
                                 round(conf if conf else 0.0, 2)])
                logfile.flush()
                print(f"[Cycle {cycle}] Done.", flush=True)

            except Exception as e:
                print(f"⚠️  Cycle {cycle} error: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                writer.writerow([cycle, "error", f"{type(e).__name__}: {e}", 0.0])
                logfile.flush()

    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped.")
    finally:
        logfile.close()
        print(f"✅ Detection log saved to {log_filename}")


if __name__ == "__main__":
    main()