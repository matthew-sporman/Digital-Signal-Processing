import os
import subprocess
import sys
# ==========================================================
# Calibration level for the hydrophone
FS_CALIBRATION_LEVEL = 170.0 

# ==========================================================
# 1. USER SETTINGS: DEFINE YOUR PATHS AND PARAMETERS BELOW.
# ==========================================================

# 1. The folder with your RAW audio files
# CHANGE THIS TO YOUR SOURCE FOLDER (e.g., C:\HydrophoneData\Raw)
RAW_AUDIO_DIR = r"D:\SWIFT_TESTING"

# 2. The folder WHERE YOU WANT the segmented files to go
# This is the OUTPUT of wav_split.py and the INPUT for hydrophone_processing.py
SEGMENT_DIR = r"D:\SWIFT_TESTING\output_segments"

# 3. The folder WHERE YOU WANT the final analysis (plots, .npz) to go
ANALYSIS_DIR = r"D:\SWIFT_TESTING\output_charts"

# --- Splitter Parameters ---
# The duration of each segment in minutes
SEGMENT_MINUTES = 12

# --- Processing Parameters ---
# Frequency range for filtering and plotting (Hz)
LOWCUT = 1000.0
HIGHCUT = 20000.0

# Segment size for Welch (FFT size)
NPERSEG = 2048

# ==========================================================
# 2. PIPELINE EXECUTION LOGIC (NO CHANGES NEEDED BELOW!)
# ==========================================================

def run_command(command, step_name):
    """Executes a command and checks for errors."""
    print(f"\n--- Running {step_name}: {' '.join(command)} ---")
    try:
        # We use check=True to raise an error if the process returns a non-zero exit code
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("--- ✅ SUCCESS ---")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"--- ⛔ ERROR: {step_name} failed! ---")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        # Exit the script on failure
        sys.exit(1)
    except FileNotFoundError:
        print(f"--- ⛔ ERROR: Python or one of the scripts was not found. ---")
        print("Ensure 'python' is in your PATH and the script files exist.")
        sys.exit(1)


if __name__ == "__main__":
    
    # 1. Ensure output directories exist before running
    os.makedirs(SEGMENT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # --- Step 1: Run wav_split.py ---
    split_command = [
        sys.executable, "wav_split.py", # sys.executable ensures we use the correct python
        "-i", RAW_AUDIO_DIR,
        "-o", SEGMENT_DIR,
        "-m", str(SEGMENT_MINUTES)
    ]
    run_command(split_command, "WAV Splitter")

    # --- Step 2: Run hydrophone_processing.py ---
    # It automatically uses SEGMENT_DIR (output of step 1) as its input
    process_command = [
        sys.executable, "snap_hydrophone_processing.py",
        "-i", SEGMENT_DIR,
        "-o", ANALYSIS_DIR,
        "--lowcut", str(LOWCUT),
        "--highcut", str(HIGHCUT),
        "--nperseg", str(NPERSEG),
        "--fs-level", str(FS_CALIBRATION_LEVEL)

    ]
    run_command(process_command, "Hydrophone Processor")

    print("\n\n--- 🎉 Pipeline Complete! ---")
    print(f"1. Split segments saved to: {SEGMENT_DIR}")
    print(f"2. Final analysis saved to: {ANALYSIS_DIR}")