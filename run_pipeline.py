import os
import subprocess
import sys
import glob
from datetime import datetime, timedelta

# ==========================================================
# Calibration level for the hydrophone
FS_CALIBRATION_LEVEL = 170.0 

# ==========================================================
# 1. USER SETTINGS
# ==========================================================

RAW_AUDIO_DIR = r"D:\SWIFT_TESTING"
SEGMENT_DIR = r"D:\SWIFT_TESTING\output_segments"
ANALYSIS_DIR = r"D:\SWIFT_TESTING\output_charts"

# --- New: Weather Output Folder ---
WEATHER_DIR = os.path.join(ANALYSIS_DIR, "weather_data_JSON")

# --- Splitter Parameters ---
SEGMENT_MINUTES = 1

# --- Processing Parameters ---
LOWCUT = 1000.0
HIGHCUT = 20000.0
NPERSEG = 2048

# --- Synchronization Parameters ---
# How many seconds to adjust the hydrophone timestamp.
TIME_OFFSET_SECONDS = 0

# ==========================================================
# 2. HELPER FUNCTIONS
# ==========================================================

def run_command(command, step_name):
    """Executes a command and checks for errors."""
    print(f"\n--- Running {step_name}: {' '.join(command)} ---")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("--- ✅ SUCCESS ---")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"--- ⛔ ERROR: {step_name} failed! ---")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        # We don't exit for weather failure, just warn
        if step_name == "Weather Fetcher":
            print("⚠️ Continuing pipeline without weather data...")
            return
        sys.exit(1)
    except FileNotFoundError:
        print(f"--- ⛔ ERROR: Script not found. ---")
        sys.exit(1)

def get_time_range(segment_dir):
    """Scans the segment directory to find the earliest and latest timestamps."""
    files = sorted(glob.glob(os.path.join(segment_dir, "*.wav")))
    if not files:
        return None, None
    
    # Parse first filename (Assumes format YYYYMMDDTHHMMSS.wav)
    try:
        first_base = os.path.basename(files[0]).split('.')[0] 
        start_time = datetime.strptime(first_base, "%Y%m%dT%H%M%S")
        
        # Parse last filename
        last_base = os.path.basename(files[-1]).split('.')[0]
        last_start = datetime.strptime(last_base, "%Y%m%dT%H%M%S")
        
        # Add segment duration to the start of the last file to get the true end
        end_time = last_start + timedelta(minutes=SEGMENT_MINUTES)
        
        return start_time, end_time
    except ValueError:
        print("Warning: Could not parse timestamps from filenames. Skipping weather fetch.")
        return None, None

# ==========================================================
# 3. PIPELINE EXECUTION LOGIC
# ==========================================================

if __name__ == "__main__":
    
    os.makedirs(SEGMENT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(WEATHER_DIR, exist_ok=True)

    # --- Step 1: Run wav_split.py ---
    split_command = [
        sys.executable, "wav_split.py",
        "-i", RAW_AUDIO_DIR,
        "-o", SEGMENT_DIR,
        "-m", str(SEGMENT_MINUTES)
    ]
    run_command(split_command, "WAV Splitter")

    # --- Step 1.5: Fetch Historical Weather (NEW) ---
    print("\n--- Determining Weather Data Range ---")
    start_t, end_t = get_time_range(SEGMENT_DIR)
    
    if start_t and end_t:
        # Format timestamps for the script arguments
        s_str = start_t.strftime("%Y%m%dT%H%M%S")
        e_str = end_t.strftime("%Y%m%dT%H%M%S")
        
        weather_command = [
            sys.executable, "weather_history.py",
            "--start", s_str,
            "--end", e_str,
            "--out", WEATHER_DIR
        ]
        run_command(weather_command, "Weather Fetcher")
    else:
        print("⚠️ Skipping weather fetch (No segments found or invalid timestamps).")

    # --- Step 2: Run hydrophone_processing.py ---
    process_command = [
        sys.executable, "snap_hydrophone_processing.py", 
        "-i", SEGMENT_DIR,
        "-o", ANALYSIS_DIR,
        "--lowcut", str(LOWCUT),
        "--highcut", str(HIGHCUT),
        "--nperseg", str(NPERSEG),
        "--fs-level", str(FS_CALIBRATION_LEVEL),
        "--offset", str(TIME_OFFSET_SECONDS) 
    ]
    run_command(process_command, "Hydrophone Processor")

    print("\n\n--- 🎉 Pipeline Complete! ---")
    print(f"1. Split segments saved to: {SEGMENT_DIR}")
    print(f"2. Weather data saved to:   {WEATHER_DIR}")
    print(f"3. Final analysis saved to: {ANALYSIS_DIR}")