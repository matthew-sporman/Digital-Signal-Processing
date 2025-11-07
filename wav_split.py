import os
from datetime import datetime, timedelta
import numpy as np
from scipy.io import wavfile
import wave
import glob
import argparse

def parse_start_time(filename: str) -> datetime:
    base = os.path.splitext(os.path.basename(filename))[0]  # removes .WAV
    timestamp_str = base.split("_")[0]  # gets "20241029T150409"
    return datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")


def get_wav_duration(filepath: str) -> float:
    """Return duration (seconds) of a .wav file."""
    with wave.open(filepath, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def batch_split(files, segment_minutes, outdir="segments"):
    """
    Split all wav files into consecutive segments of segment_minutes.
    Handles file boundaries automatically.
    """
    # --- Clears previous segments before processing ---
    if os.path.exists(outdir):
        for f in glob.glob(os.path.join(outdir, "*.wav")):
            os.remove(f)
    else:
        os.makedirs(outdir, exist_ok=True)

    # Sort files by their encoded start time
    files = sorted(files, key=parse_start_time)

    # Build a global timeline from first file start → last file end
    start_time = parse_start_time(files[0])
    last_file = files[-1]
    end_time = parse_start_time(last_file) + timedelta(seconds=get_wav_duration(last_file))

    segment_delta = timedelta(minutes=segment_minutes)
    current_start = start_time
    segment_idx = 0

    while current_start + segment_delta <= end_time:
        current_end = current_start + segment_delta
        segment_data = None
        sample_rate = None

        # Loop through files to collect overlapping pieces
        for f in files:
            file_start = parse_start_time(f)
            duration = get_wav_duration(f)
            file_end = file_start + timedelta(seconds=duration)

            # Skip files outside of range
            if file_end <= current_start or file_start >= current_end:
                continue

            # Read wav data
            sr, data = wavfile.read(f)
            if sample_rate is None:
                sample_rate = sr
                segment_samples = int(segment_minutes * 60 * sr)
                segment_data = np.zeros(segment_samples, dtype=data.dtype)
            elif sr != sample_rate:
                raise ValueError(f"Sample rate mismatch in {f}")

            # Compute overlap region
            overlap_start = max(current_start, file_start)
            overlap_end = min(current_end, file_end)

            start_idx = int((overlap_start - file_start).total_seconds() * sr)
            end_idx = int((overlap_end - file_start).total_seconds() * sr)

            if segment_data is not None:    
                seg_start = int((overlap_start - current_start).total_seconds() * sr)
                seg_end = seg_start + (end_idx - start_idx)
                segment_data[seg_start:seg_end] = data[start_idx:end_idx]

        # Save segment
        if sample_rate is not None:
            timestamp_str = current_start.strftime("%Y%m%dT%H%M%S")
            outname = os.path.join(outdir, f"{timestamp_str}.wav")
            wavfile.write(outname, sample_rate, segment_data)
            print(f"Saved {outname} : {current_start} -> {current_end}")

        # Move to next chunk
        current_start += segment_delta
        segment_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw WAV files into fixed-length segments.")
    
    parser.add_argument("-i", "--input-dir", 
                        required=True, 
                        help="Folder containing the raw .wav files")
                        
    parser.add_argument("-o", "--output-dir", 
                        required=True, 
                        help="Folder to save the segmented .wav files")
                        
    parser.add_argument("-m", "--minutes", 
                        type=int, 
                        default=60, 
                        help="The duration of each segment in minutes (default: 60)")
    
    args = parser.parse_args()

    # --- Run the logic ---
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
    
    if not files:
        print(f"Error: No .wav files found in {args.input_dir}")
    else:
        print(f"Found {len(files)} files. Starting split...")
        batch_split(files, segment_minutes=args.minutes, outdir=args.output_dir)
        print(f"\n Splitting complete. Segments saved to: {args.output_dir}")