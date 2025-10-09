import os
from datetime import datetime, timedelta
import numpy as np
from scipy.io import wavfile
import wave
import glob

def parse_start_time(filename: str) -> datetime:
    """Extract date and time from wav filename like 20241029T150409_4266917927872453_2.0.wav"""
    base = os.path.basename(filename).split("_")[0]  # "20241029T150409"
    return datetime.strptime(base, "%Y%m%dT%H%M%S")

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
            print(f"Saved {outname} : {current_start} → {current_end}")

        # Move to next chunk
        current_start += segment_delta
        segment_idx += 1


if __name__ == "__main__":

    wav_folder = r"D:\SWIFT12_hydrophone\2024-10"
    files = sorted(glob.glob(os.path.join(wav_folder, "*.wav")))

    outdir = r"D:\SWIFT12_hydrophone\2024-10\output_segments"
    batch_split(files, segment_minutes=12, outdir=outdir)