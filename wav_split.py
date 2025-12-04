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
    Handles duty cycling (skips gaps) and file boundaries automatically.
    """
    # --- Clears previous segments before processing ---
    if os.path.exists(outdir):
        for f in glob.glob(os.path.join(outdir, "*.wav")):
            try:
                os.remove(f)
            except OSError:
                pass
    else:
        os.makedirs(outdir, exist_ok=True)

    # --- Pre-calculate metadata for all files (Performance Boost) ---
    print("Indexing files...")
    file_info_list = []
    for f in sorted(files, key=parse_start_time):
        start = parse_start_time(f)
        duration = get_wav_duration(f)
        file_info_list.append({
            'path': f,
            'start': start,
            'end': start + timedelta(seconds=duration),
            'duration': duration
        })

    if not file_info_list:
        print("No valid files to process.")
        return

    # Build a global timeline
    # We start exactly at the beginning of the first file
    start_time = file_info_list[0]['start']
    
    # We end exactly at the end of the last file
    last_file = file_info_list[-1]
    global_end_time = last_file['end']

    segment_delta = timedelta(minutes=segment_minutes)
    current_start = start_time
    segment_idx = 0

    print(f"Time range: {start_time} -> {global_end_time}")

    while current_start < global_end_time:
        current_end = current_start + segment_delta
        segment_data = None
        sample_rate = None
        has_overlap = False

        # --- 1. check for overlaps in this window ---
        # We assume files are sorted. We look for any file that intersects [current_start, current_end]
        files_in_segment = []
        
        for info in file_info_list:
            # If file ends before this segment starts, skip it (too early)
            if info['end'] <= current_start:
                continue
            
            # If file starts after this segment ends, stop looking (files are sorted)
            if info['start'] >= current_end:
                break

            # If we get here, there is an overlap
            files_in_segment.append(info)

        # --- 2. DUTY CYCLE LOGIC: Handling Gaps ---
        if not files_in_segment:
            # NO files were found in this 12-minute block. We are in a "Duty Cycle Gap".
            # Instead of saving a silence file, we fast-forward to the next file.
            
            # Find the next available file start time
            next_start = None
            for info in file_info_list:
                if info['start'] > current_start:
                    next_start = info['start']
                    break
            
            if next_start:
                print(f"Skipping gap: {current_start} -> {next_start} (Duty Cycle)")
                current_start = next_start
                continue # Jump back to start of while loop with new time
            else:
                break # No more files ahead, we are done.

        # --- 3. Process the overlapping files ---
        for info in files_in_segment:
            sr, data = wavfile.read(info['path'])
            
            # Initialize the empty container (only once per segment)
            if sample_rate is None:
                sample_rate = sr
                segment_samples = int(segment_minutes * 60 * sr)
                segment_data = np.zeros(segment_samples, dtype=data.dtype)
            elif sr != sample_rate:
                raise ValueError(f"Sample rate mismatch in {info['path']}")

            # Calculate overlap indices
            overlap_start = max(current_start, info['start'])
            overlap_end = min(current_end, info['end'])

            # Source indices (File)
            start_idx = int((overlap_start - info['start']).total_seconds() * sr)
            end_idx = int((overlap_end - info['start']).total_seconds() * sr)

            # Destination indices (Segment)
            seg_start_idx = int((overlap_start - current_start).total_seconds() * sr)
            seg_end_idx = seg_start_idx + (end_idx - start_idx)

            # Handle edge case where rounding errors cause array shape mismatch (off by 1 sample)
            # We clip the indices to ensure we stay within bounds
            data_chunk = data[start_idx:end_idx]
            
            # Ensure the chunk fits into the segment slot
            write_len = min(len(data_chunk), (len(segment_data) - seg_start_idx))
            
            if write_len > 0:
                segment_data[seg_start_idx : seg_start_idx + write_len] = data_chunk[:write_len]
                has_overlap = True

        # --- 4. Save Segment ---
        if has_overlap and sample_rate is not None:
            timestamp_str = current_start.strftime("%Y%m%dT%H%M%S")
            outname = os.path.join(outdir, f"{timestamp_str}.wav")
            wavfile.write(outname, sample_rate, segment_data)
            print(f"Saved {outname}")

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