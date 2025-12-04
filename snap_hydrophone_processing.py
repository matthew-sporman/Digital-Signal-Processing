import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, butter, filtfilt, decimate
from datetime import datetime, timezone, timedelta
import matplotlib.dates as mdates
import argparse

# --- Applied bandpass filter ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', output='ba')
    return filtfilt(b, a, data)

# --- Main Processing Function ---
def main(wav_folder, save_path, lowcut, highcut, filter_order, nperseg, fs_level, time_offset):
    
    FS_LEVEL_DB = fs_level
    noverlap = nperseg // 2
    
    print(f"--- Hydrophone Processor ---")
    print(f"Input: {wav_folder}\nOutput: {save_path}")
    print(f"Filter: {lowcut/1e3}-{highcut/1e3} kHz, Order {filter_order}")
    print(f"Welch: {nperseg} points | Full-Scale Level: {FS_LEVEL_DB} dB")
    
    # --- LOG THE OFFSET ---
    print(f"Timestamp Offset Applied: {time_offset} seconds")
    
    os.makedirs(save_path, exist_ok=True)

    wav_files = [f for f in os.listdir(wav_folder) if f.lower().endswith(".wav")]
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {wav_folder}")

    f = None
    freq_mask = None
    f_kHz = None

    PSD_list = []
    segment_time_list = []
    file_level_time_list = []
    overall_levels_db = []

    # --- Processes each WAV file ---
    for file_name in sorted(wav_files):
        file_path = os.path.join(wav_folder, file_name)
        print(f"Processing {file_name}...")

        # Extract timestamp
        tstring = file_name.split('.')[0] 
        try:
            timestamp = datetime.strptime(tstring, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            
            # --- APPLY OFFSET HERE ---
            # Corrects the hydrophone time to match Met data
            timestamp = timestamp + timedelta(seconds=time_offset)
            
        except ValueError:
            print(f"Skipping {file_name} — invalid timestamp.")
            continue

        # Read audio
        try:
            fs, data = wavfile.read(file_path)
            if data.ndim > 1:
                data = data[:, 0] 
            
            if data.dtype == np.int16:
                data = data / 32768.0
            elif data.dtype == np.int32:
                data = data / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data - 128) / 128.0
            
            data = data.astype(np.float64)

        except Exception as e:
            print(f"Failed to read {file_name}: {e}")
            continue

        filtered_data = butter_bandpass_filter(data, lowcut, highcut, fs, order=filter_order)

        rms_level_norm = np.sqrt(np.mean(filtered_data**2))
        overall_level_db = 20 * np.log10(rms_level_norm + np.finfo(float).eps) + FS_LEVEL_DB 
        
        overall_levels_db.append(overall_level_db)
        file_level_time_list.append(timestamp) # This timestamp now includes the offset
        
        step = nperseg - noverlap
        num_segments = (len(filtered_data) - noverlap) // step

        for i in range(num_segments):
            segment = filtered_data[i * step : i * step + nperseg]
            if len(segment) < nperseg:
                continue

            f, psd = welch(segment, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density')
            psd_db = 10 * np.log10(psd + np.finfo(float).eps) + FS_LEVEL_DB

            if freq_mask is None:
                freq_mask = (f >= lowcut) & (f <= highcut)
                f_kHz = f[freq_mask] / 1e3

            PSD_list.append(psd_db[freq_mask])
            
            # Apply offset to the micro-segments inside the file as well
            t_seg = timestamp + timedelta(seconds=(i * step) / fs)
            segment_time_list.append(t_seg)

    if not PSD_list or f_kHz is None:
        print("No valid PSD data to process. Exiting.")
        return

    PSD = np.array(PSD_list).T 
    time_array = np.array(segment_time_list)
    overall_levels_db = np.array(overall_levels_db)
    
    median_PSD = np.median(PSD, axis=1)

    output_name = os.path.join(save_path, "SNAPhydrophonespectra_fft")
    np.savez(f"{output_name}.npz",
             f_kHz=f_kHz,
             PSD=PSD,
             median_PSD=median_PSD,
             time=time_array,
             overall_levels_db=overall_levels_db)
    print(f"\nSaved processed data to {output_name}.npz")

    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time_array, f_kHz, PSD, shading="auto")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Frequency (kHz)")
    plt.title(f"Spectrogram ({lowcut/1e3}–{highcut/1e3} kHz)\nOffset Applied: {time_offset}s")
    plt.colorbar(label="Power (dB re 1 µPa²/Hz)")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "spectrogram.png"), dpi=200)
    plt.close()

    # Plot median PSD
    dec_factor = 100
    if len(median_PSD) > dec_factor:
        median_PSD_dec = decimate(median_PSD, dec_factor)
        f_kHz_dec = decimate(f_kHz, dec_factor)
    else:
        median_PSD_dec = median_PSD
        f_kHz_dec = f_kHz

    plt.figure(figsize=(8, 6))
    plt.plot(f_kHz_dec, median_PSD_dec, color='red')
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Power (dB re 1 µPa²/Hz)")
    plt.title(f"Median Power Spectral Density ({lowcut/1e3}–{highcut/1e3} kHz)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "median_psd_decimated.png"), dpi=200)
    plt.close()

    # Plot overall RMS over time
    plt.figure(figsize=(8, 4))
    time_nums = mdates.date2num(file_level_time_list)
    plt.plot(time_nums, overall_levels_db, marker='o')
    plt.xlabel("Time (UTC)")
    plt.ylabel(f"Overall {lowcut/1e3}–{highcut/1e3} kHz Level (dB re 1 µPa)")
    plt.title(f"Hydrophone Overall Level\nOffset Applied: {time_offset}s")
    plt.grid(True)
    plt.gca().xaxis_date()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "overall_levels_over_time.png"), dpi=200)
    plt.close()

    print(f"Saved plots and data to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process segmented WAV files to create spectrograms and plots.")
    
    parser.add_argument("-i", "--input-dir", required=True, help="Folder containing segments")
    parser.add_argument("-o", "--output-dir", required=True, help="Folder to save plots")
    parser.add_argument("--fs-level", type=float, required=True, help="Full-Scale acoustic level")
    parser.add_argument("--lowcut", type=float, default=1000.0, help="Lowcut frequency")
    parser.add_argument("--highcut", type=float, default=20000.0, help="Highcut frequency")
    parser.add_argument("--order", type=int, default=4, help="Filter order")
    parser.add_argument("--nperseg", type=int, default=2048, help="Welch segment size")
    
    # --- ADDED ARGUMENT: Time Offset ---
    parser.add_argument("--offset", type=float, default=0.0, 
                        help="Time synchronization offset in seconds (Positive adds time, Negative subtracts time)")

    args = parser.parse_args()

    main(wav_folder=args.input_dir,
         save_path=args.output_dir,
         lowcut=args.lowcut,
         highcut=args.highcut,
         filter_order=args.order,
         nperseg=args.nperseg,
         fs_level=args.fs_level,
         time_offset=args.offset) # Pass the offset to main