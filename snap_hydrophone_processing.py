import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, butter, filtfilt, decimate
from datetime import datetime, timezone, timedelta
import matplotlib.dates as mdates
import json
from spec_features import save_spectrogram_png, save_spec_npz, compute_spectral_features
import event_flags as evflags
import argparse

# --- Applied bandpass filter ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', output='ba')
    return filtfilt(b, a, data)

# --- Main Processing Function ---
def main(wav_folder, save_path, lowcut, highcut, filter_order, nperseg, fs_level, time_offset, png_size=(224,224), preserve_precision=False, detect=False, detect_threshold_db=120.0):
    
    FS_LEVEL_DB = fs_level
    noverlap = nperseg // 2
    
    print(f"--- Hydrophone Processor ---")
    print(f"Input: {wav_folder}\nOutput: {save_path}")
    print(f"Filter: {lowcut/1e3}-{highcut/1e3} kHz, Order {filter_order}")
    print(f"Welch: {nperseg} points | Full-Scale Level: {FS_LEVEL_DB} dB")
    
    # --- LOG THE OFFSET ---
    print(f"Timestamp Offset Applied: {time_offset} seconds")
    
    os.makedirs(save_path, exist_ok=True)

    # --- Check for optional config.json (for TEMPEST/weather sync) ---
    config_path = os.path.join(os.getcwd(), 'config.json')
    weather_attached = False
    config = None
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as cf:
                config = json.load(cf)
                weather_attached = True
                print(f"Loaded config.json; weather data will be attached if available.")
        except Exception as e:
            print(f"Warning: failed to read config.json ({e}). Weather data will NOT be attached.")
            weather_attached = False
            config = None
    else:
        print("Warning: config.json not found — TEMPEST/weather data will NOT be attached to outputs.")

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

        # prepare per-file detections
        file_detections = []

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
            # Simple RMS-based detection per segment (if requested)
            if detect:
                seg_rms = np.sqrt(np.mean(segment ** 2))
                seg_db = 20 * np.log10(seg_rms + np.finfo(float).eps) + FS_LEVEL_DB
                if seg_db >= detect_threshold_db:
                    score = float(min(1.0, (seg_db - detect_threshold_db) / 10.0))
                    file_detections.append({'time_offset': (i * step) / fs, 'label': 'event', 'score': score})

    if not PSD_list or f_kHz is None:
        print("No valid PSD data to process. Exiting.")
        return

    PSD = np.array(PSD_list).T 
    time_array = np.array(segment_time_list)
    overall_levels_db = np.array(overall_levels_db)
    
    median_PSD = np.median(PSD, axis=1)

    output_name = os.path.join(save_path, "SNAPhydrophonespectra_fft")
    # Save compressed arrays and metadata using helper (casts to float32)
    meta = {'lowcut': lowcut, 'highcut': highcut, 'time_offset': time_offset, 'weather_attached': weather_attached}
    # If config was loaded, include minimal config info (do not embed secrets)
    if config is not None:
        # keep only non-sensitive keys if present
        minimal = {k: config.get(k) for k in ('station_id', 'weather_provider') if k in config}
        if minimal:
            meta['config'] = minimal

    # If detection was requested, save per-file flags and add them to meta
    all_flags = []
    if detect:
        # NOTE: we collected per-file detections during processing; however, per-file detections were last scoped inside loop.
        # To avoid refactor complexity, re-run a light detection pass over files here to produce per-file JSONs and aggregate flags.
        wav_files2 = [f for f in os.listdir(wav_folder) if f.lower().endswith('.wav')]
        for file_name in sorted(wav_files2):
            file_path = os.path.join(wav_folder, file_name)
            try:
                fs, data = wavfile.read(file_path)
                if data.ndim > 1:
                    data = data[:, 0]
                # normalize int -> float
                if data.dtype == np.int16:
                    data = data / 32768.0
                elif data.dtype == np.int32:
                    data = data / 2147483648.0
                elif data.dtype == np.uint8:
                    data = (data - 128) / 128.0
                data = data.astype(np.float64)
            except Exception:
                continue
            # compute file start timestamp again
            tstring = file_name.split('.')[0]
            try:
                timestamp = datetime.strptime(tstring, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc) + timedelta(seconds=time_offset)
            except Exception:
                continue
            # run the detector
            detections = evflags.detect_transients_by_rms(data, fs, nperseg, noverlap, detect_threshold_db, FS_LEVEL_DB)
            if detections:
                flags = evflags.detections_to_flags(detections, timestamp)
                # annotate wav basename for CSV/aggregation
                for f in flags:
                    f['wav'] = file_name
                all_flags.extend(flags)
                evflags.save_flags_json(flags, file_path)
                print(f"Saved {len(flags)} flags for {file_name}")
        if all_flags:
            meta['event_flags'] = all_flags

    save_spec_npz(output_name, PSD, f_kHz, time_array, median_PSD, overall_levels_db, meta=meta, preserve_precision=preserve_precision)
    print(f"\nSaved processed data to {output_name}.npz")

    # Export a resized spectrogram PNG (memory-safe) and compute/save spectral features
    png_path = os.path.join(save_path, "spectrogram.png")
    try:
        # PSD shape is (freq_bins, time_bins)
        save_spectrogram_png(PSD, png_path, img_size=png_size)
        print(f"Saved resized spectrogram PNG to {png_path}")
    except Exception as e:
        print(f"Failed to save spectrogram PNG: {e}")

    # produce overlaid spectrogram if detections exist
    if detect and meta.get('event_flags'):
        overlay_path = os.path.join(save_path, 'spectrogram_flags.png')
        try:
            evflags.overlay_flags_on_spec(PSD, time_array, f_kHz, meta['event_flags'], overlay_path, img_size=png_size)
            print(f"Saved overlaid spectrogram with flags to {overlay_path}")
        except Exception as e:
            print(f"Failed to create overlaid spectrogram: {e}")

    # Compute spectral features (centroid, bandwidth, rolloff) and save CSV
    try:
        features = compute_spectral_features(PSD, f_kHz * 1e3)
        # Build CSV: time, centroid, bandwidth, rolloff
        import csv
        csv_path = os.path.join(save_path, "spectral_features.csv")
        with open(csv_path, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['time_iso', 'centroid_Hz', 'bandwidth_Hz', 'rolloff_Hz'])
            for idx in range(len(time_array)):
                tiso = time_array[idx].isoformat()
                writer.writerow([tiso,
                                 float(features['centroid_Hz'][idx]),
                                 float(features['bandwidth_Hz'][idx]),
                                 float(features['rolloff_Hz'][idx])])
        print(f"Saved spectral features to {csv_path}")
    except Exception as e:
        print(f"Failed to compute/save spectral features: {e}")

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
    parser.add_argument("--png-size", type=str, default="224x224",
                        help="PNG output size as HxW (e.g. 224x224) or single int for square")
    parser.add_argument("--preserve-precision", action='store_true',
                        help="If set, keep float64 precision when saving .npz (default: downcast to float32)")
    parser.add_argument("--detect", action='store_true', help="Run simple RMS-based event detector and save flags")
    parser.add_argument("--detect-threshold", type=float, default=120.0, help="Detection threshold in dB (20*log10 RMS) + FS level")

    args = parser.parse_args()

    # parse png size argument
    def _parse_png_size(s):
        if 'x' in s:
            parts = s.lower().split('x')
            try:
                h = int(parts[0])
                w = int(parts[1])
                return (h, w)
            except Exception:
                raise ValueError("Invalid --png-size format. Use HxW, e.g. 224x224")
        else:
            v = int(s)
            return (v, v)

    png_size = _parse_png_size(args.png_size)

    main(wav_folder=args.input_dir,
         save_path=args.output_dir,
         lowcut=args.lowcut,
         highcut=args.highcut,
         filter_order=args.order,
         nperseg=args.nperseg,
         fs_level=args.fs_level,
         time_offset=args.offset,
            png_size=png_size,
            preserve_precision=args.preserve_precision,
            detect=args.detect,
            detect_threshold_db=args.detect_threshold) # Pass the offset to main