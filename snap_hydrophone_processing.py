import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, butter, filtfilt, decimate
from datetime import datetime, timezone, timedelta
import matplotlib.dates as mdates

# -------- USER SETTINGS -------- #
wav_folder = r"D:\SWIFT12_hydrophone\2024-10"  # Path to folder with .wav files
lowcut = 1000    # 1 kHz
highcut = 20000  # 20 kHz
filter_order = 4  # Butterworth filter order

# Welch parameters
nperseg = 2048
noverlap = nperseg // 2
# --------------------------------

save_path = r"D:\SWIFT12_hydrophone\ProcessedResults"
os.makedirs(save_path, exist_ok=True)

# Finds the WAV files
wav_files = [f for f in os.listdir(wav_folder) if f.lower().endswith(".wav")]
if not wav_files:
    raise FileNotFoundError(f"No .wav files found in {wav_folder}")

PSD_list = []
time_list = []
overall_levels_db = []

# --- Bandpass filter ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- Process each WAV file ---
for file_name in sorted(wav_files):
    file_path = os.path.join(wav_folder, file_name)
    print(f"Processing {file_name}...")

    # Extract timestamp from filename (assumes format "YYYYMMDDTHHMMSS")
    tstring = file_name[:15]
    try:
        timestamp = datetime.strptime(tstring, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"⚠️ Skipping {file_name} — invalid timestamp.")
        continue

    # Read audio
    fs, data = wavfile.read(file_path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float64)

    filtered_data = butter_bandpass_filter(data, lowcut, highcut, fs, order=filter_order)

    # --- Compute overall Root Mean Square (RMS) for the entire file ---
    rms_level = np.sqrt(np.mean(filtered_data**2))
    overall_level_db = 20 * np.log10(rms_level + np.finfo(float).eps)
    overall_levels_db.append(overall_level_db)
    time_list.append(timestamp)
    print(f"Overall 1–20 kHz level: {overall_level_db:.2f} dB (relative)")

    # --- Compute Welch per overlapping segment for smooth spectrogram ---
    step = nperseg - noverlap
    num_segments = (len(filtered_data) - noverlap) // step

    for i in range(num_segments):
        segment = filtered_data[i*step : i*step + nperseg]
        if len(segment) < nperseg:
            continue
        f, psd = welch(segment, fs=fs, nperseg=nperseg, noverlap=0, scaling='density')
        psd_db = 10 * np.log10(psd + np.finfo(float).eps)

        # Keep only 1–20 kHz
        freq_mask = (f >= lowcut) & (f <= highcut)
        PSD_list.append(psd_db[freq_mask])

        # Timestamp for this segment
        t_seg = timestamp + timedelta(seconds=(i*step)/fs)
        time_list.append(t_seg)

# --- Converts lists to arrays ---
PSD = np.array(PSD_list).T  # frequency x time
time_array = np.array(time_list)
f_kHz = f[freq_mask] / 1e3
overall_levels_db = np.array(overall_levels_db)

# --- Compute median Power Spectral Density (PSD) across all files ---
median_PSD = np.median(PSD, axis=1)

# --- Save processed data ---
output_name = os.path.join(save_path, "SNAPhydrophonespectra_fft")
np.savez(f"{output_name}.npz", f_kHz=f_kHz, PSD=PSD, median_PSD=median_PSD,
         time=time_array, overall_levels_db=overall_levels_db)
print(f"\n✅ Saved processed data to {output_name}.npz")

# --- Plot spectrogram ---
plt.figure(figsize=(10, 6))
plt.pcolormesh(time_array, f_kHz, PSD, shading="auto")
plt.xlabel("Time (UTC)")
plt.ylabel("Frequency [kHz]")
plt.title("Hydrophone Spectrogram (1–20 kHz, Welch per window)")
plt.colorbar(label="Power (dB, relative)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "spectrogram.png"), dpi=200)
plt.close()

# --- Decimate median Power Spectral Density (PSD) ---
dec_factor = 100
median_PSD_dec = decimate(median_PSD, dec_factor)
f_kHz_dec = decimate(f_kHz, dec_factor)

# --- Plot median Power Spectral Density (PSD) ---
plt.figure(figsize=(8, 6))
plt.plot(f_kHz_dec, median_PSD_dec, color='red')
plt.xlabel("Frequency [kHz]")
plt.ylabel("Power (dB, relative)")
plt.title("Median PSD (1–20 kHz, Welch, decimated)")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "median_psd_decimated.png"), dpi=200)
plt.close()

# --- Plot overall Root Mean Square (RMS) over time ---
plt.figure(figsize=(8, 4))
time_nums = mdates.date2num(time_array[:len(overall_levels_db)])  # use only one timestamp per file
plt.plot(time_nums, overall_levels_db, marker='o')
plt.xlabel("Time (UTC)")
plt.ylabel("Overall 1–20 kHz Level (dB, relative)")
plt.title("Hydrophone Overall Level (1–20 kHz)")
plt.grid(True)
plt.gca().xaxis_date()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "overall_levels_over_time.png"), dpi=200)
plt.close()

print(f"✅ Saved plots to {save_path}")