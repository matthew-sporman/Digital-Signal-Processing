import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, butter, filtfilt, decimate
from datetime import datetime, timezone, timedelta
import matplotlib.dates as mdates

# -------- USER SETTINGS -------- #
wav_folder = r"D:\SWIFT12_hydrophone\2024-10\output_segments"  # Path to folder with .wav files
lowcut = 1000    # 1 kHz
highcut = 20000  # 20 kHz
filter_order = 4  # Butterworth filter order

# Welch parameters
nperseg = 2048
noverlap = nperseg // 2
# --------------------------------

save_path = r"D:\SWIFT12_hydrophone\2024-10\ProcessedResults"
os.makedirs(save_path, exist_ok=True)

# Finds the WAV files
wav_files = [f for f in os.listdir(wav_folder) if f.lower().endswith(".wav")]
if not wav_files:
    raise FileNotFoundError(f"No .wav files found in {wav_folder}")

# Initialize varibles
f = None
freq_mask = None
f_kHz = None

# --- Lists to store data ---
PSD_list = []
segment_time_list = []         # For spectrogram -> Welch segment timestamps
file_level_time_list = []      # For overall RMS level plot
overall_levels_db = []

# --- Applied bandpass filter ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    # Second order sections for numerical stability
    b, a = butter(order, [low, high], btype='band', output='ba')
    return filtfilt(b, a, data)

# --- Processes each WAV file ---
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
    try:
        fs, data = wavfile.read(file_path)

        # --- Normalization of WAV files ---
        if data.ndim > 1:
            data = data[:, 0]  # take first channel if stereo

        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data - 128) / 128.0
        elif data.dtype in [np.float32, np.float64]:
            pass  # already float
        data = data.astype(np.float64)

    except Exception as e:
        print(f"⚠️ Failed to read {file_name}: {e}")
        continue  # skips this file

    # Step 1: Filtering (1–20 kHz)
    filtered_data = butter_bandpass_filter(data, lowcut, highcut, fs, order=filter_order)

     # Step 2: RMS level
    # --- Squares all values, takes mean + sqrt, then converts to dB. ---
    hydrophone_sensitivity_db = -170 # dBV/μPa (Volts: V)
    rms_level = np.sqrt(np.mean(filtered_data**2))
    overall_level_db = 20 * np.log10(rms_level + np.finfo(float).eps) + hydrophone_sensitivity_db # Check this ---
    overall_levels_db.append(overall_level_db)
    file_level_time_list.append(timestamp)
    print(f"Overall 1–20 kHz level: {overall_level_db:.2f} dB re 1 µPa")

    # --- Compute Welch per overlapping segment for smooth spectrogram ---
    step = nperseg - noverlap
    num_segments = (len(filtered_data) - noverlap) // step

    for i in range(num_segments):
        # Extract a segment of data for this Welch calculation
        segment = filtered_data[i * step : i * step + nperseg]
        if len(segment) < nperseg:
            continue

        # --- Step 3: Spectra / Spectrogram (per segment) ---
        f, psd = welch(segment, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='density')

        # Convert PSD to dB and apply hydrophone sensitivity
        psd_db = 10 * np.log10(psd + np.finfo(float).eps) + hydrophone_sensitivity_db # Check this ---

         # Initialize freq_mask and f_kHz only once
        if freq_mask is None:
            freq_mask = (f >= lowcut) & (f <= highcut)
            f_kHz = f[freq_mask] / 1e3

        PSD_list.append(psd_db[freq_mask])
        t_seg = timestamp + timedelta(seconds=(i * step) / fs)
        segment_time_list.append(t_seg)


if not PSD_list or f_kHz is None:
    print("⚠️ No valid PSD data to process.")
    exit(0)

    # --- Convert lists to arrays ---
PSD = np.array(PSD_list).T  # frequency x time
time_array = np.array(segment_time_list)
overall_levels_db = np.array(overall_levels_db)

# --- Compute median Power Spectral Density (PSD) across all files ---
median_PSD = np.median(PSD, axis=1)

# --- Save processed data ---
output_name = os.path.join(save_path, "SNAPhydrophonespectra_fft")
np.savez(f"{output_name}.npz",
         f_kHz=f_kHz,
         PSD=PSD,
         median_PSD=median_PSD,
         time=time_array,
         overall_levels_db=overall_levels_db)
print(f"\n✅ Saved processed data to {output_name}.npz")

# --- Print dimensions of the spectrogram data ---
print("Spectrogram plot dimensions:")
print(f"  PSD shape: {PSD.shape} (freq x time)")

# --- Plot spectrogram ---
plt.figure(figsize=(10, 6))
plt.pcolormesh(time_array, f_kHz, PSD, shading="auto")
plt.xlabel("Time (UTC)")
plt.ylabel("Frequency (kHz)")
plt.title("Spectrogram (1–20 kHz)")
plt.colorbar(label="Power (dB re 1 µPa²/Hz)")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "spectrogram.png"), dpi=200)
plt.close()

# --- Decimate median PSD to reduce frequency resolution ---
dec_factor = 100
if len(median_PSD) > dec_factor:
    median_PSD_dec = decimate(median_PSD, dec_factor)
    f_kHz_dec = decimate(f_kHz, dec_factor)
else:
    median_PSD_dec = median_PSD
    f_kHz_dec = f_kHz

# --- Plot median Power Spectral Density (PSD) ---
plt.figure(figsize=(8, 6))
plt.plot(f_kHz_dec, median_PSD_dec, color='red')
plt.xlabel("Frequency (kHz)")
plt.ylabel("Power (dB re 1 µPa²/Hz)")
plt.title("Median Power Spectral Density (1–20 kHz)")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig(os.path.join(save_path, "median_psd_decimated.png"), dpi=200)
plt.close()

# --- Plot overall RMS over time ---
plt.figure(figsize=(8, 4))
time_nums = mdates.date2num(file_level_time_list)
plt.plot(time_nums, overall_levels_db, marker='o')
plt.xlabel("Time (UTC)")
plt.ylabel("Overall 1–20 kHz Level (dB re 1 µPa)")
plt.title("Hydrophone Overall Level (1–20 kHz)")
plt.grid(True)
plt.gca().xaxis_date()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(save_path, "overall_levels_over_time.png"), dpi=200)
plt.close()

print(f"✅ Saved plots to {save_path}")