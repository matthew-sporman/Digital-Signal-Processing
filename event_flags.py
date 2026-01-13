from datetime import timedelta, datetime, timezone
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def detections_to_flags(detections, clip_start_dt):
    flags = []
    for d in detections:
        ts = clip_start_dt + timedelta(seconds=float(d['time_offset']))
        flags.append({
            'timestamp': ts.astimezone(timezone.utc).isoformat(),
            'label': d.get('label', 'event'),
            'score': float(d.get('score', 0.0)),
            'time_offset': float(d.get('time_offset', 0.0))
        })
    return flags


def save_flags_json(flags, wav_path):
    out = Path(wav_path).with_suffix(Path(wav_path).suffix + '.flags.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(flags, f, indent=2)
    return str(out)


def append_flags_csv(flags, master_csv_path):
    fieldnames = ['wav', 'timestamp', 'label', 'score']
    master = Path(master_csv_path)
    first = not master.exists()
    with open(master, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if first:
            writer.writeheader()
        for fl in flags:
            writer.writerow({
                'wav': fl.get('wav', ''),
                'timestamp': fl['timestamp'],
                'label': fl['label'],
                'score': fl['score']
            })


def overlay_flags_on_spec(spec_db, time_array, f_kHz, flags, out_path, img_size=(224, 224), vmin=None, vmax=None):
    if len(time_array) == 0:
        raise ValueError('time_array is empty')

    # convert times to matplotlib numeric format
    time_nums = mdates.date2num(time_array)
    # Ensure t0 and t1 are scalars for extent
    t0 = float(time_nums[0])
    t1 = float(time_nums[-1])

    # y extents
    y0 = float(f_kHz[0])
    y1 = float(f_kHz[-1])

    fig = plt.figure(figsize=(img_size[1] / 100.0, img_size[0] / 100.0), dpi=100)
    ax = fig.add_subplot(111)

    # show spectrogram (flip so low freq at bottom)
    im = ax.imshow(np.flipud(spec_db), aspect='auto', extent=(t0, t1, y0, y1), cmap='viridis')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Frequency (kHz)')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    # plot flags
    for fl in flags:
        try:
            ts = datetime.fromisoformat(fl['timestamp'])
        except Exception:
            continue
            
        # FIX: Explicitly cast to float to satisfy type checkers
        tn = float(mdates.date2num(ts)) 
        
        ax.axvline(tn, color='red', linestyle='--', linewidth=1)
        ax.text(tn, y1, fl.get('label', ''), color='red', rotation=90, va='top', fontsize=6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def detect_transients_by_rms(data, fs, nperseg, noverlap, threshold_db, fs_level_db):
    """Simple RMS-based detector over sliding segments.

    Returns list of detections: {'time_offset': seconds, 'label': 'event', 'score': 0-1}
    """
    detections = []
    step = nperseg - noverlap
    num_segments = (len(data) - noverlap) // step
    eps = np.finfo(float).eps
    for i in range(num_segments):
        seg = data[i * step: i * step + nperseg]
        if len(seg) < nperseg:
            continue
        rms = np.sqrt(np.mean(seg ** 2))
        seg_db = 20 * np.log10(rms + eps) + fs_level_db
        if seg_db >= threshold_db:
            score = float(min(1.0, (seg_db - threshold_db) / 10.0))
            detections.append({'time_offset': (i * step) / fs, 'label': 'event', 'score': score})
    return detections
