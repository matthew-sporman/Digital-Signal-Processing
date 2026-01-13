import numpy as np
from PIL import Image


def _normalize_db_to_uint8(spec_db, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.nanpercentile(spec_db, 1)
    if vmax is None:
        vmax = np.nanpercentile(spec_db, 99)
    spec_clipped = np.clip(spec_db, vmin, vmax)
    norm = (spec_clipped - vmin) / (vmax - vmin + 1e-12)
    return (255 * norm).astype(np.uint8)


def save_spectrogram_png(spec_db, out_path, img_size=(224, 224), vmin=None, vmax=None, flip_ud=True):
    """Save a spectrogram (freq x time) as a memory-safe resized PNG.

    spec_db: 2D array (freq_bins, time_bins) in dB (float)
    out_path: destination PNG path
    img_size: (height, width) in pixels
    """
    arr = _normalize_db_to_uint8(spec_db, vmin=vmin, vmax=vmax)
    if flip_ud:
        arr = np.flipud(arr)
    # Convert to PIL image (mode 'L' grayscale). transpose to (width, height) for resize
    img = Image.fromarray(arr, mode='L')
    # Pillow renamed resampling enums in newer versions. Use compatibility constant.
    # Try to use Pillow's Resampling enum; if unavailable, fall back to numeric 2 (BILINEAR).
    # If resize still fails, fall back to default resize without explicit resample.
    try:
        resample_mode = Image.Resampling.BILINEAR
        img = img.resize((img_size[1], img_size[0]), resample=resample_mode)
    except Exception:
        try:
            img = img.resize((img_size[1], img_size[0]), resample=2)
        except Exception:
            img = img.resize((img_size[1], img_size[0]))
    img.save(out_path, format='PNG')


def save_spec_npz(out_base, spec_db, f_kHz, time_array, median_PSD, overall_levels_db, meta=None, preserve_precision=False):
    """Save spectrogram arrays and metadata compressed as npz.

    By default casts arrays to float32 to reduce size. Set `preserve_precision=True`
    to keep original dtypes (e.g., float64).
    """
    kwargs = {}
    if preserve_precision:
        kwargs['spec_db'] = spec_db
        kwargs['f_kHz'] = f_kHz
        kwargs['median_PSD'] = median_PSD
        kwargs['overall_levels_db'] = overall_levels_db
    else:
        kwargs['spec_db'] = spec_db.astype(np.float32)
        # f_kHz is usually float; cast to float32
        kwargs['f_kHz'] = f_kHz.astype(np.float32)
        kwargs['median_PSD'] = median_PSD.astype(np.float32)
        kwargs['overall_levels_db'] = overall_levels_db.astype(np.float32)

    # time_array may be dtype datetime; store as object array for safety
    kwargs['time'] = time_array
    if meta is not None:
        kwargs['meta'] = np.array([str(meta)])

    np.savez_compressed(f"{out_base}.npz", **kwargs)


def compute_spectral_features(spec_db, f_Hz, rolloff_pct=0.85):
    """Compute spectral centroid, bandwidth, and rolloff per time slice.

    spec_db: 2D array (freq_bins, time_bins) in dB
    f_Hz: 1D array of frequency bin centers in Hz
    Returns dict of arrays (centroid_Hz, bandwidth_Hz, rolloff_Hz)
    """
    # Convert dB to linear power
    P = 10 ** (spec_db / 10.0)
    P_sum = P.sum(axis=0) + 1e-12

    centroid = (f_Hz[:, None] * P).sum(axis=0) / P_sum

    # bandwidth: sqrt of second central moment
    diff = f_Hz[:, None] - centroid[None, :]
    bandwidth = np.sqrt((diff**2 * P).sum(axis=0) / P_sum)

    # rolloff frequency
    cumsum = np.cumsum(P, axis=0)
    total = cumsum[-1, :]
    thresh = rolloff_pct * total
    rolloff_idx = np.argmax(cumsum >= thresh[None, :], axis=0)
    rolloff = f_Hz[rolloff_idx]

    return {
        'centroid_Hz': centroid.astype(np.float32),
        'bandwidth_Hz': bandwidth.astype(np.float32),
        'rolloff_Hz': rolloff.astype(np.float32)
    }
