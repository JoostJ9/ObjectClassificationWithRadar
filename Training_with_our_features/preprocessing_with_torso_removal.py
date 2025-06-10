#!/usr/bin/env python3
# ================================================================
#   INSHEP bulk-feature extractor — **multicore version**
#   ▪  Spawns one worker per logical CPU (max 12 on your machine)
#   ▪  Streams results straight into a versioned CSV file
#   ▪  Totally self-contained: just place this script beside the
#       datasets/  directory and run:  python fast_extract.py
# ================================================================
import os, math, warnings, csv, sys
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import numpy as np
import pandas as pd
from scipy import signal, stats
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte

# ──────────────────────────────────────────────────────────────
#  optional pseudo-Zernike moments  (needs  pip install mahotas)
# ──────────────────────────────────────────────────────────────
try:
    import mahotas as mh
    _HAS_MAHOTAS = True
except ImportError:
    warnings.warn("⚠️  mahotas not found – pseudo-Zernike moments will be 0.")
    _HAS_MAHOTAS = False

# Added filelock for safe concurrent versioning
try:
    import filelock
except ImportError:
    print("❌  Error: 'filelock' library not found.")
    print("   Please install it by running:  pip install filelock")
    exit(1)

# ──────────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ──────────────────────────────────────────────────────────────

### MODIFIED AND IMPROVED ###
# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).resolve().parent
# Assume the 'datasets' folder is in the parent directory of the script's folder
# This is a much more robust way to find the data.
# e.g., .../Radar/ObjectClassificationWithRadar/ (script) and .../Radar/datasets/ (data)
DATASETS_ROOT = SCRIPT_DIR.parent / "datasets"

# Add a check to ensure the directory exists and provide a helpful error if not.
if not DATASETS_ROOT.exists():
    print(f"❌  Error: The 'datasets' directory was not found at the expected location.")
    print(f"   The script looked for it here: {DATASETS_ROOT.resolve()}")
    print(f"   Please ensure your folder structure is correct.")
    sys.exit(1) # Exit the script if the data directory isn't found

CSV_BASENAME    = "INSHEP_features"
VERSION_FILE    = "inshep.version" # This .txt file tracks the version number

ACTIVITY_MAP = {
    "1": "walking",
    "2": "sitting_down",
    "3": "standing_up",
    "4": "pick_object",
    "5": "drink_water",
    "6": "fall",
}

TIME_WINDOW     = 200
OVERLAP_FRAC    = 0.95
PAD_FACTOR      = 4
BUTTER_N        = 4
BUTTER_CUT      = 0.0075         # high-pass cut-off (fraction of Nyquist)
TORSO_V_MAX     = 0.25           # ± m/s
DENSITY_THR_DB  = -3             # dB down from peak for masks

FIELDNAMES = [
    "file_id", "activity", "path",
    "mean_entropy", "mean_power", "variance", "stddev",
    "max_vel", "amp_density", "kurtosis", "zernike_moment",
    "periodicity", "mean_torso_power", "pos_neg_ratio",
    "doppler_offset", "main_lobe_width","auto_correlation",
    "envelope_width", "limb_asymmetry", "limb_power",
    "limb_smoothness", "clean_kurtosis",
    "motion_duration", "doppler_peak_velocity", "doppler_symmetry_index",
    "cepstral_entropy", "range_bin_span", "doppler_bandwidth",
    "skew_val",
    "contrast", "dissimilarity", "homogeneity", "energy",
    "correlation", "ASM"
    ]

# ──────────────────────────────────────────────────────────────
#  LOW-LEVEL UTILITIES  (top-level → picklable)
# ──────────────────────────────────────────────────────────────

def get_versioned_csv_path():
    """
    Safely gets the next available version number and returns the path.
    Uses a lock file to prevent race conditions when run concurrently.
    """
    lock = filelock.FileLock(f"{VERSION_FILE}.lock")
    with lock:
        try:
            with open(VERSION_FILE, "r") as f:
                version = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            version = 0

        next_version = version + 1

        with open(VERSION_FILE, "w") as f:
            f.write(str(next_version))

        return Path(f"{CSV_BASENAME}_v{next_version}.csv")

def iq_correction(raw_data):
    """
    Perform I/Q correction on complex radar data
    
    Args:
        raw_data (np.ndarray): Complex IQ data
        
    Returns:
        np.ndarray: Corrected complex IQ data
    """
    i_data = np.real(raw_data)
    q_data = np.imag(raw_data)
    
    i_dc = np.mean(i_data)
    q_dc = np.mean(q_data)
    i_data = i_data - i_dc
    q_data = q_data - q_dc
    
    i_amp = np.sqrt(np.mean(i_data**2))
    q_amp = np.sqrt(np.mean(q_data**2))
    amp_correction = np.sqrt(i_amp * q_amp)
    i_data = i_data * (amp_correction / i_amp) if i_amp > 1e-9 else i_data
    q_data = q_data * (amp_correction / q_amp) if q_amp > 1e-9 else q_data
    
    iq_corr = np.mean(i_data * q_data)
    phase_arg = iq_corr / (i_amp * q_amp + 1e-12)
    phase_error = np.arcsin(np.clip(phase_arg, -1.0, 1.0))
    q_data_corr = q_data * np.cos(phase_error) - i_data * np.sin(phase_error)
    
    return i_data + 1j * q_data_corr

def read_dat(path: Path):
    """Load one *.dat file and return fc [Hz], Tsweep [s], MTI-filtered range-time matrix."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f]

    fc, Tsweep_ms, NTS, Bw = map(float, lines[:4])
    Tsweep = Tsweep_ms * 1e-3
    NTS    = int(NTS)
    raw    = np.array([complex(s.replace("i", "j")) for s in lines[4:]])

    raw_corrected = iq_correction(raw)

    n_chirps = raw_corrected.size // NTS
    time_mat = raw_corrected[: n_chirps * NTS].reshape((NTS, n_chirps), order="F")

    rng_fft  = np.fft.fftshift(np.fft.fft(time_mat, axis=0), axes=0)
    rng_half = rng_fft[NTS // 2 :, :]

    b, a     = signal.butter(BUTTER_N, BUTTER_CUT, "high")
    mti      = signal.lfilter(b, a, rng_half, axis=1)

    return fc, Tsweep, mti[1:, :]


def kalman_filter_1d(observed, dt, process_noise, measurement_noise):
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0.5 * dt**2], [dt]])
    C = np.array([[1, 0]])
    Q = process_noise * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
    R = measurement_noise ** 2

    x = np.array([[observed[0]], [0]])
    P = np.eye(2)
    filtered = []

    for z in observed:
        x = A @ x
        P = A @ P @ A.T + Q
        K = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
        x = x + K @ (z - C @ x)
        P = (np.eye(2) - K @ C) @ P
        filtered.append(x[0, 0])
    
    return np.array(filtered)

def remove_torso_from_spectrogram(Sxx, velocity_axis, torso_velocity_trace, bandwidth=0.4):
    Sxx_centered = np.zeros_like(Sxx)
    zero_idx = np.argmin(np.abs(velocity_axis - 0))

    for i, torso_vel in enumerate(torso_velocity_trace):
        shift_idx = np.argmin(np.abs(velocity_axis - torso_vel))
        shift_amount = zero_idx - shift_idx
        shifted_col = np.roll(Sxx[:, i], shift=shift_amount)
        suppress_mask = np.abs(velocity_axis) <= bandwidth
        shifted_col[suppress_mask] = 0
        Sxx_centered[:, i] = shifted_col

    return Sxx_centered

def detect_envelope(Sxx_dB, velocity_axis, threshold_dB):
    n_bins, n_frames = Sxx_dB.shape
    upper_envelope = np.full(n_frames, np.nan)
    lower_envelope = np.full(n_frames, np.nan)

    for i in range(n_frames):
        above_thresh = np.where(Sxx_dB[:, i] > threshold_dB)[0]
        if len(above_thresh) > 0:
            lower_envelope[i] = velocity_axis[above_thresh[0]]
            upper_envelope[i] = velocity_axis[above_thresh[-1]]

    return lower_envelope, upper_envelope

def stft_mag(mti, prf):
    nperseg  = TIME_WINDOW
    noverlap = int(round(nperseg * OVERLAP_FRAC))
    nfft     = PAD_FACTOR * nperseg

    S_accum = None
    num_range_bins = mti.shape[0]
    start_bin = min(9, num_range_bins -1)
    end_bin = min(30, num_range_bins)

    for r in range(start_bin, end_bin):
        _, _, S = signal.spectrogram(
            mti[r, :], fs=prf, window="hann", nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, mode="complex", return_onesided=False,
        )
        S = np.fft.fftshift(S, axes=0)
        S_accum = np.abs(S) if S_accum is None else S_accum + np.abs(S)

    doppler = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / prf))
    return S_accum, doppler

def binary_mask(db_img, thresh_db):
    return db_img >= (db_img.max() + thresh_db)

def pseudo_zernike(img, radius=20, degree=4):
    if not _HAS_MAHOTAS or img.ptp() == 0:
        return 0.0
    size   = max(img.shape)
    padder = [(0, size - img.shape[0]), (0, size - img.shape[1])]
    img_n  = (np.pad(img, padder) - img.min()) / (img.ptp() + 1e-12)
    return float(np.mean(np.abs(mh.features.zernike_moments(img_n, radius, degree=degree))))

def extract_glcm_features(spectrogram_image):
    spectrogram_image = np.abs(spectrogram_image)
    if spectrogram_image.max() == spectrogram_image.min():
        return {'contrast': 0, 'dissimilarity': 0, 'homogeneity': 1, 'energy': 1, 'correlation': 0, 'ASM': 1}
    
    spectrogram_image = (spectrogram_image - spectrogram_image.min()) / (spectrogram_image.max() - spectrogram_image.min())
    spectrogram_image = img_as_ubyte(spectrogram_image)

    glcm = graycomatrix(
        spectrogram_image, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256, symmetric=True, normed=True
    )
    
    return {
        'contrast': graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'energy': graycoprops(glcm, 'energy').mean(),
        'correlation': graycoprops(glcm, 'correlation').mean(),
        'ASM': graycoprops(glcm, 'ASM').mean()
    }

def extract_features(mti, fc, Tsweep):
    prf = 1.0 / Tsweep
    S, doppler = stft_mag(mti, prf)
    if S is None: # Handle case where STFT produces no output
        return {key: 0.0 for key in FIELDNAMES if key not in ["file_id", "activity", "path"]}

    S2 = S**2
    flat = S2.ravel()
    p = flat / (flat.sum() + 1e-12)
    mean_entropy = float(-(p * np.log(p + 1e-12)).sum())
    mean_power = float(flat.mean())
    variance = float(flat.var())
    stddev = float(math.sqrt(variance))
    v_axis = doppler * 3e8 / (2 * fc)
    vmax = float(v_axis[np.unravel_index(S.argmax(), S.shape)[0]])
    amp_density = binary_mask(20 * np.log10(S + 1e-12), DENSITY_THR_DB).mean()
    kurtosis_val = float(stats.kurtosis(flat, fisher=False))
    z_moment = pseudo_zernike(S)
    pw_sweep = S2.sum(axis=0)
    acf = signal.correlate(pw_sweep, pw_sweep, mode="full")[len(pw_sweep)-1 :]
    periodicity = float(acf[1:].max() / (acf[0] + 1e-12)) if len(acf) > 1 else 0.0
    torso_mask = np.abs(v_axis) <= TORSO_V_MAX
    mean_torso_power = float(S2[torso_mask, :].mean())
    pos_power = S2[v_axis > 0, :].sum()
    neg_power = S2[v_axis < 0, :].sum()
    pos_neg_ratio = float(pos_power / (neg_power + 1e-12))
    weights = S2.sum(axis=1)
    doppler_offset = float((v_axis * weights).sum() / (weights.sum() + 1e-12))
    row_db = 20 * np.log10(S2.mean(axis=1) + 1e-12)
    mask = binary_mask(row_db, DENSITY_THR_DB)
    main_lobe_width = float(v_axis[np.where(mask)[0].max()] - v_axis[np.where(mask)[0].min()]) if mask.any() else 0.0
    auto_correlation = float(acf[1] / (acf[0] + 1e-12)) if len(acf) > 1 else 0.0
    raw_idx = np.argmax(S, axis=0)
    raw_torso_v = v_axis[raw_idx]
    kalman_torso_v = kalman_filter_1d(raw_torso_v, dt=(1 / prf), process_noise=10000.0, measurement_noise=0.1)
    S2_torso_removed = remove_torso_from_spectrogram(S2, v_axis, kalman_torso_v)
    S2_dB_torso_removed = 20 * np.log10(S2_torso_removed + 1e-12)
    threshold = np.min(S2_dB_torso_removed) + 20
    lower_env, upper_env = detect_envelope(S2_dB_torso_removed, v_axis, threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        env_width = np.nanmean(upper_env - lower_env) if np.any(~np.isnan(upper_env - lower_env)) else 0.0
        env_asymmetry = np.nanmean(upper_env + lower_env) if np.any(~np.isnan(upper_env + lower_env)) else 0.0
        smoothness = np.nanstd(np.gradient(upper_env[~np.isnan(upper_env)])) if np.any(~np.isnan(upper_env)) else 0.0
    limb_power = float(S2_torso_removed[(v_axis[:, None] >= lower_env) & (v_axis[:, None] <= upper_env)].sum())
    kurtosis_clean = float(stats.kurtosis(S2_torso_removed.ravel(), fisher=False))
    nperseg = TIME_WINDOW
    noverlap = int(round(nperseg * OVERLAP_FRAC))
    time_step = (nperseg - noverlap) / prf
    motion_thresh = pw_sweep.max() * (10**(DENSITY_THR_DB / 10))
    motion_duration = float((pw_sweep >= motion_thresh).sum() * time_step)
    avg_doppler_spectrum = S2.mean(axis=1)
    doppler_peak_velocity = float(v_axis[avg_doppler_spectrum.argmax()])
    pos_mask, neg_mask = v_axis > 0, v_axis < 0
    pos_power_avg = avg_doppler_spectrum[pos_mask].sum()
    neg_power_avg = avg_doppler_spectrum[neg_mask].sum()
    doppler_symmetry_index = float((pos_power_avg - neg_power_avg) / (pos_power_avg + neg_power_avg + 1e-12))
    log_spec = np.log(avg_doppler_spectrum + 1e-12)
    cepstrum = np.abs(np.fft.irfft(log_spec))**2
    p_cep = cepstrum / (cepstrum.sum() + 1e-12)
    cepstral_entropy = float(-(p_cep * np.log(p_cep + 1e-12)).sum())
    range_power = np.sum(np.abs(mti)**2, axis=1)
    range_thresh = range_power.max() * (10**(DENSITY_THR_DB / 10))
    active_bins_mask = range_power >= range_thresh
    range_bin_span = float(np.where(active_bins_mask)[0].max() - np.where(active_bins_mask)[0].min()) if active_bins_mask.any() else 0.0
    doppler_variance = (weights * (v_axis - doppler_offset)**2).sum() / (weights.sum() + 1e-12)
    doppler_bandwidth = float(np.sqrt(doppler_variance))
    skew_val = float(stats.skew(S2.mean(axis=1)))
    glcm_feats = extract_glcm_features(S2)
    return dict(
        mean_entropy=mean_entropy, mean_power=mean_power, variance=variance, stddev=stddev,
        max_vel=vmax, amp_density=amp_density, kurtosis=kurtosis_val, zernike_moment=z_moment,
        periodicity=periodicity, mean_torso_power=mean_torso_power, pos_neg_ratio=pos_neg_ratio,
        doppler_offset=doppler_offset, main_lobe_width=main_lobe_width, auto_correlation=auto_correlation,
        envelope_width=env_width, limb_asymmetry=env_asymmetry, limb_power=limb_power,
        limb_smoothness=smoothness, clean_kurtosis=kurtosis_clean, motion_duration=motion_duration,
        doppler_peak_velocity=doppler_peak_velocity, doppler_symmetry_index=doppler_symmetry_index,
        cepstral_entropy=cepstral_entropy, range_bin_span=range_bin_span, doppler_bandwidth=doppler_bandwidth,
        skew_val=skew_val, **glcm_feats,
    )

def process_one(path: Path):
    """Worker wrapper: returns dict ready for CSV OR raises."""
    fc, Tsweep, mti = read_dat(path)
    feats = extract_features(mti, fc, Tsweep)
    fid = path.stem
    feats.update(
        file_id=fid,
        activity=ACTIVITY_MAP.get(fid[0], "unknown"),
        path=str(path),
    )
    return feats

# ──────────────────────────────────────────────────────────────
#  MAIN — run workers & stream to CSV
# ──────────────────────────────────────────────────────────────
def main():
    """
    Main execution function. Finds files, sets up a versioned CSV,
    and processes all files in parallel.
    """
    # .rglob will search recursively through all subdirectories
    all_files = sorted(DATASETS_ROOT.rglob("*.dat"))
    n_files = len(all_files)
    if not n_files:
        print(f"❌ No .dat files found under {DATASETS_ROOT.resolve()}. Please check the path.")
        return

    output_csv_path = get_versioned_csv_path()
    print(f"🚀 Found {n_files} files. Starting feature extraction.")
    print(f"   Data source: {DATASETS_ROOT.resolve()}")
    print(f"   Output will be written to: {output_csv_path}")

    try:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
            
            max_workers = min(12, cpu_count())
            print(f"   Spawning {max_workers} worker processes...")

            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                future_to_path = {pool.submit(process_one, p): p for p in all_files}

                for i, fut in enumerate(as_completed(future_to_path), 1):
                    path = future_to_path[fut]
                    try:
                        row = fut.result()
                        writer.writerow(row)
                        csv_file.flush()
                        print(f"✓ [{i:>4}/{n_files}] Processed: {path.name}")
                    except Exception:
                        print(f"✗ [{i:>4}/{n_files}] FAILED:    {path.name}")
                        # Print full traceback for the failed file
                        traceback.print_exc()

    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user. Cleaning up...")
    except Exception:
        print(f"\n\n💥 An unexpected error occurred in the main process:")
        traceback.print_exc()
    finally:
        print(f"\n✅  Processing finished. Features saved to {output_csv_path}")

if __name__ == "__main__":
    main()