#!/usr/bin/env python3
# ================================================================
#   INSHEP bulk-feature extractor — **multicore version**
#   ▪  Spawns one worker per logical CPU (max 12 on your machine)
#   ▪  Streams results straight into CSV  (appends row-by-row)
#   ▪  Totally self-contained: just place this script beside the
#       datasets/  directory and  python fast_extract.py
# ================================================================
import os, math, warnings, csv
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy import signal, stats

# ──────────────────────────────────────────────────────────────
#  optional pseudo-Zernike moments  (needs  pip install mahotas)
# ──────────────────────────────────────────────────────────────
try:
    import mahotas as mh
    _HAS_MAHOTAS = True
except ImportError:
    warnings.warn("⚠️  mahotas not found – pseudo-Zernike moments will be 0.")
    _HAS_MAHOTAS = False

# ──────────────────────────────────────────────────────────────
#  GLOBAL CONSTANTS
# ──────────────────────────────────────────────────────────────
DATASETS_ROOT   = Path("datasets")
CSV_PATH        = "INSHEP_features.csv"

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
    "doppler_offset", "main_lobe_width",
]

# ──────────────────────────────────────────────────────────────
#  LOW-LEVEL UTILITIES  (top-level → picklable)
# ──────────────────────────────────────────────────────────────
def read_dat(path: Path):
    """Load one *.dat file and return fc [Hz], Tsweep [s], MTI-filtered range-time matrix."""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f]

    fc, Tsweep_ms, NTS, Bw = map(float, lines[:4])
    Tsweep = Tsweep_ms * 1e-3
    NTS    = int(NTS)
    raw    = np.array([complex(s.replace("i", "j")) for s in lines[4:]])

    n_chirps = raw.size // NTS
    time_mat = raw[: n_chirps * NTS].reshape((NTS, n_chirps), order="F")

    rng_fft  = np.fft.fftshift(np.fft.fft(time_mat, axis=0), axes=0)
    rng_half = rng_fft[NTS // 2 :, :]

    b, a     = signal.butter(BUTTER_N, BUTTER_CUT, "high")
    mti      = signal.lfilter(b, a, rng_half, axis=1)

    return fc, Tsweep, mti[1:, :]     # skip leakage bin


def stft_mag(mti, prf):
    """Accumulate |STFT| for range bins 10-30 → spectrogram magnitude + doppler axis."""
    nperseg  = TIME_WINDOW
    noverlap = int(round(nperseg * OVERLAP_FRAC))
    nfft     = PAD_FACTOR * nperseg

    S_accum = None
    for r in range(9, 30):  # bins 10-30  (0-based 9-29)
        _, _, S = signal.spectrogram(
            mti[r, :],
            fs            = prf,
            window        = "hann",
            nperseg       = nperseg,
            noverlap      = noverlap,
            nfft          = nfft,
            mode          = "complex",
            return_onesided=False,
        )
        S = np.fft.fftshift(S, axes=0)
        S_accum = np.abs(S) if S_accum is None else S_accum + np.abs(S)

    doppler = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / prf))
    return S_accum, doppler


def binary_mask(db_img, thresh_db):
    return db_img >= (db_img.max() + thresh_db)


def pseudo_zernike(img, radius=20, degree=4):
    if not _HAS_MAHOTAS:
        return 0.0
    size   = max(img.shape)
    padder = [(0, size - img.shape[0]), (0, size - img.shape[1])]
    img_n  = (np.pad(img, padder) - img.min()) / (img.ptp() + 1e-12)
    return float(np.mean(np.abs(mh.features.zernike_moments(img_n, radius, degree=degree))))


def extract_features(mti, fc, Tsweep):
    prf                = 1.0 / Tsweep
    S, doppler         = stft_mag(mti, prf)
    S2                 = S**2
    flat               = S2.ravel()

    p                  = flat / (flat.sum() + 1e-12)
    mean_entropy       = float(-(p * np.log(p + 1e-12)).sum())
    mean_power         = float(flat.mean())
    variance           = float(flat.var())
    stddev             = float(math.sqrt(variance))

    v_axis             = doppler * 3e8 / (2 * fc)
    vmax               = float(v_axis[np.unravel_index(S.argmax(), S.shape)[0]])
    amp_density        = binary_mask(20 * np.log10(S + 1e-12), DENSITY_THR_DB).mean()
    kurtosis_val       = float(stats.kurtosis(flat, fisher=False))
    z_moment           = pseudo_zernike(S)

    pw_sweep           = S2.sum(axis=0)
    acf                = signal.correlate(pw_sweep, pw_sweep, mode="full")[len(pw_sweep)-1 :]
    periodicity        = float(acf[1:].max() / (acf[0] + 1e-12))

    torso_mask         = np.abs(v_axis) <= TORSO_V_MAX
    mean_torso_power   = float(S2[torso_mask, :].mean())

    pos_power          = S2[v_axis > 0, :].sum()
    neg_power          = S2[v_axis < 0, :].sum()
    pos_neg_ratio      = float(pos_power / (neg_power + 1e-12))

    weights            = S2.sum(axis=1)
    doppler_offset     = float((v_axis * weights).sum() / (weights.sum() + 1e-12))

    row_db             = 20 * np.log10(S2.mean(axis=1) + 1e-12)
    mask               = binary_mask(row_db, DENSITY_THR_DB)
    if mask.any():
        idx            = np.where(mask)[0]
        main_lobe_width = float(v_axis[idx.max()] - v_axis[idx.min()])
    else:
        main_lobe_width = 0.0

    return dict(
        mean_entropy       = mean_entropy,
        mean_power         = mean_power,
        variance           = variance,
        stddev             = stddev,
        max_vel            = vmax,
        amp_density        = amp_density,
        kurtosis           = kurtosis_val,
        zernike_moment     = z_moment,
        periodicity        = periodicity,
        mean_torso_power   = mean_torso_power,
        pos_neg_ratio      = pos_neg_ratio,
        doppler_offset     = doppler_offset,
        main_lobe_width    = main_lobe_width,
    )


def process_one(path: Path):
    """Worker wrapper: returns dict ready for CSV OR raises."""
    fc, Tsweep, mti = read_dat(path)
    feats           = extract_features(mti, fc, Tsweep)
    fid             = path.stem
    feats.update(
        file_id  = fid,
        activity = ACTIVITY_MAP.get(fid[0], "unknown"),
        path     = str(path),
    )
    return feats

# ──────────────────────────────────────────────────────────────
#  MAIN — run workers & stream to CSV
# ──────────────────────────────────────────────────────────────
def main():
    all_files = sorted(DATASETS_ROOT.rglob("*.dat"))
    n_files   = len(all_files)
    if not n_files:
        print(f"No .dat files found under {DATASETS_ROOT.resolve()}")
        return

    # prepare CSV (append if exists, else create with header)
    csv_exists = os.path.exists(CSV_PATH)
    csv_file   = open(CSV_PATH, "a", newline="")
    writer     = csv.DictWriter(csv_file, FIELDNAMES)
    if not csv_exists:
        writer.writeheader()

    max_workers = min(12, cpu_count())    # 12 logical procs on your PC
    print(f"• Processing {n_files} files with {max_workers} workers …")

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_path = {pool.submit(process_one, p): p for p in all_files}

        for i, fut in enumerate(as_completed(future_to_path), 1):
            p = future_to_path[fut]
            try:
                row = fut.result()
                writer.writerow(row)
                csv_file.flush()          # ensure on-disk immediately
                print(f"✓ [{i:>4}/{n_files}] {row['file_id']}")
            except Exception as e:
                print(f"✗ [{i:>4}/{n_files}] {p.name}: {e}")

    csv_file.close()
    print(f"\n✅  Done.  All features appended to  {CSV_PATH}")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
