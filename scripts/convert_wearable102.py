# scripts/convert_wearable102.py
# Zarr-free version: writes HDF5 (chunked+compressed) or NPZ for epochs,
# plus Parquet features and QC summaries.

import json, math, re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.io import loadmat
import h5py
from scipy.signal import welch


# -------------------------------------------
# Paths / layout
ROOT = Path("recordings/public/SSVEP_Wearable102")
RAW = ROOT / "raw" / "mat"
PROC = ROOT / "processed"
H5_DIR  = PROC / "epochs_h5"
NPZ_DIR = PROC / "epochs_npz"
FEAT_DIR = PROC / "features" / "parquet"
QC_DIR   = ROOT / "quality_control" / "summaries"

# Choose one:
USE_HDF5 = True   # chunked, compressed, random access
USE_NPZ  = False  # simple, compressed, loads whole array

# Target index (1..12) -> (freq_Hz, phase_pi)
TARGETS = {
    1:(9.25,0),  2:(11.25,0), 3:(13.25,0),
    4:(9.75,0.5),5:(11.75,0.5),6:(13.75,0.5),
    7:(10.25,1), 8:(12.25,1), 9:(14.25,1),
    10:(10.75,1.5), 11:(12.75,1.5), 12:(14.75,1.5),
}

# -------------------------------------------
# Helpers

def peak_freq(trial, fs, fmin=4.0, fmax=40.0):
    """
    Robust peak frequency (Hz) for one trial [ch x time].
    - Removes DC per channel
    - Uses Welch PSD
    - Searches only in [fmin, fmax] to avoid DC/ultra-low drift
    """
    # trial: [ch, time]
    x = trial - trial.mean(axis=1, keepdims=True)  # remove DC per channel

    # Choose segment length for decent resolution (≈0.5 Hz or better)
    n = x.shape[1]
    nperseg = min(512, n)       # adjust if your epochs are short
    noverlap = nperseg // 2

    # Welch PSD per channel: returns f [F], Pxx [ch x F]
    f, Pxx = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=1,                # frequency along time axis
        return_onesided=True,
        detrend=False,
        scaling="density"
    )

    band = (f >= fmin) & (f <= fmax)
    if not band.any():
        return float("nan")

    f_band = f[band]                 # [Fb]
    P_band = Pxx[:, band]            # [ch x Fb]
    idx = np.argmax(P_band, axis=1)  # [ch]
    # per-channel peak freqs, then median across channels
    return float(np.median(f_band[idx]))

def snr_db(trial, fs, f0, bw=0.5, guard=0.5, fmin=4.0, fmax=40.0):
    """
    Narrowband SNR around f0 using sidebands as noise (Welch PSD).
    - DC removed
    - Band-limited to [fmin, fmax] for stability
    SNR computed per channel and then median combined.
    """
    x = trial - trial.mean(axis=1, keepdims=True)

    n = x.shape[1]
    nperseg = min(512, n)
    noverlap = nperseg // 2

    f, Pxx = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=1,
        return_onesided=True,
        detrend=False,
        scaling="density"
    )

    band = (f >= fmin) & (f <= fmax)
    if not band.any():
        return float("-inf")

    f_b = f[band]          # [Fb]
    P_b = Pxx[:, band]     # [ch x Fb]

    sig = (f_b >= (f0 - bw)) & (f_b <= (f0 + bw))
    # two sidebands as noise
    left  = (f_b >= (f0 - 3*bw - guard)) & (f_b < (f0 - bw - guard))
    right = (f_b >  (f0 + bw + guard))   & (f_b <= (f0 + 3*bw + guard))
    noise = left | right

    # fallbacks in case a mask is empty
    s = P_b[:, sig].mean(axis=1) if sig.any() else P_b.mean(axis=1)
    n = P_b[:, noise].mean(axis=1) if noise.any() else (P_b.mean(axis=1) + 1e-12)

    snr_ch = 10.0 * np.log10((s + 1e-12) / (n + 1e-12))   # [ch]
    return float(np.median(snr_ch))


def write_hdf5(out_path: Path, data, fs, labels, target_hz, target_phase_pi, headband, blocks, impedance_med=None):
    """
    HDF5 writer: data [trial, ch, time]. Chunk by trial; gzip+shuffle.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "data", data=data,
            chunks=(1, data.shape[1], min(data.shape[2], 710)),
            compression="gzip", compression_opts=4, shuffle=True
        )
        f.create_dataset("fs_hz", data=np.array(fs))
        f.create_dataset("labels_target_idx", data=labels.astype("i2"))
        f.create_dataset("target_hz", data=target_hz.astype("f4"))
        f.create_dataset("target_phase_pi", data=target_phase_pi.astype("f4"))
        f.create_dataset("headband_type", data=headband.astype("i1"))
        f.create_dataset("block_index", data=blocks.astype("i1"))
        if impedance_med is not None:
            f.create_dataset("impedance_med", data=impedance_med.astype("f4"))
        f.attrs["note"] = "Wearable102 epochs: (n_trials, n_channels, n_samples); fs=250 Hz; n_samples=710"

def write_npz(out_path: Path, **arrays):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

def flatten_trials(data_5d, fs=250.0):
    """
    data_5d shape: (8, 710, 2, 10, 12) = ch, time, headband, block, target.
    Returns epochs [trial, ch, time] and per-trial metadata arrays.
    """
    if data_5d.shape != (8, 710, 2, 10, 12):
        raise ValueError(f"Expected data shape (8,710,2,10,12), got {data_5d.shape}")
    ch, t, E, B, K = data_5d.shape
    trials, labels, fHz, phi, headband, blocks = [], [], [], [], [], []
    for e in range(E):      # 0=wet, 1=dry
        hb = e + 1
        for b in range(B):  # 1..10
            for k in range(K):  # 1..12
                x = data_5d[:, :, e, b, k]  # [ch, time]
                trials.append(x)
                labels.append(k + 1)
                f, p = TARGETS[k + 1]
                fHz.append(f)
                phi.append(p)
                headband.append(hb)
                blocks.append(b + 1)
    epochs = np.stack(trials, axis=0).astype(np.float32)  # [trial, ch, time]
    return (epochs, np.array(labels), np.array(fHz),
            np.array(phi), np.array(headband), np.array(blocks))

def load_impedance():
    """
    Load Impedance.mat; expected shape [8, 10, 2, 102] (ch, block, headband, subject).
    """
    imp_path = RAW / "Impedance.mat"
    if not imp_path.exists():
        return None
    mat = loadmat(imp_path, squeeze_me=True, struct_as_record=False)
    for key in ("Impedance", "impedance", "imp"):
        if key in mat:
            return np.array(mat[key])
    return None

def process_subject(mat_path: Path, imp_arr):
    """Process S###.mat -> write HDF5/NPZ + features Parquet + QC JSON."""
    m = re.search(r"S(\d{3})\.mat$", mat_path.name, re.IGNORECASE)
    subject_id = f"S{m.group(1)}" if m else "S000"

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "data" not in mat:
        raise KeyError(f"'data' key not found in {mat_path.name}")
    data = np.array(mat["data"])  # (8,710,2,10,12)
    fs = 250.0

    epochs, labels, fHz, phi, hb, blocks = flatten_trials(data, fs=fs)

    # Impedance per trial: median across channels per (block, headband), repeated for 12 targets
    if imp_arr is not None and m:
        si = int(m.group(1)) - 1  # 0-based subject idx
        if imp_arr.ndim != 4 or imp_arr.shape[:3] != (8, 10, 2):
            raise ValueError(f"Unexpected impedance shape {imp_arr.shape}")
        med_by_b_e = np.median(imp_arr[:, :, :, si], axis=0)  # [10, 2]
        imp_trials = []
        for e in range(2):
            for b in range(10):
                imp_trials.extend([med_by_b_e[b, e]] * 12)
        imp_summary = np.array(imp_trials, dtype=np.float32)
    else:
        imp_summary = np.full((epochs.shape[0],), np.nan, dtype=np.float32)

    # Write epochs (HDF5 or NPZ)
    if USE_HDF5:
        out_path = H5_DIR / subject_id / f"{subject_id}.h5"
        write_hdf5(out_path, epochs, fs, labels, fHz, phi, hb, blocks, impedance_med=imp_summary)
        epochs_ref = f"processed/epochs_h5/{subject_id}/{subject_id}.h5"
        wrote_path = out_path
    elif USE_NPZ:
        out_path = NPZ_DIR / f"{subject_id}.npz"
        write_npz(out_path,
                  data=epochs, fs_hz=fs, labels_target_idx=labels,
                  target_hz=fHz, target_phase_pi=phi,
                  headband_type=hb, block_index=blocks,
                  impedance_med=imp_summary)
        epochs_ref = f"processed/epochs_npz/{subject_id}.npz"
        wrote_path = out_path
    else:
        raise RuntimeError("Set either USE_HDF5 or USE_NPZ to True.")

    # Per-trial features -> Parquet (one file per subject)
    rows = []
    peaks, snrs = [], []
    for i in range(epochs.shape[0]):
        trial = epochs[i]  # [ch, time]
        pk = peak_freq(trial, fs)
        s = snr_db(trial, fs, f0=fHz[i])
        peaks.append(pk)
        snrs.append(s)
        rows.append({
            "subject_id": subject_id,
            "trial_idx": int(i),
            "block": int(blocks[i]),
            "headband": int(hb[i]),             # 1=wet, 2=dry
            "target_idx": int(labels[i]),       # 1..12
            "target_hz": float(fHz[i]),
            "target_phase_pi": float(phi[i]),
            "peak_hz": float(pk),
            "snr_db": float(s),
            "impedance_med": float(imp_summary[i]),
            "fs_hz": fs,
            "n_channels": 8,
            "n_samples": 710,
            "epochs_ref": epochs_ref
        })
    table = pa.Table.from_pylist(rows)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, FEAT_DIR / f"{subject_id}.parquet")

    # QC summary JSON
    QC_DIR.mkdir(parents=True, exist_ok=True)
    qc = {
        "subject_id": subject_id,
        "n_trials": int(epochs.shape[0]),
        "median_peak_hz": float(np.median(peaks)),
        "median_snr_db": float(np.median(snrs)),
        "fs_hz": fs,
        "trial_len_s": 710 / 250.0,
        "headband_counts": {
            "wet": int(np.sum(hb == 1)),
            "dry": int(np.sum(hb == 2))
        }
    }
    with open(QC_DIR / f"{subject_id}.json", "w") as f:
        json.dump(qc, f, indent=2)

    print(f" Wrote {subject_id}: {epochs.shape} -> {wrote_path}")

# -------------------------------------------
# Main

if __name__ == "__main__":
    H5_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    imp = load_impedance()

    mats = sorted(RAW.glob("S*.mat"))
    if not mats:
        print(f"No .mat files found in {RAW.resolve()}")
    for mat_path in mats:
        print("Processing", mat_path.name)
        process_subject(mat_path, imp)
