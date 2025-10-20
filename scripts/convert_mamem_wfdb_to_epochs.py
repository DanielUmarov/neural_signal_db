#!/usr/bin/env python3
"""
Convert MAMEM-SSVEP WFDB (.hea/.dat) → ML-ready epochs (HDF5) + features (Parquet) + QC JSON
Directory layout assumed:

recordings/public/SSVEP_MAMEM/
├─ raw/
│  ├─ dataset1/   # WFDB files live here (*.hea/*.dat)
│  ├─ dataset2/
│  └─ dataset3/
├─ processed/
│  ├─ epochs_h5/<SUBJECT>/<SUBJECT>.h5
│  └─ features/parquet/<SUBJECT>.parquet
└─ quality_control/summaries/<SUBJECT>.json

Run:
  py scripts\convert_mamem_wfdb_to_epochs.py --root recordings/public/SSVEP_MAMEM --overwrite
"""

import argparse, json, math, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import wfdb
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import detrend

# ---------------- Configuration ----------------

DATASET_ID = "SSVEP_MAMEM"
DEFAULT_ROOT = Path("recordings/public/SSVEP_MAMEM")

# Try to parse Hz from .hea comment lines first. If missing, fall back to these mappings.
# Adjust if your PDFs specify different frequencies per dataset.
LETTER_TO_HZ_FALLBACK: Dict[str, Dict[str, float]] = {
    # Example mappings (edit as needed)
    "dataset1": {"a": 8.0,  "b": 8.6,  "c": 10.0, "d": 12.0, "e": 15.0},
    "dataset2": {"a": 6.0,  "b": 6.66, "c": 7.5,  "d": 8.57, "e": 10.0, "f": 12.0},
    "dataset3": {"a": 8.0,  "b": 9.0,  "c": 10.0, "d": 11.0, "e": 12.0},
}

# Keep the epochs exactly as provided per WFDB file (each file = one trial)
# Optionally, you can trim/center; here we keep the full record.
CLIP_SECONDS: Optional[Tuple[float, float]] = None   # e.g., (start_sec, end_sec) to slice each record

# ---------------- Utilities ----------------

REC_RE = re.compile(r"^(S\d{3})([A-Za-z])$")  # e.g., S001a → subject=S001, letter=a
HZ_IN_TEXT = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*hz", re.I)

def read_wfdb_record(stem: Path) -> Tuple[np.ndarray, float, List[str]]:
    """
    Read WFDB record without altering original files.
    Returns (data[ch, time], fs_hz, channel_names).
    """
    rec = wfdb.rdrecord(str(stem))
    fs = float(rec.fs)
    # prefer physical signal if available
    if rec.p_signal is not None:
        X = rec.p_signal.T.astype(np.float32)
    else:
        # fallback: digital signal (rare); cast to float
        X = rec.d_signal.T.astype(np.float32)
    ch_names = [s if (s and s.strip()) else f"EEG{i+1}" for i, s in enumerate(rec.sig_name)]
    return X, fs, ch_names

def parse_hz_from_header(hea_path: Path) -> Optional[float]:
    """
    Open .hea and try to find a '... Hz' mention in comment lines.
    Returns float Hz if found, else None.
    """
    try:
        txt = hea_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    # WFDB comments usually on lines starting with '#', but we search the whole thing
    m = HZ_IN_TEXT.search(txt)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def robust_preprocess(x: np.ndarray) -> np.ndarray:
    """
    Light, safe per-channel preprocessing:
    - remove NaNs/Infs
    - detrend
    - median-center
    """
    X = x.astype(np.float32, copy=True)
    X[~np.isfinite(X)] = 0.0
    # detrend along time
    X = detrend(X, axis=1, type="constant")  # remove DC
    # extra robust center
    med = np.median(X, axis=1, keepdims=True)
    X -= med
    return X

def peak_freq(trial: np.ndarray, fs: float, fmax: float = 40.0) -> float:
    """Median FFT peak across channels for one trial [ch x time]."""
    n = trial.shape[1]
    freqs = np.fft.rfftfreq(n, d=1/fs)
    mag = np.abs(np.fft.rfft(trial, axis=1))
    mask = freqs <= fmax
    freqs = freqs[mask]; mag = mag[:, mask]
    idx = np.argmax(mag, axis=1)
    return float(np.median(freqs[idx])) if len(idx) else float("nan")

def snr_db(trial: np.ndarray, fs: float, f0: Optional[float], bw: float = 0.5, guard: float = 0.5) -> float:
    """Narrowband SNR around f0 using flanking bands as noise."""
    if f0 is None or not np.isfinite(f0):
        return float("nan")
    n = trial.shape[1]
    freqs = np.fft.rfftfreq(n, d=1/fs)
    mag = np.abs(np.fft.rfft(trial, axis=1))
    sig = (freqs >= (f0-bw)) & (freqs <= (f0+bw))
    noise = ((freqs >= (f0-3*bw-guard)) & (freqs < (f0-bw-guard))) | \
            ((freqs > (f0+bw+guard)) & (freqs <= (f0+3*bw+guard)))
    s = mag[:, sig].mean() if np.any(sig) else mag.mean()
    n = mag[:, noise].mean() if np.any(noise) else (mag.mean() + 1e-12)
    return float(10 * math.log10((s + 1e-12) / (n + 1e-12)))

def write_hdf5(out_path: Path, data: np.ndarray, fs: float, target_hz: np.ndarray):
    """
    Write epochs to HDF5 with Wearable-like schema.
    data: [epoch, ch, time]
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "data", data=data,
            chunks=(1, data.shape[1], min(data.shape[2], data.shape[2])),
            compression="gzip", compression_opts=4, shuffle=True
        )
        f.create_dataset("fs_hz", data=np.array(fs))
        f.create_dataset("target_hz", data=target_hz.astype("f4"))
        # placeholders for schema compatibility
        n = data.shape[0]
        f.create_dataset("labels_target_idx", data=np.full((n,), np.nan, dtype="f4"))
        f.create_dataset("target_phase_pi",  data=np.full((n,), np.nan, dtype="f4"))
        f.create_dataset("headband_type",    data=np.full((n,), np.nan, dtype="f4"))
        f.create_dataset("block_index",      data=np.full((n,), np.nan, dtype="f4"))
        f.create_dataset("impedance_med",    data=np.full((n,), np.nan, dtype="f4"))
        f.attrs["note"] = "MAMEM SSVEP epochs: (n_epochs, n_channels, n_samples)"

def convert_subject(subject_id: str,
                    recs: List[Tuple[str, Path, Path, str]],
                    out_root: Path,
                    dataset_name: str,
                    overwrite: bool = False):
    """
    recs: list of (letter, hea_path, dat_path, stem_str) for this subject.
    """
    # Sort by letter to keep a,b,c,... order (and numeric if multiple runs)
    recs = sorted(recs, key=lambda r: r[0])

    # Load all trials
    epochs_list, hz_list = [], []
    fs_list, ch_name_ref = [], None

    for letter, hea, dat, stem_str in recs:
        stem = Path(stem_str)
        X, fs, ch_names = read_wfdb_record(stem)

        # optional trimming
        if CLIP_SECONDS is not None:
            s0 = int(max(0, CLIP_SECONDS[0] * fs))
            s1 = int(min(X.shape[1], CLIP_SECONDS[1] * fs))
            X = X[:, s0:s1]

        X = robust_preprocess(X)

        # determine Hz: try header; then fallback mapping
        hz = parse_hz_from_header(hea)
        if hz is None:
            hz = LETTER_TO_HZ_FALLBACK.get(dataset_name, {}).get(letter.lower())
        # if still None, keep NaN (some unlabeled windows)
        hz = float(hz) if (hz is not None) else float("nan")

        epochs_list.append(X.astype(np.float32))
        hz_list.append(hz)
        fs_list.append(fs)
        if ch_name_ref is None:
            ch_name_ref = ch_names

    if not epochs_list:
        print(f"[warn] {subject_id}: no trials found; skipping.")
        return

    # Ensure consistent fs across all trials
    if not np.allclose(fs_list, fs_list[0]):
        print(f"[warn] {subject_id}: mixed sampling rates {set(fs_list)}; using first.")
    fs = float(fs_list[0])

    # Stack: [epoch, ch, time]
    # Note: lengths can differ; we pad/truncate to min length for a clean tensor
    min_len = min(e.shape[1] for e in epochs_list)
    epochs = np.stack([e[:, :min_len] for e in epochs_list], axis=0)
    target_hz = np.array(hz_list, dtype=np.float32)

    # Output paths
    h5_path = out_root / "processed" / "epochs_h5" / subject_id / f"{subject_id}.h5"
    pq_path = out_root / "processed" / "features" / "parquet" / f"{subject_id}.parquet"
    qc_path = out_root / "quality_control" / "summaries" / f"{subject_id}.json"

    if h5_path.exists() and pq_path.exists() and qc_path.exists() and not overwrite:
        print(f"[skip] {subject_id} (exists; use --overwrite)")
        return

    # Write HDF5
    write_hdf5(h5_path, epochs, fs, target_hz)
    epochs_ref = f"processed/epochs_h5/{subject_id}/{subject_id}.h5"

    # Features/parquet
    rows = []
    peaks, snrs = [], []
    for i in range(epochs.shape[0]):
        trial = epochs[i]
        pk = peak_freq(trial, fs)
        s = snr_db(trial, fs, f0=target_hz[i] if np.isfinite(target_hz[i]) else None)
        peaks.append(pk); snrs.append(s)
        rows.append({
            "dataset_id": DATASET_ID,
            "subject_id": subject_id,
            "trial_idx": int(i),
            "block": None,
            "headband": None,
            "target_idx": None,
            "target_hz": float(target_hz[i]) if np.isfinite(target_hz[i]) else None,
            "target_phase_pi": None,
            "peak_hz": float(pk) if np.isfinite(pk) else None,
            "snr_db": float(s) if np.isfinite(s) else None,
            "impedance_med": None,
            "fs_hz": fs,
            "n_channels": epochs.shape[1],
            "n_samples": epochs.shape[2],
            "epochs_ref": epochs_ref
        })
    table = pa.Table.from_pylist(rows)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, pq_path)

    # QC JSON
    qc = {
        "dataset_id": DATASET_ID,
        "subject_id": subject_id,
        "n_trials": int(epochs.shape[0]),
        "median_peak_hz": float(np.nanmedian(peaks)),
        "median_snr_db": float(np.nanmedian(snrs)),
        "fs_hz": fs,
        "trial_len_s": epochs.shape[2] / fs,
    }
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"[ok] {subject_id}: {epochs.shape} -> {h5_path}")

def scan_raw(root: Path) -> Dict[str, List[Tuple[str, Path, Path, str]]]:
    """
    Scan raw/<dataset>/ for WFDB pairs. Returns mapping subject_id -> list of (letter, hea, dat, stem_str).
    """
    raw_dir = root / "raw"
    subjects: Dict[str, List[Tuple[str, Path, Path, str]]] = defaultdict(list)

    if not raw_dir.exists():
        raise SystemExit(f"raw directory not found: {raw_dir}")

    for dataset_dir in sorted(raw_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name  # dataset1|dataset2|dataset3
        # find all .hea files and ensure .dat exists
        for hea_path in sorted(dataset_dir.glob("*.hea")):
            stem = hea_path.with_suffix("")  # remove .hea
            dat_path = stem.with_suffix(".dat")
            if not dat_path.exists():
                # Some records may use .dat16 or similar; attempt a generic match
                alt = list(dataset_dir.glob(stem.name + ".dat*"))
                if alt:
                    dat_path = alt[0]
                else:
                    print(f"[warn] Missing .dat for {hea_path.name}")
                    continue

            # Expect names like S001a
            m = REC_RE.match(stem.name)
            if not m:
                # Non-standard name; treat as its own subject file (one epoch)
                subj = stem.stem[:4] if len(stem.stem) >= 4 and stem.stem[0].upper() == "S" else stem.stem
                letter = "a"
                subjects[subj].append((letter, hea_path, dat_path, str(stem)))
                continue

            subj = m.group(1)      # S001
            letter = m.group(2)    # a
            # Store dataset name in letter (upper safe), to allow dataset-specific mapping on convert
            subjects[subj].append((letter.lower(), hea_path, dat_path, str(stem)))

    return subjects

def ensure_dirs(out_root: Path):
    (out_root / "processed" / "epochs_h5").mkdir(parents=True, exist_ok=True)
    (out_root / "processed" / "features" / "parquet").mkdir(parents=True, exist_ok=True)
    (out_root / "quality_control" / "summaries").mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Convert MAMEM WFDB to HDF5 epochs + Parquet + QC JSON")
    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="SSVEP_MAMEM root folder")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite outputs if they exist")
    args = ap.parse_args()

    out_root = Path(args.root)
    ensure_dirs(out_root)

    subj_map = scan_raw(out_root)
    if not subj_map:
        print(f"[warn] No WFDB records found under {out_root/'raw'}")
        return

    # Convert per subject
    for subject_id, recs in sorted(subj_map.items()):
        # Determine dataset name context from path of first entry
        dataset_name = Path(recs[0][1]).parent.name  # parent dir of .hea
        # Patch fallback mapping if dataset unknown
        if dataset_name not in LETTER_TO_HZ_FALLBACK:
            LETTER_TO_HZ_FALLBACK[dataset_name] = {}

        convert_subject(subject_id, recs, out_root, dataset_name, overwrite=args.overwrite)

    print("Done.")

if __name__ == "__main__":
    main()
