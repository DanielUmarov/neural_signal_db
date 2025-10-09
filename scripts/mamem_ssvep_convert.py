#!/usr/bin/env python3
"""
Convert local MAMEM-SSVEP WFDB (.hea/.dat) to EDF + sidecar JSON
without touching your raw files.

Examples (PowerShell):
  py .\scripts\mamem_ssvep_convert.py `
    --dataset dataset1 `
    --src-dir "C:/data/mssvepdb/dataset1" `
    --root "recordings/public/SSVEP_MAMEM/edf" `
    --overwrite
"""

import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import wfdb
from mne.export import export_raw

mne.set_log_level("ERROR")

PHYSIONET_BASE = "https://physionet.org/files/mssvepdb/1.0.0/"

# ---------------- EDF safety helpers ----------------

def _edf_safe_label(name: str) -> str:
    s = name.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]", "", s)
    return (s or "EEG")[:16]

def sanitize_channel_labels(raw: mne.io.Raw) -> Dict[str, str]:
    orig = [ch["ch_name"] for ch in raw.info["chs"]]
    used, mapping = set(), {}
    for nm in orig:
        base = _edf_safe_label(nm)
        cand = base
        k = 1
        while cand in used:
            tail = f"_{k}"
            cand = (base[: 16 - len(tail)]) + tail
            k += 1
        used.add(cand)
        mapping[nm] = cand
    raw.rename_channels(mapping)
    return mapping

def make_edf_friendly(raw_in: mne.io.Raw,
                      target_peak_uv: float = 500.0,
                      clip_uv: float = 1000.0) -> Tuple[mne.io.Raw, float]:
    """Detrend, robust-scale to EEG range, convert to Volts, clip; return (raw, clip_v)."""
    raw = raw_in.copy().load_data()
    X = raw.get_data().astype(float)

    # finite + DC removal
    X[~np.isfinite(X)] = 0.0
    X -= np.nanmedian(X, axis=1, keepdims=True)

    # robust per-channel scale (99.9th percentile -> target)
    perc = np.nanpercentile(np.abs(X), 99.9, axis=1)
    perc[perc <= 1e-15] = 1e-15
    X = X / perc[:, None]

    # scale to Volts + clip
    target_peak_v = float(target_peak_uv) * 1e-6
    clip_v = float(clip_uv) * 1e-6
    X = np.clip(X * target_peak_v, -clip_v, clip_v)

    info = mne.create_info(
        ch_names=[ch["ch_name"] for ch in raw.info["chs"]],
        sfreq=float(raw.info["sfreq"]),
        ch_types="eeg",
    )
    out = mne.io.RawArray(X, info, verbose=False)
    out.set_meas_date(None)
    return out, clip_v

# ---------------- IO helpers ----------------

def wfdb_to_raw(src_dir: Path, stem: str) -> mne.io.Raw:
    rec = wfdb.rdrecord(str(src_dir / stem))
    ch_names = [c or f"EEG{i+1}" for i, c in enumerate(rec.sig_name)]
    sfreq = float(rec.fs)
    # Prefer physical signal if present; fall back to adc if needed
    data = (rec.p_signal if rec.p_signal is not None else rec.d_signal).T.astype(float)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_meas_date(None)
    return raw

def ensure_dirs(root: Path, dataset: str) -> Tuple[Path, Path, Path]:
    edf_dir = root / dataset
    sidecar_dir = root.parent / "sidecar_json" / dataset
    catalog_dir = Path("data_catalog")
    edf_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    return edf_dir, sidecar_dir, catalog_dir

def load_or_init_manifest(path: Path) -> pd.DataFrame:
    cols = ["dataset","record_name","edf_path","json_path","source","created_utc"]
    if path.exists():
        try:
            df = pd.read_csv(path)
            for c in cols:
                if c not in df.columns: df[c] = pd.NA
            return df[cols]
        except Exception:
            pass
    return pd.DataFrame(columns=cols)

# ---------------- sidecar ----------------

def build_sidecar(dataset: str, rec: str, raw: mne.io.Raw,
                  source_url: str, ch_map: Dict[str,str]) -> Dict:
    return {
        "schema_version": "1.0.0",
        "dataset": f"MAMEM-SSVEP/{dataset}",
        "record_name": rec,
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "n_channels": int(raw.info["nchan"]),
        "channels": [ch["ch_name"] for ch in raw.info["chs"]],
        "channel_name_map": ch_map,  # original -> EDF-safe
        "provenance": {"source_url": source_url},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

# ---------------- main ----------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert MAMEM SSVEP WFDB to EDF + JSON (non-destructive).")
    ap.add_argument("--dataset", required=True, choices=["dataset1","dataset2","dataset3"])
    ap.add_argument("--src-dir", required=True, help="Folder with .hea/.dat files (downloaded already)")
    ap.add_argument("--root", required=True, help="Destination root for EDF (e.g., recordings/public/SSVEP_MAMEM/edf)")
    ap.add_argument("--limit", type=int, default=None, help="Process first N records only")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite EDF/JSON if they exist")
    return ap.parse_args()

def main():
    args = parse_args()
    src = Path(args.src_dir)
    if not src.exists(): raise SystemExit(f"--src-dir not found: {src}")

    edf_dir, sidecar_dir, catalog_dir = ensure_dirs(Path(args.root), args.dataset)
    manifest_path = catalog_dir / "files_manifest.csv"
    manifest = load_or_init_manifest(manifest_path)

    stems = sorted({p.stem for p in src.glob("*.hea")})
    if args.limit: stems = stems[: args.limit]
    print(f"[plan] {len(stems)} record(s) to process from {args.dataset}")

    for i, rec in enumerate(stems, 1):
        hea = src / f"{rec}.hea"
        dat = src / f"{rec}.dat"
        if not (hea.exists() and dat.exists()):
            print(f"[warn] missing .hea or .dat for {rec}")
            continue

        edf_out = edf_dir / f"{rec}.edf"
        json_out = sidecar_dir / f"{rec}.json"
        if edf_out.exists() and json_out.exists() and not args.overwrite:
            print(f"[skip] {rec} (exists; use --overwrite)")
            continue

        print(f"[{i}/{len(stems)}] converting {rec}")
        try:
            raw = wfdb_to_raw(src, rec)
        except Exception as e:
            print(f"[warn] WFDB read failed for {rec}: {e}")
            continue

        # scale to Volts, clip, sanitize labels
        raw_ef, clip_v = make_edf_friendly(raw)
        ch_map = sanitize_channel_labels(raw_ef)

        # explicit, stable physical range for EDF (symmetric across channels)
        phys_range = (-clip_v, clip_v)

        # write EDF
        try:
            export_raw(str(edf_out), raw_ef, fmt="edf", physical_range=phys_range, overwrite=True)
            print(f"[ok] EDF: {edf_out.name}")
        except Exception as e:
            print(f"[warn] EDF export failed for {rec}: {e}")
            continue

        # sidecar
        source_url = f"{PHYSIONET_BASE}{args.dataset}/{rec}.dat"
        side = build_sidecar(args.dataset, rec, raw_ef, source_url, ch_map)
        json_out.write_text(json.dumps(side, indent=2), encoding="utf-8")
        print(f"[ok] JSON: {json_out.name}")

        # manifest row
        manifest = pd.concat(
            [
                manifest,
                pd.DataFrame.from_records(
                    [{
                        "dataset": args.dataset,
                        "record_name": rec,
                        "edf_path": str(edf_out).replace("\\","/"),
                        "json_path": str(json_out).replace("\\","/"),
                        "source": source_url,
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                    }]
                ),
            ],
            ignore_index=True,
        )

    manifest.drop_duplicates(subset=["dataset","record_name"], keep="last", inplace=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"[ok] Manifest updated: {manifest_path}")
    print("Done.")

if __name__ == "__main__":
    main()
