#!/usr/bin/env python3
"""
Download PhysioNet EEG Motor Movement/Imagery (eegmmidb 1.0.0) EDFs,
create sidecar JSONs, and update a manifest CSV — all aligned with the
SynapseX project structure.

Folder layout used:
  recordings/public/physionet_MI/raw/           <- raw EDF files (immutable)
  recordings/public/physionet_MI/sidecar_json/  <- one JSON per EDF (same basename)
  data_catalog/files_manifest.csv               <- index tying raw <-> sidecar

Examples:
  python scripts/download_physionet_eegbci.py --subjects 1 --runs 3 7
  python scripts/download_physionet_eegbci.py --subjects 1 2 --runs 3 7 11
  python scripts/download_physionet_eegbci.py --subjects 1 --runs 3 7 --root recordings/public/physionet_MI

Requirements:
  pip install mne pandas

Notes:
- MNE handles fetching from PhysioNet (no manual scraping needed).
- Some PhysioNet datasets require a (free) account; EEGBCI is public.
- Run codes map to tasks; we annotate sidecars with a simple label.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import mne
import pandas as pd


# ---- EEGBCI run-code -> (task_label, short_desc)
# Source: EEGBCI documentation (approximate, useful for quick labels).
EEGBCI_RUN_MAP: Dict[int, Tuple[str, str]] = {
    1: ("baseline_eyes_open",  "Baseline, eyes open"),
    2: ("baseline_eyes_closed","Baseline, eyes closed"),
    3: ("MI_left_right_hands", "Motor imagery: left vs right hands"),
    4: ("MI_both_feet",        "Motor imagery: both feet"),
    5: ("movement_left_right", "Actual movement: left vs right hands"),
    6: ("movement_both_feet",  "Actual movement: both feet"),
    7: ("MI_left_right_hands", "Motor imagery: left vs right hands (repeat)"),
    8: ("MI_both_feet",        "Motor imagery: both feet (repeat)"),
    9: ("movement_left_right", "Actual movement: left vs right hands (repeat)"),
    10: ("movement_both_feet", "Actual movement: both feet (repeat)"),
    11: ("MI_left_right_hands","Motor imagery: left vs right hands (alt)"),
    12: ("MI_both_feet",       "Motor imagery: both feet (alt)"),
    13: ("movement_left_right","Actual movement: left vs right hands (alt)"),
    14: ("movement_both_feet", "Actual movement: both feet (alt)"),
}


def ensure_dirs(root: Path) -> Tuple[Path, Path, Path]:
    """
    Create project-structure folders if missing.

    Returns:
        raw_dir, sidecar_dir, catalog_dir
    """
    raw_dir = root / "raw"
    sidecar_dir = root / "sidecar_json"
    catalog_dir = Path("data_catalog")

    raw_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    catalog_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, sidecar_dir, catalog_dir


def load_or_init_manifest(catalog_dir: Path) -> pd.DataFrame:
    """
    Load the CSV manifest if it exists; otherwise create an empty one.
    """
    manifest_path = catalog_dir / "files_manifest.csv"
    if manifest_path.exists():
        try:
            return pd.read_csv(manifest_path)
        except Exception:
            # Corrupt or incompatible CSV — start fresh but don’t overwrite file here.
            pass
    return pd.DataFrame(
        columns=[
            "subject_id",
            "run_code",
            "task_label",
            "raw_path",
            "sidecar_path",
            "source",
            "created_utc",
        ]
    )


def save_manifest(df: pd.DataFrame, catalog_dir: Path) -> None:
    """
    De-duplicate and save the manifest to CSV.
    """
    manifest_path = catalog_dir / "files_manifest.csv"
    # Drop exact duplicates on (subject_id, run_code, raw_path) to keep things tidy
    df = df.drop_duplicates(subset=["subject_id", "run_code", "raw_path"], keep="last")
    df.to_csv(manifest_path, index=False)


def sidecar_payload(
    subject_id: str,
    run_code: int,
    raw_path: Path,
    n_channels: Optional[int],
    sfreq: Optional[float],
) -> dict:
    """
    Build a minimal, human-readable JSON payload that mirrors the raw EDF.
    """
    task_label, desc = EEGBCI_RUN_MAP.get(run_code, ("unknown", "Unknown run code"))
    return {
        "source": "physionet_eegbci (eegmmidb/1.0.0)",
        "subject_id": subject_id,
        "run_code": f"{run_code:02d}",
        "task_label": task_label,
        "task_desc": desc,
        "n_channels": n_channels,
        "sfreq_hz": sfreq,
        "device": "EDF (EEGBCI)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "path_raw": str(raw_path).replace("\\", "/"),
    }


def write_json(path: Path, payload: dict) -> None:
    """
    Write a JSON file with pretty indentation.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    """
    CLI arg parsing.
    """
    p = argparse.ArgumentParser(
        description="Download EEGBCI EDFs + sidecars + manifest (SynapseX structure)."
    )
    p.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        required=True,
        help="Subject IDs (e.g., 1 2 3).",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        type=int,
        required=True,
        help="Run codes to fetch (e.g., 3 7 11 for motor imagery).",
    )
    p.add_argument(
        "--root",
        type=str,
        default="recordings/public/physionet_MI",
        help="Root folder for this dataset within the repo.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-copy EDFs and re-write sidecars even if files already exist.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    root = Path(args.root)
    raw_dir, sidecar_dir, catalog_dir = ensure_dirs(root)
    manifest = load_or_init_manifest(catalog_dir)

    created_any = False

    for subj in args.subjects:
        subject_id = f"S{subj:03d}"

        # Download EDFs to local cache; MNE >=1.10 uses 'subjects', older used 'subject'
        try:
            edf_paths = mne.datasets.eegbci.load_data(subjects=[subj], runs=args.runs)
        except TypeError:
            edf_paths = mne.datasets.eegbci.load_data(subject=subj, runs=args.runs)
        except Exception as e:
            print(f"[WARN] Could not download subject {subj}: {e}")
            continue

        # Zip paths to requested run codes so we know which file is which.
        for edf_cached_path, run_code in zip(edf_paths, args.runs):
            basename = f"{subject_id}_run{run_code:02d}"
            dst_edf = raw_dir / f"{basename}.edf"
            dst_sidecar = sidecar_dir / f"{basename}.json"

            # Copy EDF from cache into repo, unless it exists (or overwrite requested).
            if args.overwrite or (not dst_edf.exists()):
                try:
                    shutil.copy2(edf_cached_path, dst_edf)
                    print(f"[OK] Saved {dst_edf}")
                    created_any = True
                except Exception as e:
                    print(f"[ERR] Failed to copy EDF for {subject_id} run {run_code}: {e}")
                    continue
            else:
                print(f"[SKIP] Already exists: {dst_edf}")

            # Read minimal header info for sidecar (safe to skip if it fails).
            n_channels = None
            sfreq = None
            try:
                raw = mne.io.read_raw_edf(dst_edf, preload=False, verbose=False)
                n_channels = raw.info.get("nchan")
                sfreq = float(raw.info.get("sfreq")) if raw.info.get("sfreq") else None
            except Exception as e:
                print(f"[WARN] Could not read header for {dst_edf}: {e}")

            # Write sidecar JSON (overwrite if requested or missing).
            if args.overwrite or (not dst_sidecar.exists()):
                payload = sidecar_payload(subject_id, run_code, dst_edf, n_channels, sfreq)
                try:
                    write_json(dst_sidecar, payload)
                    print(f"[OK] Sidecar {dst_sidecar}")
                    created_any = True
                except Exception as e:
                    print(f"[ERR] Failed to write sidecar for {basename}: {e}")
                    continue
            else:
                print(f"[SKIP] Already exists: {dst_sidecar}")

            # Update in-memory manifest (idempotent; we dedupe on save)
            task_label, _ = EEGBCI_RUN_MAP.get(run_code, ("unknown", ""))
            manifest = pd.concat(
                [
                    manifest,
                    pd.DataFrame.from_records(
                        [
                            {
                                "subject_id": subject_id,
                                "run_code": f"{run_code:02d}",
                                "task_label": task_label,
                                "raw_path": str(dst_edf).replace("\\", "/"),
                                "sidecar_path": str(dst_sidecar).replace("\\", "/"),
                                "source": "physionet_eegbci (eegmmidb/1.0.0)",
                                "created_utc": datetime.now(timezone.utc).isoformat(),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    # Persist manifest CSV
    try:
        save_manifest(manifest, catalog_dir)
        print(f"[OK] Manifest updated: {catalog_dir / 'files_manifest.csv'}")
    except Exception as e:
        print(f"[ERR] Failed to save manifest: {e}")

    if created_any:
        print("\n Done. New/updated EDFs in raw/, sidecars in sidecar_json/, manifest refreshed.")
    else:
        print("\n Nothing new to write; files already present. Manifest still refreshed.")


if __name__ == "__main__":
    main()
