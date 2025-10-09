# Neural Signal Processing & Database

## Overview
This repository contains the data pipeline for **Team 2: Neural Signal Processing & Database**.  
Our mission is to take **raw EEG signals** recorded with the Ultracortex Mark IV + Cyton 8-channel board and turn them into **clean, labeled, and organized datasets** that downstream teams can use for modeling and demo purposes.

The pipeline follows a **load → clean → feature → save** workflow:
1. Record EEG sessions following Team 1’s protocol.
2. Clean and preprocess signals (filters, artifact removal).
3. Extract features (band power, FFT peaks, event-related potentials).
4. Save outputs in structured folders with metadata.

---

## Folder Structure
recordings/ # raw EEG recordings (.csv)
processed/ # cleaned EEG data after filtering and artifact removal
features/ # extracted features (FFT, band power, ERP averages)
qc/ # quality control plots and sanity checks
metadata/ # sidecar JSON files with subject info, timing, labels
docs/ # protocols, notes, and READMEs
scripts/ # Python scripts for preprocessing, cleaning, and feature extraction

## Workflow
- **Data Acquisition** → headset recordings stored in `recordings/`  
- **Preprocessing & Cleaning** → filtered data saved in `processed/`  
- **Feature Extraction** → features saved in `features/`  
- **Quality Control (QC)** → quick plots and checks in `qc/`  
- **Metadata** → JSON sidecars documenting each session  

## Quality Control
Always generate QC plots (before/after filtering, FFT, etc.) to confirm:  
- Eye blinks and line noise are removed  
- Signals are within expected frequency bands  
- SSVEP peaks, P300 responses, or MI rhythms are visible  


## PhysioNet MI Scraper (not necessary anymore)

The **Motor Imagery (MI)** dataset from PhysioNet provides clean, standardized EEG recordings for left/right hand and foot movement imagination tasks.  
This data serves as a benchmark for testing signal processing and classification pipelines before applying them to our own recordings.



## SSVEP Workflow (MAMEM Dataset)

The MAMEM SSVEP dataset provides large-scale EEG recordings of steady-state visual evoked potentials — brain responses to periodic visual flickers at known frequencies.
It’s used to validate our signal-cleaning and frequency-tagging pipeline and benchmark our Ultracortex SSVEP protocols.

### Dataset Overview

Source: PhysioNet MAMEM SSVEP

#### Experiments:
```
dataset1 → 250 Hz sampling rate, 256 channels

dataset2 → 250 Hz sampling rate, 256 channels

dataset3 → 128 Hz sampling rate, 256 channels

Each subject/session includes:

.dat → raw EEG waveform

.hea → header with channel info and sampling rate

.flash → individual flash events

.win → 5-second window annotations with stimulus frequency labels
```

#### Step 1: Download Raw WFDB Datasets
``` aws s3 sync --no-sign-request s3://physionet-open/mssvepdb/1.0.0/dataset1/ recordings/public/SSVEP_MAMEM/raw/dataset1
aws s3 sync --no-sign-request s3://physionet-open/mssvepdb/1.0.0/dataset2/ recordings/public/SSVEP_MAMEM/raw/dataset2
aws s3 sync --no-sign-request s3://physionet-open/mssvepdb/1.0.0/dataset3/ recordings/public/SSVEP_MAMEM/raw/dataset3
```

Resulting Folder structure:
```
recordings/public/SSVEP_MAMEM/raw/
├── dataset1/
├── dataset2/
└── dataset3/
```

#### Step 2: Convert WFDB → EDF

py scripts/mamem_ssvep_convert.py `
  --dataset dataset1 `
  --src-dir "recordings/public/SSVEP_MAMEM/raw/dataset1" `
  --root "recordings/public/SSVEP_MAMEM/edf" `
  --overwrite


Repeat for dataset2 and dataset3.

##### Conversion process:
The mamem_ssvep_convert file converts .dat/.hea pairs to .edf using wfdb + mne + edfio.

Automatically fixes duplicate or missing channel names.

Generates JSON sidecars with:

- Sampling rate and channel count

- Subject/session IDs

- Stimulus frequency and duration

- Timestamp and file references




## Contributing
When committing:  
1. Create a new branch for your work  
2. Use clear commit messages  
3. Push and open a pull request for review  
