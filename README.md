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

## Contributing
When committing:  
1. Create a new branch for your work  
2. Use clear commit messages  
3. Push and open a pull request for review  
