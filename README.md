# SQAT4PY
Python translation of SQAT (Sound QUality Analysis Tool) for psychoacoustic and aircraft-noise metric evaluation.

## Main Entry Points
- `main_UI.py`: primary PySide6 desktop UI for interactive analysis
- `examples.py`: runnable examples that use the bundled reference WAV files
- `validation/FINAL_VAL.py`: Python-side validation runner
- `validation/run_sqat_matlab_full.m`: MATLAB-side validation runner that generates the reference JSON

## Metric Modules
- `Loudness (ISO 532-1)`: `metrics_loudness.py`
- `Loudness (ECMA 418-2)`: `metrics_ecma.py` via `metrics_shm_loudness_ecma_fast.py`
- `Sharpness (DIN 45692)`: `metrics_sharpness.py`
- `Roughness (Daniel 1997)`: `metrics_roughness.py`
- `Roughness (ECMA 418-2)`: `metrics_ecma.py` via `metrics_shm_roughness_ecma.py`
- `Fluctuation Strength (Osses 2016)`: `metrics_fluctuation.py`
- `Tonality (Aures 1985)`: `metrics_tonality.py`
- `Tonality (ECMA 418-2)`: `metrics_ecma.py` via `metrics_shm_tonality_ecma.py`
- `Annoyance (Di 2016)`: `metrics_annoyance.py`
- `Annoyance (Zwicker 1999)`: `metrics_annoyance.py`
- `Annoyance (More 2010)`: `metrics_annoyance.py`
- `EPNL (FAR Part 36)`: `metrics_epnl.py`

## Quick Start
```bash
pip install -r requirements.txt
python main_UI.py
```

## Validation Flow
1. In MATLAB, `cd original_matlab` and run `startup_SQAT`.
2. Back at the repo root, run `validation/run_sqat_matlab_full.m` to generate `validation/matlab_results/results_matlab.json`.
3. Run `python validation/FINAL_VAL.py` to compute the Python results and compare them against MATLAB.

## Repository Layout
- `original_matlab/`: original SQAT MATLAB reference implementation
- `sound_files/reference_signals/`: bundled WAV reference signals
- `validation/`: cross-check scripts plus generated reports/results
- `docs/`: Sphinx documentation
- `logos/`: shared branding assets
