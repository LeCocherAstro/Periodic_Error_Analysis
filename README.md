# Periodic Error Analysis

A [Siril](https://siril.org/) Python script that measures the **periodic error (PE)** of an equatorial telescope mount from a time series of FITS frames.

The pipeline plate-solves each frame with either [ASTAP](https://www.hnsky.org/astap.htm) (via the external CLI) or Siril's built-in GAIA-backed solver, extracts the RA / DEC drift in arcseconds vs. time, then decomposes the RA error with **Singular Spectrum Analysis** and identifies the dominant periodic components (fundamental + harmonics) via FFT. Output is a set of matplotlib figures (raw RA/DEC, drift fit, singular spectrum, reconstructed signal, per-component statistics, full FFT) and a per-component log in Siril's log panel.

The SSA implementation is based on Alonso-Sanchez (Univ. of Extremadura) and Auger (Nantes Univ.), *"The Sliding Singular Spectrum Analysis: A Data-Driven Nonstationary Signal Decomposition Tool"*, IEEE TSP vol. 66 no. 1, January 2018.

## Requirements

- **Siril ≥ 1.3.6** (provides the Python interpreter and the `sirilpy` module).
- **A plate-solver**: either ASTAP installed locally, or Siril's built-in GAIA solver (no extra install — the catalog is fetched on first use).
- All Python dependencies (`numpy`, `pandas`, `matplotlib`, `astropy`, `unidecode`, `ttkthemes`) are auto-installed on first launch into Siril's bundled venv via `sirilpy.ensure_installed`.

## Usage

1. In Siril, **set the working directory** to the folder containing your PE capture FITS frames (folder icon in the main window, or `cd /path/to/fits` in the command bar).
2. Launch the script from **Siril → Scripts** (or via Siril's Python script runner). It cannot be run with `python PE_Analysis.py` directly — it talks to a running Siril instance over IPC.
3. In the GUI:
   - Pick a plate-solver: **ASTAP** (browse to the executable if the default path doesn't match) or **SIRIL (GAIA)** (no path needed — Siril's built-in solver runs in-process). The selected paths are persisted to `config_PE.txt` next to the script.
   - Set the **begin / end frame indices** to constrain the analysis window. The window must contain at least **58 frames** — the Singular Spectrum Analysis stage extracts 10 eigencomponents, which requires the trajectory-matrix dimension `L = n / 5.71` to be ≥ 10. Narrower windows are rejected with a clear error.
   - Toggle **Run plate solve** off to reuse the WCS results from a previous **ASTAP** run (cached in a `PlateSolveAstap/` subfolder of your FITS directory). The Siril/GAIA path does not cache — it always re-solves, so this toggle must stay enabled when SIRIL is selected.
   - Click **Process**.

Progress and per-component statistics (RA / DEC drift, mean sampling period, eigencomponent grouping, fundamental period and amplitude, harmonics min/max/RMS, reconstruction RMSE) stream to Siril's log panel. Plots open in separate matplotlib windows.

## Limitations

- **Siril/GAIA path always re-solves** — cached reuse via the "Run plate solve" toggle is ASTAP-only for now (the Siril loop keeps WCS in memory and doesn't write sidecars).
- **Siril/GAIA blind-solves** without focal-length / pixel-size hints, so it may be slower (or fail more often) than ASTAP on tricky frames. Hints can be added later from the FITS header if needed.
- **Long captures freeze the GUI** during plate-solving (processing is synchronous, on the Tk main thread). Log messages still flow.
- **macOS AppleDouble files** (`._*.fits`) are filtered out automatically.

## License

Distributed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Authors

- Mickaël HILAIRET — LS2N / École Centrale de Nantes, France
- Gilles MORAIN — France
