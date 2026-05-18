# Periodic Error Analysis

A [Siril](https://siril.org/) Python script that measures the **periodic error (PE)** of an equatorial telescope mount from a time series of FITS frames.

The pipeline plate-solves each frame with [ASTAP](https://www.hnsky.org/astap.htm), extracts the RA / DEC drift in arcseconds vs. time, then decomposes the RA error with **Singular Spectrum Analysis** and identifies the dominant periodic components (fundamental + harmonics) via FFT. Output is a set of matplotlib figures (raw RA/DEC, drift fit, singular spectrum, reconstructed signal, per-component statistics, full FFT) and a per-component log in Siril's log panel.

The SSA implementation is based on Alonso-Sanchez (Univ. of Extremadura) and Auger (Nantes Univ.), *"The Sliding Singular Spectrum Analysis: A Data-Driven Nonstationary Signal Decomposition Tool"*, IEEE TSP vol. 66 no. 1, January 2018.

## Requirements

- **Siril ≥ 1.3.6** (provides the Python interpreter and the `sirilpy` module).
- **ASTAP** installed locally, used as the plate-solver CLI.
- All Python dependencies (`numpy`, `pandas`, `matplotlib`, `astropy`, `unidecode`, `ttkthemes`) are auto-installed on first launch into Siril's bundled venv via `sirilpy.ensure_installed`.

## Usage

1. In Siril, **set the working directory** to the folder containing your PE capture FITS frames (folder icon in the main window, or `cd /path/to/fits` in the command bar).
2. Launch the script from **Siril → Scripts** (or via Siril's Python script runner). It cannot be run with `python PE_Analysis.py` directly — it talks to a running Siril instance over IPC.
3. In the GUI:
   - Pick **ASTAP** as the plate-solver and browse to the ASTAP executable if the default path doesn't match. The path is persisted to `config_PE.txt` next to the script.
   - Set the **begin / end frame indices** to constrain the analysis window.
   - Toggle **Run plate solve** off to reuse the WCS results from a previous run (cached in a `PlateSolveAstap/` subfolder of your FITS directory).
   - Click **Process**.

Progress and per-component statistics (RA / DEC drift, mean sampling period, eigencomponent grouping, fundamental period and amplitude, harmonics min/max/RMS, reconstruction RMSE) stream to Siril's log panel. Plots open in separate matplotlib windows.

## Limitations

- **Siril / GAIA plate-solving** is not yet implemented — the radio is present but currently aborts with an explanatory message. Use ASTAP.
- **Long captures freeze the GUI** during ASTAP plate-solving (processing is synchronous). Log messages still flow.
- **macOS AppleDouble files** (`._*.fits`) are filtered out automatically.

## License

Distributed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Authors

- Mickaël HILAIRET — LS2N / École Centrale de Nantes, France
- Gilles MORAIN — France
