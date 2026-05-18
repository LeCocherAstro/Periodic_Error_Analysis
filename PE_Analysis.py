"""
Automatic Periodic Error Computation - Siril Python script.

GUI-based Siril script that computes the periodic error of an equatorial
mount from a time-series of FITS frames. The script collects FITS files
from Siril's working directory, lets the user pick the plate-solving
engine (ASTAP or Siril/GAIA) and the frame index range, then runs the
analysis.

The PE algorithm is the same one prototyped in PEC_Analysis_v0p2.py:
  1. plate-solve each frame with ASTAP (CLI)
  2. parse the resulting .wcs files into a DataFrame indexed by DATE-OBS
  3. plot RA / DEC drift vs time and a linear drift fit
  4. decompose the RA error signal with Singular Spectrum Analysis (SSA),
     group eigencomponents into fundamental + harmonics, and characterise
     each via FFT (amplitude, frequency, period)
  5. reconstruct the signal, plot the components and the reconstruction RMSE

Follows the official Siril Python scripting template:
https://siril.readthedocs.io/en/stable/scripts/python_gui_template
"""

# Core module imports
import csv
import math
import shutil
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import sirilpy as s
from sirilpy import tksiril, LogColor, SirilError, SirilConnectionError

# Ensure non-core modules are available in Siril's venv before importing them
s.ensure_installed("ttkthemes", "numpy", "pandas", "matplotlib", "astropy", "unidecode")

from ttkthemes import ThemedTk        # noqa: E402  (must follow ensure_installed)
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
import matplotlib                     # noqa: E402
import matplotlib.pyplot as plt       # noqa: E402
from numpy import linalg as LA        # noqa: E402
from astropy.io import fits as astrofits  # noqa: E402
from astropy.time import Time         # noqa: E402
from unidecode import unidecode       # noqa: E402


# =============================================================================
# Constants
# =============================================================================
APP_NAME       = "Automatic Periodic Error Computation"
AUTHORS        = "Mickaël HILAIRET and Gilles MORAIN"
VERSION        = "0.5.0"
REQUIRED_SIRIL = "1.3.6"

DEFAULT_ASTAP_CLI = "/Applications/ASTAP.app/Contents/MacOS/astap"
DEFAULT_SIRIL_CLI = "/Applications/Siril.app/Contents/MacOS/siril-cli"

SCRIPT_DIR  = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config_PE.txt"

# SSA / FFT tuning (kept identical to v0p2 to preserve published results)
SSA_NB_EIGEN        = 10        # Number of SSA eigencomponents to compute
SSA_GROUP_ORDER     = 5         # Fundamental + 5 harmonics
SSA_PAIR_THRESHOLD  = 0.30      # Eigenvalue ratio under which two adjacent
                                # components are merged as a sinusoid pair
SSA_WINDOW_RATIO    = 5.71      # L = N / SSA_WINDOW_RATIO  (cf. v0p2)
# Minimum number of plate-solved frames the SSA stage needs to extract
# SSA_NB_EIGEN eigencomponents (L must be >= SSA_NB_EIGEN).
SSA_MIN_FRAMES      = int(SSA_WINDOW_RATIO * SSA_NB_EIGEN) + 1
FFT_N_POINTS        = 64 * 1024
PLOT_FFT_PER_COMP   = True      # Plot FFT for every sinusoidal component

WCS_SUBDIR_NAME     = "PlateSolveAstap"


# =============================================================================
# Config file helpers
# =============================================================================
def load_config():
    """Read ASTAP and Siril CLI paths from config_PE.txt.

    If the file doesn't exist yet, write defaults so the user can later
    edit it by hand instead of having to click Browse.
    """
    paths = {"astap": DEFAULT_ASTAP_CLI, "siril": DEFAULT_SIRIL_CLI}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, newline="") as fp:
            for row in csv.reader(fp):
                if len(row) == 2 and row[0] in paths:
                    paths[row[0]] = row[1]
    else:
        save_config(paths["astap"], paths["siril"])
    return paths


def save_config(astap_path, siril_path):
    """Persist the ASTAP and Siril CLI paths to config_PE.txt."""
    with open(CONFIG_FILE, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["astap", astap_path])
        writer.writerow(["siril", siril_path])


# =============================================================================
# Plate-solving
# =============================================================================
def _plate_solve_astap(fits_paths, astap_cli, wcs_dir, log):
    """Run ASTAP CLI on each FITS file, writing .wcs results into wcs_dir.

    The directory is wiped first so the analysis stage sees exactly the
    frames that were selected in the GUI.
    """
    if not Path(astap_cli).is_file():
        raise FileNotFoundError(f"ASTAP executable not found: {astap_cli}")

    if wcs_dir.exists():
        log(f"[PE] Clearing previous results directory {wcs_dir}")
        # macOS exFAT/SMB volumes auto-remove `._*` AppleDouble companions
        # when their sibling file is deleted, so rmtree can race against
        # itself and hit ENOENT. Swallow only that case.
        def _ignore_missing(func, path, exc_info):
            if not isinstance(exc_info[1], FileNotFoundError):
                raise exc_info[1]
        shutil.rmtree(wcs_dir, onerror=_ignore_missing)
    wcs_dir.mkdir(parents=True, exist_ok=True)

    n = len(fits_paths)
    log(f"[PE] Running ASTAP on {n} frames", color=LogColor.BLUE)
    for i, fits_file in enumerate(fits_paths, 1):
        output_file = wcs_dir / (fits_file.stem + ".wcs")
        # `-update <input>` makes ASTAP update the original FITS header in place;
        # `-o <output>` controls where the .wcs sidecar is written.
        cmd = [
            str(astap_cli),
            "-f", str(fits_file),
            "-o", str(output_file),
            "-update", str(fits_file),
        ]
        log(f"[PE]   [{i}/{n}] Solving {fits_file.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # ASTAP error codes: 1=no solution, 2=not enough stars, 16=read err,
            # 32=no star db, 33=star db read err, 34=update err
            log(
                f"[PE]     ASTAP returned {result.returncode} for {fits_file.name}",
                color=LogColor.RED,
            )

    log("[PE] ASTAP plate-solving complete.", color=LogColor.GREEN)


def _extract_platesolve_hints(fits_path, log):
    """Read FOCALLEN (mm) and XPIXSZ (μm) from a FITS header to seed Siril's
    blind solve. Returns a list of cmd args; empty if neither is available.

    These are hints, not constraints — Siril still solves astrometrically
    but converges much faster (and with fewer false negatives) when it has
    a reasonable starting scale.
    """
    try:
        header = astrofits.getheader(str(fits_path))
    except Exception as exc:
        log(f"[PE] Could not read header from {fits_path.name}: {exc} "
            f"— blind solving", color=LogColor.BLUE)
        return []

    hints = []
    focal = header.get("FOCALLEN")
    if focal:
        hints.append(f"-focal={float(focal):g}")
    # Use XPIXSZ * XBINNING — Siril wants the effective pixel size on chip.
    xpixsz = header.get("XPIXSZ")
    xbin = header.get("XBINNING", 1) or 1
    if xpixsz:
        hints.append(f"-pixelsize={float(xpixsz) * int(xbin):g}")

    if hints:
        log(f"[PE] Platesolve hints from {fits_path.name}: {' '.join(hints)}",
            color=LogColor.BLUE)
    else:
        log(f"[PE] No FOCALLEN/XPIXSZ in {fits_path.name} — blind solving",
            color=LogColor.BLUE)
    return hints


def _plate_solve_siril(fits_paths, siril, log):
    """Solve each FITS via Siril's built-in GAIA solver, return a DataFrame.

    Per-frame loop — slower than seqplatesolve but simpler to reason about
    (matches the ASTAP path's per-frame progress / failure logging). The
    WCS solution lives only in Siril's in-memory image buffer; nothing is
    written to disk, so there is no cache to reuse on subsequent runs.
    """
    records = []
    n = len(fits_paths)
    log(f"[PE] Running Siril/GAIA plate-solver on {n} frames",
        color=LogColor.BLUE)

    # All selected frames come from the same capture session, so the focal
    # length and pixel size are constant — read once from the first frame.
    hints = _extract_platesolve_hints(fits_paths[0], log)

    # Siril's text command parser splits arguments on whitespace and mangles
    # extensions on long absolute paths (e.g. ".fits" gets stripped to ".f"),
    # so cmd("load", abs_path) breaks on paths with spaces or many dots.
    # Fall back to bare filenames — Siril's working directory is already the
    # FITS folder (the script discovered the .fits files there at startup).
    for i, fits_path in enumerate(fits_paths, 1):
        log(f"[PE]   [{i}/{n}] Solving {fits_path.name}")
        try:
            siril.cmd("load", fits_path.name)
            siril.cmd("platesolve", *hints)
            header = siril.get_image_fits_header(return_as="dict")
            records.append(header)
        except s.CommandError as exc:
            log(f"[PE]     platesolve failed for {fits_path.name}: {exc}",
                color=LogColor.RED)
        except SirilError as exc:
            log(f"[PE]     Siril error for {fits_path.name}: {exc}",
                color=LogColor.RED)

    if not records:
        raise RuntimeError(
            "Siril platesolve produced no results — every frame failed. "
            "Check the per-frame error messages in the log above."
        )

    log(f"[PE] Siril plate-solving complete ({len(records)}/{n} solved).",
        color=LogColor.GREEN)
    return _finalize_sequence_df(pd.DataFrame.from_records(records))


# =============================================================================
# WCS parsing
# =============================================================================
# Header keys ASTAP writes as floats / ints — parse them as such for arithmetic.
_FLOAT_KEYS = {
    "EXPOSURE", "EXPTIME", "EGAIN", "XPIXSZ", "YPIXSZ", "CCD-TEMP",
    "FOCALLEN", "FOCRATIO", "RA", "DEC", "CENTALT", "CENTAZ",
    "AIRMASS", "SITEELEV", "SITELAT", "SITELONG",
    "CRVAL1", "CRVAL2",
    "HFD", "STARS", "OBJCTROT", "CLOUDCVR", "HUMIDITY", "PRESSURE",
    "AMBTEMP", "WINDDIR", "WINDSPD",
    "EQUINOX", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2",
    "CROTA1", "CROTA2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
}
_INT_KEYS = {
    "BITPIX", "NAXIS", "BZERO", "XBINNING", "YBINNING", "GAIN", "OFFSET",
    "USBLIMIT", "FOCPOS", "FOCUSPOS", "XBAYROFF", "YBAYROFF",
}


def _parse_arcsec_offset(token):
    """Parse an ASTAP mount-offset token like '12.3"' or '0.5\\''."""
    token = token.strip()
    if token.endswith("'"):
        return float(token[:-1]) * 60.0    # arcmin → arcsec
    if token.endswith('"'):
        return float(token[:-1])
    return None


def _wcs_read_as_dict(wcs_file):
    """Parse one .wcs file into a dict (ported from v0p2 with cleanups)."""
    d = {}
    comment_count = 0
    with open(wcs_file, encoding="utf8", errors="ignore") as fp:
        for line in fp:
            if line.startswith("COMMENT 7"):
                # COMMENT 7 holds the ASTAP mount offset for RA and DEC
                parts = line.split(",")
                mount_offset_ra = None
                mount_offset_dec = None
                try:
                    mount_offset_ra = _parse_arcsec_offset(
                        parts[0].split("=")[1]
                    )
                except (IndexError, ValueError):
                    pass
                try:
                    mount_offset_dec = _parse_arcsec_offset(
                        parts[1].split("=")[1]
                    )
                except (IndexError, ValueError):
                    pass
                d["OFFSETRA"] = mount_offset_ra
                d["OFFSETDE"] = mount_offset_dec
            elif line.startswith("COMMENT"):
                d[f"COMMENT{comment_count}"] = unidecode(
                    line.split("COMMENT ", 1)[1].strip()
                )
                comment_count += 1
            else:
                try:
                    key = line.split("=")[0].strip().replace("'", "")
                    val = (
                        line.split("=")[1].split("/")[0].strip().replace("'", "")
                    )
                    if key in _FLOAT_KEYS:
                        val = float(val)
                    elif key in _INT_KEYS:
                        val = int(val)
                    d[key] = val
                except (IndexError, ValueError):
                    d["OTHER"] = line.strip()
    return d


def _finalize_sequence_df(df):
    """Index by DATE-OBS, sort, and add FRAME_NUM / TIME_DIFF / TIME_REL columns.

    Shared by both the ASTAP (load-from-.wcs-files) and Siril/GAIA
    (read-from-in-memory-headers) paths so they produce identical shapes.
    """
    if "DATE-OBS" in df.columns:
        df["DATE-OBS"] = pd.to_datetime(df["DATE-OBS"], errors="coerce")
        df.set_index("DATE-OBS", drop=False, inplace=True)
        df.sort_index(inplace=True)
        df["FRAME_NUM"] = df.reset_index(drop=True).index.values
        df["TIME_DIFF"] = df["DATE-OBS"].diff().dt.total_seconds()
        df["TIME_REL"] = (
            df["DATE-OBS"] - df["DATE-OBS"].min()
        ).dt.total_seconds()
    return df


def _load_wcs_from_folder(wcs_dir, log):
    """Load all .wcs files in wcs_dir into a DataFrame sorted by DATE-OBS."""
    # Skip dotfiles (macOS `._*` AppleDouble shadows would parse as empty rows).
    wcs_files = sorted(
        p for p in wcs_dir.glob("*.wcs")
        if p.is_file() and not p.name.startswith(".")
    )
    log(f"[PE] Loading {len(wcs_files)} WCS files from {wcs_dir}")
    if not wcs_files:
        raise FileNotFoundError(
            f"No usable .wcs files found in {wcs_dir}. "
            "Re-run with 'Run plate solve' enabled to (re)populate the cache."
        )

    records = [_wcs_read_as_dict(f) for f in wcs_files]
    return _finalize_sequence_df(pd.DataFrame.from_records(records))


def _time_axis(sequence_df):
    """Return a numpy time-from-start array (seconds), preferring DATE-LOC."""
    for key in ("DATE-LOC", "DATE-OBS"):
        if key in sequence_df.columns and sequence_df[key].notna().all():
            t_abs = Time(sequence_df[key].astype(str).tolist(), format="fits")
            return t_abs.unix - t_abs[0].unix
    raise ValueError("Neither DATE-LOC nor DATE-OBS available in WCS data")


# =============================================================================
# Plotting & analysis
# =============================================================================
def _show_figure():
    """Show the current matplotlib figure without blocking the Tk mainloop."""
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.05)


def _plot_plate_solve_data(sequence_df, log):
    """Plot RA, DEC, CRVAL1, CRVAL2 and the linear drift in arcsec."""
    log("[PE] Plotting plate-solve data")

    t = _time_axis(sequence_df)

    sequence_df["Delta_CRVAL1"] = (
        sequence_df["CRVAL1"] - sequence_df["CRVAL1"].iloc[0]
    )
    sequence_df["Delta_CRVAL2"] = (
        sequence_df["CRVAL2"] - sequence_df["CRVAL2"].iloc[0]
    )

    poly_d1 = np.polyfit(t, sequence_df["Delta_CRVAL1"], 1)
    poly_d2 = np.polyfit(t, sequence_df["Delta_CRVAL2"], 1)
    p1 = np.poly1d(poly_d1)
    p2 = np.poly1d(poly_d2)

    diff_ra = sequence_df["CRVAL1"].max() - sequence_df["CRVAL1"].min()
    diff_dec = sequence_df["CRVAL2"].max() - sequence_df["CRVAL2"].min()
    log(
        f"[PE] RA drift amplitude: {diff_ra * 3600:.2f} arcsec over "
        f"{t.max():.2f}s  ({poly_d1[0] * 3600 * 60:.3f} arcsec/min)"
    )
    log(
        f"[PE] DEC drift amplitude: {diff_dec * 3600:.3f} arcsec over "
        f"{t.max():.2f}s  ({poly_d2[0] * 3600 * 60:.3f} arcsec/min)"
    )

    # ---- Fig 1: raw RA/DEC and CRVAL1/CRVAL2 ----
    fig = plt.figure(figsize=(12, 8))

    fig.add_subplot(221)
    plt.plot(t, np.array(sequence_df["RA"]))
    plt.title("RA [°]")
    plt.grid()
    plt.xlim(0, t.max())

    fig.add_subplot(222)
    plt.plot(t, np.array(sequence_df["CRVAL1"]))
    plt.title("CRVAL1 [°]")
    plt.grid()
    plt.xlim(0, t.max())

    fig.add_subplot(223)
    plt.plot(t, np.array(sequence_df["DEC"]))
    plt.title("DEC [°]")
    plt.xlabel("time (in s)")
    plt.grid()
    plt.xlim(0, t.max())

    fig.add_subplot(224)
    plt.plot(t, np.array(sequence_df["CRVAL2"]))
    plt.title("CRVAL2 [°]")
    plt.xlabel("time (in s)")
    plt.ylabel("DEC")
    plt.grid()
    plt.xlim(0, t.max())

    _show_figure()

    # ---- Fig 2: RA/DEC error in arcsec + linear drift ----
    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(211)
    plt.plot(t, np.array(sequence_df["Delta_CRVAL1"]) * 3600, t, p1(t) * 3600)
    plt.title("RA error [arcsec]")
    plt.grid()
    plt.xlim(0, t.max())
    txt = r"$\delta\,RA=%.3f$ arcsec/min" % (poly_d1[0] * 3600 * 60,)
    y_pos = 0.20 if poly_d1[0] > 0 else 0.95
    ax.text(0.55, y_pos, txt, transform=ax.transAxes,
            fontsize=24, verticalalignment="top")

    ax = fig.add_subplot(212)
    plt.plot(t, np.array(sequence_df["Delta_CRVAL2"]) * 3600, t, p2(t) * 3600)
    plt.title("DEC error [arcsec]")
    plt.xlabel("time (in s)")
    plt.grid()
    plt.xlim(0, t.max())
    txt = r"$\delta\,DEC=%.3f$ arcsec/min" % (poly_d2[0] * 3600 * 60,)
    y_pos = 0.20 if poly_d2[0] > 0 else 0.95
    ax.text(0.55, y_pos, txt, transform=ax.transAxes,
            fontsize=24, verticalalignment="top")

    _show_figure()


def _ssa(x1, L, eigen_idx):
    """Singular Spectrum Analysis (ported verbatim from v0p2).

    Returns (reconstructed_component, residual, sorted_eigenvalues).

    Based on Matlab code from F.J. Alonso Sanchez (Univ. of Extremadura)
    improved by F. Auger (Nantes Univ.):
    "The Sliding Singular Spectrum Analysis", IEEE TSP vol. 66 no. 1, 2018.
    """
    N = len(x1)
    if L > N / 2:
        L = N - L
    K = N - L + 1

    # Step 1: trajectory matrix
    X = np.zeros((L, K))
    for i in range(K):
        X[:, i] = x1[i:L + i]

    # Step 2: SVD
    S = np.dot(X, X.T)
    eigvals, eigvecs = LA.eig(S)
    order = np.argsort(-np.real(eigvals))
    d = np.real(eigvals)[order]
    eigvecs = eigvecs[:, order]

    V = np.dot(X.T, eigvecs)

    # Step 3: grouping
    rca = np.outer(eigvecs[:, eigen_idx], V.T[eigen_idx, :])

    # Step 4: diagonal averaging (Hankelization)
    y = np.zeros(N)
    Lp, Kp = min(L, K), max(L, K)

    for k in range(0, Lp - 1):
        for m in range(0, k + 1):
            y[k] += rca[m, k - m] / (k + 1)
    for k in range(Lp - 1, Kp):
        for m in range(0, Lp):
            y[k] += rca[m, k - m] / Lp
    for k in range(Kp, N):
        for m in range(k + 1 - Kp, N - Kp + 1):
            y[k] += rca[m, k - m] / (N - k)

    return y, x1 - y, d


def _my_fft(t, signal, Ts, plot, mark_all_components, tab_info, log):
    """FFT of `signal`, optionally plot, return fundamental (amp, phase, f, T)."""
    fourier = np.fft.fft(signal, FFT_N_POINTS)
    freq = np.fft.fftfreq(FFT_N_POINTS, d=Ts)

    spectre_amp = np.abs(fourier) / signal.size
    save_dc = spectre_amp[0]
    spectre_amp = 2 * spectre_amp
    spectre_amp[0] = save_dc
    spectre_phase = np.angle(fourier)

    idx_fond = int(np.argmax(spectre_amp))
    amp_fond   = spectre_amp[idx_fond]
    phase_fond = spectre_phase[idx_fond]
    freq_fond  = float(np.abs(freq[idx_fond]))
    t_fond     = 1.0 / freq_fond if freq_fond > 0 else float("inf")

    if plot:
        xlim_f_min = 0.001
        xlim_f_max = 1 / (Ts * 2)
        xlim_t_min = 1 / xlim_f_max
        xlim_t_max = 1 / xlim_f_min

        fig = plt.figure(figsize=(12, 20))

        plt.subplot(311)
        plt.plot(t, signal)
        plt.title("RA/CRVAL1 harmonic signal")
        plt.grid()
        plt.xlabel("time (in s)")
        plt.xlim(0, t.max())

        half = FFT_N_POINTS // 2 - 1

        ax = fig.add_subplot(312)
        plt.plot(freq[0:half], spectre_amp[0:half])
        plt.title("fft spectrum")
        plt.xlabel("frequency (Hz)")
        plt.grid()
        plt.xlim(xlim_f_min, xlim_f_max)

        if not mark_all_components:
            plt.axvline(x=freq_fond, color="red", linestyle="--")
            txt = r"$F_{fond}=%.5f$Hz" % (freq_fond,)
            ax.text(
                (freq_fond - xlim_f_min) / (xlim_f_max - xlim_f_min) + 0.01,
                0.95, txt, transform=ax.transAxes,
                fontsize=12, verticalalignment="top",
            )
        else:
            k = 0
            for i in range(SSA_NB_EIGEN):
                if tab_info[0, i] == 2:  # sinusoid pair
                    plt.axvline(x=tab_info[3, i], color="red", linestyle="--")
                    txt = r"$F=%.5f$Hz" % (tab_info[3, i],)
                    ax.text(
                        (tab_info[3, i] - xlim_f_min) / (xlim_f_max - xlim_f_min) + 0.01,
                        0.95 - k * 0.05, txt, transform=ax.transAxes,
                        fontsize=12, verticalalignment="top",
                    )
                    k += 1

        ax = fig.add_subplot(313)
        # Skip the DC bin (freq[0] == 0) — its period is infinite.
        plt.plot(1 / freq[1:half], spectre_amp[1:half])
        plt.title("fft spectrum")
        plt.xlabel("time (s)")
        plt.grid()
        plt.xlim(xlim_t_min, xlim_t_max)

        if not mark_all_components:
            plt.axvline(x=t_fond, color="red", linestyle="--")
            txt = r"$T_{fond}=%.2f$s" % (t_fond,)
            ax.text(
                (t_fond - xlim_t_min) / (xlim_t_max - xlim_t_min) + 0.01,
                0.95, txt, transform=ax.transAxes,
                fontsize=12, verticalalignment="top",
            )
        else:
            k = 0
            for i in range(SSA_NB_EIGEN):
                if tab_info[0, i] == 2:
                    plt.axvline(x=tab_info[4, i], color="red", linestyle="--")
                    txt = r"$T=%.2f$s" % (tab_info[4, i],)
                    ax.text(
                        (tab_info[4, i] - xlim_t_min) / (xlim_t_max - xlim_t_min) + 0.01,
                        0.95 - k * 0.05, txt, transform=ax.transAxes,
                        fontsize=12, verticalalignment="top",
                    )
                    k += 1

        _show_figure()

    return amp_fond, phase_fond, freq_fond, t_fond


def _ssa_analysis(sequence_df, log):
    """SSA decomposition + FFT + reconstruction (ported from v0p2)."""
    log("[PE] Starting Singular Spectrum Analysis (SSA)", color=LogColor.BLUE)

    n = sequence_df["CRVAL1"].size

    # SSA builds an L x L eigenvalue problem with L = n / SSA_WINDOW_RATIO,
    # then asks for SSA_NB_EIGEN eigencomponents — so L must be >= that.
    if n < SSA_MIN_FRAMES:
        raise ValueError(
            f"Singular Spectrum Analysis needs at least {SSA_MIN_FRAMES} "
            f"successfully plate-solved frames to extract {SSA_NB_EIGEN} "
            f"eigencomponents; only {n} are available. Widen the begin/end "
            f"range, or check why ASTAP failed on so many frames (see the "
            f"[k/N] solve log above)."
        )

    deg_to_arcsec = 3600
    signal = (sequence_df["CRVAL1"] - sequence_df["CRVAL1"].iloc[0]) * deg_to_arcsec
    signal = signal.to_numpy()

    L = int(n / SSA_WINDOW_RATIO)
    y = np.zeros((n, SSA_NB_EIGEN))
    eigvals = None
    for I in range(SSA_NB_EIGEN):
        y[:, I], _, eigvals = _ssa(signal, L, I)

    t = _time_axis(sequence_df)

    # ---- Fig 3: raw RA error + normalized singular spectrum ----
    fig = plt.figure(figsize=(12, 8))

    fig.add_subplot(211)
    plt.plot(t, signal, marker="+")
    plt.title("RA error [arcsec]")
    plt.xlabel("time (in s)")
    plt.grid()
    plt.xlim(0, t.max())

    sev = float(np.sum(eigvals))
    fig.add_subplot(212)
    plt.plot(range(1, L + 1), eigvals / sev, marker="+")
    plt.title("Normalized Singular Spectrum")
    plt.xlabel("Eigenvalue Number")
    plt.ylabel("fraction of energy")
    plt.grid()
    plt.xlim(1, 10)

    _show_figure()

    # ---- Group eigencomponents into fundamental + harmonics ----
    tab_signal = np.zeros((n, SSA_NB_EIGEN))
    # Row 0 of tab_info: 1 = single component, 2 = sinusoid pair
    # Rows 1..4: amplitude, phase, frequency, period (filled by _my_fft)
    tab_info = np.zeros((5, SSA_NB_EIGEN))

    log("[PE] SSA component composition:")
    k = 0
    idx = 0
    while idx <= SSA_GROUP_ORDER:
        if eigvals[k + 1] < (1 - SSA_PAIR_THRESHOLD) * eigvals[k]:
            tab_signal[:, idx] = y[:, k]
            tab_info[0, idx] = 1
            log(f"[PE]   {idx}: component {k}")
            k += 1
        else:
            tab_signal[:, idx] = y[:, k] + y[:, k + 1]
            tab_info[0, idx] = 2
            log(f"[PE]   {idx}: components {k} + {k + 1} (sinusoid pair)")
            k += 2
        idx += 1

    # ---- FFT on each sinusoidal component ----
    Ts = float(np.mean(np.diff(t)))
    log(f"[PE] Mean sampling period Ts = {Ts:.3f} s")

    for i in range(SSA_GROUP_ORDER + 1):
        if tab_info[0, i] == 2:
            (tab_info[1, i],
             tab_info[2, i],
             tab_info[3, i],
             tab_info[4, i]) = _my_fft(
                t, tab_signal[:, i], Ts, PLOT_FFT_PER_COMP, False, tab_info, log,
            )

    # ---- Signal reconstruction + RMSE ----
    signal_rebuilt = np.sum(tab_signal, axis=1)
    err_signal = signal - signal_rebuilt

    # Fundamental period for windowing the RMSE statistic
    i = 0
    while i < SSA_NB_EIGEN and tab_info[4, i] == 0:
        i += 1
    if i < SSA_NB_EIGEN:
        T_fond = tab_info[4, i]
    else:
        T_fond = max(Ts * 2, 1e-6)

    number_of_period = n * Ts / T_fond
    n_periods_rms    = int(0.8 * number_of_period)
    start_idx        = int(0.1 * n)
    n_points         = int(n_periods_rms * T_fond / Ts)
    # err_rms slice kept for diagnostic parity with v0p2 (unused below).
    _ = err_signal[start_idx:start_idx + n_points - 1]
    rmse = math.sqrt(float(np.square(err_signal).mean()))
    log(f"[PE] RMS reconstruction error = {rmse:.3f} arcsec",
        color=LogColor.GREEN)

    # ---- Fig 4: original vs reconstructed + reconstruction error ----
    fig = plt.figure(figsize=(12, 8))

    fig.add_subplot(211)
    plt.plot(t, signal, t, signal_rebuilt, marker="+")
    plt.title("RA error and reconstructed signal [arcsec]")
    plt.ylabel("[arcsec]")
    plt.grid()
    plt.xlim(0, t.max())

    ax = fig.add_subplot(212)
    plt.plot(t, err_signal, marker="+")
    plt.title("Error of reconstruction")
    plt.xlabel("time (in s)")
    plt.ylabel("[arcsec]")
    plt.grid()
    plt.xlim(0, t.max())
    ax.text(0.55, 0.95,
            r" RMSE of reconstruction of the signal = %.3f arcsec" % (rmse,),
            transform=ax.transAxes, fontsize=12, verticalalignment="top")

    _show_figure()

    # ---- Fig 5: main components stacked vertically ----
    fig = plt.figure(figsize=(12, 20))
    nrows = SSA_GROUP_ORDER + 1

    # First subplot: the RA drift (1st grouped component)
    poly_dev = np.polyfit(t, tab_signal[:, 0], 1)
    title0 = (f"RA deviation - Slope = {poly_dev[0] * 60:.3f} arcsec/min "
              f"between t=0 and t={t[n - 1]:.1f}s")
    log(f"[PE] {title0}")
    fig.add_subplot(nrows, 1, 1)
    plt.plot(t, tab_signal[:, 0])
    plt.title(title0)
    plt.ylabel("[arcsec]")
    plt.grid()
    plt.xlim(0, t.max())

    for k in range(1, SSA_GROUP_ORDER + 1):
        window = tab_signal[start_idx:start_idx + n_points - 1, k]
        max_v = float(np.max(window)) if window.size else 0.0
        min_v = float(np.min(window)) if window.size else 0.0
        rms_v = math.sqrt(float(np.square(window).mean())) if window.size else 0.0

        if k == 1:
            title = f"Max value of Fond = {max_v:.2f} arcsec"
            log(f"[PE] Max value of Fond = {max_v:.2f} arcsec")
            log(f"[PE] Min value of Fond = {min_v:.2f} arcsec")
            log(f"[PE] RMS value of Fond = {rms_v:.2f} arcsec")
        else:
            title = f"Max value of H{k - 1} = {max_v:.2f} arcsec"
            log(f"[PE] Max value of H{k - 1} = {max_v:.2f} arcsec")
            log(f"[PE] Min value of H{k - 1} = {min_v:.2f} arcsec")
            log(f"[PE] RMS value of H{k - 1} = {rms_v:.2f} arcsec")

        fig.add_subplot(nrows, 1, k + 1)
        plt.plot(t, tab_signal[:, k])
        plt.title(title)
        plt.ylabel("[arcsec]")
        plt.grid()
        plt.xlim(0, t.max())

    plt.xlabel("time (in s)")
    _show_figure()

    # ---- Fig 6: FFT of the harmonic signal (full RA error minus linear drift)
    poly_dev = np.polyfit(t, signal, 1)
    ra_polynome = np.poly1d(poly_dev)
    signal_fond = signal - ra_polynome(t)
    _my_fft(t, signal_fond, Ts, True, True, tab_info, log)

    log("[PE] End of Singular Spectrum Analysis (SSA)", color=LogColor.GREEN)


# =============================================================================
# Top-level driver
# =============================================================================
def compute_periodic_error(fits_files, first_idx, last_idx,
                           use_astap, astap_cli, siril,
                           do_plate_solve, log):
    """Run the full PE pipeline on the selected FITS window.

    The work is synchronous (the GUI freezes during plate-solving on large
    captures); progress is reported via Siril's log panel.
    """
    selected = fits_files[first_idx - 1:last_idx]
    if not selected:
        raise ValueError("No FITS files selected (check the index range)")

    if use_astap:
        fits_folder = selected[0].parent
        wcs_dir = fits_folder / WCS_SUBDIR_NAME

        if do_plate_solve:
            _plate_solve_astap(selected, astap_cli, wcs_dir, log)
        elif not wcs_dir.exists():
            raise FileNotFoundError(
                f"Plate solve is off but no previous results exist at "
                f"{wcs_dir}. Enable 'Run plate solve' and try again."
            )

        sequence_df = _load_wcs_from_folder(wcs_dir, log)
    else:
        # Siril/GAIA path: in-memory only, so there is no cache to reuse.
        if not do_plate_solve:
            raise ValueError(
                "The Siril/GAIA path does not support cached reuse in this "
                "version. Enable 'Run plate solve', or select ASTAP for "
                "cached re-runs."
            )
        sequence_df = _plate_solve_siril(selected, siril, log)

    # Constrain the analysis to the user-selected window. v0p2 always
    # analysed from the first frame; the GUI now lets the user choose.
    if len(sequence_df) > len(selected):
        sequence_df = sequence_df.iloc[:len(selected)].copy()

    _plot_plate_solve_data(sequence_df, log)
    _ssa_analysis(sequence_df, log)
    log("[PE] Analysis complete.", color=LogColor.GREEN)


# =============================================================================
# GUI
# =============================================================================
class PEAnalysisInterface:
    """Tkinter GUI for the Periodic Error computation script."""

    PLATE_SOLVER_ASTAP = "astap"
    PLATE_SOLVER_SIRIL = "siril"

    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} - v{VERSION}")
        self.root.resizable(False, False)
        self.style = tksiril.standard_style()

        # Connect to Siril
        self.siril = s.SirilInterface()
        try:
            self.siril.connect()
        except SirilConnectionError as exc:
            messagebox.showerror("Siril connection error", str(exc))
            self.root.destroy()
            return

        # Require a recent-enough Siril
        try:
            self.siril.cmd("requires", REQUIRED_SIRIL)
        except s.CommandError:
            self.siril.error_messagebox(
                f"This script requires Siril {REQUIRED_SIRIL} or later."
            )
            self.root.destroy()
            return

        # Discover FITS files in Siril's working directory
        try:
            self.working_dir = Path(self.siril.get_siril_wd())
        except SirilError as exc:
            self.siril.error_messagebox(
                f"Could not get Siril working directory: {exc}"
            )
            self.root.destroy()
            return

        # Skip dotfiles (e.g. macOS '._*.fits' AppleDouble metadata that
        # ASTAP can't parse — see `dot_clean` to remove them on disk).
        self.fits_files = sorted(
            p for p in self.working_dir.glob("*.fits") if not p.name.startswith(".")
        )
        self.nb_fits = len(self.fits_files)
        if self.nb_fits == 0:
            self.siril.error_messagebox(
                f"No .fits files found in Siril's current working directory:\n\n"
                f"    {self.working_dir}\n\n"
                "Please set Siril's working directory to the folder that "
                "contains your PE capture frames, then run this script again.\n\n"
                "You can change it from Siril's main window "
                "(folder icon next to the working directory path), "
                "or with the Siril command:  cd /path/to/your/fits/folder"
            )
            self.root.destroy()
            return

        # Tk variables
        paths = load_config()
        self.astap_path_var      = tk.StringVar(self.root, value=paths["astap"])
        self.siril_path_var      = tk.StringVar(self.root, value=paths["siril"])
        self.solver_var          = tk.StringVar(self.root, value=self.PLATE_SOLVER_ASTAP)
        self.begin_index_var     = tk.IntVar(self.root,    value=1)
        self.end_index_var       = tk.IntVar(self.root,    value=self.nb_fits)
        self.do_plate_solve_var  = tk.BooleanVar(self.root, value=True)

        # Build and theme the UI
        self._create_widgets()
        tksiril.match_theme_to_siril(self.root, self.siril)

    # ------------------------------------------------------------------ widgets
    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main_frame, text=APP_NAME, style="Header.TLabel").pack(pady=(0, 10))

        # Plate-solving tool frame
        tools_frame = ttk.LabelFrame(main_frame, text="Plate-solving tool", padding=10)
        tools_frame.pack(fill=tk.X, padx=5, pady=5)

        self._build_solver_row(
            tools_frame, "ASTAP", self.PLATE_SOLVER_ASTAP,
            self.astap_path_var, self._browse_astap_exe,
            tooltip=("Plate-solve via the external ASTAP CLI. Writes .wcs "
                     "sidecars to PlateSolveAstap/ so 'Run plate solve' "
                     "can be unticked on subsequent runs to skip ASTAP."),
        )
        self._build_solver_row(
            tools_frame, "SIRIL (GAIA)", self.PLATE_SOLVER_SIRIL,
            self.siril_path_var, self._browse_siril_exe,
            tooltip=("Plate-solve via Siril's built-in GAIA solver. "
                     "Works without ASTAP installed, but re-solves every "
                     "run — 'Run plate solve' must stay enabled."),
        )

        # Processing parameters frame
        proc_frame = ttk.LabelFrame(main_frame, text="Processing parameters", padding=10)
        proc_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(proc_frame, text="Number of FITS files:").grid(
            row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Label(proc_frame, text=str(self.nb_fits)).grid(
            row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(proc_frame, text="Begin index:").grid(
            row=1, column=0, sticky="e", padx=5, pady=2)
        begin_entry = ttk.Entry(proc_frame, textvariable=self.begin_index_var, width=10)
        begin_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        tksiril.create_tooltip(
            begin_entry,
            f"First frame index (1..{self.nb_fits - 1}). "
            f"The selected window must contain at least {SSA_MIN_FRAMES} frames.",
        )

        ttk.Label(proc_frame, text="End index:").grid(
            row=2, column=0, sticky="e", padx=5, pady=2)
        end_entry = ttk.Entry(proc_frame, textvariable=self.end_index_var, width=10)
        end_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        tksiril.create_tooltip(
            end_entry,
            f"Last frame index (2..{self.nb_fits}). "
            f"The selected window must contain at least {SSA_MIN_FRAMES} frames.",
        )

        ttk.Label(
            proc_frame,
            text=f"(SSA requires at least {SSA_MIN_FRAMES} frames in the window)",
            foreground="gray",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 4))

        plate_solve_cb = ttk.Checkbutton(
            proc_frame, text="Run plate solve", variable=self.do_plate_solve_var,
        )
        plate_solve_cb.grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=(8, 0))
        tksiril.create_tooltip(
            plate_solve_cb,
            "Disable to reuse plate-solving results from a previous run.",
        )

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=15)

        process_btn = ttk.Button(button_frame, text="Process", command=self._on_process)
        process_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(process_btn, "Run the periodic error computation.")

        close_btn = ttk.Button(button_frame, text="Close", command=self._on_close)
        close_btn.pack(side=tk.LEFT, padx=5)
        tksiril.create_tooltip(close_btn, "Close the script (no changes to images).")

    def _build_solver_row(self, parent, label, value, path_var, browse_cmd,
                          tooltip=None):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        radio = ttk.Radiobutton(
            row, text=label, variable=self.solver_var, value=value, width=14,
        )
        radio.pack(side=tk.LEFT)
        if tooltip:
            tksiril.create_tooltip(radio, tooltip)
        ttk.Entry(row, textvariable=path_var, width=50).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True,
        )
        ttk.Button(row, text="Browse...", command=browse_cmd).pack(side=tk.LEFT)

    def _log(self, msg, color=LogColor.DEFAULT):
        """Adapter so module-level helpers can call self.siril.log()."""
        self.siril.log(msg, color=color)

    # ---------------------------------------------------------------- callbacks
    def _browse_astap_exe(self):
        path = filedialog.askopenfilename(
            title="Select ASTAP executable",
            initialdir=str(Path(self.astap_path_var.get()).parent),
        )
        if path:
            self.astap_path_var.set(path)
            save_config(path, self.siril_path_var.get())
            self.siril.log(f"[PE] ASTAP exe set to {path}", color=LogColor.BLUE)

    def _browse_siril_exe(self):
        path = filedialog.askopenfilename(
            title="Select Siril CLI executable",
            initialdir=str(Path(self.siril_path_var.get()).parent),
        )
        if path:
            self.siril_path_var.set(path)
            save_config(self.astap_path_var.get(), path)
            self.siril.log(f"[PE] Siril CLI set to {path}", color=LogColor.BLUE)

    def _validate_indices(self):
        t0, t_end = self.begin_index_var.get(), self.end_index_var.get()
        if t0 < 1 or t0 >= self.nb_fits:
            return None, None, f"Begin index must be between 1 and {self.nb_fits - 1}."
        if t_end < 2 or t_end > self.nb_fits:
            return None, None, f"End index must be between 2 and {self.nb_fits}."
        if t0 >= t_end:
            return None, None, "Begin index must be strictly less than end index."
        window = t_end - t0 + 1
        if window < SSA_MIN_FRAMES:
            return None, None, (
                f"Selected window is only {window} frames; SSA needs at "
                f"least {SSA_MIN_FRAMES}. Widen the begin/end range."
            )
        return t0, t_end, None

    def _on_process(self):
        t0, t_end, err = self._validate_indices()
        if err:
            messagebox.showwarning("Invalid index range", err)
            self.siril.log(f"[PE] {err}", color=LogColor.RED)
            return

        use_astap = self.solver_var.get() == self.PLATE_SOLVER_ASTAP
        self.siril.log(
            f"[PE] Processing frames {t0}..{t_end} with "
            f"{'ASTAP' if use_astap else 'Siril/GAIA'} "
            f"(plate solve {'on' if self.do_plate_solve_var.get() else 'off'}).",
            color=LogColor.GREEN,
        )

        try:
            compute_periodic_error(
                self.fits_files, t0, t_end, use_astap,
                self.astap_path_var.get(), self.siril,
                self.do_plate_solve_var.get(),
                self._log,
            )
        except FileNotFoundError as exc:
            self.siril.log(f"[PE] {exc}", color=LogColor.RED)
            messagebox.showerror("File not found", str(exc))
        except ValueError as exc:
            self.siril.log(f"[PE] {exc}", color=LogColor.RED)
            messagebox.showerror("Invalid input", str(exc))
        except RuntimeError as exc:
            self.siril.log(f"[PE] {exc}", color=LogColor.RED)
            messagebox.showerror("Plate solve failed", str(exc))
        except SirilError as exc:
            self.siril.log(f"[PE] Siril error: {exc}", color=LogColor.RED)
            messagebox.showerror("Siril error", str(exc))

    def _on_close(self):
        try:
            self.siril.disconnect()
        except SirilError:
            pass
        plt.close("all")
        self.root.destroy()


# =============================================================================
# Entry point
# =============================================================================
def main():
    root = ThemedTk()
    root.attributes("-topmost", True)
    PEAnalysisInterface(root)
    try:
        if not root.winfo_exists():
            return
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        root.mainloop()
    except tk.TclError:
        # __init__ aborted (no Siril, no FITS, wrong version, ...) and the
        # root window is already destroyed.
        return


if __name__ == "__main__":
    main()
