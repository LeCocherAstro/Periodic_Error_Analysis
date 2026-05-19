"""
Automatic Periodic Error Computation - Siril Python script.

Authors:
Mickaël HILAIRET, LS2N/Ecole Centrale de Nantes, France
and Gilles MORAIN, France

Distributed under a Creative Commons Attribution ǀ 4.0
International licence CC BY-NC-SA 4.0

GUI-based Siril script that computes the periodic error of an equatorial
mount from a time-series of FITS frames. The script collects FITS files
from Siril's working directory, lets the user pick the plate-solving
engine (ASTAP or Siril/GAIA) and the frame index range, then runs the
analysis.

See https://github.com/LeCocherAstro/Periodic_Error_Analysis for more information.

Follows the official Siril Python scripting template:
https://siril.readthedocs.io/en/stable/scripts/python_gui_template.html
"""

# Core module imports
import csv
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import sirilpy as s
from sirilpy import tksiril, LogColor, SirilError, SirilConnectionError

# Use Siril's nicer tkfilebrowser on Linux when available — the stock Tk
# file picker is dated and inconsistent with the rest of the Siril UI.
# macOS and Windows already get native dialogs from tkinter.filedialog.
if sys.platform.startswith("linux") and s.check_module_version(">=0.6.47"):
    import sirilpy.tkfilebrowser as filedialog  # noqa: E402
else:
    from tkinter import filedialog  # noqa: E402

# Ensure non-core modules are available in Siril's venv before importing them
s.ensure_installed("ttkthemes", "numpy", "pandas", "matplotlib", "astropy",
                   "unidecode", "reportlab")

from ttkthemes import ThemedTk        # noqa: E402  (must follow ensure_installed)
import numpy as np                    # noqa: E402
import pandas as pd                   # noqa: E402
import matplotlib                     # noqa: E402
import matplotlib.pyplot as plt       # noqa: E402
from numpy import linalg as LA        # noqa: E402
from astropy.io import fits as astrofits  # noqa: E402
from astropy.time import Time         # noqa: E402
from unidecode import unidecode       # noqa: E402

# reportlab — for the optional PDF report (imported lazily inside the
# builder so that running the script without ticking 'Save PDF report'
# doesn't pay the import cost on every launch).


# =============================================================================
# Constants
# =============================================================================
APP_NAME       = "Automatic Periodic Error Computation"
AUTHORS        = "Mickaël HILAIRET and Gilles MORAIN"
VERSION        = "0.6.2"
REQUIRED_SIRIL = "1.3.6"

# Platform-specific default paths to the ASTAP and Siril CLI executables.
# Used only as a first-launch seed for the file pickers — the user's choices
# are persisted to config_PE.txt and override these on subsequent runs.
if sys.platform == "darwin":
    DEFAULT_ASTAP_CLI = "/Applications/ASTAP.app/Contents/MacOS/astap"
    DEFAULT_SIRIL_CLI = "/Applications/Siril.app/Contents/MacOS/siril-cli"
elif sys.platform.startswith("win"):
    DEFAULT_ASTAP_CLI = r"C:\Program Files\astap\astap.exe"
    DEFAULT_SIRIL_CLI = r"C:\Program Files\Siril\bin\siril-cli.exe"
else:  # Linux and other POSIX
    DEFAULT_ASTAP_CLI = "/usr/bin/astap"
    DEFAULT_SIRIL_CLI = "/usr/bin/siril-cli"

# FITS files come in several extensions; some capture software writes
# uppercase on Linux/case-sensitive volumes, so match case-insensitively.
FITS_SUFFIXES = {".fits", ".fit", ".fts"}

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
def _sanitize_filename(name):
    """Strip characters illegal in filenames on Windows / POSIX.

    Windows reserves <>:"/\\|?* and ASCII control characters; trailing
    dots/spaces are also rejected. Collapse whitespace runs into single
    underscores so the result reads cleanly. Empty result falls back to
    "report".
    """
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = sanitized.rstrip(". ")
    return sanitized or "report"


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


def _read_capture_metadata(fits_path):
    """Read capture metadata (target, equipment, settings) from a FITS header.

    Returns a dict keyed by friendly names; values are None when the
    keyword is missing. Used both for auto-populating the GUI title field
    and for the equipment table on the PDF report's title page.
    """
    try:
        header = astrofits.getheader(str(fits_path))
    except Exception:
        return {}

    def _get(*keys):
        for key in keys:
            value = header.get(key)
            if value not in (None, "", " "):
                return value
        return None

    return {
        "object":        _get("OBJECT"),
        "date_obs":      _get("DATE-OBS"),
        "telescope":     _get("TELESCOP", "TELESCOPE"),
        "camera":        _get("INSTRUME", "INSTRUMENT", "CAMERA"),
        "focal_length":  _get("FOCALLEN"),
        "pixel_size":    _get("XPIXSZ"),
        "binning":       _get("XBINNING") or 1,
        "exposure":      _get("EXPTIME", "EXPOSURE"),
        "gain":          _get("GAIN", "EGAIN"),
        "filter":        _get("FILTER"),
        "observer":      _get("OBSERVER"),
        "site_lat":      _get("SITELAT"),
        "site_long":     _get("SITELONG"),
        "ccd_temp":      _get("CCD-TEMP"),
    }


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
    # Filenames can still embed spaces (e.g. 'Optolong L-Pro'); sirilpy's
    # cmd() just joins args with " " before shipping to Siril, so we
    # double-quote the filename to keep it as a single token in Siril's
    # command parser.
    for i, fits_path in enumerate(fits_paths, 1):
        log(f"[PE]   [{i}/{n}] Solving {fits_path.name}")
        try:
            siril.cmd("load", f'"{fits_path.name}"')
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
    # Case-insensitive suffix match; skip dotfiles (macOS `._*` AppleDouble
    # shadows on exFAT/SMB would parse as empty rows).
    wcs_files = sorted(
        p for p in wcs_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".wcs"
        and not p.name.startswith(".")
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


def _format_seconds(value):
    """Render a seconds value for the log / PDF table.

    None / non-finite / NaN  -> ``"n/a"``
    very large (>10 000 s)   -> ``"> long capture"``
    < 10 s                   -> 1-decimal seconds
    otherwise                -> integer seconds
    """
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "> long capture"
    if v > 1e4:
        return "> long capture"
    if v < 10:
        return f"{v:.1f} s"
    return f"{v:.0f} s"


def _compute_pixel_scale(sequence_df, override_arcsec_per_pixel, log):
    """Resolve the per-axis pixel scale (arcsec/pixel) from the plate-solve.

    Tries, in order: user override (both axes), CD matrix, CDELT1/CDELT2,
    and finally FOCALLEN + XPIXSZ. Per-frame values are reduced by the
    median to be robust to occasional outliers. Returns a dict with
    ``ra_arcsec_per_pixel``, ``dec_arcsec_per_pixel`` and ``source``.
    Returns ``None`` if no source could be resolved.
    """
    if override_arcsec_per_pixel is not None:
        v = float(override_arcsec_per_pixel)
        log(f"[PE] Pixel scale: {v:.3f} arcsec/pixel (source: override)")
        return {"ra_arcsec_per_pixel": v,
                "dec_arcsec_per_pixel": v,
                "source": "override"}

    def _finite_median(series):
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return float(np.median(arr))

    # CD matrix: arcsec/pixel along each image axis
    cd_cols = ("CD1_1", "CD1_2", "CD2_1", "CD2_2")
    if all(c in sequence_df.columns for c in cd_cols):
        c11 = _finite_median(sequence_df["CD1_1"])
        c12 = _finite_median(sequence_df["CD1_2"])
        c21 = _finite_median(sequence_df["CD2_1"])
        c22 = _finite_median(sequence_df["CD2_2"])
        if None not in (c11, c12, c21, c22):
            scale_x = 3600.0 * math.sqrt(c11 * c11 + c21 * c21)
            scale_y = 3600.0 * math.sqrt(c12 * c12 + c22 * c22)
            if scale_x > 0 and scale_y > 0:
                log(f"[PE] Pixel scale: {scale_x:.3f} / {scale_y:.3f} "
                    f"arcsec/pixel (source: CD matrix)")
                return {"ra_arcsec_per_pixel": scale_x,
                        "dec_arcsec_per_pixel": scale_y,
                        "source": "CD"}

    # CDELT1 / CDELT2 in degrees/pixel
    if "CDELT1" in sequence_df.columns and "CDELT2" in sequence_df.columns:
        d1 = _finite_median(sequence_df["CDELT1"])
        d2 = _finite_median(sequence_df["CDELT2"])
        if d1 is not None and d2 is not None:
            scale_x = abs(d1) * 3600.0
            scale_y = abs(d2) * 3600.0
            if scale_x > 0 and scale_y > 0:
                log(f"[PE] Pixel scale: {scale_x:.3f} / {scale_y:.3f} "
                    f"arcsec/pixel (source: CDELT)")
                return {"ra_arcsec_per_pixel": scale_x,
                        "dec_arcsec_per_pixel": scale_y,
                        "source": "CDELT"}

    # FOCALLEN (mm) + XPIXSZ (µm). 206.265 = (180/π) × 3600 / 1000.
    if ("FOCALLEN" in sequence_df.columns
            and "XPIXSZ" in sequence_df.columns):
        focal = _finite_median(sequence_df["FOCALLEN"])
        pix_um = _finite_median(sequence_df["XPIXSZ"])
        if focal and pix_um and focal > 0:
            scale = 206.265 * pix_um / focal
            log(f"[PE] Pixel scale: {scale:.3f} arcsec/pixel "
                f"(source: FOCALLEN+XPIXSZ)")
            return {"ra_arcsec_per_pixel": scale,
                    "dec_arcsec_per_pixel": scale,
                    "source": "FOCALLEN"}

    log("[PE] Pixel scale: could not resolve from headers — "
        "max-exposure-time section will be omitted", color=LogColor.SALMON)
    return None


# =============================================================================
# Plotting & analysis
# =============================================================================
def _show_figure():
    """Show the current matplotlib figure without blocking the Tk mainloop.

    Figures are created with constrained_layout=True, so we deliberately
    do NOT call tight_layout() here — the two layout engines clash.
    """
    plt.show(block=False)
    plt.pause(0.05)


def _plot_plate_solve_data(sequence_df, log):
    """Plot RA, DEC, CRVAL1, CRVAL2 and the linear drift in arcsec.

    Returns a dict with the two created figures and the headline drift
    metrics so the optional PDF report can embed both.
    """
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
    fig_raw = plt.figure(figsize=(12, 8), constrained_layout=True)

    fig_raw.add_subplot(221)
    plt.plot(t, np.array(sequence_df["RA"]))
    plt.title("RA [°]")
    plt.grid()
    plt.xlim(0, t.max())

    fig_raw.add_subplot(222)
    plt.plot(t, np.array(sequence_df["CRVAL1"]))
    plt.title("CRVAL1 [°]")
    plt.grid()
    plt.xlim(0, t.max())

    fig_raw.add_subplot(223)
    plt.plot(t, np.array(sequence_df["DEC"]))
    plt.title("DEC [°]")
    plt.xlabel("time (in s)")
    plt.grid()
    plt.xlim(0, t.max())

    fig_raw.add_subplot(224)
    plt.plot(t, np.array(sequence_df["CRVAL2"]))
    plt.title("CRVAL2 [°]")
    plt.xlabel("time (in s)")
    plt.ylabel("DEC")
    plt.grid()
    plt.xlim(0, t.max())

    _show_figure()

    # ---- Fig 2: RA/DEC error in arcsec + linear drift ----
    fig_drift = plt.figure(figsize=(12, 8), constrained_layout=True)

    ax = fig_drift.add_subplot(211)
    plt.plot(t, np.array(sequence_df["Delta_CRVAL1"]) * 3600, t, p1(t) * 3600)
    plt.title("RA error [arcsec]")
    plt.grid()
    plt.xlim(0, t.max())
    txt = r"$\delta\,RA=%.3f$ arcsec/min" % (poly_d1[0] * 3600 * 60,)
    y_pos = 0.20 if poly_d1[0] > 0 else 0.95
    ax.text(0.55, y_pos, txt, transform=ax.transAxes,
            fontsize=24, verticalalignment="top")

    ax = fig_drift.add_subplot(212)
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

    return {
        "figures": {"raw_ra_dec": fig_raw, "arcsec_drift": fig_drift},
        "metrics": {
            "time_span_s":               float(t.max()),
            "ra_drift_arcsec_per_min":   float(poly_d1[0] * 3600 * 60),
            "dec_drift_arcsec_per_min":  float(poly_d2[0] * 3600 * 60),
            "ra_pp_arcsec":              float(diff_ra * 3600),
            "dec_pp_arcsec":             float(diff_dec * 3600),
        },
    }


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
    """FFT of `signal`, optionally plot.

    Returns a tuple ``(amp_fond, phase_fond, freq_fond, t_fond, fig)`` —
    `fig` is the matplotlib Figure when `plot` is True, else None.
    """
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

    fig = None
    if plot:
        xlim_f_min = 0.001
        xlim_f_max = 1 / (Ts * 2)
        xlim_t_min = 1 / xlim_f_max
        xlim_t_max = 1 / xlim_f_min

        fig = plt.figure(figsize=(12, 20), constrained_layout=True)

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

    return amp_fond, phase_fond, freq_fond, t_fond, fig


def _max_exposure_time(t, signal_arcsec, pixel_scale_arcsec,
                       fundamental_period_s):
    """Estimate the longest exposure window keeping drift+PE under one pixel.

    Returns a dict with three complementary estimates (all in seconds):

    - ``linear_s``         pixel / |least-squares slope| (long-term drift only)
    - ``instantaneous_s``  pixel / (|slope| + 2π·rms_detrended / period)
                           — closed-form envelope of drift plus the
                           dominant periodic component, robust to
                           per-sample plate-solver scatter
    - ``sliding_window_s`` largest window whose peak-to-peak ≤ one pixel

    Plus ``binding`` (the name of the smallest finite estimate),
    ``duration_s`` (observation window), ``saturated`` (True if the
    whole capture never exceeds one pixel, so ``sliding_window_s`` is
    pinned at the duration), and ``peak_slew_arcsec_per_s`` (the slew
    rate used in ``instantaneous_s``, exposed for diagnostics).
    Pure function — no logging.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(signal_arcsec, dtype=float)
    n = s.size
    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    out = {"linear_s": None, "instantaneous_s": None,
           "sliding_window_s": None, "binding": None,
           "duration_s": duration, "saturated": False,
           "peak_slew_arcsec_per_s": None}
    if n < 2 or duration <= 0 or pixel_scale_arcsec is None \
            or pixel_scale_arcsec <= 0:
        return out
    px = float(pixel_scale_arcsec)
    Ts = duration / (n - 1)

    poly = np.polyfit(t, s, 1)
    slope = float(poly[0])
    out["linear_s"] = px / abs(slope) if abs(slope) > 1e-12 else float("inf")

    # Instantaneous envelope: drift slope + 2π · rms(detrended) / period.
    # For a pure sinusoid this slightly under-estimates the true peak slew
    # (rms = amplitude/√2 ≈ 0.71 × amplitude), but unlike np.gradient it
    # is immune to the per-sample plate-solver scatter that would
    # otherwise dominate the metric on the noisier Dec axis.
    residual = s - np.poly1d(poly)(t)
    rms_residual = float(math.sqrt(np.mean(residual * residual)))
    if fundamental_period_s and fundamental_period_s > 0:
        peak_slew = abs(slope) + (2.0 * math.pi * rms_residual
                                  / float(fundamental_period_s))
    else:
        peak_slew = abs(slope)
    out["peak_slew_arcsec_per_s"] = peak_slew
    out["instantaneous_s"] = (px / peak_slew if peak_slew > 1e-12
                              else float("inf"))

    def pp_for_k(k):
        if k < 2:
            return 0.0
        if k >= n:
            return float(s.max() - s.min())
        win = np.lib.stride_tricks.sliding_window_view(s, k)
        return float(np.max(win.max(axis=-1) - win.min(axis=-1)))

    if pp_for_k(n) <= px:
        out["sliding_window_s"] = duration
        out["saturated"] = True
    else:
        lo, hi = 2, n
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if pp_for_k(mid) <= px:
                lo = mid
            else:
                hi = mid
        out["sliding_window_s"] = (lo - 1) * Ts

    finite = [(k, v) for k, v in (
        ("linear_s", out["linear_s"]),
        ("instantaneous_s", out["instantaneous_s"]),
        ("sliding_window_s", out["sliding_window_s"]),
    ) if v is not None and math.isfinite(v)]
    if finite:
        out["binding"] = min(finite, key=lambda kv: kv[1])[0]
    return out


def _ssa_analysis(sequence_df, log):
    """SSA decomposition + FFT + reconstruction (ported from v0p2).

    Returns a dict ``{"figures": {...}, "metrics": {...}}`` so the
    optional PDF report can embed both the plots and the headline numbers.
    """
    log("[PE] Starting Singular Spectrum Analysis (SSA)", color=LogColor.BLUE)
    figures = {}        # name -> matplotlib Figure
    components = []     # per-sinusoid-pair metric dicts

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
    fig_spectrum = plt.figure(figsize=(12, 8), constrained_layout=True)

    fig_spectrum.add_subplot(211)
    plt.plot(t, signal, marker="+")
    plt.title("RA error [arcsec]")
    plt.xlabel("time (in s)")
    plt.grid()
    plt.xlim(0, t.max())

    sev = float(np.sum(eigvals))
    fig_spectrum.add_subplot(212)
    plt.plot(range(1, L + 1), eigvals / sev, marker="+")
    plt.title("Normalized Singular Spectrum")
    plt.xlabel("Eigenvalue Number")
    plt.ylabel("fraction of energy")
    plt.grid()
    plt.xlim(1, 10)

    _show_figure()
    figures["singular_spectrum"] = fig_spectrum

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
             tab_info[4, i],
             fig_fft_i) = _my_fft(
                t, tab_signal[:, i], Ts, PLOT_FFT_PER_COMP, False, tab_info, log,
            )
            if fig_fft_i is not None:
                figures[f"fft_component_{i}"] = fig_fft_i

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
    fig_recon = plt.figure(figsize=(12, 8), constrained_layout=True)

    fig_recon.add_subplot(211)
    plt.plot(t, signal, t, signal_rebuilt, marker="+")
    plt.title("RA error and reconstructed signal [arcsec]")
    plt.ylabel("[arcsec]")
    plt.grid()
    plt.xlim(0, t.max())

    ax = fig_recon.add_subplot(212)
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
    figures["reconstruction"] = fig_recon

    # ---- Fig 5: main components stacked vertically ----
    fig_stacked = plt.figure(figsize=(12, 20), constrained_layout=True)
    nrows = SSA_GROUP_ORDER + 1

    # First subplot: the RA drift (1st grouped component)
    poly_dev = np.polyfit(t, tab_signal[:, 0], 1)
    drift_slope_arcsec_per_min = float(poly_dev[0] * 60)
    title0 = (f"RA deviation - Slope = {drift_slope_arcsec_per_min:.3f} arcsec/min "
              f"between t=0 and t={t[n - 1]:.1f}s")
    log(f"[PE] {title0}")
    fig_stacked.add_subplot(nrows, 1, 1)
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
            name = "Fond"
            title = f"Max/Min values of Fond = {max_v:.2f} and {min_v:.2f} arcsec"
        else:
            name = f"H{k - 1}"
            title = f"Max/Min values of H{k - 1} = {max_v:.2f} and {min_v:.2f} arcsec"
        log(f"[PE] Max value of {name} = {max_v:.2f} arcsec")
        log(f"[PE] Min value of {name} = {min_v:.2f} arcsec")
        log(f"[PE] RMS value of {name} = {rms_v:.2f} arcsec")
        components.append({
            "name":      name,
            "period_s":  float(tab_info[4, k]) if tab_info[4, k] > 0 else None,
            "amp_arcsec": float(tab_info[1, k]) if tab_info[1, k] > 0 else None,
            "max_arcsec": max_v,
            "min_arcsec": min_v,
            "rms_arcsec": rms_v,
        })

        fig_stacked.add_subplot(nrows, 1, k + 1)
        plt.plot(t, tab_signal[:, k])
        plt.title(title)
        plt.ylabel("[arcsec]")
        plt.grid()
        plt.xlim(0, t.max())

    plt.xlabel("time (in s)")
    _show_figure()
    figures["stacked_components"] = fig_stacked

    # ---- Fig 6: FFT of the harmonic signal (full RA error minus linear drift)
    poly_dev = np.polyfit(t, signal, 1)
    ra_polynome = np.poly1d(poly_dev)
    signal_fond = signal - ra_polynome(t)
    _, _, _, _, fig_fft_all = _my_fft(t, signal_fond, Ts, True, True, tab_info, log)
    if fig_fft_all is not None:
        figures["fft_all_components"] = fig_fft_all

    log("[PE] End of Singular Spectrum Analysis (SSA)", color=LogColor.GREEN)

    return {
        "figures": figures,
        "metrics": {
            "n_frames":                   n,
            "sampling_period_s":          Ts,
            "fundamental_period_s":       float(T_fond),
            "rmse_arcsec":                rmse,
            "drift_slope_arcsec_per_min": drift_slope_arcsec_per_min,
            "components":                 components,
        },
        "arrays": {
            "time_axis":           t,
            "reconstructed_ra":    signal_rebuilt,
        },
    }


# =============================================================================
# PDF report
# =============================================================================
def _format_long_date(iso_or_dt):
    """'2024-03-03T22:53:21' -> 'March 3, 2024'  (no zero-padded day)."""
    if iso_or_dt is None:
        return ""
    if isinstance(iso_or_dt, str):
        try:
            dt = pd.to_datetime(iso_or_dt)
        except Exception:
            return iso_or_dt
    else:
        dt = iso_or_dt
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"


def _fig_to_image_flowable(fig, max_width_pt, max_height_pt):
    """Render a matplotlib Figure into a reportlab Image flowable."""
    import io
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image as RLImage

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    reader = ImageReader(buf)
    w_px, h_px = reader.getSize()
    # Preserve aspect ratio, fit within (max_width_pt, max_height_pt).
    scale = min(max_width_pt / w_px, max_height_pt / h_px)
    return RLImage(buf, width=w_px * scale, height=h_px * scale)


def _build_pdf_report(out_path, *,
                      title, capture_metadata, solver, frame_range,
                      drift_metrics, ssa_metrics, figures, log,
                      max_exposure_metrics=None):
    """Build the multi-page PDF report and write it to `out_path`.

    `figures` is the merged dict from both analysis stages (string name
    keys, matplotlib Figure values). The report embeds them inline at
    the points where the corresponding numbers are discussed.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    )
    from datetime import datetime

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=title or "Periodic Error Analysis Report",
        author=AUTHORS,
    )
    content_width = doc.width  # usable horizontal space in points
    styles = getSampleStyleSheet()
    h_title = ParagraphStyle("PEHeroTitle", parent=styles["Title"],
                             fontSize=32, leading=38, alignment=TA_CENTER,
                             spaceAfter=10)
    h_sub = ParagraphStyle("PEHeroSub", parent=styles["Normal"],
                           fontSize=16, leading=20, alignment=TA_CENTER,
                           textColor=colors.HexColor("#444"), spaceAfter=4)
    h_caption = ParagraphStyle("PECaption", parent=styles["Italic"],
                               fontSize=9, alignment=TA_CENTER,
                               textColor=colors.HexColor("#666"),
                               spaceAfter=14)
    h_section = ParagraphStyle("PESection", parent=styles["Heading2"],
                               fontSize=16, leading=20, spaceBefore=12,
                               spaceAfter=8,
                               textColor=colors.HexColor("#222"))
    h_body = ParagraphStyle("PEBody", parent=styles["BodyText"],
                            fontSize=10.5, leading=14, alignment=TA_JUSTIFY,
                            spaceAfter=6)

    story = []

    # ---- Title page -------------------------------------------------------
    story.append(Spacer(0, 1.5 * cm))
    story.append(Paragraph(title or "Periodic Error Analysis", h_title))
    capture_date = _format_long_date(capture_metadata.get("date_obs"))
    if capture_date:
        story.append(Paragraph(capture_date, h_sub))
    story.append(Paragraph("Periodic Error Analysis Report", h_sub))
    story.append(Spacer(0, 0.5 * cm))

    # Capture metadata table
    meta_rows = [["Property", "Value"]]
    def _add_meta(label, value, fmt=None):
        if value not in (None, "", " "):
            meta_rows.append([label, fmt(value) if fmt else str(value)])

    _add_meta("Target",        capture_metadata.get("object"))
    _add_meta("Capture start", capture_metadata.get("date_obs"))
    _add_meta("Telescope",     capture_metadata.get("telescope"))
    _add_meta("Camera",        capture_metadata.get("camera"))
    _add_meta("Filter",        capture_metadata.get("filter"))
    _add_meta("Focal length",  capture_metadata.get("focal_length"),
              lambda v: f"{float(v):g} mm")
    _add_meta("Pixel size",    capture_metadata.get("pixel_size"),
              lambda v: f"{float(v):g} um")
    _add_meta("Binning",       capture_metadata.get("binning"),
              lambda v: f"{int(v)}x{int(v)}")
    _add_meta("Exposure",      capture_metadata.get("exposure"),
              lambda v: f"{float(v):g} s")
    _add_meta("Gain",          capture_metadata.get("gain"))
    _add_meta("CCD temp",      capture_metadata.get("ccd_temp"),
              lambda v: f"{float(v):g} degC")
    _add_meta("Observer",      capture_metadata.get("observer"))

    meta_rows.append(["Plate-solver", solver])
    meta_rows.append(["Frames analysed",
                      f"{frame_range[0]}–{frame_range[1]} "
                      f"({frame_range[1] - frame_range[0] + 1} frames)"])
    meta_rows.append(["Mean sampling period",
                      f"{ssa_metrics['sampling_period_s']:.3f} s"])
    meta_rows.append(["Time span",
                      f"{drift_metrics['time_span_s']:.1f} s "
                      f"({drift_metrics['time_span_s'] / 60:.1f} min)"])

    table = Table(meta_rows, colWidths=[5 * cm, content_width - 5 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2e2e2e")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",       (0, 0), (-1, 0), "LEFT"),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#ccc")),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ]))
    story.append(table)

    story.append(Spacer(0, 0.6 * cm))
    story.append(Paragraph(
        f"Generated by {APP_NAME} v{VERSION} on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.", h_caption))

    # ---- Introduction -----------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Introduction", h_section))
    story.append(Paragraph(
        "Periodic error (PE) is a recurring tracking inaccuracy in equatorial "
        "telescope mounts. It originates primarily from imperfections in the "
        "right-ascension worm gear: each rotation of the worm imprints a "
        "characteristic sinusoidal drift on the right-ascension axis, with a "
        "period equal to the worm's revolution time (commonly a few minutes). "
        "Additional harmonics — at half, third, quarter the fundamental "
        "period — arise from manufacturing tolerances and load asymmetries.",
        h_body))
    story.append(Paragraph(
        "Quantifying PE matters because it directly limits unguided exposure "
        "length and, even when autoguiding, drives the residual error that "
        "the guider must correct. Measuring the fundamental period, amplitude "
        "and harmonic content informs mount tuning, gear lapping decisions, "
        "and the configuration of permanent periodic-error correction (PEC) "
        "tables.", h_body))

    # ---- Methodology ------------------------------------------------------
    story.append(Paragraph("Methodology", h_section))
    story.append(Paragraph(
        f"Each FITS frame in the selected window ({frame_range[0]}–"
        f"{frame_range[1]}) is plate-solved with {solver}. The right-"
        "ascension of the image center (FITS keyword <i>CRVAL1</i>) is "
        "extracted with its DATE-OBS timestamp, yielding a discrete RA-"
        "versus-time signal sampled at the capture cadence.", h_body))
    story.append(Paragraph(
        "The RA signal is then decomposed by Singular Spectrum Analysis "
        "(SSA), a model-free technique that builds a trajectory matrix from "
        "the signal, performs an SVD, and reconstructs the components from "
        "the leading eigentriples. Adjacent eigenpairs with similar "
        "eigenvalues are merged as sinusoid pairs, identifying the linear "
        "drift, the worm-period fundamental, and the dominant harmonics. "
        "An FFT applied to each grouped sinusoid yields its precise period "
        "and amplitude.", h_body))
    story.append(Paragraph(
        "The Singular Spectrum Analysis (SSA) function is based on "
        "Matlab code from Francisco Javier Alonso Sanchez, "
        "Department of Electronics and Electromecanical Engineering, "
        "Industrial Engineering School, University of "
        "Extremadura, Badajoz, Spain, and improved by François Auger, "
        "Nantes University, France: <i>The Sliding Singular Spectrum "
        "Analysis: A Data-Driven Nonstationary Signal Decomposition Tool</i>, "
        "IEEE TRANSACTIONS ON SIGNAL PROCESSING, vol. 66 no. 1, "
        "January 2018.", h_body))

    # ---- Results ----------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Results — drift analysis", h_section))
    story.append(Paragraph(
        f"Over the {drift_metrics['time_span_s']:.0f}-second capture, the "
        f"right-ascension drift accumulates "
        f"{drift_metrics['ra_pp_arcsec']:.1f} arcsec peak-to-peak with a "
        f"mean linear rate of "
        f"<b>{drift_metrics['ra_drift_arcsec_per_min']:.3f} arcsec/min</b>. "
        f"The declination drift is "
        f"{drift_metrics['dec_pp_arcsec']:.1f} arcsec peak-to-peak at "
        f"<b>{drift_metrics['dec_drift_arcsec_per_min']:.3f} arcsec/min</b> — "
        "small DEC drift is expected on a well-polar-aligned equatorial "
        "mount; large values point to polar-alignment error or differential "
        "flexure.", h_body))
    if "raw_ra_dec" in figures:
        story.append(_fig_to_image_flowable(figures["raw_ra_dec"],
                                            content_width, 14 * cm))
        story.append(Paragraph(
            "Figure 1 — Raw RA/DEC (telescope-reported) and CRVAL1/CRVAL2 "
            "(plate-solved) versus time.", h_caption))
    if "arcsec_drift" in figures:
        story.append(_fig_to_image_flowable(figures["arcsec_drift"],
                                            content_width, 14 * cm))
        story.append(Paragraph(
            "Figure 2 — RA and DEC error in arcseconds with the fitted "
            "linear drift overlaid.", h_caption))

    # ---- Maximum unguided exposure time ----------------------------------
    if max_exposure_metrics is not None:
        story.append(PageBreak())
        story.append(Paragraph("Maximum unguided exposure time", h_section))
        story.append(Paragraph(
            "The combined drift and periodic error sets a practical upper "
            "bound on how long a single sub-exposure can be before stars "
            "smear across more than one pixel. Three complementary estimates "
            "are reported below.", h_body))
        story.append(Paragraph(
            "<b>Linear drift only</b> assumes a constant drift rate over the "
            "exposure: <i>T = pixel / |slope|</i>. This is the long-capture "
            "limit, useful when the periodic error is small relative to the "
            "linear trend. <b>Instantaneous</b> uses a closed-form envelope "
            "of the drift plus the dominant periodic component: "
            "<i>T = pixel / (|slope| + 2π·σ / P)</i>, where σ is the RMS of "
            "the detrended signal and P is the worm-period fundamental "
            "from the SSA decomposition. Using σ rather than the numerical "
            "derivative filters out per-frame plate-solver scatter (a few "
            "tenths of an arcsec) that would otherwise dominate the metric, "
            "especially on the noisier DEC axis. <b>Sliding window</b> is "
            "the physically correct answer: the largest exposure length T "
            "such that, starting at any point in the capture, the star's "
            "total excursion within T stays under one pixel.", h_body))

        px = max_exposure_metrics["pixel_scale"]
        src_human = {
            "override":  "user override",
            "CD":        "CD matrix",
            "CDELT":     "CDELT keywords",
            "FOCALLEN":  "FOCALLEN + XPIXSZ",
        }.get(px["source"], px["source"])
        if abs(px["ra_arcsec_per_pixel"] - px["dec_arcsec_per_pixel"]) > 1e-3:
            scale_text = (f"{px['ra_arcsec_per_pixel']:.3f} arcsec/pixel (RA), "
                          f"{px['dec_arcsec_per_pixel']:.3f} arcsec/pixel (DEC)")
        else:
            scale_text = f"{px['ra_arcsec_per_pixel']:.3f} arcsec/pixel"
        story.append(Paragraph(
            f"Pixel scale used: <b>{scale_text}</b> (source: {src_human}). "
            "Because drift and periodic error are mount properties measured "
            "in arcseconds, the pixel-scale override field can also be used "
            "to simulate a different camera or binning factor: enter "
            "<i>206.265 × pixel_µm / focal_length_mm</i>, multiplied by the "
            "bin factor if applicable, to predict the tracking-limited "
            "exposure on a hypothetical setup.", h_body))

        binding_human = {
            "linear_s":         "linear drift",
            "instantaneous_s":  "instantaneous",
            "sliding_window_s": "sliding window",
        }
        exp_rows = [["Axis", "Linear drift", "Instantaneous",
                     "Sliding window", "Binding limit"]]
        for axis_label, axis_key in (("RA", "ra"), ("DEC", "dec")):
            m = max_exposure_metrics[axis_key]
            binding = (binding_human.get(m.get("binding"), "—")
                       if m.get("binding") else "—")
            exp_rows.append([
                axis_label,
                _format_seconds(m["linear_s"]),
                _format_seconds(m["instantaneous_s"]),
                _format_seconds(m["sliding_window_s"]),
                binding,
            ])
        exp_table = Table(exp_rows, hAlign="LEFT",
                          colWidths=[2 * cm, 3.5 * cm, 3.5 * cm,
                                     3.5 * cm, 3 * cm])
        exp_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2e2e2e")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN",       (1, 1), (-1, -1), "RIGHT"),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#ccc")),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ]))
        story.append(exp_table)
        story.append(Spacer(0, 0.4 * cm))

        ra_m = max_exposure_metrics["ra"]
        if ra_m.get("sliding_window_s") is not None:
            if ra_m.get("saturated"):
                story.append(Paragraph(
                    "Over the analysed window, the cumulative RA motion never "
                    "exceeded one pixel — exposure length on this capture is "
                    "not bounded by tracking error.", h_body))
            else:
                story.append(Paragraph(
                    f"On the RA axis, single sub-exposures up to "
                    f"<b>{_format_seconds(ra_m['sliding_window_s'])}</b> "
                    "should keep stars within one pixel of motion (sliding-"
                    "window criterion). Beyond that length, the drift+PE "
                    "envelope inside one exposure exceeds the pixel scale.",
                    h_body))

    story.append(PageBreak())
    story.append(Paragraph("Results — Singular Spectrum Analysis", h_section))
    story.append(Paragraph(
        f"SSA was applied to the RA error signal over "
        f"{ssa_metrics['n_frames']} samples (mean sampling period "
        f"{ssa_metrics['sampling_period_s']:.2f} s), extracting "
        f"{SSA_NB_EIGEN} eigencomponents grouped into "
        f"{SSA_GROUP_ORDER + 1} dominant components (the linear drift "
        "plus the fundamental and four harmonics).", h_body))
    if "singular_spectrum" in figures:
        story.append(_fig_to_image_flowable(figures["singular_spectrum"],
                                            content_width, 14 * cm))
        story.append(Paragraph(
            "Figure 3 — Top: raw RA error. Bottom: normalized singular "
            "spectrum (eigenvalue energy fractions) — a steep drop after "
            "the leading eigentriples confirms that the signal is dominated "
            "by a few periodic components.", h_caption))

    story.append(PageBreak())
    story.append(Paragraph("Results — periodic components", h_section))
    # Per-component metrics table
    comp_rows = [["Component", "Period (s)", "RMS (arcsec)",
                  "Min (arcsec)", "Max (arcsec)"]]
    for c in ssa_metrics["components"]:
        comp_rows.append([
            c["name"],
            f"{c['period_s']:.2f}" if c["period_s"] else "—",
            f"{c['rms_arcsec']:.2f}",
            f"{c['min_arcsec']:+.2f}",
            f"{c['max_arcsec']:+.2f}",
        ])
    comp_table = Table(comp_rows, hAlign="LEFT",
                       colWidths=[3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm])
    comp_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2e2e2e")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",       (1, 1), (-1, -1), "RIGHT"),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#ccc")),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
    ]))
    story.append(comp_table)
    story.append(Spacer(0, 0.4 * cm))
    if "stacked_components" in figures:
        story.append(_fig_to_image_flowable(figures["stacked_components"],
                                            content_width, 22 * cm))
        story.append(Paragraph(
            "Figure 4 — Stacked SSA components: linear drift plus the "
            "fundamental and four harmonics, each annotated with its peak "
            "amplitude.", h_caption))
    if "fft_all_components" in figures:
        story.append(PageBreak())
        story.append(_fig_to_image_flowable(figures["fft_all_components"],
                                            content_width, 22 * cm))
        story.append(Paragraph(
            "Figure 5 — FFT of the RA error (without the main drift) with the SSA-"
            "identified periods marked. Sharp peaks at the fundamental "
            "and integer fractions confirm the harmonic structure of the "
            "worm gear's error signature.", h_caption))

    story.append(PageBreak())
    story.append(Paragraph("Results — signal reconstruction", h_section))
    story.append(Paragraph(
        f"Summing the {SSA_GROUP_ORDER + 1} grouped SSA components yields "
        f"a reconstruction of the original RA error with a residual RMS of "
        f"<b>{ssa_metrics['rmse_arcsec']:.2f} arcsec</b>. A low RMSE "
        "indicates that the SSA retrieves the dominant sinusoidale components "
        "and drift. This reconstructed signal can be used for a "
        "Periodic Error Correction (PEC). What's left after periodic components "
        "have been removed is the unmodeled components, noise and error due "
        "to the SSA algorithm.", h_body))
    if "reconstruction" in figures:
        story.append(_fig_to_image_flowable(figures["reconstruction"],
                                            content_width, 14 * cm))
        story.append(Paragraph(
            "Figure 6 — Top: original RA error (blue) and SSA reconstruction "
            "(orange). Bottom: residual error (original minus reconstruction).",
            h_caption))

    # ---- Discussion -------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Discussion", h_section))
    fund_period = ssa_metrics["fundamental_period_s"]
    fund_comp = next((c for c in ssa_metrics["components"]
                      if c["name"] == "Fond"), None)
    fund_pp = ((fund_comp["max_arcsec"] - fund_comp["min_arcsec"])
               if fund_comp else 0.0)
    drift_rate = drift_metrics["ra_drift_arcsec_per_min"]
    fund_text = (
        f"The dominant periodic component has a period of "
        f"<b>{fund_period:.1f} seconds</b> "
        f"({fund_period / 60:.2f} min) with a peak-to-peak amplitude of "
        f"<b>{fund_pp:.1f} arcsec</b>. "
    ) if fund_comp else ""
    story.append(Paragraph(
        fund_text +
        f"The RA linear drift over the capture averages "
        f"{drift_rate:+.2f} arcsec/min; values within ±1 arcsec/min are "
        "typically consistent with a small polar-alignment residual, while "
        "larger drifts warrant a polar-alignment check before further mount "
        "tuning.", h_body))
    harmonic_text = []
    for c in ssa_metrics["components"]:
        if c["name"].startswith("H") and c["period_s"]:
            harmonic_text.append(
                f"<b>{c['name']}</b> at {c['period_s']:.1f}s "
                f"(rms {c['rms_arcsec']:.2f}″)")
    if harmonic_text:
        story.append(Paragraph(
            "Identified harmonics: " + "; ".join(harmonic_text) + ". "
            "Harmonics at integer fractions of the fundamental period are the "
            "fingerprint of worm-gear manufacturing tolerances and can be "
            "compensated by a sufficiently dense PEC table.", h_body))
    story.append(Paragraph(
        f"With a reconstruction RMSE of {ssa_metrics['rmse_arcsec']:.2f} "
        "arcsec, the residual error after removing drift and the modelled "
        f"{SSA_GROUP_ORDER + 1}-component periodic structure represents the "
        "unmodeled components, noise and error due to the SSA algorithm."
        "PEC can address the periodic part; "
        "reducing the residual requires improvements to "
        "the optical train rigidity, autoguiding, or the observing site.",
        h_body))
    if max_exposure_metrics is not None and fund_comp \
            and fund_comp.get("period_s") and fund_pp > 0:
        amp = fund_pp / 2.0
        ra_m = max_exposure_metrics["ra"]
        peak_slew = ra_m.get("peak_slew_arcsec_per_s") or 0.0
        binding_human = {"linear_s":         "linear drift",
                         "instantaneous_s":  "instantaneous envelope",
                         "sliding_window_s": "sliding window"}
        binding_label = binding_human.get(ra_m.get("binding"), "the table above")
        story.append(Paragraph(
            f"The fundamental component (amplitude {amp:.1f}″ over a "
            f"{fund_comp['period_s']:.0f}-second worm period) drives an RA "
            f"slew envelope of <b>{peak_slew:.2f} arcsec/s</b> "
            f"(|drift| + 2π·σ/P, with σ the RMS of the detrended signal). "
            f"At the pixel scale used here this dominates whenever the "
            f"linear drift is small. The {binding_label} estimate is the "
            "binding limit on the RA axis for this capture.", h_body))
    story.append(Paragraph(
        "<b>Caveats.</b> The analysis is sensitive to the chosen frame "
        "window and sampling time of the image capture. "
        "Windows shorter than two worm cycles may misidentify the "
        "fundamental period. Plate-solver scatter (typically a few tenths "
        "of an arcsecond per frame) sets a noise floor on the residual "
        "RMSE that no amount of mount improvement can lower."
        "In order to largely comply with the Shannon criterion "
        "and to represent a sinusoid by at least 20 to 40 captures, "
        "use low sampling time value such as 1 or 2s in order to capture "
        "the main harmonics of the RA error.", h_body))

    doc.build(story)
    log(f"[PE] PDF report saved to {out_path}", color=LogColor.GREEN)


# =============================================================================
# Top-level driver
# =============================================================================
def compute_periodic_error(fits_files, first_idx, last_idx,
                           use_astap, astap_cli, siril,
                           do_plate_solve, log,
                           save_pdf=False, report_title=None,
                           pixel_scale_override=None):
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

    drift_result = _plot_plate_solve_data(sequence_df, log)
    ssa_result = _ssa_analysis(sequence_df, log)

    # Maximum unguided exposure time (drift + PE under one pixel).
    # Uses the SSA-reconstructed RA signal (smoother, free of fit noise) and
    # the raw DEC drift from _plot_plate_solve_data. The mount's worm
    # period (from the RA SSA fundamental) is shared with the DEC axis
    # so the instantaneous estimate uses a consistent periodic envelope.
    max_exposure_metrics = None
    pixel_scale = _compute_pixel_scale(sequence_df, pixel_scale_override, log)
    if pixel_scale is not None:
        t_ra = ssa_result["arrays"]["time_axis"]
        ra_signal = ssa_result["arrays"]["reconstructed_ra"]
        t_dec = _time_axis(sequence_df)
        dec_signal = sequence_df["Delta_CRVAL2"].to_numpy(dtype=float) * 3600.0
        period_s = ssa_result["metrics"].get("fundamental_period_s")
        max_exposure_metrics = {
            "pixel_scale": pixel_scale,
            "ra":  _max_exposure_time(t_ra, ra_signal,
                                      pixel_scale["ra_arcsec_per_pixel"],
                                      period_s),
            "dec": _max_exposure_time(t_dec, dec_signal,
                                      pixel_scale["dec_arcsec_per_pixel"],
                                      period_s),
        }
        for axis_label, axis_key in (("RA", "ra"), ("DEC", "dec")):
            m = max_exposure_metrics[axis_key]
            if m.get("binding") is None:
                continue
            log(f"[PE] Max exposure ({axis_label}): "
                f"linear {_format_seconds(m['linear_s'])}, "
                f"instantaneous {_format_seconds(m['instantaneous_s'])}, "
                f"sliding-window {_format_seconds(m['sliding_window_s'])} "
                f"(binding: {m['binding']})",
                color=LogColor.GREEN)

    log("[PE] Analysis complete.", color=LogColor.GREEN)

    if save_pdf:
        fits_folder = selected[0].parent
        capture_metadata = _read_capture_metadata(selected[0])
        # Title precedence: explicit GUI override > OBJECT keyword > folder name
        effective_title = (report_title or "").strip()
        if not effective_title:
            effective_title = (capture_metadata.get("object")
                               or fits_folder.name)
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = _sanitize_filename(fits_folder.name)
        out_path = fits_folder / f"{safe_name}_PE_report_{stamp}.pdf"
        log(f"[PE] Building PDF report at {out_path}", color=LogColor.BLUE)
        merged_figures = {**drift_result["figures"], **ssa_result["figures"]}
        _build_pdf_report(
            out_path,
            title=effective_title,
            capture_metadata=capture_metadata,
            solver="ASTAP" if use_astap else "Siril (GAIA)",
            frame_range=(first_idx, last_idx),
            drift_metrics=drift_result["metrics"],
            ssa_metrics=ssa_result["metrics"],
            max_exposure_metrics=max_exposure_metrics,
            figures=merged_figures,
            log=log,
        )


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

        # Discover FITS files: case-insensitive on suffix (Linux ext4 cares),
        # accepting .fits/.fit/.fts, and skipping dotfiles (e.g. macOS
        # '._*.fits' AppleDouble metadata — see `dot_clean` to clean disk).
        self.fits_files = sorted(
            p for p in self.working_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in FITS_SUFFIXES
            and not p.name.startswith(".")
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
        self.save_pdf_var        = tk.BooleanVar(self.root, value=False)
        # String (not Double) so empty = auto-detect from headers.
        self.pixel_scale_var     = tk.StringVar(self.root, value="")

        # Auto-populate the report title from the first FITS file's OBJECT
        # keyword, falling back to the working-directory name. The user can
        # override it in the Report frame's text entry before clicking Process.
        try:
            self._capture_metadata = _read_capture_metadata(self.fits_files[0])
        except Exception:
            self._capture_metadata = {}
        default_title = (self._capture_metadata.get("object")
                         or self.working_dir.name)
        self.report_title_var = tk.StringVar(self.root, value=default_title)

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
            hint=("External ASTAP CLI — caches .wcs sidecars in "
                  "PlateSolveAstap/ so re-runs can skip plate-solving."),
        )
        self._build_solver_row(
            tools_frame, "SIRIL (GAIA)", self.PLATE_SOLVER_SIRIL,
            self.siril_path_var, self._browse_siril_exe,
            tooltip=("Plate-solve via Siril's built-in GAIA solver. "
                     "Works without ASTAP installed, but re-solves every "
                     "run — 'Run plate solve' must stay enabled."),
            hint=("Siril's built-in GAIA solver — no ASTAP needed, but "
                  "re-solves every run."),
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
            text=(f"(valid range 1..{self.nb_fits}; "
                  f"window must contain ≥ {SSA_MIN_FRAMES} frames for SSA)"),
            foreground="gray",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 4))

        plate_solve_cb = ttk.Checkbutton(
            proc_frame, text="Run plate solve", variable=self.do_plate_solve_var,
        )
        plate_solve_cb.grid(row=4, column=0, columnspan=2, sticky="w",
                            padx=5, pady=(8, 0))
        tksiril.create_tooltip(
            plate_solve_cb,
            "Disable to reuse plate-solving results from a previous run.",
        )
        ttk.Label(
            proc_frame,
            text="(disable to reuse cached results from a previous run)",
            foreground="gray",
        ).grid(row=5, column=0, columnspan=2, sticky="w",
               padx=(28, 5), pady=(0, 4))

        ttk.Label(proc_frame, text="Pixel scale (arcsec/pixel):").grid(
            row=6, column=0, sticky="e", padx=5, pady=(8, 2))
        pixel_scale_entry = ttk.Entry(
            proc_frame, textvariable=self.pixel_scale_var, width=10,
        )
        pixel_scale_entry.grid(row=6, column=1, sticky="w", padx=5, pady=(8, 2))
        tksiril.create_tooltip(
            pixel_scale_entry,
            "Leave empty to auto-detect from the plate solve "
            "(CD matrix / CDELT / FOCALLEN+XPIXSZ). Override only if the "
            "FITS headers are wrong. Used by the 'Maximum unguided exposure "
            "time' section of the PDF report.",
        )
        ttk.Label(
            proc_frame,
            text="(leave empty to auto-detect from the plate solve)",
            foreground="gray",
        ).grid(row=7, column=0, columnspan=2, sticky="w", padx=5, pady=(0, 4))

        # Report frame (title + save PDF checkbox)
        report_frame = ttk.LabelFrame(main_frame, text="Report", padding=10)
        report_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(report_frame, text="Title:").grid(
            row=0, column=0, sticky="e", padx=5, pady=2)
        title_entry = ttk.Entry(
            report_frame, textvariable=self.report_title_var, width=50,
        )
        title_entry.grid(row=0, column=1, sticky="we", padx=5, pady=2)
        report_frame.columnconfigure(1, weight=1)
        tksiril.create_tooltip(
            title_entry,
            "Title for the PDF report's cover page. Auto-populated from the "
            "FITS OBJECT keyword (or the FITS folder name) — edit as needed.",
        )
        ttk.Label(
            report_frame,
            text="(auto-filled from the FITS OBJECT keyword — edit as needed)",
            foreground="gray",
        ).grid(row=1, column=1, sticky="w", padx=5, pady=(0, 4))

        save_pdf_cb = ttk.Checkbutton(
            report_frame, text="Save PDF report",
            variable=self.save_pdf_var,
        )
        save_pdf_cb.grid(row=2, column=0, columnspan=2, sticky="w",
                         padx=5, pady=(6, 0))
        tksiril.create_tooltip(
            save_pdf_cb,
            "When enabled, a multi-page PDF report (cover, methodology, "
            "results with embedded figures, discussion) is written next to "
            "the FITS folder after analysis completes.",
        )
        ttk.Label(
            report_frame,
            text="(writes a multi-page PDF next to the FITS folder "
                 "after analysis)",
            foreground="gray",
        ).grid(row=3, column=0, columnspan=2, sticky="w",
               padx=(28, 5), pady=(0, 4))

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
                          tooltip=None, hint=None):
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
        if hint:
            ttk.Label(parent, text=hint, foreground="gray").pack(
                fill=tk.X, padx=(118, 5), pady=(0, 4),
            )

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

        pixel_scale_text = self.pixel_scale_var.get().strip()
        pixel_scale_override = None
        if pixel_scale_text:
            try:
                pixel_scale_override = float(pixel_scale_text.replace(",", "."))
                if pixel_scale_override <= 0:
                    raise ValueError("must be > 0")
            except ValueError:
                msg = (f"Pixel scale must be a positive number "
                       f"(got '{pixel_scale_text}'). Leave empty to "
                       "auto-detect from the plate solve.")
                messagebox.showwarning("Invalid pixel scale", msg)
                self.siril.log(f"[PE] {msg}", color=LogColor.RED)
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
                save_pdf=self.save_pdf_var.get(),
                report_title=self.report_title_var.get(),
                pixel_scale_override=pixel_scale_override,
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
