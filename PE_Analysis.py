"""
Automatic Periodic Error Computation - Siril Python script.

GUI-based Siril script that computes the periodic error of an equatorial
mount from a time-series of FITS frames. The script collects FITS files
from Siril's working directory, lets the user pick the plate-solving
engine (ASTAP or Siril/GAIA) and the frame index range, then runs the
analysis.

Follows the official Siril Python scripting template:
https://siril.readthedocs.io/en/stable/scripts/python_gui_template
"""

# Core module imports
import csv
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import sirilpy as s
from sirilpy import tksiril, LogColor, SirilError, SirilConnectionError

# Ensure non-core modules are available in Siril's venv before importing them
s.ensure_installed("ttkthemes")
# When the PE algorithm is wired up, also declare its dependencies here, e.g.:
# s.ensure_installed("numpy", "pandas", "matplotlib", "astropy")

from ttkthemes import ThemedTk  # noqa: E402  (must follow ensure_installed)


# =============================================================================
# Constants
# =============================================================================
APP_NAME       = "Automatic Periodic Error Computation"
AUTHORS        = "Mickaël HILAIRET and Gilles MORAIN"
VERSION        = "0.4.0"
REQUIRED_SIRIL = "1.3.6"

DEFAULT_ASTAP_CLI = "/Applications/ASTAP.app/Contents/MacOS/astap"
DEFAULT_SIRIL_CLI = "/Applications/Siril.app/Contents/MacOS/siril-cli"

SCRIPT_DIR  = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config_PE.txt"


# =============================================================================
# Config file helpers
# =============================================================================
def load_config():
    """Read ASTAP and Siril CLI paths from config_PE.txt (or return defaults)."""
    paths = {"astap": DEFAULT_ASTAP_CLI, "siril": DEFAULT_SIRIL_CLI}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, newline="") as fp:
            for row in csv.reader(fp):
                if len(row) == 2 and row[0] in paths:
                    paths[row[0]] = row[1]
    return paths


def save_config(astap_path, siril_path):
    """Persist the ASTAP and Siril CLI paths to config_PE.txt."""
    with open(CONFIG_FILE, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["astap", astap_path])
        writer.writerow(["siril", siril_path])


# =============================================================================
# PE algorithm placeholder
# =============================================================================
def compute_periodic_error(fits_files, first_idx, last_idx,
                           use_astap, astap_cli, do_plate_solve):
    """Compute the periodic error from the selected FITS frames.

    TODO: plug in the existing macOS / Windows PE computation algorithm here.
    The function should:
      - optionally plate-solve each frame (ASTAP if use_astap else Siril/GAIA)
      - extract RA/Dec for each frame and convert to drift in arcseconds
      - fit the periodic drift over time
      - emit a plot and a CSV of the residuals
    """
    raise NotImplementedError("PE algorithm not yet wired up.")


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

        self.fits_files = sorted(self.working_dir.glob("*.fits"))
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
        )
        self._build_solver_row(
            tools_frame, "SIRIL (GAIA)", self.PLATE_SOLVER_SIRIL,
            self.siril_path_var, self._browse_siril_exe,
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
        tksiril.create_tooltip(begin_entry, f"First frame index (1..{self.nb_fits - 1}).")

        ttk.Label(proc_frame, text="End index:").grid(
            row=2, column=0, sticky="e", padx=5, pady=2)
        end_entry = ttk.Entry(proc_frame, textvariable=self.end_index_var, width=10)
        end_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        tksiril.create_tooltip(end_entry, f"Last frame index (2..{self.nb_fits}).")

        plate_solve_cb = ttk.Checkbutton(
            proc_frame, text="Run plate solve", variable=self.do_plate_solve_var,
        )
        plate_solve_cb.grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=(8, 0))
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

    def _build_solver_row(self, parent, label, value, path_var, browse_cmd):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(
            row, text=label, variable=self.solver_var, value=value, width=14,
        ).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=path_var, width=50).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True,
        )
        ttk.Button(row, text="Browse...", command=browse_cmd).pack(side=tk.LEFT)

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
                self.astap_path_var.get(), self.do_plate_solve_var.get(),
            )
        except NotImplementedError as exc:
            self.siril.log(f"[PE] {exc}", color=LogColor.RED)
            messagebox.showinfo(
                "PE algorithm",
                "The PE computation algorithm has not been wired up yet.\n"
                "Plug it into compute_periodic_error() in this script.",
            )
        except SirilError as exc:
            self.siril.log(f"[PE] Siril error: {exc}", color=LogColor.RED)
            messagebox.showerror("Siril error", str(exc))

    def _on_close(self):
        try:
            self.siril.disconnect()
        except SirilError:
            pass
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
