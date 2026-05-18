"""

"""
# importing packages 
import os
import shutil
import glob
import csv
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from astropy.time import Time
#from unidecode import unidecode

from numpy import linalg as LA

import sirilpy as s
from sirilpy import LogColor, SirilInterface
import tkinter as tk
from tkinter import ttk, messagebox
if s.check_module_version(">=0.6.47") and sys.platform.startswith("linux"):
    import sirilpy.tkfilebrowser as filedialog
else:
    from tkinter import filedialog
import threading

# =============================================================================
# Constants
# =============================================================================
APP_NAME         = "Automatic Periodic Error Computation"
AUTHORS_NAME     = "Mickaël HILAIRET and Gilles MORAIN"
VERSION          = 0.3

# Ensure a usable Siril version
required_version = "1.3.6"

DEFAULT_ASTAP_CLI = "/Applications/ASTAP.app/Contents/MacOS/astap"
DEFAULT_SIRIL_CLI = "/Applications/Siril.app/Contents/MacOS/siril-cli"


#------------------------------------------------------------------------------
def exit_program(error):
  print("Error = {:.1f}".format(error))
  print("Program exit with error ...")
  f.close()
  sys.exit(0)


#------------------------------------------------------------------------------
def msg_warning(warning):
  print("Warning = {:.1f}".format(warning))


#------------------------------------------------------------------------------    
def close():
  print("Close script ...")
  f.close()
  sys.exit(0)

#------------------------------------------------------------------------------ 
def set_siril_tool():
   astap_plate_solve.set(0)


#------------------------------------------------------------------------------ 
def set_astap_tool():
   siril_plate_solve.set(0)


#------------------------------------------------------------------------------ 
def select_astap_exe():
  # Display the dialog for browsing files.
  store_filename = input_dir_var.get()
  print(store_filename)
  filename = filedialog.askopenfilename()
  
  if filename:
    # Print the selected file path.
    print("Selected ASTAP exe file : ", filename)
    # Print filename in the dialog box
    input_dir_var.set(filename)
    # Store path in the config_PE.txt file
    with open(config_PE_file,'w') as f:
      store_siril_path = DEFAULT_SIRIL_CLI
      f.write("{},{}\n".format("astap", filename))
      f.write("{},{}".format("siril", store_siril_path))
      f.close()
  else:
    print("No ASTAP exe file selected.")
    # Print filename in the dialog box
    input_dir_var.set(store_filename)


#------------------------------------------------------------------------------    
def process_PE():
  #------------------------------------------------------------------------------
  print("Check index of fits files")
  # Begin index
  t0 = begin_index.get()
  print(t0)
  # End index
  t_end = end_index.get()
  print(t_end)

  if (t0 < 1) | (t0 >= NbFits) :
    print("Bad begin index of the fits files")
    msg_warning(1.1)
  elif (t_end < 2) | (t_end > NbFits):
    print("Bad end index of the fits files")    
    msg_warning(1.2)
  elif (t0 >= t_end) :
    print("Bad begin/end index of the fits files")
    msg_warning(1.3)
  else :
     print("Check index OK")

  #------------------------------------------------------------------------------     
  print("Check platesolve on/off")
  PlateSolveOnOff = do_plate_solve.get()
  if(PlateSolveOnOff == 1):
    print("Plate solve on")
  elif(PlateSolveOnOff == 0):
    print("Plate solve off")


#------------------------------------------------------------------------------ 
# Connect to Siril
siril = SirilInterface()
if not siril.connect():
    raise RuntimeError("Failed to connect to Siril.")

# Check Siril version
try:
    siril.cmd("requires", required_version)
except Exception:
    raise RuntimeError(f"This script requires Siril version {required_version} or later!")

# Get working fits files directory
WorkingDirectory = os.getcwd()
#print(WorkingDirectory) 
 
# Get script file directory 
ScriptDirectory = Path(__file__).parent
#print(ScriptDirectory)

# Collect fits files name and the name of the fits file directory
print(f"Collect *fits files in {WorkingDirectory}")
listing_fits = glob.glob(WorkingDirectory + '/*.fits') 
NbFits        = len(listing_fits)
print(f"Collect {NbFits} fits files")
if NbFits == 0:
  root = tk.Tk()
  root.withdraw()
  root.destroy()
  exit_program(1)

# Read config_PE.txt file
config_PE_file = "{}/{}".format(ScriptDirectory, "config_PE.txt")
#print(config_file)
  
# Check if the config_PE.txt file exists
if os.path.exists(config_PE_file):
  print("The config_PE.txt file exists.")

  with open(config_PE_file,'r') as f:         # Open config file as CSV
       tab_tool_path = []
       fid = csv.reader(f)                   # Load data of the CSV file
       for row in fid:                      
           tab_tool_path.append(row)
    
  ASTAP_CLI = tab_tool_path[0][1]
  SIRIL_CLI = tab_tool_path[1][1]
    
#  del f, fid, row, tab_tool_path
else:
  print("The config_PE.txt file does not exist.")
  print("Create a config_PE.txt file")

  ASTAP_CLI = DEFAULT_ASTAP_CLI
  SIRIL_CLI = DEFAULT_SIRIL_CLI
  
  # Default configuration
  with open(config_PE_file,'w') as f:
   f.write("{},{}\n".format("astap", ASTAP_CLI))
   f.write("{},{}".format("siril", SIRIL_CLI))

f.close()
print("ASTAP directory : {}".format(ASTAP_CLI))
print("SIRIL directory : {}".format(SIRIL_CLI))

#------------------------------------------------------------------------------ 
# Initialize GUI
root = tk.Tk()
root.attributes("-topmost", True)
style = ttk.Style()
style.theme_use('clam')  # 'alt', 'default', or 'classic' also work

bg_color = "#2e2e2e"
fg_color = "white"

style.configure('.', background=bg_color, foreground=fg_color)
style.configure('TLabel', background=bg_color, foreground=fg_color)
style.configure('TButton', background=bg_color, foreground=fg_color)
style.configure('TEntry', fieldbackground="#3c3c3c", foreground=fg_color)

root.title(APP_NAME)

# GUI Below   
# Main Frame
main_frame = ttk.Frame(root, style="Dark.TFrame", padding=10)
main_frame.grid(row=0, column=0, sticky="nsew")

# Tools frame
tools_frame = ttk.Frame(main_frame)
tools_frame.grid(row=2, column=0, columnspan=3, pady=10)

# Process frame
process_frame = ttk.Frame(main_frame)
process_frame.grid(row=5, column=0, columnspan=5, pady=10)

# ASTAP Directory
astap_plate_solve = tk.IntVar(value=1)
ttk.Checkbutton(main_frame,text='ASTAP', variable=astap_plate_solve, command=set_astap_tool).grid(row=1, column=0)
input_dir_var = tk.StringVar(value=ASTAP_CLI)
input_entry = ttk.Entry(main_frame, textvariable=input_dir_var, width=50)
input_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
#ttk.Button(main_frame, text="Browse...", command=lambda: input_dir_var.set(filedialog.askopenfilename(initialdir=DEFAULT_ASTAP_CLI))).grid(row=1, column=2, padx=5)
ttk.Button(main_frame, text="Browse...", command=select_astap_exe).grid(row=1, column=2, padx=5)

# GAIA under SIRIL
#ttk.Label(main_frame, text="GAIA (SIRIL)").grid(row=2, column=0, sticky="e", padx=10, pady=5)
siril_plate_solve = tk.IntVar(value=0)
ttk.Checkbutton(main_frame,text='SIRIL  ', variable=siril_plate_solve, command=set_siril_tool).grid(row=2, column=0)


# First row : Print number of fits files
ttk.Label(process_frame, text="Number of fits files").grid(row=1, column=0, sticky="e", padx=(0, 5))
ttk.Label(process_frame, text=NbFits, justify="left").grid(row=1, column=1, sticky="w", padx=(0, 5))

# Second row: Begin index
begin_index = tk.IntVar(value=1)
ttk.Label(process_frame, text="Begin index").grid(row=2, column=0, sticky="e", padx=(0, 5), pady=(5, 0))
ttk.Entry(process_frame, textvariable=begin_index, width=10).grid(row=2, column=1, sticky="w", padx=(0, 15), pady=(5, 0))

# Third row: End index
end_index = tk.IntVar(value=NbFits)
ttk.Label(process_frame, text="End index").grid(row=3, column=0, sticky="e", padx=(0, 5), pady=(5, 0))
ttk.Entry(process_frame, textvariable=end_index, width=10).grid(row=3, column=1, sticky="w", pady=(5, 0))

# Plate solve on or off
do_plate_solve = tk.IntVar(value=1)
ttk.Checkbutton(process_frame,text='Plate solve on/off', variable=do_plate_solve).grid(row=4, column=0)

# Buttons
ttk.Button(process_frame, text="Process", command=process_PE).grid(row=5, column=0, padx=10, pady=(10, 10))
ttk.Button(process_frame, text="Close", command=close).grid(row=5, column=1, padx=10, pady=(10, 10), sticky="w")

# Handle exit
root.protocol("WM_DELETE_WINDOW", root.destroy)
root.mainloop()