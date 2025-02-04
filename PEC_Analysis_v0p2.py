############################################
#
# January 2024
# (C) Mickaël HILAIRET, LS2N/Ecole Centrale de Nantes, France
#     and Gilles MORAIN, France
#
# Distributed under a Creative Commons Attribution ǀ 4.0
# International licence CC BY-NC-SA 4.0
#
# Computation of the Periodic Error of a mount
#
# Version 0.1
#   Plate solve using ASTAP 
#
#   Config.txt file
#     Add the path of ASTAP tool
#     Example : 
#       astap,D:\ProgramOnD\astap\astap.exe
#       ps3,D:\ProgramOnD\Platesolve3_80\PlateSolve3.80.exe
#
#   dir_fits_file.txt file
#     Add the path of the fits files to read
#     tool,astap     : to using astap as plate solve tool
#     platesolve_on  : ask a plate solve
#     platesolve_off : no plate solve (because it has been yet done) and just
#                      a plot of the signal is ask
#
#     Example : 
#       dir_fits_file,D:\Documents\PEC_Analyse\File_Under_Test
#       tool,astap
#       platesolve_on, 
#
#
#  Version 0.2
#    * Add a decomposition of the signal in principal components in order to have
#    an estimate of the frequency and amplitude of the main components and drift.
#    The Singular Spectrum Analysis (SSA) function is based on Matlab code from 
#    Francisco Javier Alonso Sanchez, Departament of Electronics and Electromecanical Engineering
#    Industrial Engineering School, University of Extremadura, Badajoz, Spain
#    and improved by François Auger, Nantes University, France
#    The Sliding Singular Spectrum Analysis: A Data-Driven Nonstationary Signal
#    Decomposition Tool, IEEE TRANSACTIONS ON SIGNAL PROCESSING, 
#    VOL. 66, NO. 1, JANUARY 1, 2018
#  
#  To be done :
#    Plate solve using PS3
#    Plate solve for SIRIL tool
#    Add a interface
#    Add zoom in figures
#    Use can select FFT with linear or semilox, semilogy, loglog, etc
#
#  To be improved : 
#    Fig.1 is not well finsih. The RA/CRVAL1 and DEC/CRVAL2 are in degree and 
#    not in "H:M:S" (hms) and "°:M:S" (dms)
#

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
from unidecode import unidecode

from numpy import linalg as LA

version = 0.2

# User option to define the end of the signal if necessary
end_sequence_df = 70 #1200 # 70
# 1 : Plot FFT of all the main components of the signal, 0 : No plot
Plot_FFT = 1


fits_extension = 'fits'
error = 0

#------------------------------------------------------------------------------
def exit_program(error):
  print("Error = {:.1f}".format(error))
  print("Exiting the program...")
  sys.exit(0)


#------------------------------------------------------------------------------
def read_config_info():
  #----------------------------------------------------------------------------
  print("Read config.txt file")
  
  # Obtain local path
  LocalRootFolder = os.getcwd()
  config_file = LocalRootFolder + '\\config.txt'
  
  with open(config_file,'r') as f:         # Open config file as CSV
      tab_tool_path = []
      fid = csv.reader(f)                   # Load data of the CSV file
      for row in fid:                      
          tab_tool_path.append(row)
  
  # print(tab_tool_path[0][1])
  ASTAP_CLI     = tab_tool_path[0][1]
  PS3_Directory = tab_tool_path[1][1]
  
  del f, fid, row, tab_tool_path

  #----------------------------------------------------------------------------
  print("Read dir_fits_file.txt file")
  
  # Obtain path of the fit to analyse, the tool and if the astrometry
  # need to be executed
  dir_fits_file = LocalRootFolder + '\\dir_fits_file.txt'
  
  with open(dir_fits_file,'r') as f:         # Open config file as CSV
      tab_info = []
      fid = csv.reader(f)                   # Load data of the CSV file
      for row in fid:                      
          tab_info.append(row)
  
  # print(tab_tool_path[0][1])
  FitsFileDirectory  = tab_info[0][1]
  PlateSolveTool     = tab_info[1][1]
  PlateSolveOnOff    = tab_info[2][0]
  
  
  ExecutePlateSolve = 0
  if(PlateSolveOnOff == "platesolve_on"):
    ExecutePlateSolve = 1
  elif(PlateSolveOnOff == "platesolve_off"):
    ExecutePlateSolve = 0
  else:
    exit_program(1.5)

  #----------------------------------------------------------------------------
  print("Test Plate Solve Tool")
  
  PlateSolveToolNumber = 0
  if(PlateSolveTool == "astap"):
    PlateSolveToolFolder = ASTAP_CLI
    PlateSolveToolNumber = 1
    print("ASTAP tool")
    if(ASTAP_CLI == ""):
      exit_program(1.1)
  elif(PlateSolveTool == "ps3"):
    PlateSolveToolFolder = PS3_Directory
    PlateSolveToolNumber = 2
    print("PS3 tool")
    if(PS3_Directory == ""):
      exit_program(1.2)
  elif(PlateSolveTool == "ps3_auto"):
    PlateSolveToolFolder = PS3_Directory
    PlateSolveToolNumber = 3
    print("PS3 tool automatic")
    if(PS3_Directory == ""):
      exit_program(1.3) 
  else:
    exit_program(1.4)
    
  #------------------------------------------------------------------------------
  print("End fits_analysis")
  return FitsFileDirectory, PlateSolveToolFolder, PlateSolveToolNumber, ExecutePlateSolve


#------------------------------------------------------------------------------
def collect_fits(FitsFolder):      
  #----------------------------------------------------------------------------
  # Collect fits files name and the name of the fits file directory
  print("Collect *fits file") 
  listing_fits = glob.glob(FitsFolder + '\\*.fits') 
  NbFits        = len(listing_fits)
  if(NbFits == 0):
    exit_program(2)

  
  #------------------------------------------------------------------------------
  print("End collect *fits files")
  return listing_fits, NbFits


#------------------------------------------------------------------------------
def Plate_Solve_Computation(FitsFolder,listing_fits,NbFits,PlateSolveToolFolder,PlateSolveToolNumber):
  #-------------------------------------------------------------------------
  # Automatic Plate Solve with : ASTAP 
  
  plate_solve_results = []
  
  # Start ASTAP
  if(PlateSolveToolNumber == 1):
    print('Start ASTAP plate solve');
  
    # Clear result repository
    PlateSolveDirResults = FitsFolder + "\\PlateSolveAstap"
    test = os.path.isdir(PlateSolveDirResults)
    if(test):
      try:
          shutil.rmtree(PlateSolveDirResults)
          print(f"{PlateSolveDirResults} and all its contents have been deleted")
      except FileNotFoundError:
          print(f"{PlateSolveDirResults} does not exist")
      except PermissionError:
          print(f"Permission denied to delete {PlateSolveDirResults}")
      except Exception as e:
          print(f"Error occurred: {e}")
     
    # Create result repository
    print('Create new ASTAP plate solve results directory');
    os.mkdir(PlateSolveDirResults)
    
    # Start ASTAP
    for i in range(NbFits):
      fits_file = listing_fits[i]
      print(f"Solve fits file : {fits_file}")
        
      output_file = FitsFolder + "\\PlateSolveAstap\\" + os.path.basename(fits_file)  
      output_file = output_file.replace("fits", "wcs")
      
      text = r'{0}{1}{2} -f {3} -o {4} -update {5}'.format("\"",PlateSolveToolFolder,"\"", fits_file, output_file, fits_file)     
      returned_value = os.system(text)
      print('returned value:', returned_value)
      # Store plate solve result
      plate_solve_results.append(returned_value)

    # "|Error code|Description|\n",
    # "|---|---|\n",
    # "|0\t|No errors|\n",
    # "|1\t|No solution|\n",
    # "|2\t|Not enough stars detected|\n",
    # "|16\t|Error reading image file|\n",
    # "|32\t|No star database found|\n",
    # "|33\t|Error reading star database|"
    # 34	Error updating input file

    print("End ASTAP computation")


#------------------------------------------------------------------------------
def wcs_read_as_dict(wcs_file):
    #print("Start Reading .wcs files")
    d = {}
    with open(wcs_file, encoding='utf8',errors='ignore') as f:
        comment_count = 0
        for line in f:
            # extract line containing ASTAP plate-solving results
            if line.startswith('COMMENT 7'):
                #
                # TODO: need some error checking here
                #
                # try to extract Mount offset in RA
                try:
                    mount_offset_ra =line.split(',')[0].split('=')[1].strip()
                    if mount_offset_ra[-1]=='\'':
                        mount_offset_ra = float(mount_offset_ra[:-1])*60
                    elif mount_offset_ra[-1]=='"':
                        mount_offset_ra = float(mount_offset_ra[:-1])
                    else:
                        print(mount_offset_ra)
                except Exception as e:
                    mount_offset_ra = None
                # try to extract Mount offset in DEC
                try:
                    mount_offset_dec=line.split(',')[1].split('=')[1].strip()
                    if mount_offset_dec[-1]=='\'':
                        mount_offset_dec = float(mount_offset_dec[:-1])*60
                    elif mount_offset_dec[-1]=='"':
                        mount_offset_dec = float(mount_offset_dec[:-1])
                    else:
                        print(mount_offset_dec)
                except Exception as e:
                    mount_offset_dec = None
                d.update({'OFFSETRA':mount_offset_ra, 'OFFSETDE':mount_offset_dec})
            elif line.startswith('COMMENT'):
                d.update({f'COMMENT{comment_count}':unidecode(line.split('COMMENT ')[1].strip())})
                comment_count+=1
            else:
                try:
                    key = line.split('=')[0].strip().replace('\'','')
                    val = line.split('=')[1].split('/')[0].strip().replace('\'','')
                    if key in ['EXPOSURE', 'EXPTIME', 'EGAIN', 'XPIXSZ', 'YPIXSZ', 'CCD-TEMP',
                              'FOCALLEN', 'FOCRATIO', 'RA', 'DEC', 'CENTALT', 'CENTAZ',
                                'AIRMASS', 'SITEELEV', 'SITELAT', 'SITELONG',
                                'CRVAL1', 'CRVAL2', 
                              'HFD', 'STARS', 'OBJCTROT', 'CLOUDCVR', 'HUMIDITY', 'PRESSURE', 'AMBTEMP', 'WINDDIR', 'WINDSPD',
                              'EQUINOX', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CROTA1', 'CROTA2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
                        val=float(val)
                    if key in ['BITPIX', 'NAXIS', 'BZERO', 'XBINNING', 'YBINNING', 'GAIN', 'OFFSET', 
                                'USBLIMIT', 'FOCPOS', 'FOCUSPOS', 'XBAYROFF', 'YBAYROFF']:
                        val=int(val)
                    d.update({key:val})
                except:
                    d.update({'OTHER':line.strip()})
    #print("End Reading .wcs files")
    return d



#------------------------------------------------------------------------------
def load_wcs_from_folder_as_df(fits_folder):
    print("Start Loading .wcs files data")
    
    file_extension = 'wcs'
    image_filter = '*'

    # list WCS files present in folder
    wcs_folder = fits_folder + '\\PlateSolveAstap'
    
    p_wcs = Path(wcs_folder).glob(f'{image_filter}.{file_extension}')
    wcs_files = [x for x in p_wcs if x.is_file()]
    print('Found {} WCS files in folder'.format(len(wcs_files)))
    # load each WCS file as a Python dictionary
    sequence_list = []
    sequence_df = None
    for f in wcs_files:
        d = wcs_read_as_dict(f)
        sequence_list.append(d)
    sequence_df = pd.DataFrame.from_records(sequence_list)
    try:
        sequence_df['DATE-OBS'] = pd.to_datetime(sequence_df['DATE-OBS'])
        sequence_df.set_index('DATE-OBS', drop=False, inplace=True)
        sequence_df.sort_index(inplace=True)
    except Exception as e:
        pass
    sequence_df['FRAME_NUM'] = sequence_df.reset_index(drop=True).index.values
    try:
        sequence_df['TIME_DIFF'] = sequence_df['DATE-OBS'].diff()
        sequence_df['TIME_DIFF'] = sequence_df['TIME_DIFF'].dt.total_seconds()
        sequence_df['TIME_REL'] = (sequence_df['DATE-OBS'] - sequence_df['DATE-OBS'].min()).dt.total_seconds()
    except Exception as e:
        pass
    
    print("End Loading .wcs files data")
    return sequence_df
  
    
#------------------------------------------------------------------------------
def read_plate_solve_data(FitsFileDirectory,sequence_df):  
  print("Start reading plate solve data")
  
  # compute RA/DEC error 
  t_absolute = Time(sequence_df['DATE-LOC'].tolist(), format='fits')
  t = t_absolute.unix - t_absolute[0].unix   
  
  # CRVAL1/CRVAL2 in degree
  # Delta_CRVAL1/Delta_CRVAL in degree
  sequence_df['Delta_CRVAL1'] = sequence_df['CRVAL1'] - sequence_df['CRVAL1'].iloc[0]
  sequence_df['Delta_CRVAL2'] = sequence_df['CRVAL2'] - sequence_df['CRVAL2'].iloc[0]
 
  # Linear interpolation
  poly_Delta_CRVAL1 = np.polyfit(t, sequence_df['Delta_CRVAL1'], 1)
  poly_Delta_CRVAL2 = np.polyfit(t, sequence_df['Delta_CRVAL2'], 1)  
  CRVAL1_polynome = np.poly1d(poly_Delta_CRVAL1)
  CRVAL2_polynome = np.poly1d(poly_Delta_CRVAL2) 
  
  # Compute max and min
  max_CRVAL1 = sequence_df['CRVAL1'].max()
  min_CRVAL1 = sequence_df['CRVAL1'].min()
  diff_CRVAL1 = (max_CRVAL1-min_CRVAL1)
  print(f'Maximum RA drift amplitude: {diff_CRVAL1*3600:.2f} arcsec for an interval time of {t.max():.2f}s')
  print(f'i.e {poly_Delta_CRVAL1[0]*3600*60:.3f} arcsec/min')
  
  max_CRVAL2 = sequence_df['CRVAL2'].max()
  min_CRVAL2 = sequence_df['CRVAL2'].min()
  diff_CRVAL2 = (max_CRVAL2-min_CRVAL2)
  print(f'DEC drift amplitude: {diff_CRVAL2*3600:.3f} arcsec for an interval time of {t.max():.2f}s')
  print(f'i.e {poly_Delta_CRVAL2[0]*3600*60:.3f} arcsec/min')  
  
  
  # RA/CRVAL1 and DEC/CRVAL2 in hours/min/s and deg/min/sec
  fig = plt.figure(figsize=(12, 8))
  
  min_RA = min(sequence_df['RA'].min(),sequence_df['CRVAL1'].min())
  max_RA = max(sequence_df['RA'].max(),sequence_df['CRVAL1'].max())
  min_DEC = min(sequence_df['DEC'].min(),sequence_df['CRVAL2'].min())
  max_DEC = max(sequence_df['DEC'].max(),sequence_df['CRVAL2'].max())
  
  # Add RA
  ax = fig.add_subplot(221) 
  plt.plot(t,np.array(sequence_df['RA']))  # OBJCTRA 
  # plt.title("RA [h/min/s]")
  plt.title("RA [°]")
  plt.grid()
  plt.xlim(0,t.max())
  # plt.ylim(min_RA,max_RA)

  # Add CRVAL1
  ax = fig.add_subplot(222)
  plt.plot(t,np.array(sequence_df['CRVAL1']))
  #plt.title("CRVAL1 [h/min/s]")
  plt.title("CRVAL1 [°]")
  plt.grid()
  plt.xlim(0,t.max())
  # plt.ylim(min_RA,max_RA)
  
  # Add DEC
  ax = fig.add_subplot(223)
  plt.plot(t,np.array(sequence_df['DEC']))
  #plt.title("DEC [d/min/s]")
  plt.title("DEC [°]")
  plt.xlabel("time (in s)")
  plt.grid()
  plt.xlim(0,t.max())
  # plt.ylim(min_DEC,max_DEC)
  
  # Add CRVAL2
  ax = fig.add_subplot(224)
  plt.plot(t,np.array(sequence_df['CRVAL2']))
  #plt.title("CRVAL2 [d/min/s]")
  plt.title("CRVAL2 [°]")
  plt.xlabel("time (in s)")
  plt.ylabel("DEC")
  plt.grid()
  plt.xlim(0,t.max())
  # plt.ylim(min_DEC,max_DEC)
  
  plt.tight_layout()
  plt.show()
  
   
  # Plot RA and DEC error in arcsec
  fig = plt.figure(figsize=(12, 8))
  
  # Add first axes object
  ax = fig.add_subplot(211)
  plt.plot(t,np.array(sequence_df['Delta_CRVAL1'])*3600,t,CRVAL1_polynome(t)*3600)
  plt.title("RA error [arcsec]")
  plt.grid()
  plt.xlim(0,t.max())
  textstr = r'$\delta\,RA=%.3f$ arcsec/min' % (poly_Delta_CRVAL1[0]*3600*60, )
  if poly_Delta_CRVAL1[0] > 0:
    ax.text(0.55, 0.20, textstr, transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
  else:
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=24,
        verticalalignment='top')
  
  # Add second axes object
  ax = fig.add_subplot(212)
  plt.plot(t,np.array(sequence_df['Delta_CRVAL2'])*3600,t,CRVAL2_polynome(t)*3600)
  plt.title("DEC error [arcsec]")
  plt.xlabel("time (in s)")
  plt.grid()
  plt.xlim(0,t.max())
  textstr = r'$\delta\,DEC=%.3f$ arcsec/min' % (poly_Delta_CRVAL2[0]*3600*60, )
  if poly_Delta_CRVAL2[0] > 0:
    ax.text(0.55, 0.20, textstr, transform=ax.transAxes, fontsize=24,
        verticalalignment='top') 
  else:
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=24,
        verticalalignment='top')  
  
  # Make sure the elements of the plot are arranged properly
  plt.tight_layout()
  plt.show()
  

#------------------------------------------------------------------------------
def ssa(x1,L,I):  
  print("Start SSA")
  
  # Step 1: Build trajectory matrix
  N = len(x1)
  if L > N / 2:
    L = N - L
  K = N - L + 1
  X = np.zeros((L, K))

  for i in range(0, K):
    X[:,i] = x1[i:L+i]  
  
  # Step 2: SVD
  S = np.dot(X, X.T)
  EigenValues, EigenVectors = LA.eig(S)
  d = np.sort(-np.real(EigenValues)) 
  i = np.argsort(-np.real(EigenValues))
  d = -d
  EigenVectors = EigenVectors[:,i]  
  
  V = np.dot(X.T, EigenVectors)
  
  # Step 3: Grouping
  rca = np.zeros((L, K))
  Vt = V.T
  rca = np.outer(EigenVectors[:,I], Vt[I,:])

  # Step 4: Reconstruction
  y = np.zeros(N)
  Lp = min(L,K)
  Kp = max(L,K)

  for k in range(0, Lp-1):
    for m in range(0, k+1):
        y[k] += rca[m, k-m] / (k+1)

  for k in range(Lp-1, Kp):
    for m in range(0, Lp):
        y[k] += rca[m, k-m] / Lp

  for k in range(Kp, N):
    for m in range(k+1-Kp, N-Kp+1):
        y[k] += rca[m, k-m] / (N-k)
    
  r = x1 - y
    
  print("End SSA")
  return y, r, d
  

#------------------------------------------------------------------------------
def my_fft(t,signal,Ts,PlotOption,Option2,TabSignalInfo):  
  print("Start FFT")
    
  # calcul de la transformee de Fourier et des frequences
  Nspec = 64*1024
  fourier = np.fft.fft(signal,Nspec)
  freq = np.fft.fftfreq(Nspec, d=Ts)
  Spectre_amp = np.abs(fourier)/signal.size
  Save_Spectre_amp_0 = Spectre_amp[0]
  Spectre_amp = 2*Spectre_amp
  Spectre_amp[0] = Save_Spectre_amp_0;
  
  Spectre_phase = np.angle(fourier)

  # Spectrum analysis
  index_fond_est = np.argmax(Spectre_amp)
  amp_fond_est = Spectre_amp[index_fond_est]
  phase_fond_est = Spectre_phase[index_fond_est]
  Freq_fond = np.abs(freq[index_fond_est])
  T_fond    = np.abs(1/Freq_fond)

  xlim_min_fft_freq   = 0.001
  xlim_max_fft_freq   = 0.015
  xlim_min_fft_period = 1/xlim_max_fft_freq
  xlim_max_fft_period = 1/xlim_min_fft_freq

  if PlotOption == 1:
    fig = plt.figure(figsize=(12, 20))
    
    # Signal
    plt.subplot(311)
    plt.plot(t,signal)  
    plt.title("RA/CRVAL1 harmonic signal")
    plt.grid()
    plt.xlabel("time (in s)")
    plt.xlim(0,t.max())
    
    # Amplitude function of frequency
    ax = fig.add_subplot(312)
    #plt.semilogx(freq, Spectre_amp)
    plt.plot(freq, Spectre_amp)
    plt.title("fft spectrum")
    plt.xlabel("frequency (Hz)")
    plt.grid()
    plt.xlim(xlim_min_fft_freq,xlim_max_fft_freq)
    
    if Option2 == 0:
      plt.axvline(x=Freq_fond,color='red',linestyle='--')
      textstr = r'$F_{fond}=%.5f$Hz' % (Freq_fond, )
      ax.text((Freq_fond-xlim_min_fft_freq)/(xlim_max_fft_freq-xlim_min_fft_freq)+0.01, 0.95, textstr, transform=ax.transAxes, fontsize=12,
          verticalalignment='top') 
    else:
      # Put all axes of the main components
      k = 0
      for i in range(0,10):
        if TabSignalInfo[0,i] == 2: # It is a sinusoïdal component
          plt.axvline(x=TabSignalInfo[3,i],color='red',linestyle='--')
          textstr = r'$F=%.5f$Hz' % (TabSignalInfo[3,i], )
          ax.text((TabSignalInfo[3,i]-xlim_min_fft_freq)/(xlim_max_fft_freq-xlim_min_fft_freq)+0.01, 0.95-k*0.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top')    
          k = k + 1
      
    
    # Amplitude function of period
    ax = fig.add_subplot(313)
    #plt.semilogx(1/freq, Spectre_amp)
    plt.plot(1/freq, Spectre_amp)
    plt.title("fft spectrum")
    plt.xlabel("time (s)")
    plt.grid()
    plt.xlim(xlim_min_fft_period,xlim_max_fft_period)   
    
    #major_ticks = np.arange(xlim_min_fft_period, xlim_max_fft_period, 50)
    #ax.set_xticks(major_ticks)
    
    if Option2 == 0:
      plt.axvline(x=T_fond,color='red',linestyle='--')
      textstr = r'$T_{fond}=%.2f$s' % (T_fond, )
      ax.text((T_fond-xlim_min_fft_period)/(xlim_max_fft_period-xlim_min_fft_period)+0.01, 0.95, textstr, transform=ax.transAxes, fontsize=12,verticalalignment='top') 
    else:
      # Put all axes of the main components
      k = 0
      for i in range(0,10):
        if TabSignalInfo[0,i] == 2: # It is a sinusoïdal component
          plt.axvline(x=TabSignalInfo[4,i],color='red',linestyle='--')
          textstr = r'$T=%.2f$s' % (TabSignalInfo[4,i], )
          ax.text((TabSignalInfo[4,i]-xlim_min_fft_period)/(xlim_max_fft_period-xlim_min_fft_period)+0.01, 0.95-k*0.05, textstr, transform=ax.transAxes, fontsize=12,verticalalignment='top')    
          k = k + 1   
    
    
    plt.legend()
    plt.show()           
    
  print("End FFT")
  return amp_fond_est, phase_fond_est, Freq_fond, T_fond

  
#------------------------------------------------------------------------------
def ssa_plate_solve_data(sequence_df,end_sequence_df):  
  print("Start Singular Sprectrum Analysis (SSA)")

  #----------------------------------------------------
  if end_sequence_df > sequence_df['CRVAL1'].size:
    end_sequence_df = sequence_df['CRVAL1'].size
    
  CONVERT_deg_to_arcsec = 3600
  signal  = (sequence_df['CRVAL1'].iloc[0:end_sequence_df] - sequence_df['CRVAL1'].iloc[0])*CONVERT_deg_to_arcsec
   
  y = np.zeros((signal.size,10))
  L = int(signal.size / 5.71)

  for I in range(0,10):
    y[:,I], r, d = ssa(signal,L,I)
    
  #----------------------------------------------------  
  # Plot 
  t_absolute = Time(sequence_df['DATE-LOC'].tolist(), format='fits')
  time = t_absolute.unix - t_absolute[0].unix 
  t = time[0:end_sequence_df] 
  
  fig = plt.figure(figsize=(12, 8))
    
  # Add first axes object
  ax = fig.add_subplot(211)
  plt.plot(t,signal, marker = '+')
  plt.title("RA error [arcsec]")
  plt.xlabel("time (in s)")
  plt.grid()
  plt.xlim(0,t.max())

  # Add second axes object
  sev = sum(d)
  AllL = range(1,L+1) #[i for i in range(L+1)]
  ax = fig.add_subplot(212)
  plt.plot(AllL,d/sev, marker = '+')
  plt.title('Normalized Singular Spectrum')
  plt.xlabel('Eigenvalue Number')
  plt.ylabel('fraction of energy')
  plt.grid()
  plt.xlim(1,10)

  plt.tight_layout()
  plt.show() 
    
  #----------------------------------------------------
  # Regrouping of vectors to retrieve the signal
  k = 0;
  IndexTab = 0;
  TabSignal = np.zeros((signal.size,10))
  
  # Line 1 : 1 = one component, 2 = sinusoïd signal
  TabSignalInfo = np.zeros((5,10))
  
  print('Component composition :')
  Order = 5  
  threshold = 0.30  # 30%
  
  while IndexTab <= Order:
    sev = 0
    for i in range(k,L):
      sev += d[i]
      
    d_norm = d/sev
    if d[k+1] < (1-threshold)*d[k] :
      TabSignal[:,IndexTab] = y[:,k]
      TabSignalInfo[0,IndexTab] = 1 # One composant
      textstr = r'%d' % (k, )
      k = k + 1
    else:
      TabSignal[:,IndexTab] = y[:,k] + y[:,k+1]
      TabSignalInfo[0,IndexTab] = 2 # Tw0 composants => sinusoidal signal   
      textstr = r'%d + %d' % (k,k+1, )
      k = k + 2

    print(textstr);
    IndexTab = IndexTab + 1
 
  #----------------------------------------------------
  # FFT on the fondamental in order to capture the period/frequency
  Ts = np.mean(np.diff(t));  # Mean sampling period
  Fs = 1/Ts               # Sampling frequency
  print(f'Ts = {Ts:.3f} second')
  
  i = 0
  while  i <= Order:  # "fondamental+ order harmonics"
    if TabSignalInfo[0,i] == 2: # Is it a sinusoïd ?
      # Compute the frequency/period of TableSignal[:,i]
      # Store : Amplitude, Phase, Frequency and Period
      # 1st parameter : 1 = plot Fourrier spectrum, 0 = No plot 
      # 2nd parameter : 1 = Plot axes ol all main components, 0 = just one information
      TabSignalInfo[1,i], TabSignalInfo[2,i], TabSignalInfo[3,i], TabSignalInfo[4,i] = my_fft(t,TabSignal[:,i],Ts,Plot_FFT,0,TabSignalInfo)
      
    i = i + 1

  #----------------------------------------------------
  # Reconstruction of the signal
  signal_rebuilt = np.sum(TabSignal, axis=1)
  err_signal = signal - signal_rebuilt
  # Look for the fondamental period
  i = 0
  while TabSignalInfo[4,i] == 0:  
    i = i + 1
    
  T_fond = TabSignalInfo[4,i]
  
  Number_of_period = signal.size*Ts/T_fond
  Number_of_perriod_rms = int((0.8*Number_of_period))
  index_start_of_perriod_rms = int((0.1*signal.size))
  Nb_points = int((Number_of_perriod_rms*T_fond/Ts))
  err_rms = err_signal[index_start_of_perriod_rms:index_start_of_perriod_rms+Nb_points-1]
  MSE = np.square(err_signal).mean() 
  RMSE = math.sqrt(MSE)
  print(f'RMS value of the error of reconstruction = {RMSE:.3f} arcsec')
  
  
  fig = plt.figure(figsize=(12, 8))
    
  # Add first axes object
  ax = fig.add_subplot(211)
  plt.plot(t,signal,t,signal_rebuilt, marker = '+')
  plt.title("RA error and reconstructed signal [arcsec]")
  plt.ylabel('[arcsec]')
  plt.grid()
  plt.xlim(0,t.max())

  # Add second axes object
  ax = fig.add_subplot(212)
  plt.plot(t,err_signal, marker = '+')
  plt.title('Error of reconstruction')
  plt.xlabel("time (in s)")
  plt.ylabel('[arcsec]')
  plt.grid()
  plt.xlim(0,t.max())
  textstr = r' RMSE of reconstruction of the signal = %.3f arcsec' % (RMSE, )
  ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top') 

  plt.tight_layout()
  plt.show() 
  
  #----------------------------------------------------
  # Plot the main components of the signal
  fig = plt.figure(figsize=(12,20))
  
  nrows = Order + 1
  
  # 1st component - RA deviation
  poly_Deviation_RA = np.polyfit(t, TabSignal[:,0], 1)
  RA_polynome = np.poly1d(poly_Deviation_RA)  # in arcsec
  textstr = r'RA deviation - Slope =  %.5f arcsec/min' % (RA_polynome[0]*60, )
  print(textstr)
  
  ax = fig.add_subplot(nrows,1,1)
  plt.plot(t,TabSignal[:,0])
  plt.title(textstr)
  plt.ylabel('[arcsec]')
  plt.grid()
  plt.xlim(0,t.max())  
  
  for k in range(1, Order+1):
    borne_min = int(signal.size/2-T_fond/(2*Ts))
    borne_max = int(signal.size/2+T_fond/(2*Ts))
    # max
    Max_TabSignal = np.max(TabSignal[borne_min:borne_max,k])
    # min
    Min_TabSignal = np.min(TabSignal[borne_min:borne_max,k])
    # RMSE
    MSE = np.square(TabSignal[index_start_of_perriod_rms:index_start_of_perriod_rms+Nb_points-1,k]).mean() 
    RMSE = math.sqrt(MSE)

    if k == 1:
      textstr = r'Max value of Fond = %2.2f arcsec' % (Max_TabSignal, )
      print(textstr)
      textstr = r'Min value of Fond = %2.2f arcsec' % (Min_TabSignal, )  
      print(textstr)
      textstr = r'RMS value of Fond = %2.2f arcsec' % (RMSE, )  
      print(textstr)
    else:
      textstr = r'Max value of H%d = %2.2f arcsec' % (k-1, Max_TabSignal, )
      print(textstr)
      textstr = r'Min value of H%d = %2.2f arcsec' % (k-1, Min_TabSignal, )  
      print(textstr)
      textstr = r'RMS value of H%d = %2.2f arcsec' % (k-1, RMSE, )  
      print(textstr)      
  
    ax = fig.add_subplot(nrows,1,k+1)
    plt.plot(t,TabSignal[:,k])
    plt.title(textstr)
    plt.ylabel('[arcsec]')
    plt.grid()
    plt.xlim(0,t.max())   
  
  plt.xlabel("time (in s)")
  #----------------------------------------------------
  # FFT of harmonic signal of CRVAL1
  poly_Deviation_RA = np.polyfit(t, signal, 1)
  RA_polynome = np.poly1d(poly_Deviation_RA)  # in arcsec
  signal_fond = signal - RA_polynome(t)
  # 1st parameter : 1 = plot Fourrier spectrum, 0 = No plot 
  # 2nd parameter : Plot axes ol all main components
  my_fft(t,signal_fond,Ts,1,1,TabSignalInfo)   

  print("End Singular Sprectrum Analysis (SSA)")


#------------------------------------------------------------------------------
def start_script_PEC():
  print(f'Start program - version {version:.2f}')    
  print('Date : ' + str(datetime.now()))
    
  # Read files to know tool directory and the fit files directory
  (FitsFileDirectory, PlateSolveToolFolder, PlateSolveToolNumber, ExecutePlateSolve) = read_config_info()

  # List the fit files
  (listing_fits_file, NbFitsFile) = collect_fits(FitsFileDirectory)
  
  # Execute platesolve
  if(ExecutePlateSolve == 1):
    PlateSolveExecution = Plate_Solve_Computation(FitsFileDirectory,listing_fits_file,NbFitsFile,PlateSolveToolFolder,PlateSolveToolNumber)

  # Read computed date of the plate solve 
  sequence_df = load_wcs_from_folder_as_df(FitsFileDirectory)
  
  # Compute and print date of the plate solve
  read_plate_solve_data(FitsFileDirectory,sequence_df)
  
  # Singular Sprectrum Analysis (SSA)
  ssa_plate_solve_data(sequence_df,end_sequence_df)
   
  print("End program")


#------------------------------------------------------------------------------
start_script_PEC()

