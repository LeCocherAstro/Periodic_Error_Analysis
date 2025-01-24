############################################
#
# January 2024
# (C) Mickaël HILAIRET, Gilles MORAIN
#
# Distributed under a Creative Commons Attribution ǀ 4.0
# International licence CC BY-NC-SA 4.0
#
# Computation of the Periodic Error of a mount
#
# Version 0.1
#   Plate solve using ASTAP 
#   Plate solve using PS3 is under development
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
#  Comments : 
#    Fig.1 is not well finsih. The RA/CRVAL1 and DEC/CRVAL2 are in degree and 
#    not in "H:M:S" (hms) and "°:M:S" (dms)
#    
#
#  Version 0.2 is coming soon
#    * Add a decomposition of the signal in principal components in order to have
#    an estimate of the frequency and amplitude of the main components and drift
#    * Improve Fig. 1
#

# importing packages 
import os
import shutil
import glob
import csv
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from astropy.time import Time
from unidecode import unidecode


version = 0.1


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
  plt.ylim(min_RA,max_RA)

  # Add CRVAL1
  ax = fig.add_subplot(222)
  plt.plot(t,np.array(sequence_df['CRVAL1']))
  #plt.title("CRVAL1 [h/min/s]")
  plt.title("CRVAL1 [°]")
  plt.grid()
  plt.xlim(0,t.max())
  plt.ylim(min_RA,max_RA)
  
  # Add DEC
  ax = fig.add_subplot(223)
  plt.plot(t,np.array(sequence_df['DEC']))
  #plt.title("DEC [d/min/s]")
  plt.title("DEC [°]")
  plt.xlabel("time (in s)")
  plt.grid()
  plt.xlim(0,t.max())
  plt.ylim(min_DEC,max_DEC)
  
  # Add CRVAL2
  ax = fig.add_subplot(224)
  plt.plot(t,np.array(sequence_df['CRVAL2']))
  #plt.title("CRVAL2 [d/min/s]")
  plt.title("CRVAL2 [°]")
  plt.xlabel("time (in s)")
  plt.ylabel("DEC")
  plt.grid()
  plt.xlim(0,t.max())
  plt.ylim(min_DEC,max_DEC)
  
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
  ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=24,
        verticalalignment='top')  
  
  # Make sure the elements of the plot are arranged properly
  plt.tight_layout()
  plt.show()
  

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
  
  print("End program")


#------------------------------------------------------------------------------
start_script_PEC()

