# Periodic Error Analysis script in Python

## This script compute the Periodic Error of a mount based on a plate solve of fits file

* Version 0.1 : ![](PEC_Analysis_v0p1.py)
  *   Plate solve using ASTAP 
  *   Plate solve using PS3 is under development

  * Config.txt file : ![](config.txt)
     * Add the path of ASTAP tool
     * Example : 
       astap,D:\ProgramOnD\astap\astap.exe
       ps3,D:\ProgramOnD\Platesolve3_80\PlateSolve3.80.exe

   * dir_fits_file.txt file : ![](dir_fits_file.txt)
     * Add the path of the fits files to read
     * tool,astap     : to using astap as plate solve tool
     * platesolve_on  : ask a plate solve
     * platesolve_off : no plate solve (because it has been yet done) and just
                      a plot of the signal is ask

     * Example : 
       dir_fits_file,D:\Documents\PEC_Analyse\File_Under_Test
       tool,astap
       platesolve_on, 

  * Comments : 
    * Fig.1 is not well finsih. The RA/CRVAL1 and DEC/CRVAL2 are in degree and 
    not in "H:M:S" (hms) and "°:M:S" (dms)
    

*  Version 0.2 ![](PEC_Analysis_v0p2.py)
    * Add a decomposition of the signal in principal components in order to have
    an estimate of the frequency and amplitude of the main components and drift

* Coming next :
    * Improve Fig. 1
    * Add a plate solve with PS3
    * Design a version only for SIRIL tool (plate solve with GAIA)



