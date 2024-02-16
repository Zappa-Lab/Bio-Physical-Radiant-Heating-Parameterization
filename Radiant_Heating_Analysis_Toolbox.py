#computation packages
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import datetime as dt
import itertools
import math
from scipy.optimize import curve_fit
from scipy.io import loadmat
import scipy
from scipy import signal as signal

#plotting packages
import matplotlib as mpl
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.lines as mlines
from matplotlib.patches import ConnectionPatch

#physical pacakages
import gsw
import metpy
import metpy.calc
from metpy.units import units

#utilities
import glob
from tqdm import tqdm
import copy

#--------------------------------------------------------------------------------------------------#
def exp_func(x, C, a):
    '''Exponential function we will be fitting to'''
    return C*np.exp(a*x)

#--------------------------------------------------------------------------------------------------#
def import_mpro_L3a(files, rejects, cruiseID, tzoff):
    '''
    Import MultiPro L3a files output by ProSoft and collate into an xarray dataset with metadata
    
    Inputs:
        files    - [list of str] list of filepaths to L3a data
        rejects  - [list of str] list of cast IDs to reject from the dataset 
        cruiseID - [str] cruise identifier to assign as coordinate value
        tzoff    - [int] number of hours time coordinate is offset from UTC
        
    Outputs:
        dataset  - xarray dataset (dims: castID, depth, λ ; coords: cruiseID(castID))
    '''
    
    #define a date parser to deal with the time import concisely
    custom_parser = lambda d,t: dt.datetime.strptime(f'{d} {t}','%Y%j.00000000 %H:%M:%S') 

    profile_list = []
    #loop over each filepath
    for file in tqdm(files):

        #scrape castID out of file name
        castID = file.split(sep='\\')[1].split(sep='_L3a')[0] 
        #check if this castID should be rejected
        if castID in rejects:
            continue #skips the rest of the loop and starts over at next iteration

        #open file and locate blank lines that separate different data tables
        lines = open(file).readlines()
        split_inds = [idx for idx,line in enumerate(lines) if line=='\n']

        #import all metadata in file header
        Metadata = pd.read_csv(file, sep=':', engine='python', nrows=split_inds[0],header=None,index_col=0,error_bad_lines=False, warn_bad_lines=False)

        #import each data table in the L3 file separately
        Time = pd.read_csv(file, sep='\t', skiprows=split_inds[0]+2, nrows=split_inds[1]-split_inds[0]-3, usecols=[0,1,2], index_col=1, parse_dates={'Time':[0,1]},date_parser=custom_parser)
        Time['Time_UTC'] = Time.Time - pd.Timedelta(tzoff,unit='hours') #subtract because want to shift back to UTC
        if len(Time.Time) < 10:
            continue #skip this entry if there are fewer than 10 data points - don't want really short casts
        
        ED = pd.read_csv(file, sep='\t', skiprows=split_inds[0]+2, nrows=split_inds[1]-split_inds[0]-3, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], index_col=0)
        #LU = pd.read_csv(file, sep='\t', skiprows=split_inds[1]+2, nrows=split_inds[2]-split_inds[1]-3, usecols=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], index_col=0)

        #----------------------------------ED Fits---------------------------------------
        ED0s = []
        kDs = []
        ED0_errs = []
        kD_errs = []

        #loop over wavelength columns
        for λ in ED.columns.values:
            #set up inputs to fitting routine
            if float(λ) > 659: #fit to shallower depths for the long wavelengths
                if float(λ) > 700:
                    EDsub = ED.loc[0:7.5, λ] #use the top 7.5m for 705+
                else:
                    EDsub = ED.loc[0:7.5, λ] #use the top 7.5m for 660 and 683
            else:
                EDsub = ED.loc[0:10, λ] #use the top 10m for all smaller wavelengths

            depth = -EDsub.index.values
            EDsub = EDsub.values

            try: 
                #use 1 as initial guess for ED0 and 0.1 as initial guess for kD
                poptD, pcovD = curve_fit(exp_func, depth[np.isfinite(EDsub)], EDsub[np.isfinite(EDsub)], p0=(1,0.1))
                ED0s.append(poptD[0])
                kDs.append(poptD[1])
                ED0_errs.append(np.sqrt(np.diag(pcovD))[0])
                kD_errs.append(np.sqrt(np.diag(pcovD))[1])
            except: #if the curve cannot be fit, just fill the values with NaN and keep going
                ED0s.append(np.nan)
                kDs.append(np.nan)
                ED0_errs.append(np.nan)
                kD_errs.append(np.nan)

        #create xarray dataset - all metadata comes along as attributes of the castID
        coords = {'cast': (['cast'], [str(castID).replace('-','_')]),# Metadata.to_dict()[1]),
                  'cruise': (['cast'], [str(cruiseID)]),
                  'λ': (['λ'], [float(x) for x in list(ED)]),
                  'depth': (['depth'], ED.index.values),
                  'time':(['cast','depth'],[list(Time.Time_UTC.values)])}

        #note: the 0.01 factor is to convert [μW/cm^2] -> [W/m^2]
        data_vars = {'Ed':(['cast','depth','λ'], [ED.values*0.01], {'units':'W/m^2/nm', 'long_name':'Spectral Downwelling Irradiance'}),
                     'Kd':(['cast','λ'], [kDs], {'units':'1/m', 'long_name':'Downwelling Attenuation from Fit'}),
                     'Kd_err':(['cast','λ'], [kD_errs], {'units':'1/m', 'long_name':'Standard Error of Downwelling Attenuation from Fit'}),
                     'Ed0':(['cast','λ'], [[x*0.01 for x in ED0s]], {'units': 'W/m^2/nm', 'long_name':'Surface Spectral Downwelling Irradiance from Fit'}),
                     'Ed0_err':(['cast','λ'], [[x*0.01 for x in ED0_errs]], {'units': 'W/m^2/nm', 'long_name':'Standard Error of Surface Spectral Downwelling Irradiance from Fit'})}

        profile_list.append(xr.Dataset(coords=coords, data_vars=data_vars))

    profiles = xr.merge(profile_list)
    return profiles

#--------------------------------------------------------------------------------------------------#
def import_mpro_L3a_metadata(files, rejects):
    '''
    Import MultiPro L3a files output by ProSoft and collate into an xarray dataset with metadata
    
    Inputs:
        files    - [list of str] list of filepaths to L3a data
        rejects  - [list of str] list of cast IDs to reject from the dataset 
        cruiseID - [str] cruise identifier to assign as coordinate value
        tzoff    - [int] number of hours time coordinate is offset from UTC
        
    Outputs:
        dataset  - xarray dataset (dims: castID, depth, λ ; coords: cruiseID(castID))
    '''
    
    #define a date parser to deal with the time import concisely
    custom_parser = lambda d,t: dt.datetime.strptime(f'{d} {t}','%Y%j.00000000 %H:%M:%S') 

    metadata_list, castID_list = [],[]
    #loop over each filepath
    for file in tqdm(files):

        #scrape castID out of file name
        castID = file.split(sep='\\')[3].split(sep='_L3a')[0] 
        #check if this castID should be rejected
        if castID in rejects:
            continue #skips the rest of the loop and starts over at next iteration

        #open file and locate blank lines that separate different data tables
        lines = open(file).readlines()
        split_inds = [idx for idx,line in enumerate(lines) if line=='\n']

        #import all metadata in file header
        Metadata = pd.read_csv(file, sep=':', engine='python', nrows=split_inds[0],header=None,index_col=0,error_bad_lines=False, warn_bad_lines=False, names=['cast',f'{castID}'])
        
        metadata_list.append(Metadata.transpose())
        castID_list.append(castID)
        
    return pd.concat(metadata_list)





#--------------------------------------------------------------------------------------------------#
#-------------------------------TRANSMISSION PARAMETERIZATIONS-------------------------------------#
#--------------------------------------------------------------------------------------------------#

#--------------Paulson & Simpson 1977 Constant-K Calculation----------
def ps77_EdVIS(Ed0, z, vis_frac=0.42):
    '''
    Visible-wavelength downwelling radiation at depth z using Paulson & Simpson 1977 for Jerlov Type I Waters, (eq. 4, written following Manizza et al 2005 eq. 1) constant-attenuation relationship.
    Input Total Downwelling Irradiance Below Surface Ed0 [W/m^2] and desired output depth z [m].
    Optional input vis_frac specifies fraction of total downwelling SW radiation assigned to the visible band.
    '''
    
    KdVIS = 1/23
    Ed0VIS = vis_frac*Ed0
    EdVIS = Ed0VIS*np.exp(-KdVIS*z) 
    return EdVIS

#--------------Manizza et al 2005 2-band Chl-dependent Calculation----------
def mz05_EdVIS(Ed0, z, C, vis_frac=0.42, bg_frac=0.5):
    '''
    Visible-wavelength downwelling radiation at depth z using Manizza et al 2005 (eq. 2-4) Chl-dependent 2-band attenuation relationship.
    Input Total Downwelling Irradiance Below Surface Ed0 [W/m^2], Chlorophyll concentration [mg/m^3], and desired output depth z [m].
    Optional inputs: vis_frac specifies fraction of total downwelling SW radiation assigned to the visible band; bg_frac specifies fraction of visible radiation assigned to the blue-green (vs. red) band
    '''
    Ed0VIS = vis_frac*Ed0
    
    Ed0r = Ed0VIS*(1-bg_frac)
    Ed0bg = Ed0VIS*bg_frac
    
    Kdr = 0.225 + 0.037*C**0.629
    Kdbg = 0.0232 + 0.074*C**0.674
    Edr = Ed0r*np.exp(-Kdr*z)
    Edbg = Ed0bg*np.exp(-Kdbg*z)
    EdVIS = Edr + Edbg
    return EdVIS
    
#--------------Manizza et al 2005 2-band Chl-dependent Calculation----------
def mz05_EdVIS_parts(Ed0, z, C, vis_frac=0.42, bg_frac=0.5):
    '''
    Same function as mz05_EdVIS, but returns the red and blue-green fractions independently.
    Visible-wavelength downwelling radiation at depth z using Manizza et al 2005 (eq. 2-4) Chl-dependent 2-band attenuation relationship.
    Input Total Downwelling Irradiance Below Surface Ed0 [W/m^2], Chlorophyll concentration [mg/m^3], and desired output depth z [m].
    Optional inputs: vis_frac specifies fraction of total downwelling SW radiation assigned to the visible band; bg_frac specifies fraction of visible radiation assigned to the blue-green (vs. red) band
    '''
    Ed0VIS = vis_frac*Ed0
    
    Ed0r = Ed0VIS*(1-bg_frac)
    Ed0bg = Ed0VIS*bg_frac
    
    Kdr = 0.225 + 0.037*C**0.629
    Kdbg = 0.0232 + 0.074*C**0.674
    Edr = Ed0r*np.exp(-Kdr*z)
    Edbg = Ed0bg*np.exp(-Kdbg*z)
    #EdVIS = Edr + Edbg
    return Edbg, Edr
    

#-Parameterizations below require absorption values as inputs-
#--------------Kim et al 2015 2-band Chl- & CDM-dependent Calculation----------
def km15_EdVIS(Ed0, z, C, adg, vis_frac=0.42, bg_frac=0.5):
    '''
    Visible-wavelength downwelling radiation at depth z using Kim et al 2015 (eq. 5) Chl- & CDM-dependent 2-band attenuation relationship.
    Input Total Downwelling Irradiance Below Surface Ed0 [W/m^2], Chlorophyll concentration Chl [mg/m^3], absorption at 443nm due to CDOM & detritus adg [m^-1], and desired output depth z [m].
    Optional input vis_IR_split specifies fraction of total downwelling SW radiation assigned to the visible band (the rest goes to IR).
    '''
    Ed0VIS = vis_frac*Ed0
    
    Ed0r = Ed0VIS*(1-bg_frac)
    Ed0bg = Ed0VIS*bg_frac
    
    Kdr = 0.225 + 0.037*C**0.629
    Kdbg = 0.0232 + 0.0513*C**0.668 + 0.710*adg**1.13
    Edr = Ed0r*np.exp(-Kdr*z)
    Edbg = Ed0bg*np.exp(-Kdbg*z)
    EdVIS = Edr + Edbg
    return EdVIS

#----------Lee et al 2005 2-band a490-dependent calculation-----
def lee05_EdVIS(Ed0, z, a490, bb490, θ, vis_frac=0.424):
    '''
    Visible-wavelength (actually 350-700nm) downwelling radiation at depth z using Lee et al 2005 (eq 7 & 9a-b, Table 2) fitted relationship.
    Solar Zenith Angle θ input in degrees.
    '''
    #initialize all constants
    χ0 = -0.057
    χ1 = 0.482
    χ2 = 4.221
    ζ0 = 0.183
    ζ1 = 0.702
    ζ2 = -2.567
    α0 = 0.090
    α1 = 1.465
    α2 = -0.667
    
    #calculate K1 and K2
    K1 = (χ0 + χ1*a490**0.5 + χ2*bb490)*(1 + α0*np.sin(np.deg2rad(θ)))
    K2 = (ζ0 + ζ1*a490 + ζ2*bb490)*(α1 + α2*np.cos(np.deg2rad(θ)))
    
    KdVIS = K1 + K2/((1+z)**0.5)
    
    Ed0VIS = vis_frac*Ed0
    
    EdVIS = Ed0VIS*np.exp(-KdVIS*z)
    
    return EdVIS

#--------------Morel 88 Spectral Chl-dependent Calculation----------
m88_T2 = pd.read_csv('Morel 88 Table 2.txt',sep='\s+',names=['λ','Kw','χ','e'],index_col=0)
def m88_Kd(C):
    '''Morel 1988, Table 2. Chlorophyll-dependent spectral attenuation depth K'''
    Kd = m88_T2.Kw + m88_T2.χ*C**m88_T2.e
    return Kd

m94_T1 = pd.read_csv('Morel & Antoine 94 Table 1.txt',sep='\s+',names=['λ','Kw','χ','e'],index_col=0)
def m94_Kd(C):
    '''Morel & Antoine 1994, Table 1. Extrapolation from M88 Table 2. Chlorophyll-dependent spectral attenuation depth K'''
    Kd = m94_T1.Kw + m94_T1.χ*C**m94_T1.e
    return Kd

m01_T2 = pd.read_csv('Morel & Maritorena 2001 Table 2.txt',sep='\s+',names=['λ','Kw','e','χ'],index_col=0)
def m01_Kd(C):
    '''Morel & Maritorena 2001, Table 2. Revision to Morel 88 based on large new dataset of low-Chl observations. Chlorophyll-dependent spectral attenuation depth K'''
    Kd = m01_T2.Kw + m01_T2.χ*C**m01_T2.e
    return Kd

m07_T1 = pd.read_csv('coeffs_Kmod_Metal07.txt',sep='\s+',skiprows=1,names=['λ','Kw','χ','e'],index_col=0)
def m07_Kd(C):
    '''
    Morel et al 2007, revised X and e values based on additional data, downloaded from ftp at oceane.obs-vlfr.fr, cd pub/morel, file 2006-e-chi
    As written in the paper (pg. 72), the pure-water spectrum is unchanged from Morel & Maritorena 2001. 
    '''
    Kd = m07_T1.Kw + m07_T1.χ*(C**m07_T1.e)
    return Kd

    


def revised_morel(C):
    '''
    Revision to the Morel & Maritorena 2001 Parameterization for wavelengths >=560nm, for which the original χ and e values were mistakenly retained despite the use of new Kw values
    '''
    m01_T2_xr = m01_T2.to_xarray()
    m88_T2_xr = m88_T2.to_xarray()
    
    if C==0:
        χ_new=0
    else:
        χ_new = m88_T2_xr.χ + (m88_T2_xr.Kw - m01_T2_xr.Kw)/(C**m88_T2_xr.e)
    
    Kd_r = m01_T2_xr.Kw + χ_new*C**m01_T2_xr.e
    Kd_bg = m01_T2_xr.Kw + m01_T2_xr.χ*C**m01_T2_xr.e
    
    Kd = xr.merge([Kd_bg.rename('Kd').sel(λ=slice(350,555)),Kd_r.rename('Kd').sel(λ=slice(560,700))])
    return Kd


#--------------Smith & Baker 1982 Chl-dependent Calcualtion-----------
bs82_T3 = pd.read_csv('Baker & Smith 1982 Table 3.txt',sep='\s+',names=['λ','Kw','kc','kp'],index_col=0).to_xarray()

def bs82_Kd(C):
    '''
    '''
    C0 = 0.5 #[mg Chl/m^3], constant from SB82 Table 2

    Kw = bs82_T3.Kw #from table
    kc = bs82_T3.kc #from table
    kp = bs82_T3.kp #from table

    Kc = kc*C*np.exp((-kp**2)*(np.log(C)-np.log(C0))**2) + 0.001*(C**2) #eq 6., also in Table 2
    Kd = Kw + Kc
    return Kd


