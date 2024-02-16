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
#----------------------------------------------------------------------------------#
#------------------------DATA IMPORT-----------------------------------------------#
#----------------------------------------------------------------------------------#

#Import Ozone Zonal Mean Climatology from Nimbus 7
ozone_n7 = pd.read_csv('./Ozone_Nimbus7_ZonalMeanClimatology.txt',sep='\t',index_col=0).to_xarray()

#Import Starting Reference Spectra for clear-sky calculation
#d15_input_spectrum = pd.read_csv('./Diffey 2015 Solar Spectrum Parameterization/ET and AM1.5 irradiance spectra.txt',sep='\t',index_col=0)

#Import Payne 1972 Albedo Lookup Table
p72raw = pd.read_csv('./Payne72_albedoLUT.csv', index_col=0)
p72 = xr.DataArray(data=p72raw.values, coords={'T':(['T'],p72raw.index), 'elev':(['elev'],np.arange(0,92,2))})

#Import Morel07 Bio-Optical Kd Parameterization
morel07 = pd.read_csv('./Morel07_BioOpticalParameterization.csv',index_col=0)

#Import Ohlmann03 Chl-dependent modeled I parameterization
ohlmann03 = pd.read_csv('./Ohlmann_2003_Table1a.csv', index_col=0).to_xarray()

#----------------------------------------------------------------------------------#
#--------------------------FUNCTIONS-----------------------------------------------#
#----------------------------------------------------------------------------------#
def local_solar_time(time_utc,lon):
    '''
    Calculate local solar time from utc datetime and longitude (degrees, positive east)
    '''
    #convert UTC to local solar time based on lon (NOT timezone time)
    lst = time_utc + dt.timedelta(hours=lon/15)
    if lon > 180: #correct for international date line - western hemisphere should be behind UTC
        lst = lst - dt.timedelta(days=1)
        
    return lst

#----------------------------------------------------------------------------------#
def solar_spectrum_d15(time_utc,latitude,longitude,α=0.1,β=0.05):
    '''
    Calculate clear-sky solar spectrum from spatiotemporal position on the globe. Python translation of Diffey (2015) simple excel spreadsheet approach. Provides the simplest way to get a spectrum, and is shown in the publication to be reasonably comparable to much more complex approaches.
    '''

    lst = local_solar_time(time_utc,longitude)
    dayofyear = lst.timetuple().tm_yday
    month = lst.timetuple().tm_mon
    hour = lst.timetuple().tm_hour

    #------CONSTANTS-----
    #α = 0.1 #1.3  #index of aerosol size distribution (can range from 0.8 to 2.5) - maybe lower in open ocean
    #β = 0.05 #0.05 #"Angstrom Turbidity Coefficient" - index of opacity of vertical column of atmosphere (can range from 0.0 to 0.5)
    AOD500 = β*0.5**(α) #AOD = βλ^(-α) for λ in microns

    eccentricity = 0.01675 #eccentricity of the earth's orbit around the sun

    
    #Import Starting Reference Spectra - want to reimport every time the function is called, otherwise seeing weird behavior
    diffey15 = pd.read_csv('./Diffey 2015 Solar Spectrum Parameterization/ET and AM1.5 irradiance spectra.txt',sep='\t',index_col=0)

    
    #-----CALCULATIONS-----
    #single calculations
    ecliptic = 0.017214*(dayofyear-80)+2*eccentricity*(np.sin(0.017214*dayofyear)-np.sin(0.017214*80)) #ecliptic longitude of earth in its orbit
    declination = np.arcsin(0.3978*np.sin(ecliptic))
    cos_factor = np.cos(latitude*0.01745)*np.cos(declination)
    sin_factor = np.sin(latitude*0.01745)*np.sin(declination)
    solar_altitude = np.arcsin(cos_factor*np.cos(0.26175*(hour-12))+sin_factor)/0.01745
    sza = 90 - solar_altitude #solar zenith angle
    M = np.cos(sza*np.pi/180)
    Dse = 1.00011+0.034221*np.cos(2*np.pi*(dayofyear-1)/365)+0.00128*np.sin(2*np.pi*(dayofyear-1)/365)+0.000719*np.cos(4*np.pi*(dayofyear-1)/365)+0.000077*np.sin(4*np.pi*(dayofyear-1)/365) #earth-sun distance
    phi = np.sqrt(1.0266/(M*M+0.0266))-1 #eq 19 Green (1982)
    #Altitude Distribution factors eq 2 Green & Chai (1988)
    MX = np.sqrt((M*M+0.0018)/1.0018)
    MY = np.sqrt((M*M+0.0003)/1.0003)
    MZ = np.sqrt((M*M+0.0074)/1.0074)
    #Bird & Riordan (1986) diffuse irradiance factors
    ALG = np.log(1-0.65)
    BFS = ALG*(0.0783+ALG*(-0.3824-0.5874*ALG))
    AFS = ALG*(1.459+ALG*(0.1595+0.4129*ALG))
    Fs = 1 - 0.5*np.exp((AFS+BFS*M)*M)

    #spectral calculations
    wavelengths = diffey15.index.values.astype(float)
    diffey15['AM10'] = diffey15.ET*(diffey15.AM15/diffey15.ET)**(1/1.5)
    diffey15['AOD_084'] = 0.084*(wavelengths/1000)**(-α)/(0.5**(-α))
    diffey15['AOD_123'] = β*(wavelengths/1000)**(-α)
    diffey15['Direct_Normal'] = diffey15.AM10*(np.exp(-diffey15.AOD_123)/np.exp(-diffey15.AOD_084))
    diffey15['Raleigh_OD'] = 1.0456*((300/wavelengths)**4)*np.exp(0.1462*(300/wavelengths)**2) 
    diffey15['Raleigh_Trans'] = np.exp(-diffey15.Raleigh_OD/MX)
    diffey15['AOD_absorb'] = (0.067*(500/wavelengths)**-0.9369)*β*(wavelengths/1000)**-α
    diffey15['AOD_scatter'] = β*(wavelengths/1000)**(-α) - diffey15.AOD_absorb
    diffey15['Trans_scatter'] = np.exp(-diffey15.AOD_scatter/MY)
    ozone = ozone_n7[f'{month}'].interp(Latitude=latitude).values
    #two different formulas for ozone: Green & Schippnick (1982) for 290-368nm, Green & Chai (1988) for longer than 368nm
    diffey15['Ozone_OD'] = np.concatenate([ozone*11.277/(0.0355+np.exp((wavelengths[:79]-300)/7.15)), ozone*0.1274*(4*np.exp((wavelengths[79:]-594)/35.2)/(1+np.exp((wavelengths[79:]-594)/35.2))**2)]) 
    #Scatter factors from Green (1982)
    diffey15['F'] = 1/(1+84.37*(diffey15.AOD_absorb+diffey15.Ozone_OD)**0.6776)
    diffey15['F3'] = 1/(1+0.2864*(diffey15.Ozone_OD**0.8244)*(ozone**0.4166))
    diffey15['F4'] = 1/(1+(2.662*diffey15.AOD_absorb))
    diffey15['M'] = (0.8041*(diffey15.Raleigh_OD**1.389)*diffey15.F3+1.437*(diffey15.AOD_scatter**1.12)*(1+0.8041*(diffey15.Raleigh_OD**1.389)*diffey15.F3))*diffey15.F4
    diffey15['S'] = (diffey15.F+(1-diffey15.F)*np.exp(-diffey15.Ozone_OD*phi))*np.exp(-phi*(0.5346*diffey15.Raleigh_OD+0.6077*diffey15.AOD_scatter))

    #for both direct and diffuse, have to calculate the UV version and the full-spectrum version and then do the smooth average between them from 360-400
    weights_VIR = [(λ-360)/40 for λ in np.arange(360,401)]
    weights_UV = [1 - weight for weight in weights_VIR]

    Direct_UV = diffey15.ET*Dse*M*np.exp(-diffey15.Raleigh_OD/MX - diffey15.AOD_scatter/MY - diffey15.Ozone_OD/MZ - diffey15.AOD_absorb/MY)
    Direct_VIR = diffey15.ET*Dse*M*(diffey15.Direct_Normal/(diffey15.ET))**(1/M)
    Direct_merge = pd.Series(data=np.average([Direct_UV.loc[360:400], Direct_VIR.loc[360:400]],weights=[weights_UV,weights_VIR],axis=0),index=Direct_VIR.loc[360:400].index)
    diffey15['Direct'] = pd.concat([Direct_UV.loc[:359], Direct_merge, Direct_VIR.loc[401:]])

    Diffuse_UV = diffey15.ET*diffey15.S*diffey15.M*np.exp(-diffey15.Raleigh_OD - diffey15.AOD_scatter - diffey15.AOD_absorb - diffey15.Ozone_OD)
    Diffuse_VIR = (wavelengths/1000+0.55)**1.8*diffey15.Direct*(0.5*(1-(diffey15.Raleigh_Trans**0.95))+(diffey15.Raleigh_Trans**1.5)*(1-diffey15.Trans_scatter)*Fs)/(diffey15.Raleigh_Trans*diffey15.Trans_scatter)
    Diffuse_merge = pd.Series(data=np.average([Diffuse_UV.loc[360:400], Diffuse_VIR.loc[360:400]],weights=[weights_UV,weights_VIR],axis=0),index=Diffuse_VIR.loc[360:400].index)
    diffey15['Diffuse'] = pd.concat([Diffuse_UV.loc[:359], Diffuse_merge, Diffuse_VIR.loc[401:]])

    diffey15['Global'] = diffey15.Direct + diffey15.Diffuse

    return diffey15.Global, sza, Dse

#----------------------------------------------------------------------------------#
def cloud_model_s99(CI,Eclear):
    '''
    Parameterization of spectral cloud index from Siegel et al. (1999) SBDART modeling,  validated with TOGA-COARE observations
    
    Input broadband cloud index along with pandas Series of clear-sky irradiances Eclear with index of wavelengths λ, returns wavelength-specific irradiance as scaled by a wavelength-specific cloud index
    '''
    if CI<0.0001:
        A=0
        B=0
    else:
        A = -0.00130*(CI**2) + 0.00150*CI - 0.0001
        B = 0.887*(CI**2) + 0.0061*CI + 0.0562
        
    Ed0 = Eclear*(1-(A*Eclear.index)-B)
    Ed0[Ed0<0]=0 #at high cloud indices, can get negative numbers back which is non-physical and leads to bad integrations
    return Ed0

#----------------------------------------------------------------------------------#
def albedo_p72(elev, T):
    '''
    Broadband ocean albedo as a function of solar elevation (90-sza) and atmospheric transmittance from Payne 1972 lookup table. Was demonstrated to agree well with Ohlmann 2000 albedo calculations. 
    '''
    albedo = p72.interp(elev=elev, T=T).values.item()
    return albedo


#----------------------------------------------------------------------------------#

def Kd_spectrum_m07(Chl):
    '''
    Return spectral Kd with wavelengths<700nm scaled by Chlorophyll using Witte et al. 2023 Bio-Optical Parameterization
    '''
    Kd = morel07.Kw + morel07.chi*(Chl**morel07.e)
    return Kd


#----------------------------------------------------------------------------------#
def waveband_split(lat,lon,time,SW,Chl,splits):
    '''
    Run solar model for given lat/lon/time/SW/Chl, then split into bands defined by splits and return Kd and A (the fraction of total irradiance) that should be assigned to each band. splits must include the first and last wavelength (300 and 2500nm)
    '''
    if SW < 0:
        SW = 0
    
    # Get Kd spectrum 
    Kd = Kd_spectrum_w23(Chl)
    
    # Get Clear-Sky Solar Spectrum (+ sza & Dse for your troubles)
    Eclear, sza, Dse = solar_spectrum_d15(pd.to_datetime(time), lat, lon)
    Eclear = Eclear.loc[300:2500] #vanishingly small amount of energy outside these wavelengths
    # Integrate and calculate broadband cloud index
    Eclear_int = np.trapz(y=Eclear,x=Eclear.index)
    
    if (Eclear_int-SW) < 0.0001: #throws an error if they're too close together, also don't want to allow for a negative cloud index
        CI=0
    else:
        CI = (Eclear_int-SW)/Eclear_int
    
    # Calculate CI-adjusted solar spectrum
    Ed0p = cloud_model_s99(CI,Eclear)
    #reindex Ed0p onto Kd wavelengths
    Ed0p = Ed0p.reindex(Kd.index)
    # scale solar spectrum so it integrates to the SW measurement - should be very very very close to begin with
    I0p = np.trapz(y=Ed0p, x=Ed0p.index)
    scale = SW/I0p
    Ed0p = Ed0p*scale
    # Calculate broadband albedo 
    
    elev = 90-sza
    if elev <=0:
        elev=0.0001
    S = 1353 #[W/m^2], this is the value given in Payne 1972 and Aligns with NASA standards, WMO likes 1367
    T = SW/(S*np.sin(np.deg2rad(elev))*(Dse**2))
    α = albedo_p72(elev, T)
    if np.isnan(α):
        Ed0m = Ed0p
        I0m = SW
    # Apply Albedo
    else:
        Ed0m = Ed0p*(1-α)
        I0m = SW*(1-α) #same treatment for broadband input
    #loop over wavebands and calculate Kd and Ed for each
    Eds,Kds = [],[]
    for idx in np.arange(len(splits)-1):
        #if there's no energy in the waveband, just add 0 for both (weighted average throws an error otherwise)
        if np.sum(Ed0m.loc[splits[idx]:splits[idx+1]]) == 0:
            Kds.append(0.001) #cannot have 0 Kds otherwise we end up dividing by zero!
            Eds.append(0)
        else:
            Kds.append(np.average(Kd.loc[splits[idx]:splits[idx+1]], weights=Ed0m.loc[splits[idx]:splits[idx+1]]))
            Eds.append(np.trapz(Ed0m.loc[splits[idx]:splits[idx+1]], x=Ed0m.loc[splits[idx]:splits[idx+1]].index))

    #fraction of total subsurface irradiance
    if I0m==0:
        As = [E*0 for E in Eds]
    else:
        As = [E/I0m for E in Eds]
    
    return Kds,As,I0m

#----------------------------------------------------------------------------------#
def final_param(SW,Chl,z):
    '''
    Final simplified parameterization for irradiance at specific depth as a function of Chlorophyll (and SW at surface of course)
    '''
    #---------------Define Constants-------------
    #waveband partitions
    Auv,Ab,Ay,Ar,Air = 0.06,0.17,0.14,0.14,0.49
    
    #albedo
    α = 0.045
    
    #Chl-dependence parameters (UV & Vis)
    Kw_UV, χ_UV, e_UV = 0.0218, 0.1758, 0.6541
    Kw_B, χ_B, e_B = 0.0119, 0.1048, 0.633
    Kw_Y, χ_Y, e_Y = 0.0665, 0.0582, 0.5342
    Kw_R, χ_R, e_R = 0.3608, 0.0585, 0.4723
    
    #IR parameters
    C1,C2,C3,C4 = 1.87, 0.47, 0.66, 30
    
    #---------------Perform Calculations-------------
    #Chl-dependent Kds
    Kd_UV = Kw_UV + χ_UV*(Chl**e_UV)
    Kd_B = Kw_B + χ_B*(Chl**e_B)
    Kd_Y = Kw_Y + χ_Y*(Chl**e_Y)
    Kd_R = Kw_R + χ_R*(Chl**e_R)
    
    #waveband irradiances
    Ed0 = SW*(1-α)
    Ed_UV = Ed0*Auv*np.exp(-Kd_UV*z)
    Ed_B = Ed0*Ab*np.exp(-Kd_B*z)
    Ed_Y = Ed0*Ay*np.exp(-Kd_Y*z)
    Ed_R = Ed0*Ar*np.exp(-Kd_R*z)
    Ed_IR = Ed0*Air*np.exp(-C1*z)*(1 - C2*np.arctan(C3+C4*z))
    
    #final sum
    Ed = Ed_UV + Ed_B + Ed_Y + Ed_R + Ed_IR
    return Ed

#----------------------------------------------------------------------------------#
def final_param_scale(Chl,z):
    '''
    Final simplified parameterization for irradiance at specific depth as a function of Chlorophyll, to be multiplied by SW(1-albedo)
    '''
    #---------------Define Constants-------------
    #waveband partitions
    Auv,Ab,Ay,Ar,Air = 0.06,0.17,0.14,0.14,0.49
    
    #Chl-dependence parameters (UV & Vis)
    Kw_UV, χ_UV, e_UV = 0.0218, 0.1758, 0.6541
    Kw_B, χ_B, e_B = 0.0119, 0.1048, 0.633
    Kw_Y, χ_Y, e_Y = 0.0665, 0.0582, 0.5342
    Kw_R, χ_R, e_R = 0.3608, 0.0585, 0.4723
    
    #IR parameters
    C1,C2,C3,C4 = 1.87, 0.47, 0.66, 30
    
    #---------------Perform Calculations-------------
    #Chl-dependent Kds
    Kd_UV = Kw_UV + χ_UV*(Chl**e_UV)
    Kd_B = Kw_B + χ_B*(Chl**e_B)
    Kd_Y = Kw_Y + χ_Y*(Chl**e_Y)
    Kd_R = Kw_R + χ_R*(Chl**e_R)
    
    #waveband irradiances
    Ed_UV = Auv*np.exp(-Kd_UV*z)
    Ed_B = Ab*np.exp(-Kd_B*z)
    Ed_Y = Ay*np.exp(-Kd_Y*z)
    Ed_R = Ar*np.exp(-Kd_R*z)
    Ed_IR = Air*np.exp(-C1*z)*(1 - C2*np.arctan(C3+C4*z))
    
    #final sum
    Ed = Ed_UV + Ed_B + Ed_Y + Ed_R + Ed_IR
    return Ed


#----------------------------------------------------------------------------------#
#----------------------PRIOR RADIANT HEATING PARAMETERIZATIONS---------------------#
#----------------------------------------------------------------------------------#
def I_ps77(I0, z):
    '''
    Paulson & Simpson 2-band Irradiance parameterization (for Jerlov Type I waters) for a surface net solar irradiance of I0 at depth z
    '''
    I = I0*0.58*np.exp(-2.86*z) + I0*0.42*np.exp(-0.043*z)
    return I

def I_ps81(I0, z):
    '''
    Paulson & Simpson 9-band spectral irradiance parameterization for a surface net solar irradiance of I0 at depth z
    '''
    I = I0*0.237*np.exp(-0.0287*z) + I0*0.360*np.exp(-0.440*z) + I0*0.179*np.exp(-31.7*z) + I0*0.087*np.exp(-182*z) + I0*0.080*np.exp(-1200*z) + I0*0.0246*np.exp(-7940*z) + I0*0.025*np.exp(-3190*z) + I0*0.007*np.exp(-12800*z) + I0*0.0004*np.exp(-69400*z)
    return I

def I_s82(I0, z):
    '''
    Soloviev 1982 3-band Irradiance parameterization for a surface net solar irradiance of I0 at depth z
    '''
    I = I0*0.45*np.exp(-0.078*z) + I0*0.27*np.exp(-2.8*z) + I0*0.28*np.exp(-71*z)
    return I

def I_os00(I0p, z, Chl, CI, sza):
    '''
    Ohlmann & Siegel 2000 4-band Chlorophyll-dependent irradiance parameterization for a surface ABOVE-WATER solar irradiance of I0p at depth z, with CI-dependence in cloudy skies and sza-dependence in clear skies (split at CI=0.1). Albedo is implicit in this parameterization, need to provide the above-water irradiance.
    '''
    
    if CI<0.1: #clear skies
        cosza = np.cos(np.deg2rad(sza))
        A1 = 0.033*Chl - 0.025*cosza + 0.419 
        A2 = -0.010*Chl - 0.007*cosza + 0.231
        A3 = -0.019*Chl - 0.003*cosza + 0.195
        A4 = -0.006*Chl - 0.004*cosza + 0.154
        K1 = 0.066*Chl + 0.006*cosza + 0.066
        K2 = 0.396*Chl - 0.027*cosza + 0.886
        K3 = 7.68*Chl - 2.49*cosza + 17.81
        K4 = 51.27*Chl + 13.14*cosza + 665.19
        
    else: #cloudy skies
        A1 = 0.026*Chl + 0.112*CI + 0.366
        A2 = -0.009*Chl + 0.034*CI + 0.207
        A3 = -0.015*Chl - 0.006*CI + 0.188
        A4 = -0.003*Chl - 0.131*CI + 0.169
        K1 = 0.063*Chl - 0.015*CI + 0.082
        K2 = 0.278*Chl - 0.562*CI + 1.02
        K3 = 3.91*Chl - 12.91*CI + 16.62
        K4 = 16.64*Chl - 478.28*CI + 736.56

    I = I0p*(A1*np.exp(-K1*z) + A2*np.exp(-K2*z) + A3*np.exp(-K3*z) + A4*np.exp(-K4*z))
    return I



def I_O03(I0, z, Chl):
    '''
    Ohlmann 2003 Chl-dependent 2-band Irradiance parameterization for a surface net solar irradiace of I0 at depth z, derived from empirical fits to model outputs in Ohlmann et al 2000. ONLY VALID FOR DEPTHS > 2M, done with a CI=0.4 and SZA=15 because that minimized their errors. Kinda useless for us.
    '''
    #retrieve coefficients for the input chlorophyll value
    O03 = ohlmann03.interp(Chl=Chl,method='linear')
    #Calculate irradiance at this depth
    I = I0*(O03.A1*np.exp(-O03.B1*z) + O03.A2*np.exp(-O03.B2*z))
    return I
    