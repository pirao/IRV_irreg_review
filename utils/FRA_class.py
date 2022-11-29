import pandas as pd
import seaborn as sns
import numpy as np
from numpy import load
import scipy as sp
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import welch


class FRA_irregularities():
    def __init__(self,L_min=1.524,L_max=304.8,N=3000,k=0.25,signal=None,dt=None,signal_type='vert'):
        
        self.L_min=L_min # The default values of L_min and L_max are the the maximum permissible range of wavelengths for FRA standards
        self.L_max=L_max
        self.irreg_type = None
        self.N_harmonics = N
        self.k = k
        
        self.omega_max = None
        self.omega_min = None
        self.d_omega = None
        self.omega = None
        
        self.wave=None
        
        self.s_vert=None
        self.s_lat=None
        self.vert_irreg = None
        self.lat_irreg = None
        
        self.class_list = None
        
        self.signal = signal
        self.dt = dt
        self.signal_type = signal_type
        
    def PSD(self,type_irreg=6):
        
        self.omega_max = 2*np.pi/self.L_min # Maximum angular frequency (spatial wavenumber) in rad/m
        self.omega_min = 2*np.pi/self.L_max # Minimum angular frequency (spatial wavenumber) in rad/m
    
        self.d_omega = (self.omega_max-self.omega_min)/self.N_harmonics      # Frequency increment (rad/m)
        n = np.arange(1,self.N_harmonics+1,1)                      # index vector
        self.omega = self.omega_min + (n-0.5)*self.d_omega    # discrete angular frequency (rad/m)
    
        # Creating wavelength domain vector
        self.wave = 2*np.pi/self.omega
    
        if type_irreg == 6:
            Av = 0.0339*10**-4  # m^2 * (rad/m)
            Aa = 0.0339*10**-4  # m^2 * (rad/m)
            omega_c = 0.8245    # rad/m
            omega_s = 0.438     # rad/m
       
        elif type_irreg == 5:
            Av = 0.2095*10**-4  
            Aa = 0.0762*10**-4 
            omega_c = 0.8245   
            omega_s = 0.8209    
       
        elif type_irreg == 4:
            Av = 0.5376*10**-4  
            Aa = 0.3027*10**-4 
            omega_c = 0.8245   
            omega_s = 1.1312    

        elif type_irreg == 3:
            Av = 0.6816*10**-4  
            Aa = 0.4128*10**-4 
            omega_c = 0.8245   
            omega_s = 0.852    
    
        
        elif type_irreg == 2:
            Av = 1.0181*10**-4  
            Aa = 1.2107*10**-4 
            omega_c = 0.8245   
            omega_s = 0.9308    
    
        
        elif type_irreg == 1:
            Av = 1.2107*10**-4  
            Aa = 3.3634*10**-4 
            omega_c = 0.8245   
            omega_s = 0.6046
            
        else:
            print('Provide a FRA classe between 6 and 1')    
            return None
    
        self.s_vert = 2*np.pi*(self.k*Av*omega_c**2)/((self.omega**2)*(self.omega**2+omega_c**2)) # m^2/(1/m)
        self.s_lat = 2*np.pi*(self.k*Aa*omega_c**2)/((self.omega**2)*(self.omega**2+omega_c**2)) # m^2/(1/m)
        
        return self.wave,self.omega,self.s_vert,self.s_lat
        
    def _create_PSD(self,class_list=[6,5,4]):
        
        self.class_list=class_list
        self.vert_irreg = []
        self.lat_irreg = []
        
        for item in self.class_list:
            _,_,vert, lat = FRA_irregularities.PSD(self,type_irreg=item)
            self.vert_irreg.append(vert)
            self.lat_irreg.append(lat)
        
        print('Classes {} were created'.format(class_list))
        
        return self.wave,self.omega,self.vert_irreg,self.lat_irreg
        
        
def Welch_PSD(signal, fs, window_size_frac=0.1, overlap_frac=0.5):
    
    #fs = sampling frequency - samples in 1 meter

    segment_size = np.int32(window_size_frac*len(signal))
    # round up to next highest power of 2 - used for zero padding
    fft_size = 2 ** (int(np.log2(segment_size)) + 1)

    overlap_size = overlap_frac*segment_size

    f, welch_coef = welch(x=signal,
                          fs=fs,
                          nperseg=segment_size,
                          noverlap=overlap_size,
                          nfft=fft_size,
                          return_onesided=True,
                          scaling='density',
                          detrend='constant',
                          window='hann',
                          average='mean')

    return f, welch_coef 