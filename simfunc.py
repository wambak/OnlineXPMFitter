import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy.special import wofz
from lmfit.models import SkewedVoigtModel
from lmfit.models import ExponentialGaussianModel
from lmfit import Model
from lmfit import Minimizer, minimize, fit_report, conf_interval, printfuncs
from scipy import integrate
from scipy.special import erfc
from uncertainties.core import wrap


lifetime = 2000.0


def biGaus_skew(x,cat,sig_c,tau_c,sig_a,tau_a) :
  tc = 10.0
  ta = 81.9
  i_c = cat*(erfc((tc-x)/sig_c))*np.exp(-(x-tc)/tau_c)
  adjusted_an = np.exp(-(ta-tc)/lifetime)*cat
  #i_a = (sig_c/sig_a)*adjusted_an*(erfc((ta-x)/sig_a))*np.exp(-(x-ta)/tau_a)
  #i_a = (sig_c/sig_a)*adjusted_an*2.0/(1.0+np.exp((ta-x)/sig_a))*np.exp(-(x-ta)/tau_a)
  i_a = -((sig_c+tau_c)/(sig_a+tau_a))*erfc((ta-x)/sig_a)*np.exp(-(x-ta)/(tau_a))*adjusted_an
  return  i_c + i_a



def doubleBiGaus(x, cat, an, offst):
    tc = 10.0
    ta = 81.9
    sig_c = 2.0
    sig_a = 1.4
    i_c = cat*np.exp(-((x-tc)**2)/(2*sig_c**2))
    adjusted_an = np.exp(-(ta-tc)/lifetime)*an 
    i_a = (sig_c/sig_a)*adjusted_an*np.exp(-((x-ta)**2)/(2*sig_a**2))
    return i_c - i_a + offst

def skewVoigtC_skewVoigtA_yesGamma (x, cat, sig_c, gam_c, skew_c, an, sig_a, skew_a,offst):
    CF = 1.0  # feedback capacitance (not really 1, but it doesn't really matter anyway since it will divide out,
              # it's just here to turn voltage into charge in the equation)
    ta = 81.9 # where t = 1 on the anode
    tc = 10.0 # where t = 1 on the cathode
    sqrt2 = np.sqrt(2.0)
    sqrt2pi = np.sqrt(2.0*np.pi)
    #thold = 395.3 
    # anode
    z_a = (x - ta + sig_a*1j)/(sig_a*sqrt2) # first part of the Voigt dist
    adjusted_an = np.exp(-(ta-tc)/lifetime)*an 
    realw_a = adjusted_an*np.real(wofz(z_a))/(sqrt2pi*sig_a) # real component of a Faddeeva func of A 
    i_a = (realw_a)*(2.0/(1.0+np.exp((ta-x)/skew_a))) # multiply everything by the Fermi function before putting it into the integral
    anode = -(1.0/CF)*i_a

    # cathode - this is just the anode code again 
    z_c = (x - tc + gam_c*1j)/(sig_c*sqrt2)
    realw_c = -cat*np.real(wofz(z_c))/(sqrt2pi*sig_c)
    i_c = (realw_c)*(2.0/(1.0+np.exp((tc-x)/skew_c)))
    cathode = -(1.0/CF)*i_c

    return cathode + anode + offst

#wavmodel = Model(doubleBiGaus,nan_policy='raise')
#wavparams = wavmodel.make_params()
#wavmodel = Model(skewVoigtC_skewVoigtA_yesGamma,nan_policy='raise')
#wavparams = wavmodel.make_params()
#wavparams['cat'].value = 1.0   # formerly qc
#wavparams['sig_c'].value = 1.99877086
#wavparams['gam_c'].value = 0.48781508
#wavparams['skew_c'].value = 8.21680391
#wavparams['an'].value = 1.0     # formerly qa
#wavparams['sig_a'].value = 0.76546294
#wavparams['skew_a'].value = 2.81726754
#wavparams['offst'].value = 0.0

#wavmodel = Model(skewVoigtC_skewVoigtA_yesGamma,nan_policy='raise')
#wavparams = wavmodel.make_params()
#wavparams['cat'].value = 1.0   # formerly qc
#wavparams['sig_c'].value = 1.99877086
#wavparams['gam_c'].value = 0.48781508
#wavparams['skew_c'].value = 8.21680391
#wavparams['an'].value = 1.0     # formerly qa
#wavparams['sig_a'].value = 0.76546294
#wavparams['skew_a'].value = 2.81726754
#wavparams['offst'].value = 0.0

wavmodel = Model(biGaus_skew,nan_policy='raise')
wavparams = wavmodel.make_params()
wavparams['cat'].value = 1.0   # formerly qc
wavparams['sig_c'].value = 2.3331900249976414
wavparams['tau_c'].value = 1.9260917008369134
wavparams['sig_a'].value = 1.5068445074484516
wavparams['tau_a'].value = 1.3511128944360826

t = np.linspace(0.0,163.79,16380)
t_ad2 = np.linspace(0.0,163.79e-6,16380)
v_of_t = wavmodel.eval(wavparams,x=t)

plt.plot(t,v_of_t,'.',markersize=1.5)
plt.show()


