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

def skewVoigtC_skewVoigtA_yesGamma (x, cat, sig_c, gam_c, skew_c, an, sig_a, skew_a,offst):
    CF = 1.0  # feedback capacitance (not really 1, but it doesn't really matter anyway since it will divide out,
              # it's just here to turn voltage into charge in the equation)
    ta = 81.9 # where t = 1 on the anode
    tc = 10.0 # where t = 1 on the cathode
    sqrt2 = np.sqrt(2.0)
    sqrt2pi = np.sqrt(2.0*np.pi)
    thold = 395.3 
    # anode
    z_a = (x - ta + sig_a*1j)/(sig_a*sqrt2) # first part of the Voigt dist
    realw_a = an*np.real(wofz(z_a))/(sqrt2pi*sig_a) # real component of a Faddeeva func of A 
    i_a = (realw_a)*(2.0/(1.0+np.exp((ta-x)/skew_a))) # multiply everything by the Fermi function before putting it into the integral
    #anode = -(1.0/CF)*np.exp(-(x-ta)/thold)*integrate.cumulative_trapezoid( np.exp((x-ta)/thold)*i_a, x, initial=0.0)
    anode = -(1.0/CF)*i_a

    # cathode - this is just the anode code again 
    z_c = (x - tc + gam_c*1j)/(sig_c*sqrt2)
    realw_c = -cat*np.real(wofz(z_c))/(sqrt2pi*sig_c)
    i_c = (realw_c)*(2.0/(1.0+np.exp((tc-x)/skew_c)))
    #cathode = -(1.0/CF)*np.exp(-(x-tc)/thold)*integrate.cumulative_trapezoid( np.exp((x-tc)/thold)*i_c, x, initial=0.0)
    cathode = -(1.0/CF)*i_c

    return cathode + anode + offst


wavmodel = Model(skewVoigtC_skewVoigtA_yesGamma,nan_policy='raise')
wavparams = wavmodel.make_params()
wavparams['cat'].value = 50.0   # formerly qc
wavparams['cat'].vary = True
wavparams['sig_c'].value = 1.99877086
wavparams['sig_c'].vary = False
wavparams['gam_c'].value = 0.48781508
wavparams['gam_c'].vary = False
wavparams['skew_c'].value = 8.21680391
wavparams['skew_c'].vary = False
wavparams['an'].value = 50.0     # formerly qa
wavparams['an'].vary = True
wavparams['sig_a'].value = 0.76546294
wavparams['sig_a'].vary = False
wavparams['skew_a'].value = 2.81726754
wavparams['skew_a'].vary = False
wavparams['offst'].value = 0.5768164
wavparams['offst'].vary = True

t = np.linspace(0.0,1000.0,4000)
t_ad2 = np.linspace(0.0,1.0e-3,4000)
print((t[-1]),(t_ad2[-1]))

v_of_t = wavmodel.eval(wavparams,x=t)

plt.plot(t,v_of_t)
plt.show()


