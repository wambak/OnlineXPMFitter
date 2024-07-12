import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from scipy.special import wofz
from lmfit.models import SkewedVoigtModel
from lmfit.models import ExponentialGaussianModel
from lmfit.models import DoniachModel
from lmfit import Model
from lmfit import Minimizer, minimize, fit_report, conf_interval, printfuncs
from scipy import integrate
from scipy.special import erfc
from uncertainties.core import wrap





def biGaus(x, cat, an, offst):
    ta = 81.9
    tc = 10.0
    sig_a = 1.0
    sig_c = 1.0

    cathode_signal = cat*np.exp( -( ((x-tc)**2)/(2*sig_c**2) ) )
    anode_signal = -an*(sig_a/sig_c)*np.exp( -( ((x-ta)**2)/(2*sig_a**2) ) )

    return cathode_signal + anode_signal + offst
    

def skewVoigtC_skewVoigtA_yesGamma (x, cat, sig_c, gam_c, skew_c, an, sig_a, skew_a,offst):
    CF = 1.0  # feedback capacitance (not really 1, but it doesn't really matter anyway since it will divide out,
              # it's just here to turn voltage into charge in the equation)
    ta = 81.9 # where t = 1 on the anode
    tc = 10.0 # where t = 1 on the cathode
    sqrt2 = np.sqrt(2.0)
    sqrt2pi = np.sqrt(2.0*np.pi)
    thold = 395.3 
    # anode
    #z_a = (x - ta + sig_a*1j)/(sig_a*sqrt2) # first part of the Voigt dist
    z_a = (x - ta)/(sig_a*sqrt2) # first part of the Voigt dist
    realw_a = an*np.real(wofz(z_a))/(sqrt2pi*sig_a) # real component of a Faddeeva func of A 
    i_a = (realw_a)*(2.0/(1.0+np.exp((ta-x)/skew_a))) # multiply everything by the Fermi function before putting it into the integral
    #anode = -(1.0/CF)*np.exp(-(x-ta)/thold)*integrate.cumulative_trapezoid( np.exp((x-ta)/thold)*i_a, x, initial=0.0)
    anode = -(1.0/CF)*i_a

    # cathode - this is just the anode code again 
    #z_c = (x - tc + gam_c*1j)/(sig_c*sqrt2)
    z_c = (x - tc)/(sig_c*sqrt2)
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
wavparams['gam_c'].value = 0.0
wavparams['gam_c'].vary = False
wavparams['skew_c'].value = 8.21680391
wavparams['skew_c'].vary = False
wavparams['an'].value = 50.0     # formerly qa
wavparams['an'].vary = True
wavparams['sig_a'].value = 0.76546294
wavparams['sig_a'].vary = False
wavparams['skew_a'].value = 2.81726754
wavparams['skew_a'].vary = False
wavparams['offst'].value = 0.0
wavparams['offst'].vary = True



wavmodel2 = Model(biGaus,nan_policy='raise')
wavparams2 = wavmodel2.make_params()
wavparams2['cat'].value = 0.435 
wavparams2['an'].value = 0.435
wavparams2['offst'].value = 0.0


t = np.linspace(0.0,163.79,16380)
t_ad2 = np.linspace(0.0,163.79e-6,16380)
#print((t[-1]),(t_ad2[-1]))

v_of_t = wavmodel2.eval(wavparams2,x=t)

np.savetxt('bigaus.csv',v_of_t)
v = v_of_t*-1.0e-3

c= []
for row in zip(t_ad2,v):
    c.append(np.array(row))
np.savetxt('pulse.csv',c,delimiter=',')
plt.plot(t_ad2,v_of_t,'r-')
#plt.plot(t_ad2,wavmodel.eval(wavparams,x=t),'k-')
plt.xlabel('Time [s]')
plt.ylabel('Signal [V]')
plt.grid()
plt.show()


