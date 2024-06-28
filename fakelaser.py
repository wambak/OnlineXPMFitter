from WF_SDK import device, scope, wavegen, tools, error   # import instruments
import numpy as np
from scipy.special import wofz
from lmfit.models import SkewedVoigtModel
from lmfit.models import ExponentialGaussianModel
from lmfit import Model
from lmfit import Minimizer, minimize, fit_report, conf_interval, printfuncs
from scipy import integrate
from scipy.special import erfc
from uncertainties.core import wrap
import os
from pathlib import WindowsPath, Path

import matplotlib.pyplot as plt   # needed for plotting
from time import sleep            # needed for delays
from ctypes import *
import sys

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")



def doubleBiGaus(x, cat, an, offst):
    tc = 10.0
    ta = 81.9
    sig_c = 1.0
    sig_a = 1.0
    i_c = cat*np.exp(-((x-tc)**2)/(2*sig_c**2))
    lifetime = 260.0
    adjusted_an = np.exp(-(ta-tc)/lifetime)*an 
    i_a = adjusted_an*np.exp(-((x-ta)**2)/(2*sig_a**2))
    return i_c - i_a + offst


"""-----------------------------------------------------------------------"""
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
    lifetime = 200.0
    adjusted_an = np.exp(-(ta-tc)/lifetime)*an 
    realw_a = adjusted_an*np.real(wofz(z_a))/(sqrt2pi*sig_a) # real component of a Faddeeva func of A 
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


wavmodel = Model(doubleBiGaus,nan_policy='raise')
wavparams = wavmodel.make_params()
#wavmodel = Model(skewVoigtC_skewVoigtA_yesGamma,nan_policy='raise')
#wavparams = wavmodel.make_params()
wavparams['cat'].value = 1.0   # formerly qc
#wavparams['cat'].vary = True
#wavparams['sig_c'].value = 1.99877086
#wavparams['sig_c'].vary = False
#wavparams['gam_c'].value = 0.48781508
#wavparams['gam_c'].vary = False
#wavparams['skew_c'].value = 8.21680391
#wavparams['skew_c'].vary = False
wavparams['an'].value = 1.0     # formerly qa
#wavparams['an'].vary = True
#wavparams['sig_a'].value = 0.76546294
#wavparams['sig_a'].vary = False
#wavparams['skew_a'].value = 2.81726754
#wavparams['skew_a'].vary = False
wavparams['offst'].value = 0.0
#wavparams['offst'].vary = True

t = np.linspace(0.0,163.79,16380)
t_ad2 = np.linspace(0.0,163.79e-6,16380)
print((t[-1]),(t_ad2[-1]))

v_of_t = wavmodel.eval(wavparams,x=t)

try:
    device_ct = c_int()
    dwf.FDwfEnum( scope.constants.enumfilterAll , byref(device_ct) )

    # connect to the device
    hdwf0 = c_int()
    hdwf1 = c_int()
    #dwf.FDwfDeviceOpen(c_int(1),byref(hdwf1))

    device_name0 = create_string_buffer(32)
    device_name1 = create_string_buffer(32)
    dwf.FDwfEnumDeviceName( 0 , byref(device_name0) )
    dwf.FDwfEnumDeviceName( 1 , byref(device_name1) )
    device_data = device.data()
    if device_name0.value == 'Analog Discovery 3' :
        dwf.FDwfDeviceOpen(c_int(0),byref(hdwf0))
        device_data.handle = hdwf0.value
    else :
        dwf.FDwfDeviceOpen(c_int(1),byref(hdwf1))
        device_data.handle = hdwf1.value

    device_data.name = device_name0.value
    device_data = device.__get_info__(device_data)


    """-----------------------------------"""
    # handle devices without analog I/O channels
    if device_data.name != "Digital Discovery":

        # initialize the scope with default settings
        scope.open(device_data)

        # generate a 10KHz sine signal with 2V amplitude on channel 1
        wavegen.enable(device_data, channel=1)

        # set up triggering on scope channel 1
        scope.trigger(device_data, enable=True, source=scope.constants.trigsrcDetectorAnalogIn, channel=2, level=0.1)
        wavegen.dwf.FDwfAnalogInConfigure(device_data.handle, c_int(1), c_int(False), c_int(True))

        wavegen.dwf.FDwfAnalogOutNodeEnableSet(device_data.handle, c_int(0), scope.constants.AnalogOutNodeCarrier, c_bool(True))
        wavegen.dwf.FDwfAnalogOutTriggerSourceSet(device_data.handle,c_int(-1),scope.constants.trigsrcDetectorAnalogIn)
        wavegen.dwf.FDwfDeviceTriggerSet(device_data.handle,c_int(-1),scope.constants.trigsrcDetectorAnalogIn)
        wavegen.dwf.FDwfAnalogOutRunSet(device_data.handle, c_int(0) , c_double(163.8e-6))
        wavegen.dwf.FDwfAnalogOutRepeatSet(device_data.handle, c_int(0), c_int(1))
        #wavegen.dwf.FDwfAnalogOutIdleSet(device_data.handle, c_int(-1), scope.constants.DwfAnalogOutIdleInitial)
        mydata = (c_double * len(v_of_t))()
        for i in range(0, len(mydata)) :
            mydata[i] = c_double(v_of_t[i])
        wavegen.dwf.FDwfAnalogOutNodeFunctionSet(device_data.handle, c_int(0), scope.constants.AnalogOutNodeCarrier, scope.constants.funcCustom )
        wavegen.dwf.FDwfAnalogOutNodeDataSet(device_data.handle, c_int(0), scope.constants.AnalogOutNodeCarrier, mydata, c_int(len(v_of_t)) )
        wavegen.dwf.FDwfAnalogOutNodeFrequencySet(device_data.handle, c_int(0), scope.constants.AnalogOutNodeCarrier, c_double(6.105006105e3))
        wavegen.dwf.FDwfAnalogOutNodeAmplitudeSet(device_data.handle, c_int(0), scope.constants.AnalogOutNodeCarrier, c_double(0.435))
        #wavegen.generate(device_data, channel=1, function=wavegen.function.custom, offset=0, frequency=1.25e3, amplitude=0.05, data=v_of_t)

        while True :
            try :
                sleep(0.5)
                st = c_int(0)
                #print(wavegen.dwf.FDwfAnalogOutStatus(device_data.handle, c_bool(False), st),st)
                try:
                    pathExists = (WindowsPath.home() / '.shutterclosed').exists()
                except:
                    pathExists = Path('/tmp/.shutterclosed').exists()

                if pathExists == True: #shutter is closed
                    wavegen.dwf.FDwfAnalogOutConfigure(device_data.handle, c_int(0), c_int(0))
                else :
                    wavegen.dwf.FDwfAnalogOutConfigure(device_data.handle, c_int(0), c_int(1))
            except KeyboardInterrupt :
                break

        # reset the scope
        scope.close(device_data)

        # reset the wavegen
        wavegen.close(device_data)

    """-----------------------------------"""

    # close the connection
    device.close(device_data)
except error as e:
    print(e)
    # close the connection
    #device.close(device.data)
