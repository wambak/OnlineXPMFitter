import tkinter as tk
from tkinter.font import Font
import csv
from tkinter import Canvas
import socket, threading
import time
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import urllib.request
import os
from pathlib import WindowsPath, Path
from scipy.special import wofz
from lmfit.models import SkewedVoigtModel
from lmfit.models import ExponentialGaussianModel
from lmfit import Model
from lmfit import Minimizer, minimize, fit_report, conf_interval, printfuncs
from scipy import integrate
from scipy.special import erfc
from uncertainties.core import wrap
import sched
from datetime import datetime
from WF_SDK import device, scope, wavegen
from decimal import Decimal

try:
  device_data = device.open('Analog Discovery 2')     #if the AD2 is connected
  is_connected = True
  print('AD2 connected')
except:     # if it isn't, we want to ignore the code that handles it. i would make this nicer if i could find the wf_sdk docs
  is_connected = False
  print('AD2 not found')

schedule = sched.scheduler(time.time, time.sleep) 
schedule_start_time = 0.0
eventList = []
time0 = []
volt0 = []
isfibersave = True
fibersavetime = 0.0
total = 0.0
energized = False   # SHOULD start as OFF by default
in_progress = False

def startSchedule():
    try :
    #print('running schedule')
        schedule.run()
    except Exception as exc :
        return
    #print('finished schedule')
    return

def closeshutter(text,dwell) :  #energizing
  global energized
  try:
    (WindowsPath.home() / '.shutterclosed').touch()     #creates a temporary file "shutterclosed"
  except:
    Path('/tmp/.shutterclosed').touch() # for non-Windows machines
  #print('closed it')
  if not energized: 
      print('Energizing')
      
      if is_connected: # if AD2 was connected, and the shutter is open. if not, just ignore this block
        listoftimes = [0.25, 0.25]   # hardcoding the trial-and-error values from shutterGUI.py
        # code for energizing - adapted from shutterGUI.py
        for index in range(0, len(listoftimes), 2):
            wavegen.generate(device_data, channel=2, function=wavegen.function.sine, offset=4.98, frequency=1e02,
              amplitude=0.02) # on
            time.sleep(listoftimes[index])
            wavegen.generate(device_data, channel=2, function=wavegen.function.sine, offset=0.02, frequency=1e02,
              amplitude=0.02) # we briefly pulse the AD2 back off to make the spring engage/de-engage smoother 
            time.sleep(listoftimes[index + 1])
            wavegen.generate(device_data, channel=2, function=wavegen.function.sine, offset=4.98, frequency=1e02,
              amplitude=0.02) # on for good this time
            time.sleep(listoftimes[index])
      energized = True # this happens regardless of connection
  baseurl = 'http://' + str(root.graph.scopeIPText.get('1.0','end-1c'))
  urllib.request.urlopen( baseurl + '/?COMMAND=ACQUIRE:MODE+SAMPLE' ).read() #KDW 2021-1-17 doing this clears the averaging
  time.sleep(0.25)
  urllib.request.urlopen( baseurl + '/?COMMAND=ACQUIRE:MODE+AVERAGE' ).read() #and this starts it over from scratch
  return

def openshutter(text,dwell) :   #deenergizing
  global energized
  try:
    pathExists = (WindowsPath.home() / '.shutterclosed').exists()       #deletes temp file
    if pathExists == True:
      (WindowsPath.home() / '.shutterclosed').unlink()  
  except:
    pathExists = Path('/tmp/.shutterclosed').exists()
    if pathExists == True:
      Path('/tmp/.shutterclosed').unlink()

  # deenergizing - adapted from shutterGUI.py
  if energized:
      print('De-energizing')
      if is_connected: # if AD2 is connected and the shutter is closed
        listoftimes = [0.4, 0.1] 
        for index in range(0, len(listoftimes), 2):
          wavegen.generate(device_data, channel=2, function=wavegen.function.sine, offset=0.02, frequency=1e02,
            amplitude=0.02) # off
          time.sleep(listoftimes[index])
          wavegen.generate(device_data, channel=2, function=wavegen.function.sine, offset=4.98, frequency=1e02,
            amplitude=0.02) # we briefly pulse the AD2 back on to make the spring engage/de-engage smoother 
          time.sleep(listoftimes[index + 1])
          wavegen.generate(device_data, channel=2, function=wavegen.function.sine, offset=0.02, frequency=1e02,
            amplitude=0.02) # off for good this time
          time.sleep(listoftimes[index])
      energized=False
  baseurl = 'http://' + str(root.graph.scopeIPText.get('1.0','end-1c'))
  urllib.request.urlopen( baseurl + '/?COMMAND=ACQUIRE:MODE+SAMPLE' ).read() #KDW 2021-1-17 doing this clears the averaging
  time.sleep(0.25)
  urllib.request.urlopen( baseurl + '/?COMMAND=ACQUIRE:MODE+AVERAGE' ).read() #and this starts it over from scratch
  return


class append_Box(tk.simpledialog.Dialog):
    """Dialog box that pops up when starting program with a filepath that already exists. 
    Options to append data, overwrite data, or back out and start over with a different path. 
    Returns string appendMode via appendBox(): 'append', 'overwrite', or 'cancel' """

    def body(self, frame): 
        self.text = tk.Label(frame, width=30, text="File exists!")
        self.text.grid(row=0, padx=5)

        return self.text

    def buttonbox(self):
        box = tk.Frame(self)

        append_b = tk.Button(box, text = 'Append', width=5, command = self.appendMode)
        append_b.pack(side='left', ipadx=10, padx=5, pady=5)
        overwrite_b = tk.Button(box, text='Overwrite', width=5, command = self.overwriteMode)
        overwrite_b.pack(side='left', ipadx=10, padx=5, pady=5)
        cancel_b = tk.Button(box, text='Cancel', width=5, command = self.cancel)
        cancel_b.pack(side='left', ipadx=10, padx=5, pady=5)

        self.bind('<Escape>', lambda x:self.cancel()) # x is only here to catch the escape key parameter 
                                                    #and stop the program trying to pass it to self.cancel()
        box.pack()

    def appendMode(self):
        self.mode = 'append'
        self.destroy()

    def overwriteMode(self):
        self.mode = 'overwrite'
        self.destroy()

    def cancel(self):
        self.mode = 'cancel'
        self.destroy()



def appendBox(parent):
    # Function to interact with append_Box. 
    appendBox = append_Box(parent, title="File I/O")
    return appendBox.mode

class grafit(tk.Frame):

    def plotit(self,  text='' , dwell=0.0 , islaser=False ):
        baseurl = 'http://' + str(self.scopeIPText.get('1.0','end-1c'))
        if islaser : #Handle the laser traces
            urllib.request.urlopen( baseurl + '/?COMMAND=data:source+CH2' ).read()
            myurl = baseurl + '/?COMMAND=wfmpre?'
            f2 = urllib.request.urlopen( myurl )
            wfmpre = f2.read().decode()
            myurl = baseurl + '/?COMMAND=curve?'
            f = urllib.request.urlopen( myurl )
            data = f.read().decode()
            wfm = [float(u) for u in data.split(',')]
            peak1volt = [ (float(dl) - float(wfmpre.split(';')[14])) * 1.0e1 * float(wfmpre.split(';')[12]) + float( wfmpre.split(';')[13]) for dl in wfm]
            urllib.request.urlopen( baseurl + '/?COMMAND=data:source+CH3' ).read()
            myurl = baseurl + '/?COMMAND=wfmpre?'
            f2 = urllib.request.urlopen( myurl )
            wfmpre = f2.read().decode()
            myurl = baseurl + '/?COMMAND=curve?'
            f = urllib.request.urlopen( myurl )
            data = f.read().decode()
            wfm = [float(u) for u in data.split(',')]
            peak2volt = [ (float(dl) - float(wfmpre.split(';')[14])) * 1.0e1 * float(wfmpre.split(';')[12]) + float( wfmpre.split(';')[13]) for dl in wfm]
            self.dataToFile[0] = 10.0
            self.dataToFile[1] = 81.9
            self.dataToFile[2] = 1.0
            self.dataToFile[3] = 2.9
            lasermax = 100.0*max(peak1volt)
            if lasermax < 0 :
                lasermax = 0.0
            self.dataToFile[8] = round(lasermax,2) #IR
            lasermax = 100.0*max(peak2volt)
            #print([f'{dd:.4f}' for dd in peak2volt])
            #print(float(wfmpre.split(';')[12]),float(wfmpre.split(';')[13]),float(wfmpre.split(';')[14]),lasermax)
            if lasermax < 0 :
                lasermax = 0.0
            self.dataToFile[9] = round(lasermax,2) #UV
            self.IRtext.delete(1.0, tk.END)
            self.IRtext.insert( 1.0, str(self.dataToFile[8]) )
            self.UVtext.delete(1.0, tk.END)
            self.UVtext.insert( 1.0, str(self.dataToFile[9]) )
            self.plt1.plot(self.t, peak1volt, 'm-')
            self.plt1.plot(self.t, peak2volt, 'c-')
            self.plt1.grid(True)
            self.maxIR.append( self.dataToFile[8] )
            self.maxUV.append( self.dataToFile[9] )
            self.figure2.axes[0].cla()
            self.plt2.grid(True)
            self.plt2.errorbar(self.xar, self.yar, [self.el, self.eh], markersize=6, fmt='^',mec='r',mfc='None')
            self.laserX.append(self.xar[-1])
            self.plt2.plot(self.laserX, self.maxIR, 'mo', fillstyle='none')
            self.plt2.plot(self.laserX, self.maxUV, 'co', fillstyle='none')
            #self.plt2.set_title("$e^{-}$ Lifetime vs Time")
            self.plt2.set_title("$e^{-}$ Lifetime [$\mu$s] vs Time")
            self.plt2.set_ylabel('$\\tau$($\mu$s)')
            self.plt2.set_xlabel('Time (h)')
            try :
                y_plot_option = self.y_plot_option.get()
                if y_plot_option == 'Use limits' : 
                    self.plt2.set_ylim([float(self.y_lowerlimit.get()),float(self.y_upperlimit.get())])

                x_plot_option = self.x_plot_option.get()
                if x_plot_option == 'Use limits' : 
                    self.plt2.set_xlim([float(self.x_lowerlimit.get()),float(self.x_upperlimit.get())])
                if x_plot_option == 'Recent hours' :
                    self.plt2.set_xlim([self.xar[-1]-float(self.recenthr.get()),self.xar[-1]+0.1])
            except Exception as exc:
                print(exc)
            fh = open(self.savePath, 'a') 
            writer = csv.writer(fh)
            writer.writerows([self.dataToFile])
            fh.close()
            self.ctr += 1
            urllib.request.urlopen( baseurl + '/?COMMAND=horizontal:trigger:position+30' ).read()
            urllib.request.urlopen( baseurl + '/?COMMAND=horizontal:main:scale+40e-6' ).read()
            urllib.request.urlopen( baseurl + '/?COMMAND=data:source+CH1' ).read()
        else :
            urllib.request.urlopen( baseurl + '/?COMMAND=horizontal:trigger:position+30' ).read()
            urllib.request.urlopen( baseurl + '/?COMMAND=horizontal:main:scale+40e-6' ).read()
            #urllib.request.urlopen( baseurl + '/?COMMAND=ACQUIRE:MODE+SAMPLE' ).read() #KDW 2021-1-17 doing this clears the averaging
            #urllib.request.urlopen( baseurl + '/?COMMAND=ACQUIRE:MODE+AVERAGE' ).read() #and this starts it over from scratch
            urllib.request.urlopen( baseurl + '/?COMMAND=data:source+CH1' ).read()
            data = ''
            myurl = 'http://' + str(self.scopeIPText.get('1.0','end-1c')) + '/?COMMAND=curve?'
            #print(myurl)
            f = urllib.request.urlopen( myurl )
            data = f.read().decode()
            #print('received ' + data)

            wfm = [float(u) for u in data.split(',')]

            # CALLING WFMPRE TO CONVERT WFM TO MS AND VOLTS
            myurl = baseurl + '/?COMMAND=wfmpre?'
            #print(myurl)
            f2 = urllib.request.urlopen( myurl )
            wfmpre = f2.read().decode()

            # EXAMPLE WFMPRE:
            # wfmpre = '1;8;ASC;RP;MSB;500;"Ch1, AC coupling, 2.0E-2 V/div, 4.0E-5 s/div, 500 points, Average mode";Y;8.0E-7;0;-1.2E-4;"s";8.0E-4;0.0E0;-5.4E1;"V"'
            self.t = [1.0e6 * (float(wfmpre.split(';')[8]) * float(i) + float(wfmpre.split(';')[10])) for i in range(0, len(wfm))]
            volt = [1.0e3 * (((float(dl) - float(wfmpre.split(';')[14]))) * float(wfmpre.split(';')[12]) - float(wfmpre.split(';')[13])) for dl in wfm]

            try:
                pathExists = (WindowsPath.home() / '.shutterclosed').exists()
            except:
                pathExists = Path('/tmp/.shutterclosed').exists()

            if pathExists == False: #store signal+background trace in self.topHat
                self.topHat = np.array(wfm)

                # Waveform to plot
                wvPlot = self.topHat - self.nontopHat
                wvPlot = [1.0e3 * (((float(dl) - float(wfmpre.split(';')[14]))) * float(wfmpre.split(';')[12]) - float(wfmpre.split(';')[13])) for dl in wvPlot]
                if self.isStandard :
                    wts = np.zeros(len(self.t))
                    for idx,ti in zip(range(0,len(self.t)),self.t) :
                        if ( ti > -50.0 and ti < -15.0 ) or ( ti > 25.0 and ti < 65.0 ) or ( ti > 125.0 and ti < 150.0 ) :
                            wts[idx]=1.0
                    result = self.wavmodel.fit(wvPlot, self.wavparams, weights = wts, x=self.t, method='nelder', max_nfev=1000)
                    b = result.params
                    ci_txt = result.ci_report()
                else :
                    wts = np.ones(len(self.t))
                    for idx,ti in zip(range(0,len(self.t)),self.t) :
                        if ( ti > -50.0 and ti < -15.0 ) or ( ti > 25.0 and ti < 65.0 ) or ( ti > 125.0 and ti < 150.0 ) :
                            wts[idx]=1.0
                    result = self.wavmodel.fit(wvPlot, self.wavparams, weights = wts, x=self.t)
                    #result = self.wavmodel.fit(wvPlot, self.wavparams, x=t)
                    b = result.best_values
                    ci_txt = result.ci_report()

                print('results--->')
                catrow = (ci_txt.split('\n')[1].split(':')[1])
                anrow = (ci_txt.split('\n')[2].split(':')[1])
                cat = b['cat']
                an = b['an']
                offst = b['offst'] 
                print(cat,an,offst)
                cat_ll = cat + np.fromstring(catrow,dtype=float,sep=' ')[2]
                cat_ul = cat + np.fromstring(catrow,dtype=float,sep=' ')[4]
                an_ll = an + np.fromstring(anrow,dtype=float,sep=' ')[2]
                an_ul = an + np.fromstring(anrow,dtype=float,sep=' ')[4]
                an_95CL_ll = an + 2*np.fromstring(anrow,dtype=float,sep=' ')[2]
                an_95CL_ul = an + 2*np.fromstring(anrow,dtype=float,sep=' ')[4]
                cat_95CL_ll = cat + 2*np.fromstring(catrow,dtype=float,sep=' ')[2]
                cat_95CL_ul = cat + 2*np.fromstring(catrow,dtype=float,sep=' ')[4]
                #print( catrow , anrow , cat, an)

                tfine = np.arange(self.t[0], self.t[-1] + 0.8, (self.t[1] - self.t[0]) / 10.0)

                #cat=49.98262 
                #an=46.10659
                #offst=43.619015

                # adding data to list that gets printed to file ( columns 5 and 6)
                self.dataToFile[4] = round(float(cat),3)
                self.dataToFile[5] = round(float(an),3)
                self.dataToFile[6] = round(float(offst),3)
                self.dataToFile[10] = float(result.chisqr/result.nfree) #reduced chisq
                numAvgstr = urllib.request.urlopen( baseurl + '/?COMMAND=ACQuire:NUMAVG?' ).read().decode()
                self.dataToFile[11] = float(numAvgstr) #TODO: number in average
                nTrigstr = urllib.request.urlopen( baseurl + '/?COMMAND=ACQuire:NUMACQ?' ).read().decode()
                self.dataToFile[12] = float(nTrigstr) #TODO: number of triggers

                self.xar.append((time.time() - self.start_time) / 3600)
                ts = self.xar[-1]*3600.0 + self.start_time  
                self.dataToFile[7] = str( ts + 126144000.0 + 2208988800.0 )
                tau_e = (81.9 - 10.0) / np.log( cat / an )
                upper_bound = (81.9 - 10.0) / np.log( cat_ll / an_ul )
                lower_bound = (81.9 - 10.0) / np.log( cat_ul / an_ll )
                errorl = tau_e - lower_bound
                errorh = upper_bound - tau_e
                if tau_e < 0 : #unphysical lifetime
                    print('!!!! Uphysical lifetime !!!')
                    a = an
                    delta_a = an - an_ll
                    tau_lower_limit = (81.9-10.0)/np.log((a+2*delta_a)/a)
                    delta_tau_h = (81.9 - 10.0) / np.log( cat_ll / an_ul ) - tau_e 
                    number_of_sigma = (tau_lower_limit-tau_e)/errorh #approximate number of sigma
                    errorl = 0.0
                    errorh = np.inf 
                    tau_e = tau_lower_limit
                    print(errorl, errorh, tau_lower_limit , tau_e, number_of_sigma)
                print('cat and an', self.dataToFile[4], self.dataToFile[5], offst,tau_e,cat-b['cat'],an-b['an'],offst-b['offst'])
                self.yar.append(tau_e)

                # we are appending the data to the row which will be written to the file
                self.dataToFile[13] = float(cat_ll) 
                self.dataToFile[14] = float(cat_ul)
                self.dataToFile[15] = float(an_ll)
                self.dataToFile[16] = float(an_ul)
                self.el.append(errorl)
                self.eh.append(errorh)
                self.figure1.axes[0].cla()
                self.plt1.grid()
                # PLOTTTING PEAKS:
                # self.plt.subplot(211)
                # PLOTTING WAVEFORM:
                # self.plt.subplot(212)
                self.plt1.plot(self.t, wvPlot, 'g-')
                self.plt1.grid(True) 
                tfine = np.arange(self.t[0], self.t[-1] + 0.8, (self.t[1] - self.t[0]) / 10.0)
            
                if self.isStandard :
                    #self.plt1.plot(tfine,
                    #               self.wavmodel.eval(x=tfine, an=b['an'], cat=b['cat'], tcrise=b['tcrise'],
                    #               tarise=b['tarise'], offst=b['offst'], thold=b['thold'] ), 'r-', label='standard')
                    self.plt1.plot(tfine,self.wavmodel.eval(x=tfine, cat=cat, an=an, tcrise=b['tcrise'],tarise=b['tarise'], offst=offst, thold=b['thold'] ), 'r-', label='standard')
                else :
                    #self.plt1.plot(tfine,
                    #               self.wavmodel.eval(x=tfine, an=b['an'], cat=b['cat'], cent_c=b['cent_c'], tcrise=b['tcrise'],
                    #               tarise=b['tarise'], cent_a=b['cent_a'], gam_a=b['gam_a'],
                    #               gam_c=b['gam_c'], skew_a=b['skew_a'], offst=b['offst']), 'r-', label='proposed: an=42.04 mV')
                    self.plt1.plot(tfine, self.wavmodel.eval(x=tfine, cat=cat, an=an, sig_c=b['sig_c'], gam_c=b['gam_c'], sig_a=b['sig_a'], skew_c=b['skew_c'], skew_a=b['skew_a'], offst=b['offst']), 'r-', label='new voigt')
                    self.plt1.grid(True)

                #self.canvas2draw_idle()

                #Statistics labels Updates 
                # optional TODO: if anyone ever updates these fields with that wrapper, do it with these functions too
                self.eventNumLabel.config(state='normal') # disabled makes it read only but also stops us from editing, so we need to 
                                                    # switch back to update the display. 
                self.eventNumLabel.delete(1.0, tk.END)  # clear row before inserting
                self.eventNumLabel.insert(1.0,str(self.ctr))
                self.eventNumLabel.config(state='disabled')

                self.lifetimeLabel.config(state='normal')
                self.lifetimeLabel.delete(1.0, tk.END)
                self.lifetimeLabel.insert(1.0,str(round((81.9 - 10.0) / np.log( self.dataToFile[4] / self.dataToFile[5] ),1)) )
                self.lifeErrorsTxt.delete(1.0, tk.END)
                try: 
                    uppererrorbar = errorh
                    lowererrorbar = errorl
                    asymerrorboundstr = '+' + '%1.3E' % Decimal(uppererrorbar) + '\n'
                    asymerrorboundstr = asymerrorboundstr + '-' + '%1.3E' % Decimal(lowererrorbar) 
                    self.lifeErrorsTxt.insert(1.0,asymerrorboundstr)
                except Exception as exc:
                    print(exc)
                print(tau_e,lower_bound,upper_bound)
                #('+'++'-'+str(tau_e-lower_bound)
                self.lifetimeLabel.tag_add('rightjust',1.0,tk.END)
                self.lifetimeLabel.tag_config('rightjust',justify=tk.RIGHT)
                self.lifetimeLabel.config(state='disabled')

                self.cathodeLabel.config(state='normal')
                self.cathodeLabel.delete(1.0, tk.END)
                self.cathodeLabel.insert(1.0,str(self.dataToFile[4]))
                self.cathodeLabel.config(state='disabled')

                self.anodeLabel.config(state='normal')
                self.anodeLabel.delete(1.0, tk.END)
                self.anodeLabel.insert(1.0,str(self.dataToFile[5]))
                self.anodeLabel.config(state='disabled')

                self.offsetLabel.config(state='normal')
                self.offsetLabel.delete(1.0, tk.END)
                self.offsetLabel.insert(1.0,str(self.dataToFile[6]))
                self.offsetLabel.config(state='disabled')
                
                self.plt1.set_title("Most recent waveform")
                self.plt1.set_ylabel("MilliVolts")
                self.plt1.set_xlabel(u"Time (\u03bcs)")
                urllib.request.urlopen( baseurl + '/?COMMAND=horizontal:trigger:position+30' ).read()
                urllib.request.urlopen( baseurl + '/?COMMAND=horizontal:main:scale+40e-9' ).read()
            else:
                self.nontopHat = np.array(wfm)
            
        print('drawing idle')
        self.canvas1.draw_idle()
        self.canvas2.draw_idle()
        # here we check if the save file has been defined, if so write to it, if not state that it is not set
        try:
            self.saveFile
            if not self.saveFile.closed:
                print('Save file is still open')
            else:
                print('Save file has been closed')
        except :
            print('Save file is not set')
        
    def ud(self) :
        try :
            #print('schedule length',len(schedule.queue))
            if in_progress == False : #and total > 0 :
                if self.currentStatus.get(1.0,tk.END) != 'CURRENT STATUS TO BE DISPLAYED\n' : 
                    self.currentStatus.config(state='normal')
                    self.currentStatus.delete(1.0,tk.END)
                    self.currentStatus.insert(1.0,'Ending the run')
                    self.currentStatus.config(state='disabled')
                else :
                    self.currentStatus.config(state='normal')
                    self.currentStatus.delete(1.0,tk.END)
                    self.currentStatus.insert(1.0,'CURRENT STATUS TO BE DISPLAYED')
                    self.currentStatus.config(state='disabled')
            if len( schedule.queue ) > 0 :
                ct = int((schedule.queue[0].time - time.time())*100)
                #print(ct)
            if len( schedule.queue[0].argument[0] ) > 1 and ct > 0 :    
                #print('schedule length',len(schedule.queue))
                #print(schedule.queue[0].argument[0]+str(ct/100)+' sec')
                    
                # Current State update, works exactly identical to the rest of the label updates in plotit()
                self.currentStatus.config(state='normal')
                self.currentStatus.delete(1.0,tk.END)
                self.currentStatus.insert(1.0,schedule.queue[0].argument[0]+str(ct/100)+' sec')
                self.currentStatus.config(state='disabled')
                
                #Time Update
                now = datetime.now()
                self.currentTime.config(state='normal')
                self.currentTime.delete(1.0, tk.END)
                self.currentTime.insert(1.0,now.strftime('%m-%d-%Y %H:%M:%S'))
                self.currentTime.config(state='disabled')
            if len( schedule.queue ) == 0 :
                print('Schedule is depopulated')
        except Exception as exc:
            pass
            #print(exc)
            #for event in schedule.queue :
            #    schedule.cancel(event)
            #self.saveFile.close()
            #openshutter('',0.0)
            #os._exit(0)
        try :
            y_plot_option = self.y_plot_option.get()
            if y_plot_option == 'Use limits' : 
                self.plt2.set_ylim([float(self.y_lowerlimit.get()),float(self.y_upperlimit.get())])
            else :
                self.plt2.autoscale(enable=True, axis='y')

            x_plot_option = self.x_plot_option.get()
            if x_plot_option == 'Use limits' : 
                self.plt2.set_xlim([float(self.x_lowerlimit.get()),float(self.x_upperlimit.get())])
            elif x_plot_option == 'Recent hours' :
                self.plt2.set_xlim([self.xar[-1]-float(self.x_recenthr.get()),self.xar[-1]+0.1])
            else :
                self.plt2.autoscale(enable=True, axis='x')
            
            self.canvas2.draw_idle()
        except Exception as exc:
            print(exc)
        self.parent.after(1000,self.ud)

    def on_closing(self):
        try :
            for event in schedule.queue :
                schedule.cancel(event)
            print("File closed")
            self.saveFile.close()
            openshutter('',0.0)
            baseurl = 'http://' + str(self.scopeIPText.get('1.0','end-1c'))
            urllib.request.urlopen( baseurl + '/?COMMAND=horizontal:main:scale+40e-6' ).read()
        except Exception as exc:
            return
            #os._exit(0)

    def togglefibersave(self):
        global isfibersave
        isfibersave = ( self.fibersaveval == 1 )
        return



    def end_it(self):
        global in_progress
        global total
        if in_progress == False :
            return
        events_remaining = len( schedule.queue )
        print('Progress',events_remaining)
        total = 0.0
        for event in schedule.queue :
            schedule.cancel(event)

        schedule.enter( total, 1, openshutter, argument=('Ending the run ',5.0) ) 
        total = total + 5
        schedule.run()
        
        print("File closed")
        self.saveFile.close()
        self.ctr = 0
        self.xar = []
        self.yar = []
        self.el = []
        self.eh = []
        self.maxIR = []
        self.maxUV = []
        self.laserX = []
        self.dataToFile = np.zeros(17)
        self.t = []

        self.scheduThread = threading.Thread(target=startSchedule)
        self.scheduThread.daemon = True
        baseurl = 'http://' + str(self.scopeIPText.get('1.0','end-1c'))
        urllib.request.urlopen( baseurl + '/?COMMAND=horizontal:main:scale+40e-6' ).read()

        in_progress = False

    def fitter_func(self, x, cat, an, tcrise, tarise, offst,thold ):
        global err
        #thold = 395.3
        z = np.array(x)
        x_beg = z[z<10.0]
        x_mid = z[(z>=10.0)*(z<81.9)]
        x_end = z[z>=81.9]
        y_beg = 0.5*cat*erfc(-(x_beg-10.0)/tcrise) - 0.5*an*erfc(-(x_beg-81.9)/tarise)
        y_mid = 0.5*cat*erfc(-(x_mid-10.0)/tcrise)*np.exp(-(x_mid-10.0)/thold) - 0.5*an*erfc(-(x_mid-81.9)/tarise)
        y_end = 0.5*cat*erfc(-(x_end-10.0)/tcrise)*np.exp(-(x_end-10.0)/thold) - 0.5*an*erfc(-(x_end-81.9)/tarise)*np.exp(-(x_end-81.9)/thold)
        y = np.concatenate((y_beg,y_mid,y_end),axis=None)
        y = y + offst
        return y

    def extra_smeared(self, x, cat, an, tcrise, cent_c, gam_c, tarise, cent_a, gam_a, skew_a, offst):
        self.catpars['amplitude'].value = cat
        self.catpars['sigma'].value = tcrise
        self.catpars['center'].value = cent_c
        self.catpars['gamma'].value = gam_c
        # tfine = np.arange(x[-1]-1000.0,x[-1],0.08)
        integrand_c = self.catmodel.eval(self.catpars, x=x)
        integral_c = integrate.cumulative_trapezoid(integrand_c, x)
        integral_c = np.append(integral_c, integral_c[-1])
        y = integral_c * np.exp(-(x - 10.0) / 395.3)
        self.pars['amplitude'].value = an
        self.pars['sigma'].value = tarise
        self.pars['center'].value = cent_a
        self.pars['gamma'].value = gam_a
        self.pars['skew'].value = skew_a
        integrand_a = self.pkmodel.eval(self.pars, x=x)
        integral_a = integrate.cumulative_trapezoid(integrand_a, x)
        integral_a = np.append(integral_a, integral_a[-1])
        y = y - integral_a * np.exp(-(x - 81.9) / 395.3)
        y = y + offst
        return y
    
    def skewVoigtC_skewVoigtA_yesGamma (self, x, cat, sig_c, gam_c, skew_c, an, sig_a, skew_a,offst):
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
        anode = -(1.0/CF)*np.exp(-(x-ta)/thold)*integrate.cumulative_trapezoid( np.exp((x-ta)/thold)*i_a, x, initial=0.0)

        # cathode - this is just the anode code again 
        z_c = (x - tc + gam_c*1j)/(sig_c*sqrt2)
        realw_c = -cat*np.real(wofz(z_c))/(sqrt2pi*sig_c)
        i_c = (realw_c)*(2.0/(1.0+np.exp((tc-x)/skew_c)))
        cathode = -(1.0/CF)*np.exp(-(x-tc)/thold)*integrate.cumulative_trapezoid( np.exp((x-tc)/thold)*i_c, x, initial=0.0)

        return cathode + anode + offst

    def control(self):
        #print("Self.control...")
        try :
            global total
            global fibersavetime
            dwellclosed = 32.0 ### 32.0
            dwellopen = float(self.waitT_input.get())
            fibersavetime = float(self.fibersavesb.get())  #for fibersave
            tbc = dwellclosed
            tf = 0 
            if ( len( schedule.queue ) == 0 and in_progress == True ) or ( len(schedule.queue) == 6  and root.graph.ctr > 0 ) :
                print( 'length of queue',len( schedule.queue ) )
                for iii in range(0,10) :
                    text = '*Initializing acquisition ---SHUTTER CLOSED--- '
                    #print(text)
                    if isfibersave and root.graph.ctr > 0 and iii == 0 : 
                        text = '*Fiber-saving mode: ---SHUTTER CLOSED--- resume in '
                    schedule.enter( total, 1, closeshutter, argument=(text,5.0) ) 
                    text = '*Acquisition mode ---SHUTTER CLOSED--- capture background trace in '
                    #if isfibersave and root.graph.ctr > 0 :
                    #  text = '*Fiber-saving mode: ---SHUTTER CLOSED--- next acquisition in '
                    total = total + 5 + dwellclosed
                    schedule.enter( total, 1, root.graph.plotit, argument=(text,dwellclosed+5) )
                    total = total + 5 
                    text = '*Capturing (signal+background) ---OPENING SHUTTER--- '
                    schedule.enter( total, 1, openshutter, argument=(text,5.0) )
                    text = 'Acquisition mode ---SHUTTER OPEN--- capture (signal+background) in '
                    total = total + dwellopen 
                    schedule.enter( total, 1, root.graph.plotit, argument=(text,dwellopen))
                    #text = 'Acquisition mode ---SHUTTER OPEN--- capturing laser traces '
                    total = total + 32 + 5
                    schedule.enter( total, 1, root.graph.plotit , argument = ('Getting IR, UV Laser traces in ',32.0,True) )
                    #total = total + 1 
                    #schedule.enter( total, 1, root.graph.plotit , argument = ('Getting UV Laser trace ',1.0,True) )
                    if isfibersave and iii == 9 : 
                        text = '*Fiber-saving mode: ---CLOSING SHUTTER--- '
                        #print(text)
                        total = total + 5 
                        fibersavetime = float(self.fibersavesb.get())  #for fibersave
                        schedule.enter( total, 1, closeshutter, argument=(text,5.0) )
                        total = total + fibersavetime
                    else : 
                        total = total + 5
                        text = '*Acquisition mode ---CLOSING SHUTTER--- preparing next acquisition '
                        schedule.enter( total, 1, closeshutter, argument=(text,5.0) )
                        total = total + tbc 
                #print('schedule has '+str( len(schedule.queue) ))
                if isfibersave : 
                    fibersavetime = float(self.fibersavesb.get())  #for fibersave
                    total = fibersavetime
                else :
                    total = tbc
                #print(schedule.queue)
        except Exception as exc :
            print(exc)
            for event in schedule.queue :
                print( event )
                schedule.cancel(event)
            self.saveFile.close()
            #os._exit(0)
        self.parent.after(5, self.control) 

    def set_saveFile(self):
        global in_progress
        if in_progress == True : # stops the program from trying to start if it's already running
            tk.messagebox.showerror(title="Error", message='Run already in progress')
            return
        in_progress = True
        self.savePath=self.fileSaveInput.get('1.0', 'end-1c')
        if os.path.isfile(self.savePath): # if file exists, check for start mode 
            state = appendBox(self.parent)
 
            if state == 'append':
                if self.ctr == 0 :
                    if self.scheduThread.is_alive() == False :
                        self.control() 

                print("Opening in append mode")
                self.saveFile = open(r'%s' % (self.savePath), "r+")  # opens in append-and-read mode 
                count = 0
                newtime = 0
                self.xar = []
                self.yar = []
                self.el = []
                self.eh = []
                self.maxIR = []
                self.maxUV = []
                self.laserX = []
                self.dataToFile = np.zeros(17)
                self.t = []
                while True: # read existing data to graph 
                    self.ctr = count
                    count += 1
                    line = self.saveFile.readline()
                    if line == '\n': # skips the blank lines
                        continue
                    elif line == '':  # for eof
                        break
                    else: # lines with data
                        line.strip() # take the newline off so it doesn't become its own, empty list item
                        linearr = line.split(',')
                        linearr = [float(i) for i in linearr]
                        # line[4], line[5] are cat, an 
                        # line[7] is ts (time (seconds)) 
                        # line[13] - line[16] are cat_ll, cat_ul, an_ll, an_ul
                    if count == 1:
                        newtime = linearr[7] - (126144000+2208988800) # start time of the existing file data in Unix time
                    ts = linearr[7] - (126144000+2208988800)
                    self.xar.append( (ts - newtime) / 3600)
                    tau_e = (81.9 - 10.0) / np.log(linearr[4] / linearr[5])
                    upper_bound = -(81.9 - 10.0) / np.log(linearr[16] / linearr[13])
                    lower_bound = -(81.9 - 10.0) / np.log(linearr[15] / linearr[14])
                    errorl = tau_e - lower_bound
                    errorh = upper_bound - tau_e
                    
                    if tau_e < 0 : #unphysical lifetime
                        a = linearr[4]
                        delta_a = linearr[5] - linearr[14]
                        tau_lower_limit = (81.9-10.0)/np.log((a+delta_a)/a)
                        number_of_sigma = (tau_lower_limit-tau_e)/errorh #approximate number of sigma
                        print(tau_lower_limit , number_of_sigma)
                        errorl = 0.0
                        errorh = self.plt2.get_ylim()[1] - tau_lower_limit
                        tau_e = tau_lower_limit

                    self.yar.append(tau_e)
                    self.el.append(errorl)
                    self.eh.append(errorh)
                    self.maxIR.append( linearr[8] )
                    self.maxUV.append( linearr[9] )
                    self.figure2.axes[0].cla()
                    self.figure1.axes[0].cla()
                    self.plt1.grid(True)
                    self.plt2.grid(True)
                    self.plt2.errorbar(self.xar, self.yar, [self.el, self.eh], markersize=6, fmt='^',mec='r',mfc='None')
                    #self.plt2.set_ylim([self.plt2.get_ylim()[0],])
                    self.laserX.append(self.xar[-1])
                    self.plt2.plot(self.laserX, self.maxIR, 'mo', fillstyle='none')
                    self.plt2.plot(self.laserX, self.maxUV, 'co', fillstyle='none')
                    self.plt1.set_title("Most recent waveform")
                    self.plt1.set_ylabel("MilliVolts")
                    self.plt1.set_xlabel(u"Time (\u03bcs)")
                    self.plt2.set_title("$e^{-}$ Lifetime [$\mu$s] vs Time")
                    self.plt2.set_ylabel('')#$\\tau$($\mu$s)')
                    self.plt2.set_xlabel('Time (h)')
                    self.canvas1.draw_idle()
                    self.canvas2.draw_idle()
                self.start_time = newtime # extending the time backwards to accomodate old data so graph renders correctly 
                # bumping the display clock back to match 
                self.startTime.config(state='normal')
                self.startTime.delete(1.0, tk.END)
                self.startTime.insert('1.0',time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(newtime)))
                self.startTime.config(state='disabled')
                if self.scheduThread.is_alive() == False :
                    self.scheduThread.start()
            elif state == 'overwrite':
                if self.ctr == 0 :
                    if self.scheduThread.is_alive() == False :
                        self.control() 

                print("Opening in overwrite mode")
                self.saveFile = open(r'%s' % (self.savePath), "w+")  # deletes and overwrites old data
                self.start_time = time.time()
                self.startTime.config(state='normal')
                self.startTime.delete(1.0, tk.END)
                self.startTime.insert('1.0',time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(self.start_time)))
                self.startTime.config(state='disabled')
                self.xar = []
                self.yar = []
                self.el = []
                self.eh = []
                self.maxIR = []
                self.maxUV = []
                self.laserX = []
                self.dataToFile = np.zeros(17)
                self.t = []
                self.figure2.axes[0].cla()
                self.figure1.axes[0].cla()
                self.plt1.grid()
                self.plt2.grid()
                self.plt1.set_title("Most recent waveform")
                self.plt1.set_ylabel("MilliVolts")
                self.plt1.set_xlabel(u"Time (\u03bcs)")
                #self.plt2.set_title("$e^{-}$ Lifetime vs Time")
                self.plt2.set_title("$e^{-}$ Lifetime [$\mu$s] vs Time")
                self.plt2.set_ylabel('$\\tau$($\mu$s)')
                self.plt2.set_xlabel('Time (h)')
                self.canvas1.draw_idle()
                self.canvas2.draw_idle()
                if self.ctr == 0 :
                    if self.scheduThread.is_alive() == False :
                        self.scheduThread.start()
            else:
                # cancel. go back to the start screen to enter a new file and try again
                in_progress = False
                return
        
        else:  
        # if file does not already exist
            if self.ctr == 0:
                if self.scheduThread.is_alive() == False :
                    self.control()
            print("Creating file")
            self.saveFile = open(r'%s' % (self.savePath), "w+") 
            if self.ctr == 0 :
                if self.scheduThread.is_alive() == False :
                    self.scheduThread.start()
            self.figure2.axes[0].cla()
            self.plt1.grid()
            self.plt2.grid()
            self.canvas2.draw_idle()
            self.ctr = 0
            self.start_time = time.time()
            self.startTime.config(state='normal')
            self.startTime.delete(1.0, tk.END)
            self.startTime.insert('1.0',time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(self.start_time)))
            self.startTime.config(state='disabled')
        in_progress = True


    def __init__(self, parent):
        self.ctr = 0
        self.start_time = time.time()
        self.topHat = []
        self.nontopHat = []
        try:
                self.saveFile = open('/dev/null','r')
        except:
                self.saveFile = open('NUL','r')
        tk.Frame.__init__(self, parent)
        # Set up figure and plot
        #self.figure = Figure(figsize=(3, 5), dpi=100)
        #self.figure = Figure(figsize=(6, 5), dpi=100)

        #self.plt = self.figure.add_subplot(111)

        # Create parent, which is the class onlineXPMFitter from down below
        self.parent = parent
        self.parent.configure(bg="lightgray") # set the background color
        #self.T = tk.Text(self.parent, height=1, width=5, font=("Courier", 64))
        #self.T.grid(row=0, column=1)
        #self.T.config(foreground="blue")
        self.isStandard = True
        self.scheduThread = threading.Thread(target=startSchedule)
        self.scheduThread.daemon = True

        if self.isStandard : # old fitter func 
          self.p_i = [49.98262, 46.10659, 10.0, 1.0, 2.9, 81.9, 395.3, 0.8, 0.9, 43.619015]
          self.wavmodel = Model(self.fitter_func,nan_policy='raise')
          self.wavparams = self.wavmodel.make_params()
          self.wavparams['cat'].value = self.p_i[0]
          self.wavparams['cat'].vary = True
          self.wavparams['an'].value = self.p_i[1]
          self.wavparams['an'].vary = True
          self.wavparams['thold'].value = self.p_i[6]
          self.wavparams['thold'].vary = False
          self.wavparams['tcrise'].value = self.p_i[3]
          self.wavparams['tcrise'].vary = False
          self.wavparams['tarise'].value = self.p_i[4]
          self.wavparams['tarise'].vary = False
          self.wavparams['offst'].value = self.p_i[9]
          self.wavparams['offst'].vary = True
        else :  # newer version of the fitter func
          """ self.p_i = [37.873185672822736, 40.81570955383812, 10.0, 3.598, 0.980325759727434, 81.9, 1.80825, 0.8, 0.9, 0.2]
          self.wavmodel = Model(self.extra_smeared, nan_policy='raise')
          self.wavparams = self.wavmodel.make_params()
          self.wavparams['cat'].value = self.p_i[0]
          self.wavparams['cat'].vary = True
          self.wavparams['an'].value = self.p_i[1]
          self.wavparams['an'].vary = True
          self.wavparams['cent_c'].value = self.p_i[2]
          self.wavparams['cent_c'].vary = False
          # self.wavparams['thold'].value = self.p_i[3]
          # self.wavparams['thold'].vary = False
          self.wavparams['tcrise'].value = self.p_i[3]
          self.wavparams['tcrise'].vary = False
          self.wavparams['tarise'].value = self.p_i[4]
          self.wavparams['tarise'].vary = False
          self.wavparams['cent_a'].value = self.p_i[5]
          self.wavparams['cent_a'].vary = False
          self.wavparams['gam_a'].value = self.p_i[6]
          self.wavparams['gam_a'].vary = False
          self.wavparams['skew_a'].value = self.p_i[7]
          self.wavparams['skew_a'].vary = False
          self.wavparams['gam_c'].value = self.p_i[8]
          self.wavparams['gam_c'].vary = False
          self.wavparams['offst'].value = self.p_i[9]
          self.wavparams['offst'].vary = True """
          self.wavmodel = Model(self.skewVoigtC_skewVoigtA_yesGamma,nan_policy='raise')
          self.wavparams = self.wavmodel.make_params()
          self.wavparams['cat'].value = 30.0   # formerly qc
          self.wavparams['cat'].vary = True
          self.wavparams['sig_c'].value = 1.99877086
          self.wavparams['sig_c'].vary = False
          self.wavparams['gam_c'].value = 0.48781508
          self.wavparams['gam_c'].vary = False
          self.wavparams['skew_c'].value = 8.21680391
          self.wavparams['skew_c'].vary = False
          self.wavparams['an'].value = 15.0     # formerly qa
          self.wavparams['an'].vary = True
          self.wavparams['sig_a'].value = 0.76546294
          self.wavparams['sig_a'].vary = False
          self.wavparams['skew_a'].value = 2.81726754
          self.wavparams['skew_a'].vary = False
          self.wavparams['offst'].value = 0.5768164
          self.wavparams['offst'].vary = True
        
        self.pkmodel = SkewedVoigtModel()
        self.catmodel = ExponentialGaussianModel()
        self.pars = self.pkmodel.make_params()
        self.catpars = self.catmodel.make_params()

        self.xar = []
        self.yar = []
        self.el = []
        self.eh = []
        self.maxIR = []
        self.maxUV = []
        self.laserX = []
        self.dataToFile = np.zeros(17)
        self.t = []
	
	# seconds to wait between captures
        self.dummyspacer3 = tk.Label(height=1,width=1)
        self.dummyspacer3.config(text=' ')
        self.dummyspacer3.config(bg=parent['background'])
        self.dummyspacer3.grid(row=18,column=0)
        self.waitT_label = tk.Label(height=1, width=30)
        self.waitT_label.config(text="Seconds to wait between captures")
        self.waitT_label.grid(row=18, column=1, sticky = tk.E)

        # seconds to wait input 
        self.defaultTime = tk.StringVar(self.parent)
        self.defaultTime.set('33.0')  ### 33.0
        self.waitT_input = tk.Spinbox(self.parent, increment=1.0, foreground='black', background='white', from_ = 33.0 , to = 10800.0 , textvariable = self.defaultTime)
        self.waitT_input.grid(row=19, column=1, sticky=tk.W)

        # next two lines are for the texbox for entries
        self.fileSaveInput = tk.Text( height=3, width=24) # text box( where user enters path)
        self.fileSaveInput.insert(tk.END,os.getcwd() + os.sep + 'xpm_fitter_data' + os.sep + 'testData')
        self.fileSaveInput.grid( row=24, column=1, sticky=tk.W)

        # the scope IP address LABEL
        self.scopeIPLabel = tk.Label(height=1, width=24)
        self.scopeIPLabel.config(text='Scope IP Address')
        self.scopeIPLabel.grid( row = 30 , column=1, sticky=tk.W)
        # the scope IP address TEXT
        self.scopeIPText = tk.Text( height=1, width=18 )
        self.scopeIPText.insert(tk.END, '134.79.229.21')
        self.scopeIPText.grid( row=31, column=1,sticky=tk.W)
        self.scopeIPText.tag_add('rightjust',1.0,tk.END)
        self.scopeIPText.tag_config('rightjust',justify=tk.RIGHT)
        
        self.dummyspacer11 = tk.Label(height=1,width=1)
        self.dummyspacer11.config(text=' ')
        self.dummyspacer11.config(bg=parent['background'])
        self.dummyspacer11.grid(row=32,column=0)
        # 'Start' button starts a run
        self.StartButton = tk.Button(text="Start", command=lambda:self.set_saveFile())
        self.StartButton.grid(row=34, column=1)
        self.dummyspacer12 = tk.Label(height=1,width=1)
        self.dummyspacer12.config(text=' ')
        self.dummyspacer12.config(bg=parent['background'])
        self.dummyspacer12.grid(row=36,column=0)

        # 'End' button ends a run
        self.EndButton = tk.Button(text="End", command=lambda:self.end_it())
        self.EndButton.grid(row=38, column=1)

        # positioning of the graphs
        self.figure1 = Figure(figsize=(5, 4), dpi=100)
        self.figure2 = Figure(figsize=(6, 4), dpi=100)

        self.plt1 = self.figure1.add_subplot(111)
        self.plt1.set_title("Most recent waveform")
        self.plt1.set_ylabel("MilliVolts")
        self.plt1.set_xlabel(u"Time (\u03bcs)")
        self.plt1.grid(True)
        #self.figure1.tight_layout()

        self.plt2 = self.figure2.add_subplot(111)
        self.plt2.set_title("$e^{-}$ Lifetime [$\mu$s] vs Time")
        #self.plt2.set_title("$e^{-}$ Lifetime vs Time")
        self.plt2.set_ylabel('$\\tau$($\mu$s)')
        self.plt2.set_xlabel('Time (h)')
        self.plt2.grid(True)

        self.y_plotLabel = tk.Label(height=1,width=14)
        self.y_plotLabel.config(text='Y-axis config',font='Helvetica 12 bold')
        self.y_plotLabel.config(bg=parent['background'])
        self.y_plotLabel.grid(row=65, column=17,sticky=tk.E)
        self.y_plot_option = tk.StringVar(master=self.parent)
        self.y_plot_option.set('Autoscale')
        self.y_pd_menu = tk.OptionMenu(self.parent,self.y_plot_option,'Autoscale','Use limits')
        self.y_pd_menu.grid(row=65, column=18,rowspan=3,sticky=tk.N)
        self.y_upperlimitstr = tk.StringVar(self.parent)
        self.y_upperlimitstr.set('100000.0')  ### 33.0
        self.y_upperlimit = tk.Spinbox(self.parent, increment=1.0, foreground='black', background='white', from_ = -1.0e99 , to = +1.0e99 , textvariable = self.y_upperlimitstr,width=14)
        self.y_upperlimit.grid(row=65, column=19,sticky=tk.W)
        self.y_upperlimitLabel = tk.Label(height=1,width=15)
        self.y_upperlimitLabel.config(text='upper limit [us]')
        self.y_upperlimitLabel.config(bg=parent['background'])
        self.y_upperlimitLabel.grid(row=65,column=19,sticky=tk.E)

        self.y_lowerlimitstr = tk.StringVar(self.parent)
        self.y_lowerlimitstr.set('100.0')  ### 33.0
        self.y_lowerlimit = tk.Spinbox(self.parent, increment=1.0, foreground='black', background='white', from_ = -1.0e99 , to = +1.0e99 , textvariable = self.y_lowerlimitstr,width=14)
        self.y_lowerlimit.grid(row=66, column=19,sticky=tk.W)
        self.y_lowerlimitLabel = tk.Label(height=1,width=15)
        self.y_lowerlimitLabel.config(text='lower limit [us]')
        self.y_lowerlimitLabel.config(bg=parent['background'])
        self.y_lowerlimitLabel.grid(row=66,column=19,sticky=tk.E)
        self.midrightdummyspacer = tk.Label(height=1)

        self.midrightdummyspacer.config(text=' ',bg=parent['background'])
        self.midrightdummyspacer.grid(row=67,column=19)
        
        self.x_plotLabel = tk.Label(height=1,width=14)
        self.x_plotLabel.config(text='X-axis config',font='Helvetica 12 bold')
        self.x_plotLabel.config(bg=parent['background'])
        self.x_plotLabel.grid(row=68, column=17,sticky=tk.E)
        self.x_plot_option = tk.StringVar(master=self.parent)
        self.x_plot_option.set('Autoscale')
        self.x_pd_menu = tk.OptionMenu(self.parent,self.x_plot_option,'Autoscale','Use limits','Recent hours')
        self.x_pd_menu.grid(row=68, column=18,rowspan=3,sticky=tk.N)
        self.x_upperlimitstr = tk.StringVar(self.parent)
        self.x_upperlimitstr.set('10.0')  ### 33.0
        self.x_upperlimit = tk.Spinbox(self.parent, increment=1.0, foreground='black', background='white', from_ = -1.0e99 , to = +1.0e99 , textvariable = self.x_upperlimitstr,width=14)
        self.x_upperlimit.grid(row=68, column=19,sticky=tk.W)
        self.x_upperlimitLabel = tk.Label(height=1,width=15)
        self.x_upperlimitLabel.config(text='upper limit [h]')
        self.x_upperlimitLabel.config(bg=parent['background'])
        self.x_upperlimitLabel.grid(row=68,column=19,sticky=tk.E)

        self.x_lowerlimitstr = tk.StringVar(self.parent)
        self.x_lowerlimitstr.set('0.0')  ### 33.0
        self.x_lowerlimit = tk.Spinbox(self.parent, increment=1.0, foreground='black', background='white', from_ = -1.0e99 , to = +1.0e99 , textvariable = self.x_lowerlimitstr,width=14)
        self.x_lowerlimit.grid(row=69, column=19,sticky=tk.W)
        self.x_lowerlimitLabel = tk.Label(height=1,width=15)
        self.x_lowerlimitLabel.config(text='lower limit [h]')
        self.x_lowerlimitLabel.config(bg=parent['background'])
        self.x_lowerlimitLabel.grid(row=69,column=19,sticky=tk.E)

        self.x_recenthrstr = tk.StringVar(self.parent)
        self.x_recenthrstr.set('1.0')  ### 33.0
        self.x_recenthr = tk.Spinbox(self.parent, increment=1.0, foreground='black', background='white', from_ = -1.0e99 , to = +1.0e99 , textvariable = self.x_recenthrstr,width=14)
        self.x_recenthr.grid(row=70, column=19,sticky=tk.W)
        self.x_recenthrLabel = tk.Label(height=1,width=15)
        self.x_recenthrLabel.config(text='recent hours [h]')
        self.x_recenthrLabel.config(bg=parent['background'])
        self.x_recenthrLabel.grid(row=70,column=19,sticky=tk.E)
        
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=self.parent)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=self.parent)
        # self.graph = Graph(self)
        # self.canvas = self.graph.canvas
        self.plot_widget1 = self.canvas1.get_tk_widget()
        self.plot_widget2 = self.canvas2.get_tk_widget()

        self.plot_widget1.grid(row=1, rowspan=64, column=5, columnspan=8)
        self.plot_widget2.grid(row=1, rowspan=64, column=14, columnspan=8)

        self.canvas1.draw_idle()
        self.canvas2.draw_idle()

        # Lower Labels Field
        # Optional TODO: might be better if these text fields are made into their own class, since right now they all function basically
        # identically and there's a lot of repeated lines, both here and in plotit/ud 

        self.dummyspacerright = tk.Label(height=1,width=1)
        self.dummyspacerright.config(text=' ')
        self.dummyspacerright.config(bg=parent['background'])
        self.dummyspacerright.grid(row=71,column=27)
        
        self.currentStatus = tk.Text(height=1, width=78, bg='white',fg='#CF9FFF', wrap='none')  # current status is displayed
        self.currentStatus.insert(tk.END,'CURRENT STATUS TO BE DISPLAYED')
        self.currentStatus.config(state='disabled')
        self.currentStatus.grid( row=72, column=16, columnspan=5)
        
        self.fibersaveval = tk.IntVar(value=1)
        self.fibersavecheck = tk.Checkbutton( text ='Fiber-saving mode' , command = self.togglefibersave, variable=self.fibersaveval, onvalue=1, offvalue=0)
        self.fibersavecheck.config(bg=parent['background'])
        self.fibersavecheck.grid( row = 74, column = 17 )
        self.fstimestr = tk.StringVar(value='10800.0')
        self.fibersavesb = tk.Spinbox( self.parent , width = 34 , increment=1.0 , foreground='black', background='white', from_ = 33.0 , to = 259200.0 , textvariable = self.fstimestr )
        self.fibersavesb.grid( row = 74, column = 19 , sticky = tk.W )
        global fibersavetime
        fibersavetime = float(self.fibersavesb.get())  #for fibersave
        self.fibersavelabel = tk.Label(height=1,width=24)
        self.fibersavelabel.config(text='Closed-shutter time [s]',bg=parent['background'])
        self.fibersavelabel.grid(row=74,column=18, sticky=tk.E)

        self.dummyspacer8 = tk.Label(height=1,width=1)
        self.dummyspacer8.config(text=' ')
        self.dummyspacer8.config(bg=parent['background'])
        self.dummyspacer8.grid(row=71,column=27)
        
        now = datetime.now()
        self.currentTime = tk.Text(height=1, width=21, bg='white',fg='#7393B3', wrap='none')  # current time / time the interface was launched
        self.currentTime.insert(tk.END, now.strftime('%m-%d-%Y %H:%M:%S'))
        self.currentTime.config(state='disabled')
        self.currentTime.grid( row=71, column=8,columnspan=3,sticky = tk.W)
        
        self.timeLabel2 = tk.Label(height=1, width=12)
        self.timeLabel2.config(text="Current Time")
        self.timeLabel2.grid(row=70, column=8,sticky = tk.SW)
        self.dummyspacer = tk.Label(height=1,width=1)
        self.dummyspacer.config(text=' ')
        self.dummyspacer.config(bg=parent['background'])
        self.dummyspacer.grid(row=68,column=5)
        self.timeLabel1 = tk.Label(height=1, width=12)
        self.timeLabel1.config(text="Start Time")
        self.timeLabel1.grid(row=70, column=5,sticky = tk.SW)
        
        self.startTime = tk.Text(height=1, width=21, bg='white',fg='#7393B3', wrap='none')  # code start time
        self.startTime.insert(tk.END, now.strftime('%m-%d-%Y %H:%M:%S'))
        self.startTime.config(state='disabled')
        self.startTime.grid( row=71, column=5, columnspan=3, sticky = tk.W)

        self.dummyspacer5 = tk.Label(height=1,width=1)
        self.dummyspacer5.config(text=' ')
        self.dummyspacer5.config(bg=parent['background'])
        self.dummyspacer5.grid(row=72,column=5)

        self.IRLabel = tk.Label( height =1, width=12 )
        self.IRLabel.config(text='IR signal [mV]')
        self.IRLabel.grid(row=73,column=6, sticky=tk.E)
        self.IRtext = tk.Text(height=1, width=12 )
        self.IRtext.insert(tk.END,'0.00')
        self.IRtext.grid(row=74,column=6, sticky=tk.E)

        self.UVLabel = tk.Label( height =1, width=12 )
        self.UVLabel.config(text='UV signal [mV]')
        self.UVLabel.grid(row=73,column=8,sticky=tk.W)
        self.UVtext = tk.Text(height=1, width=12 )
        self.UVtext.insert(tk.END,'0.00')
        self.UVtext.grid(row=74,column=8,sticky=tk.W)

        self.dummyspacer2 = tk.Label(height=1,width=1)
        self.dummyspacer2.config(text=' ')
        self.dummyspacer2.config(bg=parent['background'])
        self.dummyspacer2.grid(row=75,column=5)
        
        #Stats Info Display 
        self.eventNumName = tk.Label(height=1, width=6)
        self.eventNumName.config(text='Event# ')
        self.eventNumName.grid(row=65, column=5, sticky=tk.E)
        self.eventNumLabel = tk.Text(height=1, width=12, wrap='none') 
        self.eventNumLabel.insert(1.0,str(0))
        self.eventNumLabel.config(state='disabled')
        self.eventNumLabel.grid(row=65, column=6)
        
        self.lifetimeName = tk.Label(height=1, width=7)
        self.lifetimeName.config(text='Lifetime')
        self.lifetimeName.grid(row=65, column=7)
        self.lifetimeLabel = tk.Text(height=1, width=12, wrap='none') 
        self.lifetimeLabel.insert(1.0,'0.00')
        self.lifetimeLabel.tag_add('rightjust',1.0,tk.END)
        self.lifetimeLabel.tag_config('rightjust',justify=tk.RIGHT)
        self.lifetimeLabel.config(state='disabled')
        self.lifetimeLabel.grid(row=65, column=8,sticky=tk.E)
        self.lifeErrorsTxt = tk.Text(height=2, width=12, font=Font(size=6))  
        self.lifeErrorsTxt.insert(1.0,'0.000\n0.000')
        self.lifeErrorsTxt.grid(row=65,column=9,sticky=tk.W)
        
        self.anodeName = tk.Label(height=1, width=6)
        self.anodeName.config(text='Anode')
        self.anodeName.grid(row=66, column=5, sticky=tk.E)
        self.anodeLabel = tk.Text(height=1, width=12, wrap='none') 
        self.anodeLabel.insert(1.0,'0.00')
        self.anodeLabel.config(state='disabled')
        self.anodeLabel.grid(row=66, column=6)
        
        self.tcName = tk.Label(height=1, width=7)
        self.tcName.config(text='Tc')
        self.tcName.grid(row=66, column=7)
        self.tcLabel = tk.Text(height=1, width=12, wrap='none') 
        self.tcLabel.insert(1.0,'10.0')
        self.tcLabel.config(state='disabled')
        self.tcLabel.grid(row=66, column=8, sticky=tk.W)

        self.cathodeName = tk.Label(height=1, width=6)
        self.cathodeName.config(text='Cathode')
        self.cathodeName.grid(row=65, column=10)
        self.cathodeLabel = tk.Text(height=1, width=10, wrap='none') 
        self.cathodeLabel.insert(1.0,'0.00')
        self.cathodeLabel.config(state='disabled')
        self.cathodeLabel.grid(row=65, column=11)
        
        self.tcriseName = tk.Label(height=1, width=6)
        self.tcriseName.config(text='Tcrise')
        self.tcriseName.grid(row=66, column=10)
        self.tcriseLabel = tk.Text(height=1, width=10, wrap='none') 
        self.tcriseLabel.insert(1.0,'1.0')
        self.tcriseLabel.config(state='disabled')
        self.tcriseLabel.grid(row=66, column=11)
        
        self.offsetName = tk.Label(height=1, width=6)
        self.offsetName.config(text='Offset  ')
        self.offsetName.grid(row=67, column=10)
        self.offsetLabel = tk.Text(height=1, width=10, wrap='none') 
        self.offsetLabel.insert(1.0,'0.00')
        self.offsetLabel.config(state='disabled')
        self.offsetLabel.grid(row=67, column=11)
        
        self.taName = tk.Label(height=1, width=6)
        self.taName.config(text='Ta')
        self.taName.grid(row=67, column=5, sticky=tk.E)
        self.taLabel = tk.Text(height=1, width=12, wrap='none') 
        self.taLabel.insert(1.0,'81.9')
        self.taLabel.config(state='disabled')
        self.taLabel.grid(row=67, column=6)
        
        self.tariseName = tk.Label(height=1, width=7)
        self.tariseName.config(text='Tarise')
        self.tariseName.grid(row=67, column=7)
        self.tariseLabel = tk.Text(height=1, width=12, wrap='none') 
        self.tariseLabel.insert(1.0,'2.9')
        self.tariseLabel.config(state='disabled')
        self.tariseLabel.grid(row=67, column=8, sticky=tk.E)
        
        """self.offsetName = tk.Label(height=1, width=6)
        self.offsetName.config(text='Offset  ')
        self.offsetName.grid(row=67, column=9)
        self.offsetLabel = tk.Text(height=1, width=8, wrap='none') 
        self.offsetLabel.insert(1.0,'0.00')
        self.offsetLabel.grid(row=67, column=10)"""

        # self.fig.canvas.draw()

        # self.plotter = threading.Thread(target=self.plotit)
        # self.plotter.setDaemon(True) # MAKES CODE THREAD SAFE

    def increment(self):
        self.value.set(self.value.get() + 1)

    def decrement(self):
        self.value.set(self.value.get() - 1)

    def filter_key(self, event):
        if not event.char.isdigit() and event.keysym not in ('BackSpace', 'Delete'):
            return 'break'



class onlineXPMFitter(tk.Tk):
    """ Class instance of main/root window. Mainly responsible for showing the
        core components and setting other properties related to the main window."""

    def __init__(self):
        tk.Tk.__init__(self)

        # Set title and screen resolutions
        tk.Tk.wm_title(self, 'XPM Fitter')
        tk.Tk.minsize(self, width=1480, height=530)
        # Optional TODO: Set a custom icon for the XPM application
        # tk.Tk.iconbitmap(self, default="[example].ico")

        # Show window and control bar
        self.graph = grafit(self)
        schedule_start_time = time.time()
        self.graph.ud()
        time.sleep(1.0)

        # self.graph.pack(side='top', fill='both', expand=True)

root = onlineXPMFitter()


#scheduThread.setDaemon(True)
#scheduThread.setDaemon(True)
root.mainloop()
