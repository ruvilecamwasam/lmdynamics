# testApproximation.py

#%{[{ Initial variables and functions

import numpy as np
from scipy.integrate import odeint
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

import multiprocessing as mp

import pickle
import time

#Simulation parameters
tf  = 500 #Time interval over which to simulate
nt  = 100 #Number of data points per one unit of time.
eta = 100  #Ratio of optical to mechanical timescales
t = np.linspace(0,tf,tf*nt)
deAccuracy = 1e-4 #We make sure that the derivatives at each potential well are less than this
hugeNumber = 1e5 #Substitute for infinity
xEscaped = 1e2 #Distance at which we consider the mirror to have escaped the potential

plotFigs = False

#Load the search results
searchResults = pickle.load(open('searchTrappingCooling.p','rb'))

#Import potential variables and functions from searchTrappingCooling.py
from searchTrappingCooling import xMax, xMin, xNum, xRange, xDelta, \
  VL, Vg, V2, HL, H2

#Variables to store data

#Numerically found maximum cooling region for a given set of parameters
wellNumericParams = np.zeros((len(searchResults),4)) 
#Analytically calculated trapping cooling regions. 
wellAnalyticParams = {}

#Axes for plotting data
if plotFigs:
  fig,axes = plt.subplots(nrows=3,ncols=2,figsize=(20,15))

#Debugging functions

logfile = 'testingApprox_' + time.strftime('%y%b%d_%H%M',time.localtime()) + '.txt'
print('Log file: ' + logfile)

#Write a message to the log file
def dbg(message):
  print('Writing to log file> ' + message)

  f=open(logfile,'a+')
  f.write(time.strftime('%y %b %d %H:%M:%S',time.localtime()) + ': ')
  f.write(message)
  f.write('\n')
  f.close()

#Write a message to the log file, including parameter values
def dbgGBDelta(message,g,B,Delta):
  dbg('(g,B,Delta)=({:.2f},{:.2f},{:.2f})> '.format(g,B,Delta)+message)

#Like assert, but will write the message to the log file rather than throwing an error
def dbgAssert(cond,message):
  try:
    assert cond
  except AssertionError:
    dbg('Failed Assertion: '+message)

dbg('Beginning computation')
dbg('eta = {:1.2e}, tf={:d}, nt={:d}, deAccuracy={:1.2e}'.format(eta,tf,nt,deAccuracy))

#%}]}

#%{[{ analyseParameters() Analyse and plot the potential well

#Analyse the trapping/cooling regions for a given set of parameters
#Plot the results
#Save the data to wellAnalyticParams
def analyseParameters(g,B,Delta): 

  #Potential shapes
  potential = V2(xRange,g,B,Delta) #Net potential
  potAlpha = (g/2)*xRange+VL(xRange,0) #Potential due to alpha
  potBeta = (g/2)*xRange+(B**2)*VL(xRange,Delta) #Potential due to beta

  #Heating rates
  heating = np.array(H2(xRange,B,Delta)) #Net heating
  heatingAlpha = HL(xRange,0) #Heating due to alpha
  heatingBeta  = (B**2)*HL(xRange,Delta) #Heating due to beta

  #Find the peaks (maxima) in the potential
  peaksI,_ = sig.find_peaks(potential) #Indexes of peaks
  peaks   = xRange[peaksI] #x values of peaks

  #Find the widths of the well and cooling regions
  minimaI,_ = sig.find_peaks(-potential) #Indexes of minima
  minima   = xRange[minimaI] #x values of minima
  tcminimaI = minimaI[heating[minimaI]<0] #Trapping and cooling minima indexes
  tcminima   = xRange[tcminimaI] #Trapping and cooling minima x values
  wellWidthsRaw = sig.peak_widths(-potential,tcminimaI,rel_height=1) #Data about well widths
  wellWidthsI = wellWidthsRaw[0] #Width of wells in terms of list indexes
  wellWidths = xDelta*wellWidthsRaw[0] #Width of wells in terms of x
  #wellHeights = - wellWidthsRaw[1] #Height of the potential well
  #wellLefts = xMin + xDelta * wellWidthsRaw[2] #Left edge of potential well
  #wellRights = xMin + xDelta * wellWidthsRaw[3] #Right edge of potential well

  #Find the widths of the cooling regions
  coolingPlusI = 0*tcminimaI #Blank variable to hold the right edges of cooling regions
  coolingMinusI = 0*tcminimaI#Blank variable to hold the left edges of cooling regions
  for n,tcmin in enumerate(tcminimaI):
    coolingPlusI[n] = tcmin + np.argmax(heating[tcmin:]>0)
    coolingMinusI[n] = tcmin - np.argmax(np.flip(heating[:tcmin])>0)
  coolingPlus = xRange[coolingPlusI]
  coolingMinus = xRange[coolingMinusI]
  coolingWidths = coolingPlus - coolingMinus
  #Depth of cooling region
  coolingDepths = [np.max(potential[coolingMinusI[n]:coolingPlusI[n]])-potential[tcminI] \
    for n,tcminI in enumerate(tcminimaI)]

  if plotFigs:
    #Plot the data

    #Plot full potential V2, and show how it is made from the individual potentials
    axes[0,0].plot(xRange,potential,'k',label='Full')
    axes[0,0].plot(peaks,potential[peaksI],'rx')
    axes[0,0].plot(tcminima,potential[tcminimaI],'gx')
    axes[0,0].plot(xRange,potAlpha,'--',label='alpha')
    axes[0,0].plot(xRange,potBeta,'--',label='beta')
    axes[0,0].legend(loc='upper left')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('Potential')
    axes[0,0].set_title('Dimensionless potentials'+' g={:.2f}, B={:.2f}, Delta={:.2f}'.format(g,B,Delta))

    #Plot heating, and show how it is made from the individual heating terms
    axes[1,0].plot(xRange,heating,'k',label='Full')
    axes[1,0].plot(xRange,heatingAlpha,'--',label='alpha')
    axes[1,0].plot(xRange,heatingBeta,'--',label='beta')
    axes[1,0].legend(loc='upper left')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('dE/dt')
    axes[1,0].set_title('Dimensionless heating rate '+' g={:.2f}, B={:.2f}, Delta={:.2f}'.format(g,B,Delta))
    
    #Combine the above to show trapping cooling regions
    axes[2,0].plot(xRange,potential,'k',label='Full')
    axes[2,0].plot(xRange[heating<0],heating[heating<0],'co',label='Full',markersize=0.5)
    axes[2,0].plot(xRange[heating>0],heating[heating>0],'ro',label='Full',markersize=0.5)
    axes[2,0].plot(peaks,potential[peaksI],'rx')
    axes[2,0].plot(tcminima,potential[tcminimaI],'gx')
    #axes[,02].hlines(wellHeights,wellLefts,wellRights,'g')
    axes[2,0].vlines(coolingMinus,-2,2,'c')
    axes[2,0].vlines(coolingPlus,-2,2,'c')
    for n,(width,depth) in enumerate(zip(coolingWidths,coolingDepths)):
      axes[2,0].text(coolingPlus[n]+0.05,heating[tcminimaI[n]]+0.1,'Width: {:1.1e}'.format(width),color='c')
      axes[2,0].text(coolingPlus[n]+0.05,heating[tcminimaI[n]]-0.3,'Depth {:1.1e}'.format(depth),color='c')
    axes[2,0].set_xlabel('x')
    axes[2,0].set_ylabel('Potential')
    axes[2,0].set_title('Well analysis '+' g={:.2f}, B={:.2f}, Delta={:.2f}, width={:1.2e}'.format(g,B,Delta,wellWidths[0]))

  #Save the calculated well parameters 
  wellAnalyticParams[(g,B,Delta)] = [tcminima,coolingPlus,coolingMinus]

#%}]}

#%{[{ Simulate a given trajectory and find heating

#Differential equation for the two-colour system
#Have to make alpha, beta into two real variables
def twoColourDE(y,t,g,B,Delta):
  x, px, ar, ai, br, bi = y

  a2 = ar**2 + ai**2
  b2 = br**2 + bi**2

  #Mirror
  dxdt  = px
  dpxdt = -g + a2 + (B**2) * b2

  #Alpha
  dardt = eta*( -x*ai-ar+1 )
  daidt = eta*( x*ar-ai )

  #Beta
  dbrdt = eta*( -(Delta+x)*bi-br+1 )
  dbidt = eta*( (Delta+x)*br-bi )

  return [dxdt, dpxdt, dardt, daidt, dbrdt, dbidt]

#For a given mirror position x0 find the 'steady-state' values of 
#the other variables.
def getSSForX0(Delta,x0):

  p0 = 0
  a0 = 1/(1-1j*x0)
  b0 = 1/(1-1j*(x0+Delta))

  ar0 = np.real(a0)
  ai0 = np.imag(a0)
  br0 = np.real(b0)
  bi0 = np.imag(b0)

  return np.array([x0, p0, ar0, ai0, br0, bi0])

#Unpack a combined vector y of dynamical variables into the individual vars.
#Works if y is a single point or a timeseries from odeint.
def unpack(y):
  if y.ndim == 1:
    x   = y[0]
    px  = y[1]
    ar  = y[2]
    ai  = y[3]
    br  = y[4]
    bi  = y[5]
  elif y.ndim == 2:
    x   = y[:,0]
    px  = y[:,1]
    ar  = y[:,2]
    ai  = y[:,3]
    br  = y[:,4]
    bi  = y[:,5]
  else:
      raise Exception('Tried to unpack y, but had 3 or more dimensions!')

  return x,px,ar,ai,br,bi

#Find the energy of a given point or trajectory
def getEnergy(y,g,B,Delta):
  x,px,ar,ai,br,bi = unpack(y)
  V = g*x-np.arctan(x)-(B**2)*np.arctan(x+Delta)
  return (px**2)/2+V

#Colour trajectories based on the perturbation
def colourTraj(axes):

  #Store the graphs for positive and negative perturbations
  posLines = {}
  negLines = {}

  #Loop over all objects in the axis
  for child in axes.get_children():

    #Select the plots
    if not isinstance(child,matplotlib.lines.Line2D):
      continue
      
    #Get the perturbation for that plot from the label
    label = child.get_label()
    pert  = float(label.split(',')[0])

    #Add the plot to the appropriate dictionary based on the sign of the trajectory
    if pert>0:
      posLines[pert] = child
    if pert<0:
      negLines[pert] = child

  #Find the number of positive and negative trajectories
  nPos = len(posLines)
  nNeg = len(negLines)
  #Create an array of colour values (between 0 and 1) for each sign of perturbation
  posColours = np.linspace(0,1,nPos)
  negColours = np.linspace(0,1,nNeg)
  #The colour maps for each sign
  negMap = cm.get_cmap('cool')
  posMap = cm.get_cmap('autumn')

  #Colour the trajectories appropriately
  for n,pert in enumerate(sorted(posLines.keys())):
    posLines[pert].set_color(posMap(posColours[n])[0:3])
  for n,pert in enumerate(sorted(negLines.keys())):
    negLines[pert].set_color(negMap(negColours[n])[0:3])
    

#Find the heating rate for a given g,B,Delta, and starting position
def getHeatingFromParams(y0,g,B,Delta,wellParams):
  xL = wellParams['xL'] #Leftmost trapping/cooling region
  xR = wellParams['xR'] #Rightmost trapping/cooling region
  x0 = wellParams['x0'] #Starting position
  xD = wellParams['DX'] #Perturbation
  xWell = wellParams['xWell'] #Well minimum

  #If we are starting in the cooling region (pert<1), heck x0, xR, xL have been ordered properly
  #if abs(pert) <= 1:
  #  dbgAssert(xL <= x0 <= xR, 'xL<=x0<=xR, {:1.2e}<={:1.2e}<={:1.2e}'.format(xL,x0,xR))
  #  dbgAssert(xL <= xWell <= xR, 'xL<=xWell<=xR, {:1.2e}<={:1.2e}<={:1.2e}'.format(xL,xWell,xR))

  #Solve the DE and then unpack the results
  sol = odeint(twoColourDE,y0,t,args=(g,B,Delta))
  x,px,ar,ai,br,bi = unpack(sol)


  #Plot the trajectory

  if plotFigs:
    fig.suptitle('g={:.2f}, B={:.2f}, Delta={:.2f}, eta={:1.1e}\n (xL,xW,xR)=({:1.2e},{:1.2e},{:1.2e})'.format(g,B,Delta,eta,xL,xWell,xR))

    #Plot the position and momentum
    axes[0,1].plot(t,x-xWell,label='{:1.2e}'.format(xD))
    axes[0,1].set_xlabel('t')
    axes[0,1].set_title('x-xWell')

    #Plot the energy
    energy = getEnergy(sol,g,B,Delta)
    axes[1,1].plot(t,energy,label='{:1.2e}'.format(xD))
    axes[1,1].set_title('Energy')

  #Plot how the amplitude of oscillations is changing

  #Find the local maxima and minima of the mirror positions, i.e. extrema of oscillations
  xMaxI,_ = sig.find_peaks(x) #Indexes of maxima
  xMax    = x[xMaxI] #x values of maxima
  xMinI,_ = sig.find_peaks(-x) #Indexes of minima
  xMin    = x[xMinI] #x values of minima

  num = min(len(xMax),len(xMin)) #Number of complete oscillations
  #if num <= 3:
  #  dbg('Less than 3 oscillations for (g,B,Delta)=({:.2f},{:.2f},{:.2f})!'.format(g,B,Delta))

  amplitudes = xMax[0:num]-xMin[0:num]
  amplitudes = amplitudes[1:] #In case the first peak is messed up, if it thinks the starting value is a max/min
  if len(xMaxI) == num:
    ampTimes = t[xMaxI[1:num]]
  else:
    ampTimes = t[xMinI[1:num]]

  #Find the median change in amplitude
  #Median to disregard outliers, such as if we mis-matched a max and min peak of one oscillation.
  if len(np.diff(amplitudes)>0):
    meanHeating = np.median(np.diff(amplitudes))
  else:
    #dbgGBDelta('Zero elements in amplitudes! x0={:1.2e}'.format(y0[0]),g,B,Delta)
    meanHeating = hugeNumber

  if np.isnan(x[-1]):
    meanHeating = hugeNumber

  if abs(x[-1]) > xEscaped:
    meanHeating = hugeNumber

  if plotFigs:
    #Need a comma after the perturbation in the label, colourTraj can read the perturbation off it
    axes[2,1].plot(ampTimes,amplitudes,'-o',label='{:1.2e}, Heating = {:1.2e}'.format(xD,meanHeating))
    axes[2,1].set_title('Oscillation Amplitudes')


  return meanHeating

#%}]}

#%{[{ findWidthNumerically Simulate trajectories to find the width numerically

#Print a message, prepended with the simulation parameters
#We need this due to the asynchronous execution
def printWithParams(message,g,B,Delta):
  pass
  #print('(g,B,Delta)=({:.2f},{:.2f},{:.2f})> '.format(g,B,Delta) + message)

#Simulate trajectories to find the width numerically
def findWidthNumerically(nsearchResult):
  n,searchResult = nsearchResult
  g,B,Delta = searchResult
  analyseParameters(g,B,Delta)
  wells = np.transpose(np.array(wellAnalyticParams[(g,B,Delta)]))
  bestDX = 0
  xCentre = 1e2

  for well in wells:
    xWell = well[0]
    xR = well[1]
    xL = well[2]
    dbgAssert(xL <= xWell <= xR,'xL<=xWell<=xR Failed for (g,B,Delta)=({:.2f},{:.2f},{:.2f})'.format(g,B,Delta))
    #print('\nxWell={:1.2e}, xL={:1.2e}, xR={:1.2e}'.format(xWell,xR,xL))
    
    #Check that the well really is a well
    ySS = getSSForX0(Delta,xWell)
    if np.max(twoColourDE(ySS,t,g,B,Delta)[1:]) > deAccuracy:
      dbg('For ({:.2f},{:.2f},{:.2f}), we do not have a steady state (error={:.2e})! Increase xNum in searchTrappingCooling.py'.format(g,B,Delta,np.max(twoColourDE(ySS,t,g,B,Delta))))

    #Simulate the trajectory 
    pertAbs = [1.5, 2, 3, 2.5, 0.99, 4, 6, 8, 10, 5, 7, 9, 0.95, 0.8]
    pertList = pertAbs + [-p for p in pertAbs]
    
    DXList = [xR*pert*(xR-xWell) for pert in pertAbs] + [xL*pert*(xL-xWell) for pert in pertAbs] \
      + [-xR*pert*(xR-xWell) for pert in pertAbs] + [-xL*pert*(xL-xWell) for pert in pertAbs]

    minFailedPertPos = hugeNumber
    minFailedPertNeg = -hugeNumber
    minFailedDXPos = hugeNumber
    minFailedDXNeg = -hugeNumber

    for pert in pertList:
      #Turn the perturbation into a DX
      #Skip perturbations if we have failed a smaller perturbation
      if pert >= 0:
        if pert > minFailedPertPos:
          printWithParams('  Skipping pert={:.2f} as we have already failed p={:.2f}'.format(pert,minFailedPertPos),g,B,Delta)
          continue
        DX = pert*(xR-xWell)
      if pert < 0:
        if pert < minFailedPertNeg:
          printWithParams('  Skipping pert={:.2f} as we have already failed p={:.2f}'.format(pert,minFailedPertNeg),g,B,Delta)
          continue
        DX = -pert*(xL-xWell)
      
      #Skip if we have failed a smaller DX
      if abs(DX) <= bestDX:
        printWithParams('  Skipping DX={:1.2e} as it is less than bestDX'.format(DX),g,B,Delta)
        continue

      wellParams = {'xWell': xWell, 'xL': xL, 'xR':  xR, 'DX': DX, 'x0': xWell+DX}
      y0 = ySS+np.array([DX,0,0,0,0,0])
      heatingPos = getHeatingFromParams(y0,g,B,Delta,wellParams)
      
      if heatingPos < 0:
        
        #Try the other side
        DX = -DX
        wellParams = {'xWell': xWell, 'xL': xL, 'xR':  xR, 'DX': DX, 'x0': xWell+DX}
        y0 = ySS+np.array([DX,0,0,0,0,0])
        heatingNeg = getHeatingFromParams(y0,g,B,Delta,wellParams)
        heating = max(heatingPos,heatingNeg)

        #If the trajectory was cooling
        if heating < 0:
          dbgAssert(abs(DX)>bestDX,'abs(DX) was less than bestDX, (g,B,Delta)=({:.2f},{:.2f},{:.2f})'.format(g,B,Delta))
          bestDX = abs(DX)
          xCentre = xWell
          printWithParams('  >DX={:1.2e} is our new best DX!'.format(DX),g,B,Delta)

          if DX == max(np.abs(DXList)):
            dbg('Maxed out on (g,B,Delta)=({:.2f},{:.2f},{:.2f})'.format(g,B,Delta))
        #If the trajectory was heating
        else:
          printWithParams('  DX={:.2f} was heating'.format(DX),g,B,Delta)
          if DX > 0:
            minFailedDXPos = min(minFailedDXPos,DX)
          else:
            minFailedDXNeg = max(minFailedDXNeg,DX)
      else:
        printWithParams('  DX={:.2f} was heating'.format(DX),g,B,Delta)
        if DX > 0:
          minFailedDXPos = min(minFailedDXPos,DX)
        else:
          minFailedDXNeg = max(minFailedDXNeg,DX)

      #Clear the plot of this trajectory, leaving the potential plots alone.


    if plotFigs:
      colourTraj(axes[0,1])
      colourTraj(axes[1,1])
      colourTraj(axes[2,1])
      axes[0,1].legend(loc='upper left',ncol=2)
      axes[1,1].legend(loc='upper left',ncol=2)
      axes[2,1].legend(loc='upper left',ncol=2)

      plt.savefig('testingPlots/well_({:.2f},{:.2f},{:.2f}).png'.format(g,B,Delta))
      axes[0,1].cla()
      axes[1,1].cla()
      axes[2,1].cla()

  coolingDepth = min(V2(xWell-bestDX,g,B,Delta),V2(xWell+bestDX,g,B,Delta))-V2(xWell,g,B,Delta)

  if plotFigs:
    #Clear the potential plots
    axes[0,0].cla()
    axes[1,0].cla()
    axes[2,0].cla()

  if n % 5 == 0:
    print('Just finished n={:d}!'.format(n))
    #np.savetxt('wellNumericParams.csv',wellNumericParams,delimiter=',')
  return [g, B, Delta, bestDX, xCentre, coolingDepth]
    
#%}]}

#Multiprocessing
pool = mp.Pool(mp.cpu_count())
results = pool.map_async(findWidthNumerically,enumerate(searchResults))
pool.close()
wellNumericParams = results.get()
print('Finished simulations!')

print('Saving data...',end='')
np.savetxt('wellNumericParams(eta_{:.2f}).csv'.format(eta),wellNumericParams,delimiter=',')
pickle.dump(wellAnalyticParams,open('wellAnalyticParams(eta_{:.2f}).p'.format(eta),'w+b'))
print(' done!')
dbg('Done!')
