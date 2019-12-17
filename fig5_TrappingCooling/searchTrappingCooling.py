## searchTrappingCooling.py - Search for regions which are simultaneously trapping and cooling %{[{
import numpy as np
import scipy.signal as sig
import pickle
import time
import multiprocessing as mp

from itertools import product

#%}]}

## Potential and heating functions %{[{

#Potential due to a single laser
def VL(x,Delta):
  return -np.arctan(x+Delta)
#Potential due to gravity
def Vg(x,g):
  return g*x
#Potential due to two lasers
def V2(x,g,B,Delta):
  return g*x - np.arctan(x)-(B**2)*np.arctan(x+Delta)
#Heating due to a single laser
def HL(x,Delta):
  return 4*(x+Delta)/( (1+(x+Delta)**2)**3 )
#Potential due to two lasers
def H2(x,B,Delta):
  return 4*x/( (1+x**2)**3 ) + (B**2)*4*(x+Delta)/( (1+(x+Delta)**2)**3 )

#%}]}

## Define parameters %{[{
#Define test parameters
ng  = 500
nB  = 500
nD  = 1000
gRange = np.linspace(0,2,ng)
BRange = np.linspace(0,1,nB)
DeltaRange = np.linspace(-10,10,nD)

#Range to plot over
xMax   = 10
xMin   = -xMax
xNum   = 1e4
xRange = np.linspace(xMin,xMax,xNum)
xDelta = xRange[1]-xRange[0]

xRangeToStore = ['xMax', 'xMin', 'xNum', 'xRange', 'xDelta']
xRangeData = {v:eval(v) for v in xRangeToStore}

#%}]}

## Perform a grid search %{[{

def searchBD(g):
  print('Commencing search over g={:.2f}'.format(g))
  found = 0
  results = np.zeros((nB*nD,3),dtype=float)
  for B,Delta in product(BRange,DeltaRange):
    if g > 1+B**2:
      continue

    potential = V2(xRange,g,B,Delta)
    minimaI,_ = sig.find_peaks(-potential)
    minima   = xRange[minimaI]
    
    wellHeating = H2(minima,B,Delta)
    goodPoint = minima[wellHeating < 0]
    
    if len(goodPoint) >0:
      results[found] = (g,B,Delta)
      found +=1

  return results[0:found]

#We want to import variables such as xNum into testApproximation.py, but doing this triggers
#this whole script to run. The below if statement stops the below code from running unless we explicitly call searchTrappingCooling.py
if __name__ == "__main__":
  nVals = ng*nB*nD
  print('Searching {:d} values!'.format(nVals))
  
  pool = mp.Pool(mp.cpu_count())
  results = pool.map_async(searchBD,gRange)
  pool.close()
  paramsFound = np.concatenate(results.get())
  numParamsFound = len(paramsFound)
  
  
  filename='searchTrappingCooling.p'
  print('Writing results to '+filename)
  pickle.dump(paramsFound,open(filename,'wb'))
  
  np.savetxt('searchTrappingCooling.csv', paramsFound,delimiter=',')
