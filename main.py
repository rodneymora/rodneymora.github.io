import numpy as np
import matplotlib.pyplot as plt

## JONSWAp spectrum
def JONSWAP_HsTp(f, Tp = 10, Hs = 1.5, gamma = 3.3):
  g = 9.807
  sigma_a = 0.07
  sigma_b = 0.08
  fp = 1 / Tp

  shape_par = np.hstack(( gamma**np.exp(-(f[f <= fp] - fp)**2 / (2 * sigma_a**2 * fp * 2)),
                         gamma**np.exp(-(f[f > fp] - fp)**2 / (2 * sigma_b**2 * fp * 2)) ))

  E = (g**2 * (2 * np.pi)**-4 * fp**-1 * f**-4) * np.exp(-(f / fp)**-4) * shape_par

  Hm0 = np.trapz(E, x = f)
  
  E = Hs**2/16/Hm0*E

  return E

def DirSpec_HsTp(f, dirs, Hs = 1.5, Tp = 10, dirp = 90, spr = 60, gamma = 3.3, model = 'D', sf = 'sech'):
  nd = len(dirs)
  ## Compute one dimensional spectrum
  E = JONSWAP_HsTp(f, Tp, Hs, gamma, model)

  frac = 0.9 

  Beta = 2 * np.arctanh(frac) / (spr * np.pi / 180)
  nd2 = int(np.fix(nd / 2))
  dirs2 = np.hstack((dirs[nd2:nd] - 360, dirs, dirs[:nd2] + 360)) 
  D = 0.5 * Beta * (1 / np.cosh(Beta * (dirs2 - dirp) * np.pi / 180))**2
    
  ## dir spread in degrees
  D = D * np.pi / 180 
  D = D[nd2:nd2+nd] + np.hstack((D[nd2+nd:], D[:nd2]))

  ## Prevent energy loss caused by integration from 355 to 0
  D = D / np.trapz(D, x = dirs)

  S = np.reshape(E, [-1, 1]) * np.reshape(D, [1, -1])

  return S
  
  
## Frequency and directions
g = 9.81
df = 0.001
fN = 1
f = np.arange(df, fN + df, df)
ddir = 5
dirs = np.arange(0, 360, ddir)

## Set parameters for wave conditions
Hs = [1, 1.5, 0.5] #prueba=18
Tp = [10, 5, 15]
dirp = [45, 180, 90]
spr = [30, 90, 20]

## Compute directional spectrum (Sinthetic)
DS = np.zeros((len(f), len(dirs)))
for ii in range(0, len(Hs)):
  DS = DS + DirSpec_HsTp(f, dirs, Hs[ii], Tp[ii], dirp[ii], spr[ii], 3.3, model = 'D', sf = 'cos2s')

## Make plot
fig, ax = plt.subplots(figsize = (10, 6), facecolor = 'w',
           edgecolor = 'k', constrained_layout = True)

hc1 = ax.contourf(dirs, f, DS, cmap = cmap0)
    ## Colorbar
    #hc1.set_clim(0, 0.2)
    cb = fig.colorbar(hc1, ax = ax)
    cb.set_label(r'$m^2/Hz/rad$')
