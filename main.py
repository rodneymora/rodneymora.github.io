#import pyodide_js
#await pyodide_js.loadPackage('numpy')
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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

##
## Color palette to use in contourf or pcolor
#1. White
Whi = ['#FFFFFF']
#2. Pinks to Blues
PinBlu = ['#C9BFCD', '#B19FBB', '#A78DB4', '#A480B3', '#A470B8', '#B34AD7', '#C50EFB', '#AE0AFE']#, '#9415FE', '#7320FD']
#3. Blues to Greens
BluGree = ['#562BFD', '#3C35FC', '#1F3FFC', '#004CFC', '#0070F9', '#008EF7', '#02AAF4', '#03CDF2', '#03EBF0']
#4. Greens to Yellows
GreeYell = ['#25ED5B', '#57F846', '#7BFF36', '#94FE2C', '#ABFF21', '#C7FE15', '#DCFF0E', '#F2FF03', '#F4F105']
#5. Orange to Reds
OraRed = ['#F7E205', '#FAD305', '#FDC600', '#FFB800', '#FFA900', '#FF9B00', '#FF7C01', '#FF3E00', '#FF0000']#, '#E50001']

colors_palette = Whi + PinBlu + BluGree + OraRed

## Define Log Levels
bounds1 = np.arange(0.0001, 0.001, 0.0001)
bounds2 = np.arange(0.001, 0.01, 0.001)
bounds3 = np.arange(0.01, 0.1, 0.01)
bounds4 = np.arange(0.1, 1, 0.1)
bounds5 = np.arange(1, 11, 1)
levels = np.hstack([bounds2, bounds3, bounds4])

## Parameters for colorbar
cmap = colors.LinearSegmentedColormap.from_list('mymap', colors_palette)
cmap = colors.ListedColormap(cmap(np.linspace(0, 1, 27)))
# Logaritmic scale colors
norm = colors.BoundaryNorm(np.log(levels), cmap.N, clip = True)

## Make plot
fig, ax = plt.subplots(figsize = (10, 6), facecolor = 'w',
           edgecolor = 'k', constrained_layout = True, subplot_kw = dict(polar = True))

hc1 = ax.pcolormesh(dirs*pi/180, f, np.log(DS), norm = norm, cmap = cmap, shading = 'gouraud')

## Radii and angles labels
ax.set_theta_zero_location('N')

## Clockwise
ax.set_theta_direction(-1)
ax.set_thetagrids(np.arange(0, 360, 30),\
			labels = [str(i)+'º' if i > 0 else str(i)+'ºN' for i in range(0, 360, 30)],
			fontsize = 8, fontweight = 'bold')

## Frequency space
if False:
	ax.set_rmax(0.5)
	ax.set_rgrids([0.05, 0.1, 0.2, 0.3, 0.4],\
				labels = [r'0.05 Hz', r'0.1 Hz', r'0.2 Hz',\
				r'0.3 Hz', r'0.4 Hz'],\
				fontweight = 'bold', fontsize = 6)
## Period space
if True:
	ax.set_rmax(12)
	ax.set_rgrids([2, 4, 6, 8, 10],\
				labels = [r'2', r'4', r'6', r'8',\
				r'10s'], angle = 45,\
				fontweight = 'bold', fontsize = 6)

ticklabel = [r'$10^{-3}$', r'$10^{-2}$',\
			r'$10^{-1}$', r'$10^0$']
cbar = fig.colorbar(hpc, ax = axs, shrink = 0.8, pad = 0.01, spacing = 'proportional',\
ticks = np.log([0.001, 0.01, 0.1, 1]))
cbar.set_ticklabels(ticklabel)
cbar.set_label(r'$\log_{10}[(S(f,\theta)/S(f,\theta)^{max}]$')

## Legend and grid
ax.grid(True, linestyle = '--', alpha = .5, color = 'black',linewidth = .5)

display(fig, target="mpl")
## Colorbar
cb = fig.colorbar(hc1, ax = ax)
cb.set_label(r'$m^2/Hz/rad$')
