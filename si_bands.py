#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 9 13:30:47 2019

@author: aditi
"""

import numpy as np
from matplotlib import pyplot as plt

def phase_factor(k, nbs):    
    p,q,r,s = [np.exp(1j*k @ nb) for nb in nbs]
    pf = np.array([p+q+r+s,    p+q-r-s,    p-q+r-s,    p-q-r+s])
    return (1 / 4) * pf

def band_energy_eigval(g, E_s, E_p, V_ss, V_sp, V_xx, V_xy):
    gc = np.conjugate(g)
    H = np.array([[         E_s,  V_ss * g[0],            0,            0,            0, V_sp * g[1], V_sp * g[2], V_sp * g[3]],
                  [V_ss * gc[0],          E_s, -V_sp * gc[1], -V_sp * gc[2], -V_sp * gc[3],          0,          0,          0],
                  [          0, -V_sp * g[1],           E_p,            0,            0, V_xx * g[0], V_xy * g[3], V_xy * g[1]],
                  [          0, -V_sp * g[2],            0,           E_p,            0, V_xy * g[3], V_xx * g[0], V_xy * g[1]],
                  [          0, -V_sp * g[3],            0,            0,           E_p, V_xy * g[1], V_xy * g[2], V_xx * g[0]],
                  [V_sp * gc[1],           0,  V_xx * gc[0],  V_xy * gc[3],  V_xy * gc[1],         E_p,         0,           0],
                  [V_sp * gc[2],           0,  V_xy * gc[3],  V_xx * gc[0],  V_xy * gc[2],          0,        E_p,           0],
                  [V_sp * gc[3],           0,  V_xy * gc[1],  V_xy * gc[1],  V_xx * gc[0],          0,         0,          E_p]])
    eigen_values = np.linalg.eigvalsh(H)
    return np.sort(eigen_values)

def band_structure(par, nbs, kpath):
    bands = []
    path=np.vstack(kpath)
    for k in path:
        g=phase_factor(k,nbs)
        bands.append(band_energy_eigval(g,*par))
    return np.stack(bands, axis=-1)

def linpath(x,y,n=50, endpoint=True):
    spacings = [np.linspace(start, end, num=n, endpoint=endpoint) for start, end in zip(x,y)]
    return np.stack(spacings, axis=-1)

parameters = (-4.03, 3.17, -8.13, 5.88, 3.17, 7.51)
n = 1000
a = 1
nbs =np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])*a/4
[G,L,K,X,W,U]=(2*np.pi/a)*np.array([[0, 0, 0],[1/2, 1/2, 1/2],[3/4, 3/4, 0],[0, 0, 1],[1, 1/2, 0],[1/4, 1/4, 1]])
Lambda = linpath(L, G, n, endpoint=False)
Delta = linpath(G, X, n, endpoint=False)
Xuk = linpath(X, U, n // 4, endpoint=False)
Sigma = linpath(K, G, n, endpoint=True)
kpath=[Lambda,Delta,Xuk,Sigma]
bands = band_structure(parameters,nbs,kpath)

plt.figure(figsize=(15,10))
ax = plt.subplot(111)
plt.xlim(0, len(bands))
plt.ylim(min(bands[0]) - 1, max(bands[7]) + 1)
x_ticks=np.array([0, 0.5, 1, 1.5, 2, 2.25, 2.75, 3.25])*n
plt.xticks(x_ticks, ('$L$','$\Lambda$','$\Gamma$','$\Delta$','$X$','$U,K$','$\Sigma$','$\Gamma$'), fontsize=18)
plt.yticks(fontsize=18)
for y in np.arange(-25, 25, 2.5):
    plt.axhline(y, ls='--', lw=0.3, color='b',alpha=0.3)
plt.xlabel('k-path', fontsize=20)
plt.ylabel('Energy(eV)', fontsize=20)
ax.set_facecolor('white')
plt.grid(color='whitesmoke', linestyle='--', linewidth=2)
c =np.array(['black','c','gold','m','limegreen','royalblue','orange','red','brown','cyan'])
for band, color in zip(bands, c):
    plt.plot(band, lw=2.0, color=color)
plt.savefig('phy403.png')
