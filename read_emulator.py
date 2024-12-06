import math, sys, os, glob
import pandas as pd

import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.interpolate import griddata
import scipy.interpolate
import scipy.linalg as sp
from scipy.stats import qmc

import timeit

import numpy as np

import h5py

#from readfile import *
#from params import *

from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator

cm = 1.0/2.54

##########################################################################################
# emulation data

nucl48 = 'Ca48'
nucl52 = 'Ca52'

c1 = - 0.81


class eigenvector_continuation:
    def __init__(self):
        sample = 4

        basedir = ''
        filename1 = basedir + 'Ca48_EM1.8_2.0_eMax12_E3Max16_hwHO016_EC_sample' + str(sample) + '.h5'
        f = h5py.File(filename1, 'r')
        print(list(f.keys()))

        self.H0_ca48 = np.array(f['H0'])
        self.N_ca48 = np.array(f['N'])
        self.c1_ca48 = np.array(f['c1'])
        self.c3_ca48 = np.array(f['c3'])
        self.c4_ca48 = np.array(f['c4'])
        self.cD_ca48 = np.array(f['cD'])
        self.cE_ca48 = np.array(f['cE'])
        self.Rch_ca48 = np.array(f['Rch'])

        filename2 = basedir + 'Ca52_EM1.8_2.0_eMax12_E3Max16_hwHO016_EC_sample' + str(sample) + '.h5'
        f_ca52 = h5py.File(filename2, 'r')
        print(list(f_ca52.keys()))

        self.H0_ca52 = np.array(f_ca52['H0'])
        self.N_ca52 = np.array(f_ca52['N'])
        self.c1_ca52 = np.array(f_ca52['c1'])
        self.c3_ca52 = np.array(f_ca52['c3'])
        self.c4_ca52 = np.array(f_ca52['c4'])
        self.cD_ca52 = np.array(f_ca52['cD'])
        self.cE_ca52 = np.array(f_ca52['cE'])
        self.Rch_ca52 = np.array(f_ca52['Rch'])

        self.energies_ca48 = []
        self.radii_ca48 = []
        self.good_samples = []

        self.energies_ca52 = []
        self.radii_ca52 = []
        self.delta = []


        self.U_ca48, self.s_ca48, self.Vh_ca48 = sp.svd(self.N_ca48)
        while self.s_ca48[-1]<0.000001:
            self.s_ca48 = np.delete(self.s_ca48, -1)
            self.Vh_ca48 = np.delete(self.Vh_ca48, -1, 0)
            self.U_ca48 = np.delete(self.U_ca48, -1, 1)
        self.Ntrunc_ca48 = np.dot(np.transpose(self.U_ca48),np.dot(self.N_ca48,np.transpose(self.Vh_ca48)))
        print("Truncated SVD of Ca48 data [kep %i singular values]" % len(self.s_ca48.shape))

        self.U_ca52, self.s_ca52, self.Vh_ca52 = sp.svd(self.N_ca52)
        while self.s_ca52[-1]<0.000001:
            self.s_ca52 = np.delete(self.s_ca52,-1)
            self.Vh_ca52 = np.delete(self.Vh_ca52,-1,0)
            self.U_ca52 = np.delete(self.U_ca52,-1,1)

        self.Ntrunc_ca52 = np.dot(np.transpose(self.U_ca52),np.dot(self.N_ca52,np.transpose(self.Vh_ca52)))
        print("Truncated SVD of Ca52 data [kep %i singular values]" % len(self.s_ca52.shape))

    def emulate(self, lecs, scan=False):
        H_ca48 = self.H0_ca48 + c1 * self.c1_ca48 + lecs[0] * self.c3_ca48 + lecs[1] * self.c4_ca48 + lecs[2] * self.cD_ca48 + lecs[3] * self.cE_ca48
        H_ca52 = self.H0_ca52 + c1 * self.c1_ca52 + lecs[0] * self.c3_ca52 + lecs[1] * self.c4_ca52 + lecs[2] * self.cD_ca52 + lecs[3] * self.cE_ca52

        Htrunc_ca48 = np.dot(np.transpose(self.U_ca48), np.dot(H_ca48, np.transpose(self.Vh_ca48)))
        Htrunc_ca52 = np.dot(np.transpose(self.U_ca52), np.dot(H_ca52, np.transpose(self.Vh_ca52)))

        EW_ca48, EVl_ca48, EVr_ca48 = sp.eig(Htrunc_ca48, self.Ntrunc_ca48, left=True, right=True)
        EW_ca52, EVl_ca52, EVr_ca52 = sp.eig(Htrunc_ca52, self.Ntrunc_ca52, left=True, right=True)

        idx_ca48 = np.argmin(EW_ca48)
        idx_ca52 = np.argmin(EW_ca52)

        if np.iscomplex(EW_ca48[idx_ca48]) or np.iscomplex(EW_ca52[idx_ca52]):
            print('Ground-state energy is complex, try changing the regularisation of the norm matrix')
            sys.exit(1)

        EVr_ca48 = np.dot(np.transpose(self.Vh_ca48), EVr_ca48)
        overlap_ca48 = np.dot(np.transpose(np.real(EVr_ca48[:, idx_ca48])), np.matmul(self.N_ca48, np.real(EVr_ca48[:, idx_ca48])))
        rch_ca48 = np.dot(np.real(EVr_ca48[:, idx_ca48]), np.dot(self.Rch_ca48, np.real(EVr_ca48[:, idx_ca48]))) / overlap_ca48
        E_ca48 = np.real(EW_ca48[idx_ca48])

        EVr_ca52 = np.dot(np.transpose(self.Vh_ca52), EVr_ca52)
        overlap_ca52 = np.dot(np.transpose(np.real(EVr_ca52[:, idx_ca52])), np.matmul(self.N_ca52, np.real(EVr_ca52[:, idx_ca52])))
        rch_ca52 = np.dot(np.real(EVr_ca52[:, idx_ca52]), np.dot(self.Rch_ca52, np.real(EVr_ca52[:, idx_ca52]))) / overlap_ca52
        E_ca52 = np.real(EW_ca52[idx_ca52])

        dR = rch_ca52 ** 2 - rch_ca48 ** 2

        # individual Ca48 contributions to energy
        E_H0_ca48 = 1.0 * np.dot(np.real(EVr_ca48[:, idx_ca48]), np.dot(self.H0_ca48, np.real(EVr_ca48[:, idx_ca48]))) / overlap_ca48
        E_c1_ca48 = c1 * np.dot(np.real(EVr_ca48[:, idx_ca48]), np.dot(self.c1_ca48, np.real(EVr_ca48[:, idx_ca48]))) / overlap_ca48
        E_c3_ca48 = lecs[0] * np.dot(np.real(EVr_ca48[:, idx_ca48]), np.dot(self.c3_ca48, np.real(EVr_ca48[:, idx_ca48]))) / overlap_ca48
        E_c4_ca48 = lecs[1] * np.dot(np.real(EVr_ca48[:, idx_ca48]), np.dot(self.c4_ca48, np.real(EVr_ca48[:, idx_ca48]))) / overlap_ca48
        E_cD_ca48 = lecs[2] * np.dot(np.real(EVr_ca48[:, idx_ca48]), np.dot(self.cD_ca48, np.real(EVr_ca48[:, idx_ca48]))) / overlap_ca48
        E_cE_ca48 = lecs[3] * np.dot(np.real(EVr_ca48[:, idx_ca48]), np.dot(self.cE_ca48, np.real(EVr_ca48[:, idx_ca48]))) / overlap_ca48

        # individual Ca52 contributions to energy
        E_H0_ca52 = 1.0 * np.dot(np.real(EVr_ca52[:, idx_ca52]), np.dot(self.H0_ca52, np.real(EVr_ca52[:, idx_ca52]))) / overlap_ca52
        E_c1_ca52 = c1 * np.dot(np.real(EVr_ca52[:, idx_ca52]), np.dot(self.c1_ca52, np.real(EVr_ca52[:, idx_ca52]))) / overlap_ca52
        E_c3_ca52 = lecs[0] * np.dot(np.real(EVr_ca52[:, idx_ca52]), np.dot(self.c3_ca52, np.real(EVr_ca52[:, idx_ca52]))) / overlap_ca52
        E_c4_ca52 = lecs[1] * np.dot(np.real(EVr_ca52[:, idx_ca52]), np.dot(self.c4_ca52, np.real(EVr_ca52[:, idx_ca52]))) / overlap_ca52
        E_cD_ca52 = lecs[2] * np.dot(np.real(EVr_ca52[:, idx_ca52]), np.dot(self.cD_ca52, np.real(EVr_ca52[:, idx_ca52]))) / overlap_ca52
        E_cE_ca52 = lecs[3] * np.dot(np.real(EVr_ca52[:, idx_ca52]), np.dot(self.cE_ca52, np.real(EVr_ca52[:, idx_ca52]))) / overlap_ca52


        if rch_ca48 > 3.427 and rch_ca48 < 3.527 and rch_ca52 < 3.603 and rch_ca52 > 3.503 and max(
                [E_ca48, E_ca52]) < -250 and min([E_ca48, E_ca52]) > -350 and (E_ca48 - E_ca52) < 20 and (
                E_ca48 - E_ca52) > 10 and dR > 0.3:
            self.radii_ca48.append(rch_ca48)
            self.energies_ca48.append(E_ca48)
            self.radii_ca52.append(rch_ca52)
            self.energies_ca52.append(E_ca52)
            self.good_samples.append(lecs)
            self.delta.append(dR)

            print("   c1: % 2.4f | c3: % 2.4f | c4: % 2.4f | cD: % 2.4f | cE: % 2.4f " % (
                -0.81, lecs[0], lecs[1], lecs[2], lecs[3]))
            print("   Rch[Ca48]    = % 2.3f fm " % rch_ca48)
            print("   Rch[Ca52]    = % 2.3f fm " % rch_ca52)
            print("   E[Ca48]      = % 2.3f MeV " % E_ca48)
            print("   E[Ca52]      = % 2.3f MeV " % E_ca52)
            print("   delta R^2    = % 2.3f fm " % dR)
            print("")
            print("   <H0> [Ca48]  = % 2.3f MeV " % E_H0_ca48)
            print("   <c1> [Ca48]  = % 2.3f MeV " % E_c1_ca48)
            print("   <c3> [Ca48]  = % 2.3f MeV " % E_c3_ca48)
            print("   <c4> [Ca48]  = % 2.3f MeV " % E_c4_ca48)
            print("   <cD> [Ca48]  = % 2.3f MeV " % E_cD_ca48)
            print("   <cE> [Ca48]  = % 2.3f MeV " % E_cE_ca48)
            print("")
            print("   <H0> [Ca52]  = % 2.3f MeV " % E_H0_ca52)
            print("   <c1> [Ca52]  = % 2.3f MeV " % E_c1_ca52)
            print("   <c3> [Ca52]  = % 2.3f MeV " % E_c3_ca52)
            print("   <c4> [Ca52]  = % 2.3f MeV " % E_c4_ca52)
            print("   <cD> [Ca52]  = % 2.3f MeV " % E_cD_ca52)
            print("   <cE> [Ca52]  = % 2.3f MeV " % E_cE_ca52)
            print("")

            # if scan == True:
            #     self.scan_neighbourhood(lecs)



    def scan_neighbourhood(self, lecs):
        scal = 0.1
        self.emulate(lecs - [scal, 0, 0, 0], scan=False)
        self.emulate(lecs + [scal, 0, 0, 0], scan=False)

        self.emulate(lecs - [0, scal, 0, 0], scan=False)
        self.emulate(lecs + [0, scal, 0, 0], scan=False)

        self.emulate(lecs - [0, 0, scal, 0], scan=False)
        self.emulate(lecs + [0, 0, scal, 0], scan=False)

        self.emulate(lecs - [0, 0, 0, scal], scan=False)
        self.emulate(lecs + [0, 0, 0, scal], scan=False)


        # print("Scannig neighbourhood of sample: ", lecs)
        # nsamples = 100
        # sample = sampler.random(n=nsamples)
        #
        # l_bounds = lecs - 0.1
        # u_bounds = lecs + 0.1
        #
        # print("Upper bounds: ", u_bounds)
        # print("Lower bounds: ", l_bounds)
        #
        # sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        #
        #
        # for lecs in sample_scaled:
        #     print(lecs)
        #     self.emulate(lecs, scan=False)




print("Initialize LHS sampler")
sampler = qmc.LatinHypercube(d=4)

nsamples = 100000
print("Sampling %i LEC combinations" %  nsamples)
sample = sampler.random(n=nsamples)


l_bounds = [-6.7,0.4,0.,0.]
u_bounds = [-0.7,9.4,15.,4.]

magic =  [-3.2, +5.4, 1.264, -0.12]
interval = [2.5, 3.5, 7.5, 2.0]


print("Rescale hypercube to fit bounds \n")
print("   Lower bounds: ", l_bounds)
print("   Upper bounds: ", u_bounds)
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

start = timeit.default_timer()
i = 0
j = 0

ec = eigenvector_continuation()

steps = np.arange(0, nsamples, 1000)
while i < nsamples:
    if i % 10 == 0:
        print("Current sample number: %i [found %i good samples so far]" % (i, j))

    lecs = sample_scaled[i]

    ec.emulate(lecs,scan=True)

    # H_ca48 = H0_ca48 - 0.81 * c1_ca48 + lecs[0] * c3_ca48 + lecs[1] * c4_ca48 + lecs[2] * cD_ca48 + lecs[3] * cE_ca48
    # H_ca52 = H0_ca52 - 0.81 * c1_ca52 + lecs[0] * c3_ca52 + lecs[1] * c4_ca52 + lecs[2] * cD_ca52 + lecs[3] * cE_ca52
    #
    # Htrunc_ca48 = np.dot(np.transpose(U_ca48), np.dot(H_ca48, np.transpose(Vh_ca48)))
    # Htrunc_ca52 = np.dot(np.transpose(U_ca52), np.dot(H_ca52, np.transpose(Vh_ca52)))
    #
    # EW_ca48, EVl_ca48, EVr_ca48 = sp.eig(Htrunc_ca48, Ntrunc_ca48, left=True, right=True)
    # EW_ca52, EVl_ca52, EVr_ca52 = sp.eig(Htrunc_ca52, Ntrunc_ca52, left=True, right=True)
    #
    # idx_ca48 = np.argmin(EW_ca48)
    # idx_ca52 = np.argmin(EW_ca52)
    #
    # if np.iscomplex(EW_ca48[idx_ca48]) or np.iscomplex(EW_ca52[idx_ca52]) :
    #     print('Ground-state energy is complex, try changing the regularisation of the norm matrix')
    #     sys.exit(1)
    #
    # EVr_ca48 = np.dot(np.transpose(Vh_ca48),EVr_ca48)
    # overlap_ca48 = np.dot(np.transpose(np.real(EVr_ca48[:,idx_ca48])),np.matmul(N_ca48,np.real(EVr_ca48[:,idx_ca48])))
    # rch_ca48 = np.dot(np.real(EVr_ca48[:,idx_ca48]),np.dot(Rch_ca48,np.real(EVr_ca48[:,idx_ca48])))/overlap_ca48
    # E_ca48 = np.real(EW_ca48[idx_ca48])
    #
    # EVr_ca52 = np.dot(np.transpose(Vh_ca52),EVr_ca52)
    # overlap_ca52 = np.dot(np.transpose(np.real(EVr_ca52[:,idx_ca52])),np.matmul(N_ca52,np.real(EVr_ca52[:,idx_ca52])))
    # rch_ca52 = np.dot(np.real(EVr_ca52[:,idx_ca52]),np.dot(Rch_ca52,np.real(EVr_ca52[:,idx_ca52])))/overlap_ca52
    # E_ca52 = np.real(EW_ca52[idx_ca52])
    #
    # # Radien = [Radius,Radius1]
    # # Energien = [np.real(EW[idx]),np.real(EW1[idx1])]
    # dR = rch_ca52**2 - rch_ca48**2

    if i in steps:
        print('i= ', i)

    i = i+1


print('Samples total', i)
print('Good samples', j)
end = timeit.default_timer()
print('Time', end-start)

print("Final overview of emulated data:")
print(" idx:       c1:      c3:     c4:     cD:     cE:     Rch[Ca48]:      Rch[Ca52]:      E[Ca48]:        E[Ca52]:        deltaR:")
for idx in range(len(delta)):
    print(" %i      % 2.4f      % 2.4f      % 2.4f      % 2.4f      % 2.4f      % 2.4f      % 2.4f      % 2.4f      % 2.4f      % 2.4f" %
          (idx,
           -0.81, good_samples[idx][0], good_samples[idx][1], good_samples[idx][2], good_samples[idx][3],
          radii_ca48[idx], radii_ca52[idx],
          energies_ca48[idx], energies_ca52[idx],
          delta[idx])
    )



