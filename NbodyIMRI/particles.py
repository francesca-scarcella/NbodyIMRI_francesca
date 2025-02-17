import warnings
from math import sqrt

from os.path import join
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import signal

from NbodyIMRI import distributionfunctions as DF
from NbodyIMRI import tools
from NbodyIMRI import units as u
import NbodyIMRI

import random

import h5py
import copy


class particles():
    def __init__(self, M_1, M_2, N_DM=2, M_DM = 0, dynamic_BH=True):

        self.M_1 = M_1
        self.M_2 = M_2

        print("10:01pm version")


        #self.M_tot = M_1 + M_2

        self.M_DM = M_DM*np.ones(N_DM)
        self.N_DM = N_DM

        self.xBH1 = np.zeros((3), dtype=np.float64)
        self.vBH1 = np.zeros((3), dtype=np.float64)

        self.xBH2 = np.zeros((3), dtype=np.float64)
        self.vBH2 = np.zeros((3), dtype=np.float64)

        self.xDM = np.zeros((N_DM, 3))
        self.vDM = np.zeros((N_DM, 3))

        # define the variables that will contain the DM particels in the inner and outer region
        self.xDM_in = None
        self.vDM_in = None

        self.xDM_out = None
        self.vDM_out = None

        self.dvdtBH1 = None
        self.dvdtBH2 = None

        self.mask = None

        #these are not really used?
        self.dxdtDM = None
        self.dxdtDM_in = None
        self.dxdtDM_out = None

        self.dynamic_BH = dynamic_BH

        #Null values for the spike parameters
        self.rho_6    = 0.0
        self.gamma_sp = 0.0
        self.alpha    = 0.0
        self.r_t      = -1.0


    def M_tot(self):
        return self.M_1 + self.M_2

    def xstep(self, h):
            if (self.dynamic_BH):
                self.xBH1 += self.vBH1*h
            self.xBH2 += self.vBH2*h
            self.xDM += self.vDM*h

    def vstep(self, h):
        if (self.dynamic_BH):
            self.vBH1 += self.dvdtBH1*h
        self.vBH2 += self.dvdtBH2*h
        self.vDM += self.dvdtDM*h


    #during the short steps, keep both the BHs fixed
    def xstep_in(self, h):
        self.xDM_in += self.vDM_in*h

    def vstep_in(self, h):
        self.vDM_in += self.dvdtDM_in*h

    #BH states are updated during long time step only
    def xstep_out(self, h):
        if (self.dynamic_BH):
            self.xBH1 += self.vBH1*h
        self.xBH2 += self.vBH2*h
        self.xDM_out += self.vDM_out*h

    def vstep_out(self, h):
            if (self.dynamic_BH):
                self.vBH1 += self.dvdtBH1*h
            self.vBH2 += self.dvdtBH2*h
            self.vDM_out += self.dvdtDM_out*h


    def orbital_elements(self):
        return tools.calc_orbital_elements(self.xBH1 - self.xBH2, self.vBH1 - self.vBH2, self.M_tot())

    def T_orb(self):
        a_i, e_i = self.orbital_elements()
        return tools.calc_Torb(a_i, self.M_tot())

    def initialize_spike(self, rho_6=1e15*u.Msun/u.pc**3, gamma_sp=7/3, r_max=1e-6*u.pc, r_t = -1, alpha  = 2, circular=0, r_soft=-1):

        print("hi")

        self.rho_6    = rho_6
        self.gamma_sp = gamma_sp
        self.r_t      = r_t
        self.alpha    = alpha

        if (self.dynamic_BH):
            M1_eff = self.M_1
        else:
            M1_eff = self.M_1 + self.M_2

        if (self.N_DM > 0):
            if (r_t < 0):
                SpikeDF = DF.PowerLawSpike(M1_eff, rho_6, gamma_sp)
            else:
                SpikeDF = DF.GeneralizedNFWSpike(M1_eff, rho_6, gamma_sp, r_t, alpha, r_soft)

            r, v = SpikeDF.draw_particle(r_max, N = self.N_DM)

            for i in range(self.N_DM):
                rhat = tools.get_random_direction()

                self.xDM[i,:] = r[i]*rhat
                if (circular == 0):
                    vhat = tools.get_random_direction()
                    self.vDM[i,:] = v[i]*vhat

                if (circular == 1):
                    #Generate an orthonormal basis
                    v1 = 1.0*rhat
                    v2 = np.cross(rhat, np.array([0, 0, 1]))
                    v3 = np.cross(rhat, np.array([0, 1, 0]))

                    u1 = 1.0*v1
                    u2 = v2 - np.dot(u1, v2)*u1
                    u3 = v3 - np.dot(u1, v3)*u1 - np.dot(u2, v3)*u2

                    e2 = u2/np.sqrt(np.dot(u2, u2))
                    e3 = u3/np.sqrt(np.dot(u3, u3))

                    #print(np.dot(v1, e2), np.dot(v1, e3))
                    phi = 2*np.pi*np.random.rand()

                    vhat = np.cos(phi)*e2 + np.sin(phi)*e3
                    self.vDM[i,:] = (SpikeDF.v_max(r[i])/np.sqrt(2))*vhat

                if (circular == 2):
                    zhat = np.array([0, 0, 1])
                    chat = np.cross(rhat, zhat)
                    chat /= np.sqrt(np.dot(chat,chat))
                    v1 = tools.get_random_direction()
                    v2 = np.dot(v1, rhat)*rhat
                    v3 = np.dot(v1, chat)*chat
                    vnew = v1 - v2 - v3
                    vhat = vnew/np.sqrt(np.dot(vnew,vnew))

                    sgn = 2*(random.randint(0,1) - 0.5)
                    self.vDM[i,:] = sgn*(SpikeDF.v_max(r[i])/np.sqrt(2))*vhat

            self.xDM += self.xBH1
            #self.vDM += self.vBH1

            #assign paricles to inner and outer arrays
            self.mask=tools.norm(self.xDM - self.xBH1)<r_soft

            self.xDM_in=self.xDM[self.mask]
            self.xDM_out=self.xDM[~self.mask]

            self.vDM_in=self.vDM[self.mask]
            self.vDM_out=self.vDM[~self.mask]





    def summary(self):
        print("> Particle set:")
        print(f">     M_1 [M_sun] = {self.M_1/u.Msun}")
        if (self.M_2 > 0):
            print(f">     M_2 [M_sun] = {self.M_2/u.Msun}")
            a, e = self.orbital_elements()
            print(f">     (a [pc], e) = ({a/u.pc}, {e})")
        if (np.sum(self.M_DM) > 0):
            print(" ")
            print(f">     N_DM = {self.N_DM}")
            print(f">     M_DM [M_sun] = {self.M_DM[0]/u.Msun}")



    def plot(self):
        if (np.sum(self.M_DM) > 0):
            ncols = 3
        else:
            ncols = 2

        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(16, 5))

        axes = ax[:]
        #----------------------------------------

        if (np.sum(self.M_DM) > 0):
            axes[0].scatter(self.xDM[:,0]/u.pc, self.xDM[:,1]/u.pc, color='C0', marker='o', alpha=0.75)

        axes[0].scatter(self.xBH1[0]/u.pc, self.xBH1[1]/u.pc, color='k', marker='o', s=250)

        if (self.M_2 > 0):
            a_pc = self.orbital_elements()[0]/u.pc
            axes[0].scatter(self.xBH2[0]/u.pc, self.xBH2[1]/u.pc, color='k', marker='o', s=40)
            axes[0].set_xlim(-1.5*a_pc, 1.5*a_pc)
            axes[0].set_ylim(-1.5*a_pc, 1.5*a_pc)

        axes[0].set_xlabel(r"$x$ [pc]")
        axes[0].set_ylabel(r"$y$ [pc]")
        axes[0].set_aspect('equal')

        #----------------------------------------

        if (np.sum(self.M_DM) > 0):
            axes[1].scatter(self.xDM[:,0]/u.pc, self.xDM[:,2]/u.pc, color='C0', marker='o', alpha=0.75)

        axes[1].scatter(self.xBH1[0]/u.pc, self.xBH1[2]/u.pc, color='k', marker='o', s=250)

        if (self.M_2 > 0):
            a_pc = self.orbital_elements()[0]/u.pc
            axes[1].scatter(self.xBH2[0]/u.pc, self.xBH2[2]/u.pc, color='k', marker='o', s=40)
            axes[1].set_xlim(-1.5*a_pc, 1.5*a_pc)
            axes[1].set_ylim(-1.5*a_pc, 1.5*a_pc)


        axes[1].set_xlabel(r"$x$ [pc]")
        axes[1].set_ylabel(r"$z$ [pc]")
        axes[1].set_aspect('equal')

        #---------------------------------------
        if (np.sum(self.M_DM) > 0):

            r_vals = tools.norm(self.xDM - self.xBH1)
            axes[2].hist(np.log10(r_vals/u.pc), 50, density=True)

            axes[2].set_xlabel(r"$\log_{10}(r/\mathrm{pc})$")
            axes[2].set_ylabel(r"$P(\log_{10}(r/\mathrm{pc}))$")

            if (self.M_2 <= 0):
                r_max = 1e4*tools.calc_risco(self.M_1)/u.pc
                axes[0].set_xlim(-r_max, r_max)
                axes[0].set_ylim(-r_max, r_max)
                axes[1].set_xlim(-r_max, r_max)
                axes[1].set_ylim(-r_max, r_max)


        plt.tight_layout()
        return fig, ax


def load_particles_from_file(fileID, which="initial"):
    """
    Load a particles object using data from an NbodyIMRI output file.

    Parameters:
        fileID (string)         : fileID which identifies the file to be read
        which (string)          : Specifies whether to load in the "initial" or "final" particle configuration from the file

    Returns:
        p (particles): a `particles` object containing the state of the system from file

    """

    f = tools.open_file_for_read(fileID)

    M_1 = f['data'].attrs["M_1"]*u.Msun
    M_2 = f['data'].attrs["M_2"]*u.Msun
    N_DM = f['data'].attrs["N_DM"]
    M_DM_i = f['data'].attrs["M_DM"]*u.Msun
    dynamic = f['data'].attrs["dynamic"]

    try:
        M1_list  = np.array(f['data']['M_1'])
        M2_list  = np.array(f['data']['M_2'])
    except:
        M1_list = M_1*np.ones_like(ts)
        M2_list = M_2*np.ones_like(ts)

    if (dynamic == 1):
        dynamic_BH = True
    else:
        dynamic_BH = False

    if (N_DM <= 0):
        N_DM = 2 #Keep N_DM = 2 so that all the arrays work as expected...
        M_DM = 0.0

    p = particles(M_1, M_2, N_DM=N_DM, M_DM=M_DM_i, dynamic_BH=dynamic_BH)

    if (which == "initial"):
        p.M_1 = M1_list[0]
        p.M_2 = M2_list[0]
        p.xBH1 = np.array(f['data']['xBH1'])[0,:]
        p.vBH1 = np.array(f['data']['vBH1'])[0,:]

        p.xBH2 = np.array(f['data']['xBH2'])[0,:]
        p.vBH2 = np.array(f['data']['vBH2'])[0,:]

        try:
            p.xDM  = np.array(f['data']['xDM_i'])
            p.vDM  = np.array(f['data']['vDM_i'])
        except:
            print("Initial DM positions and velocities not found in output file... (setting to zero)")
            p.xDM = np.zeros((N_DM, 3))
            p.vDM = np.zeros((N_DM, 3))


    elif (which == "final"):
        p.M_1 = M1_list[-1]
        p.M_2 = M2_list[-1]
        p.xBH1 = np.array(f['data']['xBH1'])[-1,:]
        p.vBH1 = np.array(f['data']['vBH1'])[-1,:]

        p.xBH2 = np.array(f['data']['xBH2'])[-1,:]
        p.vBH2 = np.array(f['data']['vBH2'])[-1,:]

        try:
            p.xDM  = np.array(f['data']['xDM_f'])
            p.vDM  = np.array(f['data']['vDM_f'])
            p.M_DM = np.array(f['data']['M_DM'])
        except:
            print("Final DM positions and velocities not found in output file... (setting to zero)")
            p.xDM = np.zeros((N_DM, 3))
            p.vDM = np.zeros((N_DM, 3))



    return p


def single_BH(M_1, N_DM=0, rho_6=1e15*u.Msun/u.pc**3, gamma_sp=7/3, r_max=-1, r_t = -1, alpha = 2, circular=0, r_soft = -1):
    """
    Initialise a `particles` object which consists of a single BH surrounded by a DM halo.

    Parameters:
        M_1 (float)     : Mass of the central BH
        N_DM (int)      : Number of DM pseudoparticles. Set N_DM = 0 in order to neglect DM particles.
        rho_6 (float)   : density normalisation of the DM spike
        gamma_sp (float): power-law slope of the spike
        r_max (float)   : maximum radius to include for DM density profile (useful for profiles which are formally infinite). Default: 1e5*r_isco(M_1)
        r_t (float)     : Smooth truncation radius of the spike. Default = -1 (no truncation)
        alpha (float)   : Power-law slope for truncating the outer parts of the spike. Default = 2
        circular (int)  : Set circular = 1 in order to initialise DM particles on circular orbits. Default is 0 (isotropic orbits).

    Returns:
        p (particles)   : Set of particles
    """

    if (r_max < 0):
        if (r_t < 0):
            r_max = 1e5*tools.calc_risco(M_1)
        else:
            r_max = 1e3*r_t


    if (N_DM > 0):
        if (r_t < 0):
            SpikeDF = DF.PowerLawSpike(M_1, rho_6, gamma_sp)
        else:
            SpikeDF = DF.GeneralizedNFWSpike(M_1, rho_6, gamma_sp, r_t, alpha, r_soft)
        M_spike = SpikeDF.M_DM_ini(r_max)
        M_DM    = (M_spike/N_DM)
    else:
        M_DM = 0.0

    p = particles(M_1, M_2=0, N_DM=N_DM, M_DM=M_DM, dynamic_BH=False)

    if (N_DM > 0):
        p.initialize_spike(rho_6, gamma_sp, r_max, r_t, alpha, circular, r_soft)

    return p



def particles_in_binary(M_1, M_2, a_i, e_i=0.0, N_DM=0, dynamic_BH=True, rho_6=1e16*u.Msun/u.pc**3, gamma_sp=7/3, r_max=-1, r_t = -1, alpha = 2, circular = 0, include_DM_mass=False, r_soft = -1):
    """
    Initialise a `particles` object which consists of a BH binary, which may be surrounded by a DM halo.

    Parameters:
        M_1 (float)     : Mass of the central BH
        M_2 (float)     : Mass of secondary BH.
        a_i (float)     : Initial semi-major axis of the binary.
        e_i (float)     : Initial eccentricity of the binary
        N_DM (int)      : Number of DM pseudoparticles. Set N_DM = 0 in order to neglect DM particles.
        dynamic_BH (bool): Set dynamic_BH=True in order to evolve both the central and orbiting BHs (dynamic_BH = False fixes the central BH and evolves the imaginary 'reduced mass' particle.)
        rho_6 (float)   : density normalisation of the DM spike
        gamma_sp (float): power-law slope of the spike
        r_max (float)   : maximum radius to include for DM density profile (useful for profiles which are formally infinite). Default: 1e5*r_isco(M_1)
        r_t (float)     : Smooth truncation radius of the spike. Default = -1 (no truncation)
        alpha (float)   : Power-law slope for truncating the outer parts of the spike. Default = 2
        circular (int)  : Set circular = 1 in order to initialise DM particles on circular orbits. Default is 0 (isotropic orbits).
        include_DM_mass (bool): Set to True in order to include the enclosed DM mass in the calculation of the initial velocity (for a given a_i, e_i)

    Returns:
        p (particles)   : Set of particles
    """

    if (r_max < 0):
        if (r_t < 0):
            r_max = 1e5*tools.calc_risco(M_1)
        else:
            r_max = 1e3*r_t

    if (N_DM > 0):
        if (r_t < 0):
            SpikeDF = DF.PowerLawSpike(M_1, rho_6, gamma_sp)
        else:
            SpikeDF = DF.GeneralizedNFWSpike(M_1, rho_6, gamma_sp, r_t, alpha, r_soft)

        M_spike = SpikeDF.M_DM_ini(r_max)
        M_DM    = (M_spike/N_DM)
    else:
        M_DM = 0.0

    p = particles(M_1, M_2, N_DM=N_DM, M_DM=M_DM, dynamic_BH=dynamic_BH)

    #Initialise BH properties
    r_i = a_i * ( 1 + e_i)

    if (include_DM_mass):
        M_tot = p.M_tot() + SpikeDF.M_DM_ini(a_i)
    else:
        M_tot = p.M_tot()

    mu = u.G_N*M_tot
    v_i = np.sqrt( mu * (2.0/r_i - 1.0/a_i) )

    if (dynamic_BH):
        factor = M_2/M_tot
    else:
        factor = 0

    p.xBH1[:] = np.atleast_2d([-r_i*factor,   0, 0])
    p.xBH2[:] = np.atleast_2d([r_i*(1-factor),   0, 0])

    p.vBH1[:] = np.atleast_2d([0.0, v_i*factor, 0])
    p.vBH2[:] = np.atleast_2d([0.0, -v_i*(1-factor), 0])

    if (N_DM > 0):
        p.initialize_spike(rho_6, gamma_sp, r_max, r_t, alpha, circular, r_soft)

    return p
