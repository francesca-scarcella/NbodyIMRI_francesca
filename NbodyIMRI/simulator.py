import warnings
from math import sqrt

from os.path import join
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import signal

from NbodyIMRI import tools, particles
from NbodyIMRI import units as u
import NbodyIMRI

import h5py

import copy



# Leapfrog stepsize parameters
#-----------------------------
theta = 1/(2 - 2**(1/3))

xi  = +0.1786178958448091e+00
lam = -0.2123418310626054e+00
chi = -0.6626458266981849e-01
#-----------------------------

#NB: For PN corrections, see https://arxiv.org/abs/1312.1289
#NB: Deal with saving of M1 and M2 values for wheel and spoke


# Time-stepping scheme of this version (2026-11-25). See simulator.full_step_split.
SCHEME = "force_split"


class simulator():
    """
    Class for evolving the N-body system.

    Attributes:
        p (particles)       : A particles object, which specifies the initial conditions of the objects to simulate (a deep copy is made and used internally by the simulator)
        r_soft_sq2 (float)   : The square of the softening length to be used for the DM particle interactions with the secondary, m2
        r_soft_sq1 (float)   : The square of the softening length to be used for the DM particle interactions with the primary, m1. Default is set to r_soft_sq2.
        soft_method (string): Softening method to be used. Options are: "plummer", "plummer2", "uniform", "truncate", "empty_shell".
                            Default is "uniform" which computes the softening assuming that each DM particle is a finite sphere of uniform density.
        IDhash (string)     : A hash made up of 5 hexadecimal digits which identifies the simulation (and the output files). This is generated and saved when he simulation is run.
        check_state (function): A function which will be called in between each timestep to check the state of the simulation and perform any required operations.
                                (For example, removing certain particles). Must have the signature `check_state(simulator)`. Default is None.

    """
    
    def __init__(self, particle_set, r_soft_sq2 = 0.0, r_soft_sq1 = -1, soft_method1="uniform", soft_method="uniform", check_state = None, r_partition2 = -1, react_M1 = False):
            
        self.p = copy.deepcopy(particle_set)
        self.r_soft_sq2 = r_soft_sq2
        self.r_soft_sq1 = r_soft_sq1
        if (self.r_soft_sq1 < 0):
            self.r_soft_sq1 = 1.0*r_soft_sq2
        self.soft_method = soft_method
        
        #Softening with the primary is always set to "uniform"    
        self.soft_method1 = soft_method1

        # ---- sub-steps (used when run_simulation is called with N_step_partition > 1) ----
        # The inner partition takes the short steps. It is made of the two black holes and of the particles within
        #   p.r_partition  of the primary   (set when the particles are created), or
        #   r_partition2   of the secondary (-1: one Hill radius, a (q/3)^(1/3), set at the start of the run; 0: off).
        self.r_partition2 = r_partition2
        # If True the primary feels the dark matter (every kick is applied as a pair). Default False, as in all runs so far:
        # with the spike at rest in the centre-of-mass frame the primary moves through it, so this changes the physical setup.
        self.react_M1 = react_M1
        self.check_state = check_state
        self.background_field = None





    def full_step(self, dt, method="PEFRL"):
        """
        Perform a full leapfrog step, See e.g. https://arxiv.org/abs/2007.05308, http://physics.ucsc.edu/~peter/242/leapfrog.pdf

        Parameters:
            dt (float)      : size of the timestep (the leapfrog is made up of many sub-steps, with dt being the size of one full leapfrog step)
            method (string) : leapfrog method to use. Options are: "DKD", "FR", "PEFRL" [default]. These corresponds to 2nd, 4th and 4th order methods.

        Returns:
            None

        """


        #2nd order 'standard' leapfrog
        if (method == "DKD"):
            self.p.xstep(0.5*dt)
            self.update_acceleration()
            self.p.vstep(1.0*dt)
            self.p.xstep(0.5*dt)

        #4th order Ruth-Forest (FR) leapfrog
        elif (method == "FR"):
            self.p.xstep(theta*dt/2)
            self.update_acceleration()
            self.p.vstep(theta*dt)
            self.p.xstep((1-theta)*dt/2)
            self.update_acceleration()
            self.p.vstep((1-2*theta)*dt)
            self.p.xstep((1-theta)*dt/2)
            self.update_acceleration()
            self.p.vstep(theta*dt)
            self.p.xstep(theta*dt/2)

        #Improved 4th order "Position Extended Forest-Ruth Like" (PEFRL) leapfrog
        elif (method == "PEFRL"):
            self.p.xstep(xi*dt)
            self.update_acceleration()
            self.p.vstep((1-2*lam)*dt/2)
            self.p.xstep(chi*dt)
            self.update_acceleration()
            self.p.vstep(lam*dt)
            self.p.xstep((1-2*(chi + xi))*dt)
            self.update_acceleration()
            self.p.vstep(lam*dt)
            self.p.xstep(chi*dt)
            self.update_acceleration()
            self.p.vstep((1-2*lam)*dt/2)
            self.p.xstep(xi*dt)


    def update_acceleration(self):
        """
        Update the acceleration of all particles in p, based on the current positions. Used by full_step (no sub-steps).


        Returns:
            None
        """

        #Calculate separations between DM particles and central BH
        dx1     = (self.p.xDM - self.p.xBH1)
        r1      = np.linalg.norm(dx1, axis=-1, keepdims=True)
        dx1    /= r1
        r1_sq   = r1**2

        if (self.p.dynamic_BH):
            M1_eff  = self.p.M_1
            M2_eff  = self.p.M_2
        else:
            #M1_eff  = self.p.M_1
            #M2_eff  = self.p.M_2
            M1_eff  = self.p.M_1 + self.p.M_2
            M2_eff  = (self.p.M_1*self.p.M_2)/(self.p.M_1 + self.p.M_2)

        #Calculate forces (including softening)
        if (self.soft_method1 == "plummer"):
            acc_DM1 = -u.G_N*M1_eff*dx1*r1*(r1_sq + self.r_soft_sq1)**(-3/2)
            
        elif (self.soft_method1 == "plummer2"):
            acc_DM1 = -u.G_N*M1_eff*r1*(dx1/2)*(2*r1_sq + 5*self.r_soft_sq1)*(r1_sq + self.r_soft_sq1)**(-5/2)

        elif (self.soft_method1 == "uniform_old"):
            x = np.sqrt(r1_sq/self.r_soft_sq1)
            acc_DM1 = -u.G_N*M1_eff*dx1*(r1_sq)**-1
            inds = x < 1
            if (np.sum(inds) >= 1):
                inds = inds.flatten()
                acc_DM1[inds] = -u.G_N*M1_eff*dx1[inds,:]*x[inds]*(8 - 9*x[inds] + 2*(x[inds])**3)/(self.r_soft_sq1)

        elif (self.soft_method1 == "uniform"):
            x = np.sqrt(r1_sq/self.r_soft_sq1)
            acc_DM1 = -u.G_N*M1_eff*dx1*(r1_sq)**-1
            inds = x < 1
            if (np.sum(inds) >= 1):
                inds = inds.flatten()
                acc_DM1[inds] = -u.G_N*M1_eff*dx1[inds,:]*x[inds]/(self.r_soft_sq1)

        elif (self.soft_method1 == "truncate"):
            r1_sq = np.clip(r1_sq, self.r_soft_sq1, 1e50)
            acc_DM1 = -u.G_N*M1_eff*dx1/r1_sq

        elif (self.soft_method1 == "empty_shell"):
            x = np.sqrt(r1_sq/self.r_soft_sq1)
            acc_DM1 = -u.G_N*M1_eff*dx1*(r1_sq)**-1
            inds = x < 1
            if (np.sum(inds) >= 1):
                inds = inds.flatten()
                acc_DM1[inds] *= 0.0

        else:
            raise ValueError("Invalid softening method:" + self.soft_method1)

        #Calculate forces on second BH (if it exists)
        if (self.p.M_2 > 0):
            dx2     = (self.p.xDM - self.p.xBH2)
            r2      = np.linalg.norm(dx2, axis=-1, keepdims=True)
            dx2     /= r2
            r2_sq   = r2**2


            if (self.soft_method == "plummer"):
                acc_DM2 = -u.G_N*M2_eff*dx2*r2*(r2_sq + self.r_soft_sq2)**(-3/2)

            elif (self.soft_method == "plummer2"):
                acc_DM2 = -u.G_N*M2_eff*r2*(dx2/2)*(2*r2_sq + 5*self.r_soft_sq2)*(r2_sq + self.r_soft_sq2)**(-5/2)

            elif (self.soft_method == "uniform_old"):
                x = np.sqrt(r2_sq/self.r_soft_sq2)
                acc_DM2 = -u.G_N*M2_eff*dx2*(r2_sq)**-1
                inds = x < 1
                if (np.sum(inds) >= 1):
                    inds = inds.flatten()
                    acc_DM2[inds] = -u.G_N*M2_eff*dx2[inds,:]*x[inds]*(8 - 9*x[inds] + 2*(x[inds])**3)/(self.r_soft_sq2)

            elif (self.soft_method == "uniform"):
                x = np.sqrt(r2_sq/self.r_soft_sq2)
                acc_DM2 = -u.G_N*M2_eff*dx2*(r2_sq)**-1
                inds = x < 1
                if (np.sum(inds) >= 1):
                    inds = inds.flatten()
                    acc_DM2[inds] = -u.G_N*M2_eff*dx2[inds,:]*x[inds]/(self.r_soft_sq2)

            elif (self.soft_method == "truncate"):
                r2_sq = np.clip(r2_sq, self.r_soft_sq2, 1e50)
                acc_DM2 = -u.G_N*M2_eff*dx2/r2_sq

            elif (self.soft_method == "empty_shell"):
                x = np.sqrt(r2_sq/self.r_soft_sq2)
                acc_DM2 = -u.G_N*M2_eff*dx2*(r2_sq)**-1
                inds = x < 1
                if (np.sum(inds) >= 1):
                    inds = inds.flatten()
                    acc_DM2[inds] *= 0.0

            else:
                raise ValueError("Invalid softening method:" + self.soft_method)

            #Calculate forces between the 2 BHs
            dx12    = (self.p.xBH1 - self.p.xBH2)
            r12_sq  = np.linalg.norm(dx12, axis=-1, keepdims=True)**2
            acc_BH = -u.G_N*M2_eff*dx12*(r12_sq)**-1.5
        else:
            acc_BH = 0.0
            acc_DM2 = 0.0

        #Save the values of the acceleration
        if (self.p.dynamic_BH):
            #Acceleration of central BH due only to m2
            self.p.dvdtBH1 = acc_BH #- (1/M1_eff)*np.sum(np.atleast_2d(self.p.M_DM).T*acc_DM1, axis=0)
        else:
            self.p.dvdtBH1 = 0.0

        if (self.p.M_2 > 0):
            self.p.dvdtBH2 = -(M1_eff/M2_eff)*acc_BH - (1/M2_eff)*np.sum(np.atleast_2d(self.p.M_DM).T*acc_DM2, axis=0)
        else:
            self.p.dvdtBH2 = 0.0

        self.p.dvdtDM  = acc_DM1 + acc_DM2

        #Now, if a background force field has been set, calculate the acceleration
        if self.background_field is not None:
            self.p.dvdtBH1 += self.background_field(self.p.xBH1)
            self.p.dvdtBH2 += self.background_field(self.p.xBH2)
            self.p.dvdtDM  += self.background_field(self.p.xDM)


    def _softened_acceleration(self, dx, M, method, r_soft_sq):
        """
        Acceleration of particles at separation dx (N x 3) from a black hole of mass M, with the softening `method`.
        Same expressions as in update_acceleration.
        """
        r    = np.linalg.norm(dx, axis=-1, keepdims=True)
        r_sq = r**2
        dxh  = dx/r                                                    # unit vectors

        if (method == "plummer"):
            return -u.G_N*M*dxh*r*(r_sq + r_soft_sq)**(-3/2)

        elif (method == "plummer2"):
            return -u.G_N*M*r*(dxh/2)*(2*r_sq + 5*r_soft_sq)*(r_sq + r_soft_sq)**(-5/2)

        elif (method == "uniform_old"):
            x   = np.sqrt(r_sq/r_soft_sq)
            acc = -u.G_N*M*dxh/r_sq
            inds = (x < 1).flatten()
            if (np.sum(inds) >= 1):
                acc[inds] = -u.G_N*M*dxh[inds,:]*x[inds]*(8 - 9*x[inds] + 2*(x[inds])**3)/r_soft_sq
            return acc

        elif (method == "uniform"):
            x   = np.sqrt(r_sq/r_soft_sq)
            acc = -u.G_N*M*dxh/r_sq
            inds = (x < 1).flatten()
            if (np.sum(inds) >= 1):
                acc[inds] = -u.G_N*M*dxh[inds,:]*x[inds]/r_soft_sq
            return acc

        elif (method == "truncate"):
            return -u.G_N*M*dxh/np.clip(r_sq, r_soft_sq, 1e50)

        elif (method == "empty_shell"):
            acc = -u.G_N*M*dxh/r_sq
            inds = (np.sqrt(r_sq/r_soft_sq) < 1).flatten()
            if (np.sum(inds) >= 1):
                acc[inds] *= 0.0
            return acc

        raise ValueError("Invalid softening method:" + method)


    def full_step_split(self, dt):
        """
        One main step with sub-steps (force-split scheme, 2026-11-25):

            K_out(dt/2)  [ short step of the inner partition ]^N  D_out(dt)  K_out(dt/2)

        Inner partition : the particles selected by self.p.mask AND the two black holes.
        K_out           : kick from the interaction between the OUTER particles and the black holes, in both directions
                          (the black holes kick the outer particles, the outer particles kick the black holes),
                          all positions taken at the same time.
        short step      : 4th-order (PEFRL) step of length dt/N of the inner particles and the black holes, with the forces
                          of the black holes on the inner particles, of the inner particles on the black holes, and of the
                          black holes on each other.
        D_out           : free drift of the outer particles.

        The outer drift may come after the short steps because the two do not involve the same coordinates.
        Every force of the dark matter on a black hole is evaluated with both at the same time, so no partition lags the
        other, and there are no shadow black holes: the real ones take the short steps.
        The primary feels the dark matter only if self.react_M1 is True.
        """
        p = self.p
        if not p.dynamic_BH:
            raise ValueError("sub-steps need dynamic black holes")
        if self.background_field is not None:
            raise ValueError("sub-steps do not include a background field")

        inner, outer = p.mask, ~p.mask
        M_1, M_2 = p.M_1, p.M_2
        m_in  = np.atleast_1d(p.M_DM)[inner][:, None]
        m_out = np.atleast_1d(p.M_DM)[outer][:, None]

        def kick_outer(h):
            "outer particles <-> black holes, as a pair"
            x = p.xDM[outer]
            a1 = self._softened_acceleration(x - p.xBH1, M_1, self.soft_method1, self.r_soft_sq1)
            a2 = self._softened_acceleration(x - p.xBH2, M_2, self.soft_method,  self.r_soft_sq2)
            p.vDM[outer] += (a1 + a2)*h
            p.vBH2 = p.vBH2 - np.sum(m_out*a2, axis=0)/M_2*h
            if self.react_M1:
                p.vBH1 = p.vBH1 - np.sum(m_out*a1, axis=0)/M_1*h

        # inner particles: work on copies, written back at the end
        x_in = p.xDM[inner]
        v_in = p.vDM[inner]

        def drift_inner(h):
            nonlocal x_in
            x_in   = x_in + v_in*h
            p.xBH1 = p.xBH1 + p.vBH1*h
            p.xBH2 = p.xBH2 + p.vBH2*h

        def kick_inner(h):
            "inner particles <-> black holes as a pair, and the black holes on each other"
            nonlocal v_in
            a1 = self._softened_acceleration(x_in - p.xBH1, M_1, self.soft_method1, self.r_soft_sq1)
            a2 = self._softened_acceleration(x_in - p.xBH2, M_2, self.soft_method,  self.r_soft_sq2)
            dx12 = p.xBH1 - p.xBH2
            r12  = np.sqrt(np.sum(dx12**2))
            aBH1 = -u.G_N*M_2*dx12/r12**3
            aBH2 = +u.G_N*M_1*dx12/r12**3 - np.sum(m_in*a2, axis=0)/M_2
            if self.react_M1:
                aBH1 = aBH1 - np.sum(m_in*a1, axis=0)/M_1
            v_in   = v_in + (a1 + a2)*h
            p.vBH1 = p.vBH1 + aBH1*h
            p.vBH2 = p.vBH2 + aBH2*h

        kick_outer(0.5*dt)

        h = dt/self.N_partition
        for it2 in range(self.N_partition):
            drift_inner(xi*h)
            kick_inner((1-2*lam)*h/2)
            drift_inner(chi*h)
            kick_inner(lam*h)
            drift_inner((1-2*(chi + xi))*h)
            kick_inner(lam*h)
            drift_inner(chi*h)
            kick_inner((1-2*lam)*h/2)
            drift_inner(xi*h)

        p.xDM[inner] = x_in
        p.vDM[inner] = v_in

        p.xDM[outer] += p.vDM[outer]*dt

        kick_outer(0.5*dt)


    def run_simulation(self, dt, t_end, method="PEFRL", save_to_file = False, add_to_list = False, show_progress=False, save_DM_states=False, N_save=1, label=None, N_step_partition=1, IDhash=None):
        """
        Run the simulator, starting from the current state of particles in p, running for a time t_end.
        Times and timesteps are in physical times (as opposed to being in terms of number of orbits etc.)
        BEWARE: run_simulation will erase a previous version of the simulation with the same IDhash before starting.

        Parameters:
            dt (float)      : Size of the individual timesteps
            t_end (float)   : End time of the simulation
            method (string) : Leapfrog method. See `simulator.full_step` for more details.
            save_to_file (bool):    Set to True in order to output the simulation data to file. Default = False.
            add_to_list (bool):     Set to True in order to save metadata about the simulation to `SimulationList.txt`. Default = False.
            show_progress (bool):   Set to True in order to show a progress bar during the simulation. Default = False
            save_DM_states (bool):  Set to True in order to save the initial and final configuration of the DM particles in the output. Default = False
            N_save (int):    Number of time steps in between saving the data to file. Default = 1
            label (str):    String to be used in the name of the output file (along with the IDhash).
            N_step_partition (int): Number of timesteps to be performed on the inner set of particles for each main step. Default = 1.
            IDhash (str): passing a IDhash can be useful when restarting a simulation. Remember to also pass "reload" as label
        Returns:
            None

        """

        print("> Simulating...")

        print("soft method 2 =", self.soft_method)

        #Determine total number of steps
        self.t_end   = t_end
        self.dt      = dt
        self.current_step = 0
        self.runID   = label

        self.IDhash=IDhash

        if (self.IDhash is None):
            self.IDhash = tools.generate_hash()


        self.N_partition=N_step_partition
        if self.N_partition > 1:
            if self.r_partition2 < 0:
                #default: one Hill radius of the secondary
                self.r_partition2 = self.p.orbital_elements()[0]*(self.p.M_2/(3*self.p.M_1))**(1./3.)
            print("sub-steps: force split, SCHEME =", SCHEME,
                  "| sub-step region around M2:", ("off" if self.r_partition2 <= 0 else "%.3f a_i" % (self.r_partition2/self.p.orbital_elements()[0])),
                  "| primary feels the dark matter:", self.react_M1)

        if self.p.r_partition==-1:
            self.p.r_partition=9*self.r_soft
        print("simulationg with r partition", self.p.r_partition/tools.calc_risco(self.p.M_1), "r isco")

        if (self.runID is None):
            self.fileID = self.IDhash
        else:
            self.fileID = f"{self.runID}_{self.IDhash}"

        N_step = int(np.ceil(t_end/dt))

        #Initialise lists to save the BH positions
        self.ts        = np.linspace(0, t_end, N_step)
        self.xBH1_list = np.zeros((N_step, 3))
        self.vBH1_list = np.zeros((N_step, 3))

        self.xBH2_list = np.zeros((N_step, 3))
        self.vBH2_list = np.zeros((N_step, 3))

        self.M1_list   = np.zeros(N_step)
        self.M2_list   = np.zeros(N_step)


        #N_save = 100 #Save only every 100 timesteps
        #N_save = 1
        #N_out = int(N_step/N_save)
        print("N_save is", N_save)


        N_out = len(self.ts[::N_save])

        print("N_out is", N_out)

        #N_update = 100_000 #Update the output file only every 100_000 steps
        #N_update = 1
        N_update = N_save

        print("N update= ", N_update)


        N_update_mask=1
        print("N update mask= ", N_update_mask)
        #Determine initial orbital parameters of the system
        if (self.p.M_2 > 0):

            a_i, e_i = self.p.orbital_elements()
            self.a_i = float(a_i)
            self.e_i = float(e_i)
            T_orb    = self.p.T_orb()
        else:
            self.a_i = 0
            self.e_i = 0


        self.M_2_ini = self.p.M_2

        self.method = method
        self.finished = False


        #Add information to SimulationList.txt if required
        if (add_to_list):
            self.output_metadata_initial()

        #Open output file
        if (save_to_file):
            fname = f"{NbodyIMRI.snapshot_dir}/{self.fileID}.hdf5"

            try:
                os.remove(fname)
                print("Old file removed successfully:", fname)
            except:
                print("No old snapshot file found...")
            f = self.open_outputfile(fname, N_out, save_DM_states)



        #Save the time steps and the initial DM configuration
        if (save_to_file):
            print(N_step, len(self.t_data[:]), len(1.0*self.ts[::N_save]))
            self.t_data[:] = 1.0*self.ts[::N_save]
            self.t_data.flush() #force write to file
            self.M1_list[0] = self.p.M_1
            self.M2_list[0] = self.p.M_2

            if (save_DM_states):
                self.xDM_i_data[:,:] = 1.0*self.p.xDM
                self.vDM_i_data[:,:] = 1.0*self.p.vDM


        #Define a dummy in case we're not using a progress bar
        stepper = lambda x: x
        if (show_progress):
            stepper = tqdm

        #print("N_steps:", N_step)
        #Simulate for N_step time-steps
        for it in stepper(range(N_step)):

            #Do any checks of the state of the system in between timesteps
            if (self.check_state is not None):
                self.check_state(self)

            #Save current binary configuration to array
            self.M1_list[it]     = self.p.M_1
            self.M2_list[it]     = self.p.M_2

            self.xBH1_list[it,:] = self.p.xBH1
            self.vBH1_list[it,:] = self.p.vBH1

            self.xBH2_list[it,:] = self.p.xBH2
            self.vBH2_list[it,:] = self.p.vBH2

            #Update data saved in file
            if ((it%N_update == 0) and (save_to_file)):
                #print(N_step, N_save, N_out, N_update)
                self.M_1_data[:]    = 1.0*self.M1_list[::N_save]
                self.M_2_data[:]    = 1.0*self.M2_list[::N_save]

                self.xBH1_data[:,:] = 1.0*self.xBH1_list[::N_save,:]
                self.vBH1_data[:,:] = 1.0*self.vBH1_list[::N_save,:]

                self.xBH2_data[:,:] = 1.0*self.xBH2_list[::N_save,:]
                self.vBH2_data[:,:] = 1.0*self.vBH2_list[::N_save,:]

                self.M_1_data.flush()
                self.M_2_data.flush()

                self.xBH1_data.flush()
                self.vBH1_data.flush()

                self.xBH2_data.flush()
                self.vBH2_data.flush()

            #Step forward by dt
            if self.N_partition>1:
                #the inner partition: the particles near M1 (and near M2 if r_partition2 > 0), together with the black holes
                if (it%N_update_mask == 0):
                    self.p.mask = tools.norm(self.p.xDM - self.p.xBH1) < self.p.r_partition
                    if self.r_partition2 > 0:
                        self.p.mask = self.p.mask | (tools.norm(self.p.xDM - self.p.xBH2) < self.r_partition2)
                self.full_step_split(dt)

            else:

                self.full_step(dt, method)





            #Increment the current step number (this is primarily so that the
            #check_state function has some idea about how far in the simulation we are...)
            self.current_step += 1


        #One final update of the output data
        if (save_to_file):
            self.M_1_data[:]    = 1.0*self.M1_list[::N_save]
            self.M_2_data[:]    = 1.0*self.M2_list[::N_save]

            self.xBH1_data[:,:] = 1.0*self.xBH1_list[::N_save,:]
            self.vBH1_data[:,:] = 1.0*self.vBH1_list[::N_save,:]

            self.xBH2_data[:,:] = 1.0*self.xBH2_list[::N_save,:]
            self.vBH2_data[:,:] = 1.0*self.vBH2_list[::N_save,:]

            if (save_DM_states):
                self.xDM_f_data[:,:] = 1.0*self.p.xDM
                self.vDM_f_data[:,:] = 1.0*self.p.vDM

                self.M_DM_data[:] =  1.0*self.p.M_DM

        print("> Simulation completed.")


        #Add information to SimulationList.txt if required
        if (add_to_list):
            self.output_metadata()

        if (save_to_file):
            f.close()

        self.finished = True



    def open_outputfile(self, fname, N_step, save_DM_states):
        """
        ...

        """
        f = h5py.File(fname, "w", libver="latest")

        grp = f.create_group("data")
        grp.attrs['M_1'] = self.p.M_1/u.Msun
        grp.attrs['M_2'] = self.M_2_ini/u.Msun

        a_i, e_i = self.p.orbital_elements()

        #Still need to add other stuff here!

        grp.attrs['a_i'] = a_i/u.pc
        grp.attrs['e_i'] = e_i
        grp.attrs['N_DM'] = self.p.N_DM
        grp.attrs['M_DM'] = self.p.M_DM[0]/u.Msun
        grp.attrs['r_soft'] = np.sqrt(self.r_soft_sq2)/u.pc
        grp.attrs['scheme'] = SCHEME
        grp.attrs['r_partition2'] = max(self.r_partition2, 0.0)/u.pc
        grp.attrs['react_M1'] = int(self.react_M1)
        grp.attrs['r_partition'] = self.p.r_partition/u.pc
        if (self.p.dynamic_BH):
            grp.attrs['dynamic'] = 1
        else:
            grp.attrs['dynamic'] = 0


        datatype = np.float64
        # These arrays are rewritten in full at every save. They must NOT be compressed: a compressed array is stored
        # in chunks, each rewrite allocates new chunks, and the space of the old ones is not reused while the file is
        # open, so the file grew by ~57 kB per orbit (626 MB after 11000 orbits for 2 MB of data). Uncompressed arrays
        # are overwritten in place and the file keeps its size.
        self.t_data   = grp.create_dataset("t", (N_step,), dtype=datatype)
        self.M_1_data = grp.create_dataset("M_1", (N_step,), dtype=datatype)
        self.M_2_data = grp.create_dataset("M_2", (N_step,), dtype=datatype)

        self.xBH1_data = grp.create_dataset("xBH1", (N_step,3), dtype=datatype)
        self.vBH1_data = grp.create_dataset("vBH1", (N_step,3), dtype=datatype)

        self.xBH2_data = grp.create_dataset("xBH2", (N_step,3), dtype=datatype)
        self.vBH2_data = grp.create_dataset("vBH2", (N_step,3), dtype=datatype)



        if (save_DM_states):
            self.xDM_i_data = grp.create_dataset("xDM_i", (self.p.N_DM,3), dtype=datatype, compression="gzip")
            self.xDM_f_data = grp.create_dataset("xDM_f", (self.p.N_DM,3), dtype=datatype, compression="gzip")

            self.vDM_i_data = grp.create_dataset("vDM_i", (self.p.N_DM,3), dtype=datatype, compression="gzip")
            self.vDM_f_data = grp.create_dataset("vDM_f", (self.p.N_DM,3), dtype=datatype, compression="gzip")

            self.M_DM_data = grp.create_dataset("M_DM", (self.p.N_DM,), dtype=datatype, compression="gzip")

        f.swmr_mode = True  # Activate SWMR mode
        f.flush()


        return f

    def output_metadata_initial(self):
        """
        ...
        """

        listfile = f'{NbodyIMRI.snapshot_dir}/SimulationListAll.txt'
        hdrtxt = "Columns: FileID, M_1/MSUN, M_2/MSUN, a_i/r_isco(M1), e_i, N_DM, M_DM/MSUN, Nstep_per_orb, N_orb, r_soft/PC, method, rho_6/(MSUN/PC**3), gamma, alpha, r_t/PC"

        T_orb = 2*np.pi*np.sqrt(self.a_i**3/(u.G_N*self.p.M_tot()))

        meta_data = np.array([self.fileID, self.p.M_1/u.Msun, self.M_2_ini/u.Msun,
                            self.a_i/tools.calc_risco(self.p.M_1), self.e_i, self.p.N_DM, self.p.M_DM[0]/u.Msun,
                            int(np.round(T_orb/self.dt)), int(np.round(self.t_end/T_orb)), np.sqrt(self.r_soft_sq2)/u.pc, self.method,
                            self.p.rho_6/(u.Msun/u.pc**3), self.p.gamma_sp, self.p.alpha, self.p.r_t/u.pc])

        meta_data = np.reshape(meta_data, (1,  len(meta_data)))


        if (os.path.isfile(listfile)):
            with open(listfile,'a') as g:
                np.savetxt(g, meta_data, fmt='%s')
            g.close()
        else:
            np.savetxt(listfile, meta_data, header=hdrtxt, fmt='%s')



    def output_metadata(self):
        """
        ...
        """

        listfile = f'{NbodyIMRI.snapshot_dir}/SimulationListCompleted.txt'
        hdrtxt = "Columns: FileID, M_1/MSUN, M_2/MSUN, a_i/r_isco(M1), e_i, N_DM, M_DM/MSUN, Nstep_per_orb, N_orb, r_soft/PC, method, rho_6/(MSUN/PC**3), gamma, alpha, r_t/PC"

        T_orb = 2*np.pi*np.sqrt(self.a_i**3/(u.G_N*self.p.M_tot()))

        meta_data = np.array([self.fileID, self.p.M_1/u.Msun, self.M_2_ini/u.Msun,
                            self.a_i/tools.calc_risco(self.p.M_1), self.e_i, self.p.N_DM, self.p.M_DM[0]/u.Msun,
                            int(np.round(T_orb/self.dt)), int(np.round(self.t_end/T_orb)), np.sqrt(self.r_soft_sq2)/u.pc, self.method,
                            self.p.rho_6/(u.Msun/u.pc**3), self.p.gamma_sp, self.p.alpha, self.p.r_t/u.pc])

        meta_data = np.reshape(meta_data, (1,  len(meta_data)))


        if (os.path.isfile(listfile)):
            with open(listfile,'a') as g:
                np.savetxt(g, meta_data, fmt='%s')
            g.close()
        else:
            np.savetxt(listfile, meta_data, header=hdrtxt, fmt='%s')

    def plot_orbital_elements(self):
        """
        ...
        """

        if (self.finished == False):
            print("Simulation has not been finished. Please run using `rum_simulation()`.")
            return 0

        else:
            xBH_list = self.xBH1_list - self.xBH2_list
            vBH_list = self.vBH1_list - self.vBH2_list
            a_list, e_list = tools.calc_orbital_elements(xBH_list, vBH_list, self.M1_list + self.M2_list)

            delta_a = (a_list - a_list[0])/a_list
            delta_e = (e_list - e_list[0])

            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))

            axes = ax[:]


            axes[0].plot(self.ts, delta_a)
            axes[0].set_xlabel(r"$t$ [s]")
            axes[0].set_ylabel(r"$\Delta a/a$")

            axes[1].plot(self.ts, delta_e)
            axes[1].set_xlabel(r"$t$ [s]")
            axes[1].set_ylabel(r"$\Delta e$")

            #plt.title(self.fileID)
            plt.tight_layout()

            plt.show()
            return fig

    def plot_trajectory(self):
        """
        ...
        """
        if (self.finished == False):
            print("Simulation has not been finished. Please run using `rum_simulation()`.")
            return 0

        else:
            fig = plt.figure()

            plt.plot(self.xBH1_list[:,0]/u.pc, self.xBH1_list[:,1]/u.pc, lw=2)
            #plt.plot(self.xBH2_list[:,0]/u.pc, self.xBH2_list[:,1]/u.pc)

            plt.xlabel(r"$x$ [pc]")
            plt.ylabel(r"$y$ [pc]")
            plt.gca().set_aspect('equal')

            plt.show()
            return fig

    def plot(self):
        self.p.plot()
