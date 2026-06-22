"""Ensemble square-root filters for the Lorenz 96 model"""
import datetime
import numpy as np
import scipy.stats as stats
import os, sys
import time
from scipy import interpolate
from L96 import L96
from enkf import *
from rank_histogram import *
from hovmuller_diagram import *
from plot import *
import pickle
import traceback
import matplotlib.pyplot as plt

#np.seterr(all='raise')                                   # raise error when floating-point errors occur
#mpl.style.use('default')

class L96enkf:
    def __init__(self, nts=2000, enkf_method=3, nEnKF=1, dt_da=0.05, N=40, F=8.0, nem=10, nem_sclfct=1, sp_dist=1, proc_q_covinfl_par=1.0, proc_errstddev=1.0, obs_errstddev=1.0, cov_loc_dist=1.0, cov_loc_dist_pert=None, nobs=100, obs_pos_dst_seed=592, truth_seed=42, obs_pos_dst=1, h_type=0, obs_loc_dist=1.0, row_rank_thresh = 0.99, cov_infl_method=-1, cov_infl_par1=0.4, cov_infl_par2=1.0, K_solver=3, plot_show=False, plot_save=True, plot_data_save=False, mpi=False, loop_test=False):
        """
        Initialize 
        """
        # 1) time step
        self.nts_spinup = 1000                           # time steps to spin up truth run (for steady state)
        self.nts = nts                                   # time steps

        # 2) L96 model parameters
        self.dt = 0.05                                   # model interval (dt = 0.05 means 6 hours, since unit timestep dt = 1 means 5 days in L96 model)
        self.dt_da = dt_da                               # assimilation interval
        self.N = N                                       # the number of grid points along a latitude circle (each grid has one state variable). Moreover, N is dimension of L96 model (global model).
                                                         # x_0 (= x_40), x_1, x_2, ..., x_38, x_39 for N = 40
                                                         #           x_0
                                                         #      x_39     x_1
                                                         #  x_38            x_2
                                                         #   :               :
                                                         #   :               :
                                                         #  x_20            x_18
                                                         #      x_21     x_19
                                                         #           x_20
        self.ndim = self.N                               # global dimension
        self.ndim_loc = 1                                # local dimension (each grid has one state variable)
        self.F = F                                       # forcing (resulting in chaotic behavior in the system dynamics)
        self.deltaF = 1./8.                              # [optional] scale parameter of gamma distribution
        #self.deltaF = 0.                                # [optional] scale parameter of gamma distribution
        self.Fcorr = np.exp(-1)**(1./3.)                 # [optional] efolding over n timesteps, n=3

        # 3) [optional] zonally varying advection diffusion and obaservation localization distance
        #    - when grid point is closer to the first grid point, diffusion is stronger and localization distance is longer.
        #    - when grid point is more distant from the first grid point, diffusion is weaker and localization distance is shorter.
        #    - If blend_min and blend_max are set to  1., this function is not available.
        self.blend_min = 1.                 
        self.blend_max = 1.
        #self.blend = np.cos(np.linspace(0., np.pi, self.N))**4
        self.blend = self.blend_min + (self.blend_max - self.blend_min)*(np.cos(np.linspace(0., np.pi, self.N))**4)
        
        # 4) ensemble Kalman filter (EnKF) method
        #    - 0:  serial EnSRF (serial Potter method (one obs at a time))
        #    - 1:  EnSRF (bulk Potter method (all obs at once))
        #    - 2:  ETKF (no localization applied)
        #    - 3:  LETKF (using observation localization) (with observation space horizontal covariance localization; R localization)
        #    - 4:  UTKF (no localization applied)
        #    - 5:  LUTKF (using observation localization)  
        #    - 6:  SUTKF (no localization applied)
        #    - 7:  SLUTKF (using observation localization)
        #    - 8:  PF (no localization applied): Due to resampling for full state, estimation accuracy deteriorate. Therefore, EPF is recommended for global data assimilation.
        #    - 9:  EPF (no localization applied): PF but each state element is estimated individually similar to LPF        
        #    - 10: LPF (using observation localization)         
        #    - 11: serial EnSRF (serial Potter method with localization via modulation ensemble)
        #    - 12: ETKF with modulation ensemble
        #    - 13: ETKF with modulation ensemble and perturbed obs
        #    - 14: serial EnSRF (serial Potter method using sqrt of localized Pb ensemble)
        #    - 15: EnKF (all obs at once)
        #    - 16: EnKF (all obs at once) with perturbed obs
        #    - 17: GETKF (with no localization)
        #    - 18: GETKF (with modulated ensemble for model space vertical covariance localization (B localization))
        #    - 19: GETKF (with modulated ensemble for model space vertical covariance localization (B localization) and R localization)        
        #    - 20: ETKF with modulation ensemble and stochastic subsampling
        #    - 21: ETKF with modulation ensemble and 'adjusted' perturbed obs
        #    - 22: 'DEnKF' approx to ETKF with modulated ensemble
        #    - 23: 'DEnKF' approx to bulk potter method.   
        self.enkf_method = enkf_method
        self.enkf_method_list = ["serial_ensrf","bulk_ensrf","etkf","letkf","utkf","lutkf","sutkf","slutkf","pf","epf","lpf","serial_ensrf_modens","etkf_modens", \
                                 "etkf_modens_ptbobs","serial_ensrf_modens","bulk_enkf","bulk_enkf_ptbobs","getkf","getkf_modens","etkf_modens","etkf_modens_ptbobs","denkf","bulk_denkf"]
        self.enkf_method_name =  self.enkf_method_list[self.enkf_method]
        self.enkf_method_name_c = self.enkf_method_name.upper()

        # 5) number of ensemble members
        self.nem = nem 
        self.nem_sclfct = nem_sclfct                            # scaling factor of ensemble size for UTKF, LUTKF, SUTKF and SLUTKF
        self.sp_dist = sp_dist                                  # distribution of ensemble (sigma point) for UTKF, LUTKF, SUTKF and SLUTKF (0: uniform distribution, 1: normal distribution, 2: normal distribution using box muller transform)
        self.sp_dist_nrml = 0                                   # normal distribution of ensemble (sigma point) for UTKF and SUTKF (when self.sp_dist = 1) (0: each element, 1: entire element)

        if self.enkf_method == 4:                               # UTKF (global dimension = self.N)
            self.nem = 2*self.nem_sclfct*self.N                        
        elif self.enkf_method == 5:                             # LUTKF (since each grid point has a state varialbe, local dimenstion = 1)
            self.nem = 2*self.nem_sclfct*self.ndim_loc
        elif self.enkf_method == 6:                             # SUTKF (global dimension = self.N)
            self.nem = (2*self.nem_sclfct*self.N) + 1
        elif self.enkf_method == 7:                             # SLUTKF (since each grid point has a state varialbe, local dimenstion = 1)
            self.nem = (2*self.nem_sclfct*self.ndim_loc) + 1

        if self.sp_dist == 2:
            u = (np.arange(1, self.nem_sclfct+2, 1)/float(self.nem_sclfct+1))[0:self.nem_sclfct]
            self.nrml_bmt = np.sqrt(-2.0*np.log(u))
            self.nrml_bmt = self.nrml_bmt / self.nrml_bmt.max() # normal distribution using box muller transform
        else:
            self.nrml_bmt = None

        # 6) Gaussian sum filter
        self.nEnKF = nEnKF                                      # number of multiple EnKF methods
        self.same_wgt_nEnKF = True                              # same weights for multiple EnKF methods 
        
        # 7) process error
        self.proc_errstddev = proc_errstddev                    # process error standard deviation for forecast ensemble
        self.proc_q = self.proc_errstddev**2                    # process error variance for forecast ensemble

        if self.enkf_method in [4,5,6,7,8,9,10]:
            self.proc_q_covinfl_par = proc_q_covinfl_par

        # 8) background covariance localization (model space localization or B localization)
        self.cov_loc_method = 0                                  # localization function of background error covariance Pb (0: Gaspari-Cohn polynomial (best), 1: Bohman taper (good))
        self.cov_loc_dist = cov_loc_dist                          # localization distance (more precisely, cutoff distance)
        self.cov_loc_dist_pert = cov_loc_dist_pert
        self.locmtx_nmz = 0                                     # localization matrix normalization 
                                                                # 0: localization matrix normalization so that the sum of eigen values of low-rank localization matrix is the same as that of full-rank localization matrix)
                                                                # 1: localization matrix normalization so that each of diagonal elements of low-rank localization matrix is one
        
        # 9) observation size and location
        self.nobs = nobs #self.N                                # number of observations
        self.obs_pos_dst_seed = obs_pos_dst_seed
        self.obs_pos_dst = obs_pos_dst                          # 0: one observation over each state location
                                                                # 1: Gaussian distribution centered on variable 20 with a standard deviation of 1/5 the domain length
                                                                # 2: uniform distribution over the domain length 
        if self.obs_pos_dst == 0:
            self.nobs = self.N 
            self.obs_pos = np.arange(self.N)                 
        elif self.obs_pos_dst == 1:
            #self.obs_pos = (self.N/2 - 1) + (self.N/5.)*np.random.RandomState(seed=1).standard_normal(size=self.nobs) 
            #self.obs_pos = (self.N/2 - 1) + (self.N/3.)*np.random.RandomState(seed=2).standard_normal(size=self.nobs)
            #self.obs_pos = (self.N/2 - 1) + (self.N/3.)*np.random.RandomState(seed=592).standard_normal(size=self.nobs) [best]
            self.obs_pos = (self.N/2 - 1) + (self.N/3.)*np.random.RandomState(seed=self.obs_pos_dst_seed).standard_normal(size=self.nobs)
            self.obs_pos = np.where(self.obs_pos < 0., self.obs_pos + float(self.N - 1), self.obs_pos)
            self.obs_pos = np.where(self.obs_pos > float(self.N - 1), self.obs_pos - float(self.N - 1), self.obs_pos)
        elif self.obs_pos_dst == 2:   
            self.obs_pos = np.random.RandomState(seed=1).uniform(0,self.N-1,self.nobs)            
            
        # 10) observation operator H (Gaussian kernel smoothing or heaviside kernel smoothing (boxcar smoothing))
        #    - the weighted average of neighboring data. the weight of neighbors is determined by gaussian kernel or heaviside kernel
        #    - 1) gaussian kernel smoothing
        #         - http://iskim3068.tistory.com/41
        #    - 2) heaviside kernel smoothing
        #         - a.k.a. boxcar smoothing (moving average filter or running average filter)
        #         - it is used to smooth the data and replaces each data value with the average of neighboring values
        #         - https://www.wavemetrics.com/products/igorpro/dataanalysis/signalprocessing/smoothing.htm
        #    - 3) kernel smoothing 
        #         - https://en.wikipedia.org/wiki/Kernel_(statistics)
        #         - https://en.wikipedia.org/wiki/Kernel_smoother#Gaussian_kernel_smoother
        self.kernel_smooth = 0                                  # 0: Gaussian kernel smoothing, 1: heaviside kernel smoothing (boxcar smoothing)
        self.smooth_len = 1.                                    # for gaussian kernel smoothing, smooth_len is standard deviation. 
                                                                # for heaviside kernel smoothing (boxcar smoothing), smooth_len is half-width of boxcar.
        self.h_type = h_type                                    # 0: linear h(x)=x, 1: nonlinear h(x)=|x|, 2: nonlinear h(x)=ln(|x|) 
        
        # 11) observation error
        self.obs_errstddev = obs_errstddev                      # observation error standard deviation
        if self.enkf_method == 7:
            if obs_errstddev == 0.1:
                self.obs_errstddev = 0.06
        self.obs_r = self.obs_errstddev**2                      # observation error variance

        # 12) observation localization (observation space localization or R localization)
        self.obs_rloc_method = 0                                # localization weight function of observation error covariance R (0: Gaspari-Cohn polynomial (best), 1: Bohman taper (good))
        self.obs_loc_dist = obs_loc_dist                        # localization distance (more precisely, cutoff distance)

        # 13) posterior covariance inflation for ETKF and LETKF
        self.cov_infl_method = cov_infl_method                    # -1: off, 0: Hodyss and Campbell inflation (best), 1: relaxation to prior spread (RTPS) inflation (good)
        self.cov_infl_par1 = cov_infl_par1                        # default: 1., cov_infl_par1 corresponding to parameter a in RTPS inflation (https://doi.org/10.1175/MWR-D-15-0329.1).
        self.cov_infl_par2 = cov_infl_par2                        # default: 1., cov_infl_par1 and cov_infl_par2 corresponding to parameters a (<=1) and b (>=1) in Hodyss and Campbell inflation (https://doi.org/10.1175/MWR-D-15-0329.1).

        # 14) scaling parameters for SUTKF and SLUTKF
        self.alpha = 1.                                         # controls size of sigma point distribution; 0 <= self.alpha <= 1
        self.beta = 2.                                          # knowledge of the higher order moments of sigma point distribution; self.beta = 2
        self.kappa = 0.                                         # gurantees positive semi-definiteness of the covariance matrix; self.kappa = 0
        
        if self.enkf_method == 6:                               # SUTKF (global dimension = self.N)
            self.en = self.nem_sclfct*self.N
            self.lamda = (self.alpha**2)*(float(self.en) + self.kappa) - float(self.en)
        elif self.enkf_method == 7:                             # SLUTKF (since each grid point has a state varialbe, local dimenstion = 1)
            self.en = self.nem_sclfct*self.ndim_loc
            self.lamda = (self.alpha**2)*(float(self.en) + self.kappa) - float(self.en)
        else:
            self.en = 0.
            self.lamda = 0.

        # 15) parameters for PF, EPF, and LPF
        self.ds_func = 0                                        # distribution function for sample weighting (0:t-distribution, 1:normal distribution)
        self.nu = 100.0                                         # default: 0.1, t-distribution normality parameter (as nu increases, t-distribution becomes closer to normal distribution)
        self.rs_thrd = 0.5                                      # resampling threshold (0 <= rs_thrd <= 1)                      
        self.rs_mode = 0                                        # resampling mode (0:systematic, 1:stratified)
        self.w_pf = np.zeros(self.nem, float)                   # weight of samples of PF
        self.w_epf = np.zeros((self.nem, self.N), float)        # weight of samples of EPF
        self.w_lpf = np.zeros((self.nem, self.N), float)        # weight of samples of LPF
        self.w_acc = False                                      # accumulate weight of samples
        self.prior_additive_noise = False                       # prior additive noise

        # 16) parameter to verify steady state of da method
        self.verf_da_steady_state = True
        self.nts_da_steady_state = self.nts_spinup

        if self.verf_da_steady_state:
            if self.nts_da_steady_state >= self.nts:
                self.nts += self.nts_da_steady_state
        
        # 17) [optional] threshold for modulated ensemble eigenvalue truncation (low-rank vertical covariance localization matrix) for 18: GETKF (with modulated ensemble).
        #self.thresh = 0.99
        self.row_rank_thresh = row_rank_thresh

        # 18) [optional] random seed for random number generator
        self.rs_truth = np.random.RandomState(seed=truth_seed)  # fixed seed for truth run
        self.rs_ens = np.random.RandomState(seed=20)            # varying seed for ob noise and ensemble initial conditions

        # 19) [optional] Kalman gain solver for UKF da method (UTKF, LUTKF, SUTKF, SLUTKF)
        self.K_solver = K_solver                                # 0: Kalman gain solution using matrix inverse (slow)
                                                                # 1: standard linear solver (fast, safe)
                                                                # 2: linear solver using cholesky factorization (fastest, unsafe)
                                                                # 3: computation in the subspace spanned by the ensemble (fastest, safe)
        # 20) [optional] rank histogram
        self.rank_hist = []
        
        for iEnKF in range(self.nEnKF):
            self.rank_hist.append(np.zeros((self.nem+1, self.N), int))
        
        # 21) [optional] print error stats every time if True
        self.verbose = False

        # 22) [optional] plot
        self.plot_show = plot_show        
        self.plot_outdir = 'fig'
        self.plot_save = plot_save
        self.plot_data_outdir = 'data'        
        self.plot_data_save = plot_data_save
                
        if self.plot_show or self.plot_save or self.plot_data_save:
            if os.name != 'nt':                                 # in case of linux system
                import matplotlib
                matplotlib.use('Agg')                           # Generating matplotlib graphs without a running X server
            import matplotlib.pyplot as plt
            self.plt = plt

            if sys.platform.startswith('win'):
                set_plot_font(self.plt,0)                       # plot font: Arial
                
            if (self.plot_show or self.plot_save) and (not os.path.exists(self.plot_outdir)):
                os.makedirs(self.plot_outdir)

            if (self.plot_data_save) and (not os.path.exists(self.plot_data_outdir)): 
                os.makedirs(self.plot_data_outdir)
                
        # 23) [optional] mpi                                    # LETKF, UTKF, LUTKF, SUTKF, SLUTKF, EPF, LPF
        self.mpi = mpi
        
        if self.mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.myrank = self.comm.Get_rank()
            self.nproc = self.comm.Get_size()
        else:
            self.comm = None
            self.myrank = 0
            self.nproc = 1

        # 24) [optional] Maximal Lyapunov exponent (MLE) test
        self.mle = False
        
        # 25) [optional] loop performance test (skip errors that happen at timestep)
        self.loop_test = loop_test
        
        self.gen_L96_instance()
        self.gen_truth_data()
        self.gen_h_and_obs()
        self.gen_locmtx()           # generatre localization matrix for B localization (model space localization)      
        self.gen_obs_rloc()         # generatre localization weight matrix for R localization (observation space localization)

    def gen_L96_instance(self):
        """
        Generate L96 model instance 
        """
        # model instance for truth (nature) run
        self.model = L96(dt=self.dt, N=self.N, ndim_loc=self.ndim_loc, blend=self.blend, F=self.F, deltaF=self.deltaF, Fcorr=self.Fcorr, \
                         proc_q=self.proc_q, en=self.en, lamda=self.lamda, rs=self.rs_truth)

        # model instance for forecast ensemble
        self.ensemble = []
        
        for iEnKF in range(self.nEnKF):
            self.ensemble.append(L96(enkf_method=self.enkf_method, nem=self.nem, nem_sclfct=self.nem_sclfct, sp_dist=self.sp_dist, sp_dist_nrml=self.sp_dist_nrml, \
                                nrml_bmt=self.nrml_bmt, dt=self.dt, N=self.N, ndim_loc=self.ndim_loc, blend=self.blend, F=self.F, deltaF=self.deltaF, Fcorr=self.Fcorr, \
                                proc_q=self.proc_q, en=self.en, lamda=self.lamda, rs=self.rs_ens))


    def gen_truth_data(self):
        """
        Obtain truth data by running L96 model 
        """
        # 1) spin up truth run
        for its in range(self.nts_spinup): 
            self.model.fcst()

        # 2) truth run    
        x = []
        t = []

        for its in range(self.nts):
            self.model.fcst()
            x.append(self.model.x[0,:])                                            # single ensemble member
            t.append(float(its)*self.dt)

        self.x_truth = np.array(x, float)
        #self.x_truth = np.loadtxt(os.path.join('data',"3_data_over_time_enkf_method_7_h_type_2_nem_3_nobs_200_x_truth.txt"))
        self.t_truth = np.array(t, float)

        if self.mle:
            import nolds
            
            mle_list = []
            
            for i in range(self.N):
                mle_list.append(nolds.lyap_e(self.x_truth[:,i]).max())

            mle_mean = np.array(mle_list, float).mean()
            mle_max = np.array(mle_list, float).max()
            mle_min = np.array(mle_list, float).min()
            
            print("Maximal Lyapunov exponent (MLE) = {}".format(mle_mean))
        
        # mean and standard deviation of x truth data 
        if self.verbose:                                    
            x_truth_mean = self.x_truth.mean(axis=0)
            x_truth_var = ((self.x_truth - x_truth_mean)**2).sum(axis=0)/float(self.nts-1)
            print("x_truth mean = {}".format(x_truth_mean.mean()))
            print("x_truth stddev = {}".format(np.sqrt(x_truth_var.mean())))

            
    def gen_h_and_obs(self):
        """
        Determine observation operator H and generate satellite radiance observations from true data (cf. in this model, f: nonliear, H: linear)
        """
        self.h = np.zeros((self.nobs, self.N), float)

        # determine H using weights for satellite radiance observations as in Fig. 1 in Bishop et al. 2017 (doi.org/10.1175/MWR-D-17-0102.1)
        if float(self.smooth_len) > 0.:
            for j in range(self.nobs):
                for i in range(self.N):
                    jp = self.obs_pos[j]
                    rr = float(i-jp)

                    if rr < -float(self.N/2): 
                        rr = float(i-jp+self.N)                                   

                    if rr > float(self.N/2): 
                        rr = float(i-jp-self.N)                                   

                    r = np.fabs(rr)/float(self.smooth_len)


                    if self.kernel_smooth == -1:  # point obs
                        jp = int(self.obs_pos[j])
                        self.h[j, jp] = 1.0
                        continue
                    elif self.kernel_smooth == 0:                                  # Gaussian kernel
                        self.h[j,i] = np.exp(-r**2) 
                    elif self.kernel_smooth == 1:                                # running average (heaviside kernel)
                        if r <= 1.:
                            self.h[j,i] = 1.
                        else:
                            self.h[j,i] = 0.
                            
                # normalize H so sum of weight is 1 (https://en.wikipedia.org/wiki/Kernel_(statistics))
                self.h[j,:] = self.h[j,:]/self.h[j,:].sum()
        
        self.obs = np.zeros((self.x_truth.shape[0], self.nobs), float)
        
        for its in range(self.x_truth.shape[0]):
            self.obs[its] = np.dot(self.h, self.x_truth[its])                    # kernel smoothing (the weighted average of neighboring x_truth data. the weight of neighbors is determinded by gaussian or heaviside kernel)
            self.obs[its] = nonlinear_h(self.obs[its], self.h_type)
            
        self.obs = self.obs + self.obs_errstddev*self.rs_ens.standard_normal(size=self.obs.shape)

    def gen_locmtx_and_z(self, cov_loc_dist):
        locmtx = np.zeros((self.N, self.N), float)

        if float(cov_loc_dist) < float(2*self.N):
            for j in range(self.N):
                for i in range(self.N):
                    rr = float(i-j)
                    if rr < -float(self.N/2):
                        rr = float(i-j+self.N)
                    if rr > float(self.N/2):
                        rr = float(i-j-self.N)

                    r = np.fabs(rr)/float(cov_loc_dist)
                    taper = 0.0

                    if self.cov_loc_method == 0:
                        rr2 = 2.*r
                        if r <= 0.5:
                            taper = (((-0.25*rr2 + 0.5)*rr2 + 0.625)*rr2 - 5.0/3.0)*rr2**2 + 1.
                        elif (r > 0.5) and (r < 1.):
                            taper = ((((rr2/12.0 - 0.5)*rr2 + 0.625)*rr2 + 5.0/3.0)*rr2 - 5.0)*rr2 + 4.0 - 2.0/(3.0*rr2)
                    elif self.cov_loc_method == 1:
                        if r < 1.:
                            taper = (1.-r)*np.cos(np.pi*r) + np.sin(np.pi*r)/np.pi

                    locmtx[j, i] = taper

        evals, evecs = np.linalg.eigh(locmtx)
        evalsum = evals.sum()
        neig = 0
        frac = 0.0
        while frac < self.row_rank_thresh:
            frac = evals[self.N-neig-1:self.N].sum()/evalsum
            neig += 1

        if self.locmtx_nmz == 0:
            z = (evecs[:, self.N-neig:self.N] * np.sqrt(evals[self.N-neig:self.N]/frac)).T
        else:
            evecs2 = evecs[:, self.N-neig:self.N] * np.sqrt(evals[self.N-neig:self.N])
            zz = np.dot(evecs2, evecs2.T)
            z = np.dot(np.diag(1./np.sqrt(np.diag(zz))), evecs2).T

        return locmtx, z, neig

    def gen_locmtx(self):
        """
        Generate localization matrix for background error covariance Pb (for model space localization or B localization)  
        """
        self.locmtx, self.z, self.neig = self.gen_locmtx_and_z(self.cov_loc_dist)

        if getattr(self, "cov_loc_dist_pert", None) is None:
            self.locmtx_pert = self.locmtx
            self.z_pert = self.z
            self.neig_pert = self.neig
        else:
            self.locmtx_pert, self.z_pert, self.neig_pert = self.gen_locmtx_and_z(self.cov_loc_dist_pert)

    def gen_obs_rloc(self):
        """
        Generatre localization weight matrix for observation error covariance R (for observation space localization or R localization)
        """
        self.obs_rloc = np.zeros((self.N, self.nobs), float)                    # localization weighting matrix for observation error covariance R
        self.obs_dist = np.zeros((self.N, self.nobs), float)

        if float(self.obs_loc_dist) < float(2*self.N):
            for j in range(self.N):
                for i in range(self.nobs):
                    # method 1 
                    ip = self.obs_pos[i]
                    rr = float(ip-j)
                    #dd = abs(rr)                                               # vertical covariance localization matrix
                    
                    if rr < -float(self.N/2): 
                        rr = float(ip-j+self.N)                                   
                        
                    if rr > float(self.N/2): 
                        rr = float(ip-j-self.N)                                   
                        
                    dd = np.fabs(rr)
                    self.obs_dist[j,i] = dd

                    #r = np.fabs(rr)/(self.blend[i]*float(self.obs_loc_dist))   # due to model.blend[i], when grid point is closer to the first grid point, localization distance is longer.
                    r = dd/float(self.obs_loc_dist)
                    taper = 0.0
                    
                    if self.obs_rloc_method == 0:                               # Gaspari-Cohn polynomial (best) (Gneiting 2002, doi:10.1006/jmva.2001.2056, equation, eq 23). Figure 3 of that paper compares Bohman and GC tapers.          
                        rr = 2.*r                                               # a piecewise polynomial approximation of a Gaussian localization function with compact support                

                        if r <= 0.5:
                            taper = (((-0.25*rr + 0.5)*rr + 0.625)*rr - 5.0/3.0)*rr**2 + 1.
                        elif (r > 0.5) and (r < 1.):
                            taper = ((((rr/12.0 - 0.5)*rr + 0.625)*rr + 5.0/3.0)*rr - 5.0)*rr + 4.0 - 2.0 / (3.0*rr)                            
                    elif self.obs_rloc_method == 1:                              # Bohman taper (good) (Gneiting 2002, doi:10.1006/jmva.2001.2056, equation, eq 21)
                        if r < 1.:
                            taper = (1.-r)*np.cos(np.pi*r) + np.sin(np.pi*r)/np.pi
                                                
                    self.obs_rloc[j,i] = taper
                    
        if self.plot_show or self.plot_save or self.plot_data_save:
            #self.plt.matshow(self.obs_rloc)
            #self.plt.colorbar()
            #self.plt.show()
            pass
        

    def get_mean_ptb_cov_err_msd_for_gsf(self, xmean_mf, xptb_mf, xcov_mf, x_truth):
        """
        Get mean, perturbation, covariance, error, and rmsd for multiple filters (Gaussian sum filter)
        """    
        if self.nEnKF > 1:
            if self.enkf_method in [3,5,7,10]:                  # Local da method (LETKF, LUTKF, SLUTKF, LPF)
                xmean = (xmean_mf * self.alpha_mf).sum(axis=0)
                xptb = xmean_mf - xmean
                xcov = ((xcov_mf + xptb**2) * self.alpha_mf).sum(axis=0)                        
            else:
                xmean = (xmean_mf.T * self.alpha_mf).sum(axis=1)
                xptb = xmean_mf - xmean
                xcov = ((xcov_mf + xptb**2).T * self.alpha_mf).sum(axis=1)
        else:
            xmean = xmean_mf[0].copy()
            xptb = xptb_mf[0].copy()           
            xcov = xcov_mf[0].copy()

        xerr = xmean - x_truth                                  # background or analysis - truth data
        xmsd = (xerr**2).mean()                                 # background or analysis msd against truth data (zonally)

        return xmean, xptb, xcov, xerr, xmsd 


    def da(self, x, xmean, xptb, obs):
        """
        Data assimilation using ensemble Kalman filtering
        """
        if self.enkf_method == 3:  # letkf
            return letkf(xmean, xptb, self.h, self.h_type, obs, self.obs_r, self.obs_rloc, self.mpi, self.comm, self.myrank)
        elif self.enkf_method == 17: # original getkf with modulated ensemble
            return getkf_modens(xmean, xptb, self.h, self.h_type, obs, self.obs_r, self.z)
        elif self.enkf_method == 18: # MLETKF
            return mletkf(xmean, xptb, self.h, self.h_type, obs, self.obs_r, self.locmtx, self.obs_rloc, self.z, self.mpi, self.comm, self.myrank, self.obs_dist, self.locmtx_pert)
        else:
            raise ValueError('illegal value for enkf method flag')


    def fcst_and_da(self):
        """
        Forecast and data assimilation
        """
        self.t_acc = []
        self.xmean_b_acc = []
        self.xmean_a_acc = []
        self.x_truth_acc = []
        self.x_truth_zmean = []
        self.obs_zmean = []
        self.zrsd_acc = []
        self.xmean_b_zmean = []
        self.xmsd_b_acc = []
        self.xerr_b_acc = []
        self.xcov_b_zmean = []
        self.xmean_a_zmean = []
        self.xmsd_a_acc = []
        self.xerr_a_acc = []
        self.xcov_a_zmean = []
        self.diverged = False
        self.xcov_b_acc = np.zeros(self.N, float)
        self.zcov_acc = np.zeros(self.N, float)
        self.xxcov_b_acc = np.zeros(self.N, float)
        self.xzcov_acc = np.zeros(self.N, float)

        # 1. spin up ensemble
        for its in range(self.nts_spinup):
            for iEnKF in range(self.nEnKF):
                self.ensemble[iEnKF].fcst()                                                             # forecast
            
        # 2. forecast and data assimilation
        nstep = int(self.dt_da/self.dt)                                                                 # time steps in assimilation interval

        if self.enkf_method in [3,5,7,10]:                                                              # Local da method (LETKF, LUTKF, SLUTKF, LPF)
            self.alpha_mf = np.zeros((self.nEnKF, self.N), float)
            self.beta_mf = np.zeros((self.nEnKF, self.N), float)
        else:
            self.alpha_mf = np.zeros(self.nEnKF, float)
            self.beta_mf = np.zeros(self.nEnKF, float)

        self.alpha_mf[:] = 1./float(self.nEnKF)
        
        xmean_b_mf = np.zeros((self.nEnKF, self.N), float)
        xptb_b_mf = np.zeros((self.nEnKF, self.nem, self.N), float)
        xcov_b_mf = np.zeros((self.nEnKF, self.N), float)                                            # assume that xcov_b_mf is diagonal matrix
        xmean_a_mf = np.zeros((self.nEnKF, self.N), float)
        xptb_a_mf = np.zeros((self.nEnKF, self.nem, self.N), float)
        xcov_a_mf = np.zeros((self.nEnKF, self.N), float)                                            # assume that xcov_a_mf is diagonal matrix
        zrsd_mf = np.zeros((self.nEnKF, self.nobs), float)
        
        for its in range(0,self.nts,nstep):
            try:
                if self.myrank == 0:
                    if not self.loop_test:
                        print("nts_spinup = {}, nts = {}, its = {}".format(self.nts_spinup, self.nts, its))
                          
                #------------------------------------------------------------------------------------------------------------------- 
                # 1) data assimilation (background/analysis mean and covaricane)
                #-------------------------------------------------------------------------------------------------------------------
                for iEnKF in range(self.nEnKF):
                    xmean_b_mf[iEnKF] = self.ensemble[iEnKF].x.mean(axis=0)                   
                    xptb_b_mf[iEnKF] = self.ensemble[iEnKF].x - xmean_b_mf[iEnKF]                                                      # xptb_b: nems x self.N                                            

                    if self.enkf_method in [4,5,6,7]:                                                                                  # UKF da method (UTKF, LUTKF, SUTKF, SLUTKF)
                        xmean_a_mf[iEnKF], xptb_a_mf[iEnKF], xcov_b_mf[iEnKF], xcov_a_mf[iEnKF], zrsd_mf[iEnKF], self.beta_mf[iEnKF] = \
                        self.da(self.ensemble[iEnKF].x.copy(), xmean_b_mf[iEnKF].copy(), xptb_b_mf[iEnKF].copy(), self.obs[its,:])
                    else:
                        xcov_b_mf[iEnKF] = (xptb_b_mf[iEnKF]**2).sum(axis=0)/float(self.nem-1)                                         # cov(x_i): background error covariance (i: variable index) (consider only the diagonal elements of covariance matrix)

                        if self.enkf_method in [2,3]:                                                                                  # EKF da method (ETKF, LETKF)
                            xmean_a_mf[iEnKF], xptb_a_mf[iEnKF], zrsd_mf[iEnKF], self.beta_mf[iEnKF] = \
                            self.da(self.ensemble[iEnKF].x.copy(), xmean_b_mf[iEnKF].copy(), xptb_b_mf[iEnKF].copy(), self.obs[its,:])                    
                        else:
                            xmean_a_mf[iEnKF], xptb_a_mf[iEnKF], self.beta_mf[iEnKF] = \
                            self.da(self.ensemble[iEnKF].x.copy(), xmean_b_mf[iEnKF].copy(), xptb_b_mf[iEnKF].copy(), self.obs[its,:])

                        xcov_a_mf[iEnKF] = (xptb_a_mf[iEnKF]**2).sum(axis=0)/float(self.nem-1)                                         # cov(x_i): analysis error covariance (i: variable index)

                #------------------------------------------------------------------------------------------------------------------- 
                #  - save data for background verification
                #------------------------------------------------------------------------------------------------------------------- 
                # get statistics for multiple filters (Gaussina sum filter)
                xmean_b, xptb_b, xcov_b, xerr_b, xmsd_b = self.get_mean_ptb_cov_err_msd_for_gsf(xmean_b_mf, xptb_b_mf, xcov_b_mf, self.x_truth[its])

                # check filter divergence through background msd against truth data
                if np.isnan(xmsd_b):                                                            
                    self.diverged = True
                    break
       
                # save data for performance verification                            
                if (not self.verf_da_steady_state) or (self.verf_da_steady_state and (its >= self.nts_da_steady_state)):
                    # 1) residual distribution
                    for iEnKF in range(self.nEnKF):
                        if self.enkf_method in [2,3,4,5,6,7]:                                                                          # EKF da method (ETKF, LETKF) or UKF da method (UTKF, LUTKF, SUTKF, SLUTKF)
                            zrsd = zrsd_mf[iEnKF].copy()                        
                        else:
                            z = np.zeros((self.nem, self.nobs), float)

                            for iem in range(self.nem):
                                z[iem] = np.dot(self.h, self.ensemble[iEnKF].x[iem])
                                z[iem] = nonlinear_h(z[iem], self.h_type)

                            zmean = z.mean(axis=0)    
                            zrsd = self.obs[its,:] - zmean
                            
                        self.zrsd_acc.append(zrsd)    

                    # 2) correlation (x vs x, x vs hx)
                    xcov_b_tmp = (xptb_b**2).sum(axis=0)/float(self.nem-1)
                    xxcov_b = (xptb_b.T*xptb_b[:,int(self.N/2)]).sum(axis=1)/float(self.nem-1)              # cov(x_i,x_N/2): for corr(x_i, x_N/2) correlation between x_i and x_N/2 (N: number of variables)
                                                                                                            #            |  .  |     |     |
                                                                                                            # cov matrix |  .  | or  |. . .|
                                                                                                            #            |  .  |     |     |
                    #zptb = np.dot(xptb_b, self.h)                                                          # originally, zptb.T = np.dot(self.h,xptb.T), where self.h is symmetic matrix 
                    #zcov = (zptb**2).sum(axis=0)/float(self.nem-1) + self.obs_r                            # cov(z_i): predicted observation error covariance (i: variable index)
                    #xzcov = (xptb_b.T*zptb[:,int(self.N/2)]).sum(axis=1)/float(self.nem-1)                 # cov(x_i,z_N/2): for corr(x_i, z_N/2) correlation between x_i and z_N/2 (N: number of variables)
                    self.xcov_b_acc +=  xcov_b_tmp
                    self.xxcov_b_acc += xxcov_b
                    #self.zcov_acc += zcov
                    #self.xzcov_acc += xzcov

                    self.xmean_b_acc.append(xmean_b)
                    self.x_truth_acc.append(self.x_truth[its])
                    self.xerr_b_acc.append(xerr_b)
                    
                    # 3) mean, error covariance, and msd (zonally)
                    self.x_truth_zmean.append(self.x_truth[its].mean())
                    self.obs_zmean.append(self.obs[its].mean())                
                    self.xmean_b_zmean.append(xmean_b.mean())
                    self.xcov_b_zmean.append(xcov_b.mean())                                                 # background error cov mean (zonally)
                    self.xmsd_b_acc.append(xmsd_b)                                                          # background msd against truth data (zonally)

                    # 4) rank histogram
                    for iEnKF in range(self.nEnKF):
                        calc_rank_histogram(self.x_truth[its], self.ensemble[iEnKF].x.copy(), self.rank_hist[iEnKF])
                        
                #------------------------------------------------------------------------------------------------------------------- 
                #  - save data for analysis verification
                #------------------------------------------------------------------------------------------------------------------- 
                # calculate weights for multiple filters
                if self.same_wgt_nEnKF:
                    self.beta_mf[:] = 1./float(self.nEnKF)
                    
                self.alpha_mf = (self.alpha_mf * self.beta_mf)/(self.alpha_mf * self.beta_mf).sum(axis=0)

                # get statistics for multiple filters (Gaussina sum filter)
                xmean_a, xptb_a, xcov_a, xerr_a, xmsd_a = self.get_mean_ptb_cov_err_msd_for_gsf(xmean_a_mf, xptb_a_mf, xcov_a_mf, self.x_truth[its])
                
                # check filter divergence through analysis msd against truth data
                if np.isnan(xmsd_a):                                                             
                    self.diverged = True
                    break
                    
                # save data for performance verification
                if (not self.verf_da_steady_state) or (self.verf_da_steady_state and (its >= self.nts_da_steady_state)):
                    # mean, error covariance, and msd
                    self.xmean_a_acc.append(xmean_a)
                    self.xmean_a_zmean.append(xmean_a.mean())
                    self.xcov_a_zmean.append(xcov_a.mean())                                                 # analysis error cov mean (zonally)
                    self.xerr_a_acc.append(xerr_a)
                    self.xmsd_a_acc.append(xmsd_a)                                                          # analysis msd against truth data (zonally)
                    #elf.t_acc.append((its+1)*self.dt_da)
                    self.t_acc.append(its)
                    
                #-------------------------------------------------------------------------------------------------------------------     
                # 2) posterior covariance inflation
                #------------------------------------------------------------------------------------------------------------------- 
                for iEnKF in range(self.nEnKF):               
                    #if self.enkf_method not in [4,5,6,7,8,9,10]:
                    if self.cov_infl_method >= 0:
                        if self.cov_infl_method == 0:                                                        # Hodyss and Campbell inflation        
                            inc = xmean_a_mf[iEnKF] - xmean_b_mf[iEnKF]
                            inf_fact = np.sqrt(self.cov_infl_par1 + (xcov_a_mf[iEnKF]/xcov_b_mf[iEnKF]**2)*((xcov_b_mf[iEnKF]/float(self.nem)) + self.cov_infl_par2*(2.*inc**2/float(self.nem-1))))
                        elif self.cov_infl_method == 1:                                                      # relaxation to prior spread (RTPS) inflation
                            xsprd_a = np.sqrt(xcov_a_mf[iEnKF])
                            xsprd_b = np.sqrt(xcov_b_mf[iEnKF])
                            inf_fact = 1. + self.cov_infl_par1*(xsprd_b-xsprd_a)/xsprd_a

                        xptb_a_mf[iEnKF] *= inf_fact

                    self.ensemble[iEnKF].x = xmean_a_mf[iEnKF] + xptb_a_mf[iEnKF]

                    if self.enkf_method in [8,9,10]:                                                        # posterior additive noise for PF, EPF and LPF
                        self.ensemble[iEnKF].x = self.ensemble[iEnKF].x + np.sqrt(self.proc_q*self.proc_q_covinfl_par)*self.rs_ens.standard_normal(size=(self.nem, self.N))
                        
                #-------------------------------------------------------------------------------------------------------------------      
                # 3) forecast
                #-------------------------------------------------------------------------------------------------------------------  
                for iEnKF in range(self.nEnKF):
                    for istep in range(nstep):
                        if self.enkf_method in [8,9,10]:                                                    # PF da method (PF, EPF, LPF)
                            self.ensemble[iEnKF].fcst(self.prior_additive_noise)
                        else:
                            self.ensemble[iEnKF].fcst()
            except:
                if self.loop_test:
                    self.diverged = True
                    continue
                else:
                    traceback.print_exc()
                    sys.exit()

    def show_perf(self, start_time):        
        """
        Performance evaluation
        """
        ncount = len(self.xmsd_b_acc)

        if self.diverged:
            #print("Filter divergence (filter enkf_method:{}, ncount:{}, obs_loc_dist:{}, cov_infl_par1:{}, cov_infl_par2:{}, obs_errstddev:{}, neig:{})".format( \
            #                          self.enkf_method, ncount, self.obs_loc_dist, self.cov_infl_par1, self.cov_infl_par2, self.obs_errstddev, self.neig))
            if self.myrank == 0:
                elapsed_time = time.time() - start_time
                #print("enkf_method: {} Time(s): {} dt: {} dt_da: {} nem: {} nobs: {} obs_loc_dist: {} divergence".format(self.enkf_method, elapsed_time, self.dt, self.dt_da, self.nem, self.nobs, self.obs_loc_dist))                                      
                print("enkf_method: {} Time(s): {} dt: {} dt_da: {} nem: {} cov_loc_dist: {} nobs: {} obs_pos_dst: {} h_type: {} obs_loc_dist: {} cov_infl_par1: {} truth: {} mean_b: {} stddev_b: {} rmse_b: {} mean_a: {} stddev_a: {} rmse_a: {} divergence".format( \
                                                    self.enkf_method, elapsed_time, self.dt, self.dt_da, self.nem, self.cov_loc_dist, self.nobs, self.obs_pos_dst, self.h_type, self.obs_loc_dist, self.cov_infl_par1, -999.99, -999.99, -999.99, -999.99, -999.99, -999.99, -999.99))
        else:
            if self.myrank == 0:
                #-------------------------------------------------------------------------------------------------------------------     
                # 1. execution time
                #-------------------------------------------------------------------------------------------------------------------              
                elapsed_time = time.time() - start_time
                #print("Elapased Time:{} s".format(elapsed_time))
                #print("Elapased Time:{}".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
                
                #-------------------------------------------------------------------------------------------------------------------     
                # 2. covariance localization verification
                #-------------------------------------------------------------------------------------------------------------------     
                # cov and corr for background - background_mean 
                xcov_b_mean = self.xcov_b_acc/float(ncount)                                                                         # cov(x_i)                                                              
                xstd_b_mean = np.sqrt(xcov_b_mean)                                                                                  # stddev(x_i) = sqrt(cov(x_i))                                              |         |
                xxcov_b_mean = self.xxcov_b_acc/float(ncount)                                                                       # cov(x_i,x_N/2)                                                            |         |
                xxcorr_b_mean = xxcov_b_mean/(xstd_b_mean*xstd_b_mean[int(self.N/2)])                                               # corr(x_i,x_N/2) = cov(x_i,x_N/2) / (stddev(x_i) * stddev(x_N/2)) --> x_N/2|.........|
                                                                                                                                    #                                                                           |         |
                                                                                                                                    #                                                                           |         | 
                                                                                                                                    #                                                                               x_i
                f_intp = interpolate.interp1d(np.arange(self.N), xxcorr_b_mean, kind='cubic')                                       # interpolation for xxcorr_b_mean
                x_intp = np.arange(0., self.N-1, 0.1)
                y_intp = f_intp(x_intp)
                #np.savetxt('xxcorr_b_mean_enkf_method_{}_nEnKF_{}_nem_{}_nobs_{}_obs_loc_dist_{}_h_type_{}_x'.format(self.enkf_method, self.nEnKF, self.nem, self.nobs, self.obs_loc_dist, self.h_type), np.arange(self.N))
                #np.savetxt('[xxcorr_b_mean_x20]enkf_method_{}_nEnKF_{}_nem_{}_nobs_{}_obs_loc_dist_{}_h_type_{}_y'.format(self.enkf_method, self.nEnKF, self.nem, self.nobs, self.obs_loc_dist, self.h_type), xxcorr_b_mean)
                #np.savetxt('[obs_rloc_x20]enkf_method_{}_nEnKF_{}_nem_{}_nobs_{}_obs_loc_dist_{}_h_type_{}_y'.format(self.enkf_method, self.nEnKF, self.nem, self.nobs, self.obs_loc_dist, self.h_type), self.obs_rloc[:,int(self.N/2)])
                #np.savetxt('xxcorr_b_mean_enkf_method_{}_nEnKF_{}_nem_{}_nobs_{}_obs_loc_dist_{}_h_type_{}_x_cubic_interpolation'.format(self.enkf_method, self.nEnKF, self.nem, self.nobs, self.obs_loc_dist, self.h_type), x_intp)
                #np.savetxt('xxcorr_b_mean_enkf_method_{}_nEnKF_{}_nem_{}_nobs_{}_obs_loc_dist_{}_h_type_{}_y_cubic_interpolation'.format(self.enkf_method, self.nEnKF, self.nem, self.nobs, self.obs_loc_dist, self.h_type), y_intp)
                    
                # cov and corr for predicted observation error                                                                                                                          
                #zcov_mean = self.zcov_acc/float(ncount)                                                                            # cov(z_i)                                                                      
                #zstd_mean = np.sqrt(zcov_mean)                                                                                     # stddev(z_i) = sqrt(cov(z_i))
                #xzcov_mean = self.xzcov_acc/float(ncount)                                                                          # cov(x_i,z_N/2)
                #xzcorr_mean = xzcov_mean/(xstd_b_mean*zstd_mean[int(self.N/2)])                                                    # corr(x_i, z_N/2) = cov(x_i,z_N/2) / (stddev(x_i) * stddev(z_N/2))

                # cov and corr for background_mean - truth
                self.t_acc = np.array(self.t_acc)
                self.xerr_b_acc = np.array(self.xerr_b_acc)
                xcov_b = (self.xerr_b_acc**2).sum(axis=0)/float(self.xerr_b_acc.shape[0]-1)                                         # cov(x_i)
                xstd_b = np.sqrt(xcov_b)                                                                                            # stddev(x_i) = sqrt(cov(x_i))    
                xxcov_b = (self.xerr_b_acc.T*self.xerr_b_acc[:,int(self.N/2)]).sum(axis=1)/float(self.xerr_b_acc.shape[0]-1)        # cov(x_i,x_N/2)
                xxcorr_b = xxcov_b/(xstd_b*xstd_b[int(self.N/2)])                                                                   # corr(x_i,x_N/2) = cov(x_i,x_N/2) / (stddev(x_i) * stddev(x_N/2))

                self.xerr_a_acc = np.array(self.xerr_a_acc)

                # correlation between mean and truth 
                self.xmean_b_acc = np.array(self.xmean_b_acc)
                xmean_b_mean = self.xmean_b_acc.mean(axis=0)
                xmean_b_err = self.xmean_b_acc - xmean_b_mean
                xmean_b_cov = (xmean_b_err**2).sum(axis=0)/float(self.xmean_b_acc.shape[0]-1) 
                xmean_b_std = np.sqrt(xmean_b_cov)

                self.xmean_a_acc = np.array(self.xmean_a_acc)
                xmean_a_mean = self.xmean_a_acc.mean(axis=0)
                xmean_a_err = self.xmean_a_acc - xmean_a_mean
                xmean_a_cov = (xmean_a_err**2).sum(axis=0)/float(self.xmean_a_acc.shape[0]-1) 
                xmean_a_std = np.sqrt(xmean_a_cov)

                self.xmean_a_inc_acc = self.xmean_a_acc - self.xmean_b_acc          # analysis increment 
                
                self.x_truth_acc = np.array(self.x_truth_acc)
                x_truth_mean = self.x_truth_acc.mean(axis=0)
                x_truth_err = self.x_truth_acc - x_truth_mean
                x_truth_cov = (x_truth_err**2).sum(axis=0)/float(self.x_truth_acc.shape[0]-1) 
                x_truth_std = np.sqrt(x_truth_cov)
                
                xtcov_b = (xmean_b_err*x_truth_err).sum(axis=0)/float(self.x_truth_acc.shape[0]-1)
                self.xtcorr_b = xtcov_b/(xmean_b_std*x_truth_std)

                xtcov_a = (xmean_a_err*x_truth_err).sum(axis=0)/float(self.x_truth_acc.shape[0]-1)
                self.xtcorr_a = xtcov_a/(xmean_a_std*x_truth_std) 
                
                if self.plot_show or self.plot_save or self.plot_data_save:
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 1) plot: hovmuller diagram (time-longitude snapshot) of L96 model
                    #---------------------------------------------------------------------------------------------------------------------------
                    snplt = 2000; enplt = 3001
                    colorbar_extend = 'neither'                # 'neither', 'both', 'min', 'max'

                    # 1. Background mean
                    title = 'Background ensemble mean'
                    fn = '1_1_background_ensemble_mean_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs,self.obs_loc_dist)
                    plot_hovmuller_diagram(np.arange(self.N)+1, self.t_acc[snplt:enplt], self.xmean_b_acc[snplt:enplt], np.linspace(-18,18,41), colorbar_extend,\
                                           self.plt, title, fn, self.plot_outdir, self.plot_save, self.plot_show, self.plot_data_outdir, self.plot_data_save)
                                           
                    # 2. Analysis mean
                    title = 'Analysis ensemble mean'
                    fn = '1_2_analysis_ensemble_mean_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs,self.obs_loc_dist)
                    plot_hovmuller_diagram(np.arange(self.N)+1, self.t_acc[snplt:enplt], self.xmean_a_acc[snplt:enplt], np.linspace(-18,18,41), colorbar_extend,\
                                           self.plt, title, fn, self.plot_outdir, self.plot_save, self.plot_show, self.plot_data_outdir, self.plot_data_save)
                                           
                    # 3. Background error
                    title = 'Background error from the true state'
                    fn = '1_3_background_error_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs,self.obs_loc_dist)
                    plot_hovmuller_diagram(np.arange(self.N)+1, self.t_acc[snplt:enplt], self.xerr_b_acc[snplt:enplt], np.linspace(-18,18,41), colorbar_extend,\
                                           self.plt, title, fn, self.plot_outdir, self.plot_save, self.plot_show, self.plot_data_outdir, self.plot_data_save)
                    # 4. Analysis error
                    title = 'Analysis error from the true state'
                    fn = '1_4_analysis_error_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs,self.obs_loc_dist)
                    plot_hovmuller_diagram(np.arange(self.N)+1, self.t_acc[snplt:enplt], self.xerr_a_acc[snplt:enplt], np.linspace(-18,18,41), colorbar_extend,\
                                           self.plt, title, fn, self.plot_outdir, self.plot_save, self.plot_show, self.plot_data_outdir, self.plot_data_save)
                   
                    # 5. Analysis increment
                    title = 'Analysis increment from the true state'
                    fn = '1_5_analysis_increment_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs,self.obs_loc_dist)
                    plot_hovmuller_diagram(np.arange(self.N)+1, self.t_acc[snplt:enplt], self.xmean_a_inc_acc[snplt:enplt], np.linspace(-4.0,4.0,41), colorbar_extend,\
                                           self.plt, title, fn, self.plot_outdir, self.plot_save, self.plot_show, self.plot_data_outdir, self.plot_data_save)
                                           
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 2) plot: climatological covariance and correlation matrix for L96 model
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 1. climatological covariance
                    ncount = self.xmean_a_acc.shape[0]
                    err = self.xmean_a_acc - self.xmean_a_acc.mean(axis=0)
                    cov = np.dot(err.T,err)/float(ncount-1)
                    self.plt.figure()
                    self.plt.pcolormesh(np.arange(self.N)+1,np.arange(self.N)+1,cov,cmap=self.plt.cm.bwr,vmin=-15,vmax=15)
                    self.plt.title('climatological covariance matrix for L96 model')
                    self.plt.xlabel('$\mathrm{x}$')
                    self.plt.ylabel('$\mathrm{x}$')
                    self.plt.colorbar()
                    fn = '2_1_climatological_covariance_matrix_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs,self.obs_loc_dist)

                    if self.plot_save:
                        save_plot(self.plt,os.path.join(self.plot_outdir,fn),['png','svg','eps'])
                    if self.plot_show:     
                        self.plt.show()
                    if self.plot_data_save:
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_x_axis.txt'.format(fn)), np.arange(self.N)+1)
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_y_axis.txt'.format(fn)), np.arange(self.N)+1)
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_value.txt'.format(fn)), cov)   

                    # 2. climatological correlation    
                    err = self.xmean_a_acc - self.xmean_a_acc.mean(axis=0)
                    cov = np.dot(err.T,err)/float(ncount-1)
                    #std = np.array([np.sqrt((err**2).sum(axis=0)/float(ncount-1))])
                    std = np.array([np.sqrt(cov.diagonal())])
                    cor = cov/(std.T@std)
                    self.plt.figure()
                    self.plt.pcolormesh(np.arange(self.N)+1,np.arange(self.N)+1,cor,cmap=self.plt.cm.bwr,vmin=-1,vmax=1)
                    self.plt.title('climatological correlation matrix for L96 model')
                    self.plt.xlabel('$\mathrm{x}$')
                    self.plt.ylabel('$\mathrm{x}$')
                    self.plt.colorbar()
                    fn = '2_2_climatological_correlation_matrix_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs,self.obs_loc_dist)
                    if self.plot_save:
                        save_plot(self.plt,os.path.join(self.plot_outdir,fn),['png','svg','eps'])
                    if self.plot_show:     
                        self.plt.show()
                    if self.plot_data_save:
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_x_axis.txt'.format(fn)), np.arange(self.N)+1)
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_y_axis.txt'.format(fn)), np.arange(self.N)+1)
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_value.txt'.format(fn)), cor)
                        
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 3) plot: correlation (x vs x)
                    #---------------------------------------------------------------------------------------------------------------------------                
                    self.plt.figure()
                    self.plt.plot(np.arange(self.N),xxcorr_b_mean,color='r',label='Correlation (x vs x)')                           # color='C1'
                    self.plt.plot(x_intp,y_intp,color='g',label='Correlation (x vs x) cubic interpolation')
                    #self.plt.plot(np.arange(self.N),xzcorr_mean,color='r',label='Correlation (x vs Hx)')                    
                    #self.plt.plot(np.arange(self.N),xxcorr_b,color='g',label='Correlation (x vs x) using x_truth')
                    #self.plt.plot(np.arange(self.N),self.h[:,int(self.N/2)]/self.h.max(),color='g',label= 'Operator H')             # color='C2'
                    if self.obs_pos_dst == 0:
                        self.plt.plot(np.arange(self.N),self.obs_rloc[:,int(self.N/2)],'k:',label='Localization function for R')
                    self.plt.xlim(0,self.N)
                    self.plt.legend(loc='upper right')

                    fn = '3_correlation_enkf_method_{}_h_type_{}_nem_{}_nobs_{}_obs_loc_dist_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs,self.obs_loc_dist)
                    if self.plot_save:
                        save_plot(self.plt,os.path.join(self.plot_outdir,fn),['png','svg','eps'])
                    if self.plot_show:     
                        self.plt.show()
                    if self.plot_data_save:
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_x.txt'.format(fn)), np.arange(self.N))
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_y.txt'.format(fn)), xxcorr_b_mean)
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_y_intp.txt'.format(fn)), y_intp)
                        if self.obs_pos_dst == 0:
                            np.savetxt(os.path.join(self.plot_data_outdir,'{}_rloc.txt'.format(fn)), self.obs_rloc[:,int(self.N/2)])
                            
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 4) plot: covariance (x vs x)
                    #---------------------------------------------------------------------------------------------------------------------------                            
                    self.plt.figure()
                    self.plt.plot(np.arange(self.N),xxcov_b_mean,color='r',label='Cov (ens)')               # color='C1'
                    self.plt.plot(np.arange(self.N),xxcov_b,color='g',label='Cov (truth)')                   # color='C2'
                    self.plt.xlim(0,self.N)
                    self.plt.legend(loc='upper right')

                    fn = '4_covariance_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs)
                    if self.plot_save:
                        save_plot(self.plt, os.path.join(self.plot_outdir,fn), ['png','svg','eps'])
                    if self.plot_show:     
                        self.plt.show()
                    if self.plot_data_save:
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_x.txt'.format(fn)), np.arange(self.N))
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_xxcov_b_mean.txt'.format(fn)), xxcov_b_mean)
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_xxcov_b.txt'.format(fn)), xxcov_b)
                        
                #-------------------------------------------------------------------------------------------------------------------  
                # 3. Ensemble square-root filter performance verification
                #------------------------------------------------------------------------------------------------------------------- 
                self.x_truth_zmean = np.array(self.x_truth_zmean)
                self.obs_zmean = np.array(self.obs_zmean)
                self.xmean_b_zmean = np.array(self.xmean_b_zmean)
                self.xcov_b_zmean = np.array(self.xcov_b_zmean)
                self.xstd_b_zmean = np.sqrt(self.xcov_b_zmean.mean())                                                               # background standard deviation (spread) against background mean    
                self.xmsd_b_acc = np.array(self.xmsd_b_acc)                                       
                self.xrmsd_b = np.sqrt(self.xmsd_b_acc.mean())                                                                      # background rmsd against truth data
                self.xmean_a_zmean = np.array(self.xmean_a_zmean)
                self.xcov_a_zmean = np.array(self.xcov_a_zmean)
                self.xstd_a_zmean = np.sqrt(self.xcov_a_zmean.mean())                                                               # analysis standard deviation (spread) against analysis mean
                self.xmsd_a_acc = np.array(self.xmsd_a_acc)                                       
                self.xrmsd_a = np.sqrt(self.xmsd_a_acc.mean())                                                                      # analysis rmsd against truth data
                #print("Overall filter performance (filter enkf_method:{}, ncount:{}, obs_loc_dist:{}, cov_infl_par1:{}, cov_infl_par2:{}, obs_errstddev:{}, truth:{}, mean_b:{}, stddev_b:{}, rmse_b:{}, mean_a:{}, stddev_a:{}, rmse_a:{}, neig:{})".format( \
                #                                   self.enkf_method, ncount, self.obs_loc_dist, self.cov_infl_par1, self.cov_infl_par2, self.obs_errstddev, self.x_truth_zmean.mean(), self.xmean_b_zmean.mean(), self.xstd_b_zmean, self.xrmsd_b, self.xmean_a_zmean.mean(), self.xstd_a_zmean, self.xrmsd_a, self.neig))
                #print("Overall filter performance (filter enkf_method:{}, elapased Time:{} s, nobs:{}, obs_loc_dist:{}, truth:{}, mean_b:{}, stddev_b:{}, rmse_b:{}, mean_a:{}, stddev_a:{}, rmse_a:{})".format( \
                #                                   self.enkf_method, elapsed_time, self.nobs, self.obs_loc_dist, self.x_truth_zmean.mean(), self.xmean_b_zmean.mean(), self.xstd_b_zmean, self.xrmsd_b, self.xmean_a_zmean.mean(), self.xstd_a_zmean, self.xrmsd_a))
                print("enkf_method: {}, Time(s): {}, dt: {}, dt_da: {}, nem: {}, cov_loc_dist: {}, nobs: {}, obs_pos_dst: {}, h_type: {}, obs_loc_dist: {}, row_rank_thresh: {}, neig: {}, cov_infl_par1: {}, truth: {}, mean_b: {}, stddev_b: {}, rmse_b: {}, corr_b: {}, mean_a: {}, stddev_a: {}, rmse_a: {}, corr_a: {}".format( \
                                                    self.enkf_method, elapsed_time, self.dt, self.dt_da, self.nem, self.cov_loc_dist, self.nobs, self.obs_pos_dst, self.h_type, self.obs_loc_dist, self.row_rank_thresh, self.neig, self.cov_infl_par1, self.x_truth_zmean.mean(), self.xmean_b_zmean.mean(), self.xstd_b_zmean, self.xrmsd_b, self.xtcorr_b.mean(), self.xmean_a_zmean.mean(), self.xstd_a_zmean, self.xrmsd_a, self.xtcorr_a.mean()))

                # temporary save [start]
                """
                self.plot_data_outdir = '/home/kiaps/kjsung/test/python/L96/data/'
                fn = '5_data_over_time_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs)
                #np.savetxt(os.path.join(self.plot_data_outdir,'{}_x_truth.txt'.format(fn)), self.x_truth)
                np.savetxt(os.path.join(self.plot_data_outdir,'{}_xmean_b_zmean_da_steady_5000_obserrstd_{}_rev1[best].txt'.format(fn,self.obs_errstddev)), self.xmean_b_zmean)  
                np.savetxt(os.path.join(self.plot_data_outdir,'{}_xmean_a_zmean_da_steady_5000_obserrstd_{}_rev1[best].txt'.format(fn,self.obs_errstddev)), self.xmean_a_zmean)
                np.savetxt(os.path.join(self.plot_data_outdir,'{}_x_truth_zmean_da_steady_5000_obserrstd_{}_rev1[best].txt'.format(fn,self.obs_errstddev)), self.x_truth_zmean)
                np.savetxt(os.path.join(self.plot_data_outdir,'{}_obs_zmean_da_steady_5000_obserrstd_{}_rev1[best].txt'.format(fn,self.obs_errstddev)), self.obs_zmean)

                fn = '6_rmse_over_time_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs)
                np.savetxt(os.path.join(self.plot_data_outdir,'{}_xmsd_b_acc_da_steady_5000_obserrstd_{}_rev1[best].txt'.format(fn,self.obs_errstddev)), self.xmsd_b_acc)  
                np.savetxt(os.path.join(self.plot_data_outdir,'{}_xmsd_a_acc_da_steady_5000_obserrstd_{}_rev1[best].txt'.format(fn,self.obs_errstddev)), self.xmsd_a_acc)
                """
                # temporary save [end]
                
                if self.plot_show or self.plot_save or self.plot_data_save:
                    import matplotlib
                    
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 1) plot: gues mean, anal mean, true state, and observation over time
                    #---------------------------------------------------------------------------------------------------------------------------                    
                    self.plt.figure(figsize=(12,4))
                    #self.plt.subplot(211)
                    self.plt.plot(self.xmean_b_zmean, color='r', label='Background mean')                                           # color='C1'
                    self.plt.plot(self.xmean_a_zmean, color='g', label='Analysis mean')                                             # color='C2'
                    self.plt.plot(self.x_truth_zmean, color='k', linestyle='--', label='True state')                                # color='k'
                    self.plt.plot(self.obs_zmean, color='b', marker='+', linestyle='None', label='Observation')                     # color='b'
                    self.plt.xlabel('Time steps')
                    self.plt.ylabel('$x$')
                    self.plt.legend(loc='upper right')
                    #self.plt.xlim(0, self.obs_zmean.mean(axis=1).shape[0])
                    
                    fn = '5_data_over_time_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs)
                    if self.plot_save:
                        save_plot(self.plt, os.path.join(self.plot_outdir,fn), ['png','svg','eps'])
                    if self.plot_show: 
                        self.plt.show()
                    if self.plot_data_save:
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_xmean_b_zmean.txt'.format(fn)), self.xmean_b_zmean)  
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_xmean_a_zmean.txt'.format(fn)), self.xmean_a_zmean)
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_x_truth_zmean.txt'.format(fn)), self.x_truth_zmean)
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_obs_zmean.txt'.format(fn)), self.obs_zmean)
                        
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 2) plot: RMSE over time
                    #---------------------------------------------------------------------------------------------------------------------------                    
                    self.plt.figure(figsize=(12,4))
                    #self.plt.subplot(212)
                    self.plt.plot(np.sqrt(self.xmsd_b_acc), color='r', label='Background')                                          # color='C1'
                    self.plt.plot(np.sqrt(self.xmsd_a_acc), color='g', label='Analysis')                                            # color='C2'
                    self.plt.yscale('log')
                    self.plt.xlabel('Time steps')
                    self.plt.ylabel('RMSE ($x$)')
                    self.plt.legend(loc='upper right')
                    #self.plt.xlim(0, self.xmsd_b_acc.shape[0])

                    fn = '6_rmse_over_time_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs)
                    if self.plot_save:
                        save_plot(self.plt, os.path.join(self.plot_outdir,fn), ['png','svg','eps'])
                    if self.plot_show: 
                        self.plt.show()
                    if self.plot_data_save:
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_xmsd_b_acc.txt'.format(fn)), self.xmsd_b_acc)  
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}_xmsd_a_acc.txt'.format(fn)), self.xmsd_a_acc) 
                        
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 3) plot: residual distribution
                    #---------------------------------------------------------------------------------------------------------------------------                    
                    self.plt.figure()
                    self.zrsd_acc = np.array(self.zrsd_acc)
                    self.zrsd_acc = self.zrsd_acc.flatten()
                    self.zrsd_acc = sorted(self.zrsd_acc)
                    
                    pdf = stats.norm.pdf(self.zrsd_acc, np.mean(self.zrsd_acc), np.std(self.zrsd_acc))
                    self.plt.plot(self.zrsd_acc, pdf)
                    
                    if float(matplotlib.__version__[:3]) >= 2.0:
                        self.plt.hist(self.zrsd_acc, 30, density=True)                 # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.normal.html
                    else:
                        self.plt.hist(self.zrsd_acc, 30, normed=True)

                    (p_value, result) = ks_test_white_noise(self.zrsd_acc, alpha = 0.0001)  # Kolmogorov-Smirnov test
                    self.plt.title('KS test (p-value: {}, white nosie: {})'.format(p_value, result))                        
                    self.plt.xlabel('$\widetilde{z} = O-h(\overline{B})$')
                    self.plt.ylabel('PDF')
                    
                    fn = '7_residual_distribution_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs)
                    if self.plot_save:
                        save_plot(self.plt, os.path.join(self.plot_outdir,fn), ['png','svg','eps'])
                    if self.plot_show: 
                        self.plt.show()
                    if self.plot_data_save:
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}.txt'.format(fn)), self.zrsd_acc)
                    
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 4) plot: rank histogram
                    #---------------------------------------------------------------------------------------------------------------------------
                    for iEnKF in range(self.nEnKF):
                        #---------------------------------------------------------------------------------------------------------------------------
                        # 4.1) rank histogram for each state variable
                        #---------------------------------------------------------------------------------------------------------------------------
                        plot_rank_histogram_each_state(iEnKF, self.enkf_method, self.h_type, self.nem, self.nobs, self.rank_hist[iEnKF], \
                                                       self.plot_show, self.plot_outdir, self.plot_save, self.plot_data_outdir, self.plot_data_save)


                        #---------------------------------------------------------------------------------------------------------------------------
                        # 4.2) rank histogram for all state variables
                        #---------------------------------------------------------------------------------------------------------------------------
                        plot_rank_histogram_all_state(iEnKF, self.enkf_method, self.h_type, self.nem, self.nobs, self.rank_hist[iEnKF], \
                                                      self.plot_show, self.plot_outdir, self.plot_save, self.plot_data_outdir, self.plot_data_save)

                    #---------------------------------------------------------------------------------------------------------------------------
                    # 5) plot: observation histogram and operator
                    #---------------------------------------------------------------------------------------------------------------------------
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 5.1) histogram for observation locations
                    #---------------------------------------------------------------------------------------------------------------------------    
                    self.plt.figure()
                    self.obs_pos = sorted(self.obs_pos)
                    
                    if self.obs_pos_dst == 1:
                        pdf = stats.norm.pdf(self.obs_pos, np.mean(self.obs_pos), np.std(self.obs_pos))
                        self.plt.plot(self.obs_pos, pdf)
                        
                    if float(matplotlib.__version__[:3]) >= 2.0:    
                        self.plt.hist(self.obs_pos, 20, density=True)                  # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.normal.html
                    else:
                        self.plt.hist(self.obs_pos, 20, normed=True)

                    self.plt.title('Histogram for observation locations')
                    self.plt.xlabel('observation locations')
                    self.plt.ylabel('PDF')
                    self.plt.tight_layout()

                    fn = '9_1_obspos_histogram_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs)
                    if self.plot_save:
                        save_plot(self.plt, os.path.join(self.plot_outdir,fn), ['png','svg','eps'])
                    if self.plot_show:     
                        self.plt.show()            
                    if self.plot_data_save:
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}.txt'.format(fn)), self.obs_pos) 
                        
                    #---------------------------------------------------------------------------------------------------------------------------
                    # 5.2) observation operator H
                    #---------------------------------------------------------------------------------------------------------------------------   
                    self.plt.matshow(self.h)
                    #self.plt.matshow(self.obs_rloc)
                    
                    self.plt.title('Observation operator {}'.format(h_mathexp_str(self.h_type)))
                    self.plt.colorbar()
                    self.plt.tight_layout()

                    fn = '9_2_obs_operator_enkf_method_{}_h_type_{}_nem_{}_nobs_{}'.format(self.enkf_method,self.h_type,self.nem,self.nobs)
                    if self.plot_save:
                        save_plot(self.plt, os.path.join(self.plot_outdir,fn), ['png','svg','eps'])
                    if self.plot_show:     
                        self.plt.show()            
                    if self.plot_data_save:
                        np.savetxt(os.path.join(self.plot_data_outdir,'{}.txt'.format(fn)), self.h)                     

    # save background RMSE data for plot figure 3, 4, 5
    def save_rmse_b(obj, tag, outdir="./", save_txt=True):
        its = np.asarray(obj.t_acc, dtype=int)
        msd_b = np.asarray(obj.xmsd_b_acc, dtype=float)
        rmse_b = np.sqrt(msd_b)

        m = (its >= 3000) & (its <= 4000)
        its_sel = its[m]
        rmse_sel = rmse_b[m]

        base = f"{tag}_rmse_b"

        npz_path = os.path.join(outdir, base + ".npz")
        np.savez(npz_path, step=its_sel, rmse_b=rmse_sel)
        print(f"[Saved] {npz_path}")

        if save_txt:
            txt_path = os.path.join(outdir, base + ".txt")
            data = np.column_stack([its_sel, rmse_sel])
            np.savetxt(txt_path, data, fmt=["%d", "%.10e"], header="step rmse_b")
            print(f"[Saved] {txt_path}")

    # plot background RMSE for figure 3, 4, 5
    def plot_state_timeseries(obj, var_idx=0):
        ftname = "MLETKF-Z"
        save = True
        g_color = "tab:red"
        its = np.asarray(obj.t_acc)
        x_true = np.asarray(obj.x_truth_acc)   
        x_a    = np.asarray(obj.xmean_a_acc)   
        x_b    = np.asarray(obj.xmean_b_acc)  

        m = (its >= 3000) & (its <= 4000)
        t = its[m]

        plt.figure(figsize=(12, 2.5))
        plt.plot(t, x_true[m, var_idx], color="black", linewidth=1.5)
        plt.plot(t, x_b[m, var_idx], color=g_color, label=ftname, linestyle="--", linewidth=1.5)

        plt.xlabel("Time steps")
        plt.ylabel("x")
        plt.yticks([-5, 0, 5, 10])
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()

        if save:
            fname = f"state_var{var_idx}.png"
            plt.savefig(f"./{fname}", dpi=400, bbox_inches="tight")
            print(f"[Saved] ./{fname}")     

if __name__ == "__main__":
    # [GETKF] LETKF, GETKF (with modulated ensemble) 
    errstddev = 1.0                                 # observation error standard deviation
    cov_infl_method = -1           
    cov_infl_par1 = 0.2    
    
    enkf_method_list = [18]                         # 3: LETKF (with observation space horizontal covariance localization; R localization), 17: GETKF (with no localization), 18: GETKF (with modulated ensemble for model space vertical covariance localization; B localization), 19: GETKF (with modulated ensemble for model space vertical covariance localization (B localization) and R localization) 
    
    #nem_list = np.arange(2,10,1).tolist() + np.arange(10,110,10).tolist()
    nem_list = [50]

    cov_loc_dist = 3.0                                 # background covariance localization distance
    cov_loc_dist_pert = None                         # for Z-loc perturbation (Set to None if Z-loc is not used)

    #obs_loc_dist = np.arange(1,11,1).tolist()       # for LETKF and GETKF (with modulated ensemble)                             
    obs_loc_dist = [3.0]                             # observation localization distance

    #row_rank_thresh =np.arange(0.1,1.0,0.1).tolist()  # for GETKF (with modulated ensemble)
    row_rank_thresh = [0.4]

    mpi = True
    
    for h_type in [2]:                               # 0: linear, 1: nonlinear (absolute value), 2: nonlinear (natural log)
        for nobs in [100]:                           # number of observations
            for enkf_method in enkf_method_list:     
                for nem in nem_list:
                    for dist in obs_loc_dist:
                        for thresh in row_rank_thresh:
                            if enkf_method == 3:        # LETKF
                                object = L96enkf(nts=5000, enkf_method=enkf_method, nEnKF=1, dt_da=0.05, N=40, nem=nem, nem_sclfct=1, proc_errstddev=errstddev, obs_errstddev=errstddev, cov_loc_dist=cov_loc_dist, nobs=nobs, obs_pos_dst_seed=592, obs_pos_dst=1, h_type=h_type, obs_loc_dist=dist, row_rank_thresh=thresh, cov_infl_method=cov_infl_method, cov_infl_par1=cov_infl_par1, K_solver=1, plot_show=False, plot_save=False, plot_data_save=False, mpi=mpi, loop_test=True)
                            else:
                                object = L96enkf(nts=5000, enkf_method=enkf_method, nEnKF=1, dt_da=0.05, N=40, nem=nem, nem_sclfct=1, proc_errstddev=errstddev, obs_errstddev=errstddev, cov_loc_dist=cov_loc_dist, cov_loc_dist_pert = cov_loc_dist_pert, nobs=nobs, obs_pos_dst_seed=592, obs_pos_dst=1, h_type=h_type, obs_loc_dist=dist, row_rank_thresh=thresh, cov_infl_method=cov_infl_method, cov_infl_par1=cov_infl_par1, K_solver=1, plot_show=False, plot_save=False, plot_data_save=False, mpi=mpi, loop_test=True)
                            

                            start_time = time.time()
                            object.fcst_and_da()    
                            #save_rmse_b(object, tag="CASE3_MLETKF_Z_Ne50", outdir="./", save_txt=True)
                            #plot_state_timeseries(object, var_idx=1, use_time_in_steps=True, show_background=False)
                            object.show_perf(start_time)
