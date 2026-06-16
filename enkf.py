import numpy as np
import sys
#from numpy.linalg import eigh # scipy.linalg.eigh broken on my mac
from scipy.linalg import eigh, cho_solve, cho_factor, svd, pinvh, solve_triangular, cholesky
import scipy.stats as stats

def ks_test_normal_dist(x, alpha = 0.05):
    """
    stats.kstest returns:

              1. KS statistic
              2. pvalue

    if pvalue > 0.05 (5%) we accept the null hypothesis:

    H0: our random variable from simulation follow the distribution with
        the parameters obtained from 'stats.dist.fit'. 
    """
    ks = stats.kstest(x, 'norm', stats.norm.fit(x))
    p_value = ks[1]
    result = 'accept' if p_value > alpha else 'reject'
    
    return (p_value, result)    
    
def ks_test_white_noise(x, alpha = 0.05):
    """ Kolmogorov-Smirnov test """
    ks = stats.kstest(x, 'norm', (0.0, np.std(x)))
    p_value = ks[1]
    result = 'accept' if p_value > alpha else 'reject'
    
    return (p_value, result)
    
def nonlinear_h(x, h_type):
    """ observation operator H """
    if h_type == 0:                 # linear h(x)=x
        return x
    elif h_type == 1:               # nonlinear h(x)=|x|
        return np.abs(x)
    elif h_type == 2:               # nonlinear h(x)=ln(|x|) 
        return np.log(np.abs(x))

def nonlinear_h_deriv(y, h_type, eps=1.e-8):
    if h_type == 0:
        return np.ones_like(y)
    elif h_type == 1:
        return np.sign(y)
    elif h_type == 2:
        y_safe = np.where(np.abs(y) < eps, eps*np.sign(y + eps), y)
        return 1.0 / y_safe

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    stats.random_state = np.random.RandomState(seed=20)
    return stats.truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
        
def symsqrt_psd(a, inv=False):
    """symmetric square-root of a symmetric positive definite matrix"""
    evals, eigs = eigh(a)
    symsqrt =  (eigs * np.sqrt(np.maximum(evals,0))).dot(eigs.T)
    if inv:
        inv =  (eigs * (1./np.maximum(evals,0))).dot(eigs.T)
        return symsqrt, inv
    else:
        return symsqrt
        
def linalg_solve(a,b,method):
    """determine x in linear equation ax=b"""
    if method == 1:
        return np.linalg.solve(a,b) 
    elif method == 2:
        f = cho_factor(a)
        return cho_solve(f,b)    

def symsqrtinv_psd(a):
    """inverse and inverse symmetric square-root of a symmetric positive definite matrix"""
    try:
        evals, evecs = eigh(a)
        symsqrtinv =  (evecs * (1./np.sqrt(np.maximum(evals,0)))).dot(evecs.T)
        inv =  (evecs * (1./np.maximum(evals,0))).dot(evecs.T)
    except:
        symsqrtinv = 0.
        inv = 0.
        
    return symsqrtinv, inv

def syminv_psd(a):
    """inverse of a symmetric positive definite matrix"""
    try:
        evals, evecs = eigh(a)
        inv =  (evecs * (1./np.maximum(evals,0))).dot(evecs.T)
    except:
        inv = 0.
        
    return inv

def letkf(xmean, xptb, h, h_type, obs, obs_r, obs_rloc, mpi, comm, myrank):
    """LETKF (with R-localization)"""

    nems, ndim = xptb.shape                                                     # xptb: nems x ndim
    nobs = obs.shape[-1]                                                        # obs: 1 x nobs(ndim)
    zptb = np.zeros((nems, nobs), xptb.dtype)

    x = xmean + xptb
    z = np.zeros((nems, nobs), x.dtype)
    
    for iens in range(nems):
        z[iens] = np.dot(h, x[iens])
        z[iens] = nonlinear_h(z[iens], h_type)
   
    zmean = z.mean(axis=0)
    zptb = z - zmean
    zrsd = obs - zmean    
    
    obs_rloc = np.where(obs_rloc < 1.e-13, 1.e-13, obs_rloc)                    # e.g., x = array([[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]), y = np.where(x < 5, x, -1) => y = array([[ 0.,1.,2.],[ 3.,4.,-1.],[-1.,-1.,-1.]]) 
    
    # calculate analysis mean and perturbation of each grid (each grid has one state variable) for local data assimilation (refer to ETKF for global data assimilation)
    if mpi:
        xptb_T = np.empty((ndim,nems), float)
        n = myrank                                                      
        if n < ndim:
            Rinv = np.diag(obs_rloc[n,:]/obs_r)
            C = np.dot(zptb, Rinv)                                                  # zptb: (Yb)^T
            sqrt_pa, pa = symsqrtinv_psd((nems-1)*np.eye(nems)+np.dot(C,zptb.T))    # symmetric matrix inverse and its square root using eigen decomposition
            K = np.dot(xptb[:,n].T, np.dot(pa,C))          
            Wa = np.sqrt(nems-1)*sqrt_pa
            xmean[n] = xmean[n] + np.dot(K, zrsd)
            xptb[:,n] = np.dot(Wa.T, xptb[:,n])       

            comm.Allgather(np.array(xmean[n]), xmean)
            comm.Allgather(np.array(xptb[:,n]), xptb_T)
            xptb = xptb_T.T
    else:
        for n in range(ndim):                                                       # sequential operation for each state (as in sequential KF)
            Rinv = np.diag(obs_rloc[n,:]/obs_r)
            C = np.dot(zptb, Rinv)                                                  # zptb: (Yb)^T
            sqrt_pa, pa = symsqrtinv_psd((nems-1)*np.eye(nems)+np.dot(C,zptb.T))    # symmetric matrix inverse and its square root using eigen decomposition
            K = np.dot(xptb[:,n].T, np.dot(pa,C))          
            Wa = np.sqrt(nems-1)*sqrt_pa
            xmean[n] = xmean[n] + np.dot(K, zrsd)
            xptb[:,n] = np.dot(Wa.T, xptb[:,n])
   
    beta_mf = np.empty(ndim, float)
    beta_mf[:] = 1.                                                                 # weight for Gaussian sum filter
    
    return xmean, xptb, zrsd, beta_mf

def getkf_modens(xmean, xptb, h, h_type, obs, obs_r, z):
    """GETKF with modulated ensemble (implicit B-localization)"""
    nems, ndim = xptb.shape
    nobs = obs.shape[-1]
    svd_calc = True
    
    if z is None:
        raise ValueError('z not specified')                                         # z = W^T
        
    # modulation ensemble
    neig = z.shape[0]                                                               # number of eigenvalues
    nems_modens = neig*nems
    iens_modens = 0

    xptb_modens = np.zeros((nems_modens,ndim),xptb.dtype)
    
    for j in range(neig):
        for iens in range(nems):
            xptb_modens[iens_modens,:] = xptb[iens,:]*z[neig-j-1,:]
            #xptb_modens[iens_modens,:] = xptb[iens,:]*z[j,:]                       # same as upper line            
            iens_modens += 1
            
    xptb_modens = np.sqrt(float(nems_modens-1)/float(nems-1))*xptb_modens
        
    # data assimilation
    if h_type == 0:                                                                 # linear h(x)=x (for simulation)
        zptb = np.empty((nems, nobs), xptb.dtype)
        zptb_modens = np.empty((nems_modens, nobs), xptb_modens.dtype)

        for iens in range(nems):
            zptb[iens] = np.dot(h,xptb[iens])
            zptb[iens] = nonlinear_h(zptb[iens], h_type)
            
        for iens in range(nems_modens):
            zptb_modens[iens] = np.dot(h,xptb_modens[iens])
            zptb_modens[iens] = nonlinear_h(zptb_modens[iens], h_type)

        zmean_modens = np.dot(h,xmean)
        zmean_modens = nonlinear_h(zmean_modens, h_type)
        zrsd_modens = obs - zmean_modens
    else:                                                                           # nonlinear h(x)=|x| or nonlinear h(x)=ln(|x|) (for operational mode)
        x = xmean + xptb
        z = np.empty((nems, nobs), xptb.dtype)
    
        for iens in range(nems):
            z[iens] = np.dot(h, x[iens])
            z[iens] = nonlinear_h(z[iens], h_type)
       
        zmean = z.mean(axis=0)
        zptb = z - zmean   
        
        x_modens = xmean + xptb_modens
        z_modens = np.empty((nems_modens, nobs), x_modens.dtype)

        for iens in range(nems_modens):
            z_modens[iens] = np.dot(h, x_modens[iens])
            z_modens[iens] = nonlinear_h(z_modens[iens], h_type)

        zmean_modens = z_modens.mean(axis=0)
        zptb_modens = z_modens - zmean_modens
        zrsd_modens = obs - zmean_modens 

    if svd_calc:                                                                    # singular value decomposition (for simulation)
        Rsqrt_inv = 1./np.sqrt(obs_r)                                               # R^(-1/2)
        YbRsqrtinv = zptb_modens * Rsqrt_inv                                        # (R^(-1/2)*H*Xp)^T = (Yp*H)^T*R^(-1/2)
        u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')         # u, v: orthogonal matrix, s: singular values
        sp = (nems_modens-1) + (s**2)
        painv =  (u*(1./sp)).dot(u.T)
        
        kfgain = np.dot(xptb_modens.T, np.dot(painv, YbRsqrtinv*Rsqrt_inv))         # kalman gain
        xmean = xmean + np.dot(kfgain, zrsd_modens)                                 # analysis mean
        
        reducedgain = np.dot(xptb_modens.T, u)*(1.-np.sqrt((nems_modens-1)/sp))     # reduced kalman gain (modified kalman gain)
        reducedgain = np.dot(reducedgain, (v.T/s).T)*Rsqrt_inv
    else:                                                                           # eigenvalue decompostion (for operational mode)
        Rinv = 1./obs_r                                                             # R^(-1)
        YbRinv = zptb_modens*Rinv                                                   # Yp^T*R^(-1)
        a = np.dot(YbRinv, zptb_modens.T)                                           # Yp^T*R^(-1)*Yp
        evals, evecs = np.linalg.eigh(a)                                            # evals: eigenvalue, evecs: eigenvector
        
        b = (nems_modens-1) + evals
        painv =  np.dot(evecs*(1./b), evecs.T)
        
        kfgain = np.dot(xptb_modens.T, np.dot(painv, YbRinv))                       # kalman gain
        xmean = xmean + np.dot(kfgain, zrsd_modens)                                 # analysis mean
        
        reducedgain = np.dot(xptb_modens.T, evecs)*(1.-np.sqrt((nems_modens-1)/b))*(1./evals)   # reduced kalman gain (modified kalman gain)
        reducedgain = np.dot(reducedgain, np.dot(evecs.T, YbRinv))
        
    xptb = xptb - np.dot(reducedgain, zptb.T).T                                     # analysis ensemble perturbation

    return xmean, xptb, 1.

def mletkf(xmean, xptb, h, h_type, obs, obs_r, locmtx, obs_rloc, z, mpi=False, comm=None, myrank=None, obs_dist=None, locmtx_pert=None):
    """mletkf (mean with B-localization & perturbation update with R-localization / Z-localization)"""
    nems, ndim = xptb.shape
    nobs = obs.shape[-1]
    svd_calc = True        # True: singular value decomposition, False: eigenvalue decomposition
    noLoc = False          # True: MLETKF-noLoc, False: MLETKF
    use_zloc = True       # True: use Z-localization for perturbation update, False: use R-localization for perturbation update

    if z is None:
        raise ValueError('z not specified')                                         # z = W^T

    # modulation ensemble
    neig = z.shape[0]                                                               # number of eigenvalues
    nems_modens = neig*nems
    iens_modens = 0

    xptb_modens = np.zeros((nems_modens,ndim),xptb.dtype)

    for j in range(neig):
        for iens in range(nems):
            xptb_modens[iens_modens,:] = xptb[iens,:]*z[neig-j-1,:]
            #xptb_modens[iens_modens,:] = xptb[iens,:]*z[j,:]                       # same as upper line
            iens_modens += 1

    xptb_modens = np.sqrt(float(nems_modens-1)/float(nems-1))*xptb_modens
    
    # data assimilation
    if h_type == 0:                                                                 # linear h(x)=x (for simulation)
        zptb = np.empty((nems, nobs), xptb.dtype)
        zptb_modens = np.empty((nems_modens, nobs), xptb_modens.dtype)

        for iens in range(nems):
            zptb[iens] = np.dot(h,xptb[iens])
            zptb[iens] = nonlinear_h(zptb[iens], h_type)

        for iens in range(nems_modens):
            zptb_modens[iens] = np.dot(h,xptb_modens[iens])
            zptb_modens[iens] = nonlinear_h(zptb_modens[iens], h_type)

        zmean_modens = np.dot(h,xmean)
        zmean_modens = nonlinear_h(zmean_modens, h_type)
        zrsd_modens = obs - zmean_modens
    else:                                                                           # nonlinear h(x)=|x| or nonlinear h(x)=ln(|x|) (for operational mode)
        x = xmean + xptb
        z = np.empty((nems, nobs), xptb.dtype)

        for iens in range(nems):
            z[iens] = np.dot(h, x[iens])
            z[iens] = nonlinear_h(z[iens], h_type)

        zmean = z.mean(axis=0)
        zptb = z - zmean

        x_modens = xmean + xptb_modens
        z_modens = np.empty((nems_modens, nobs), x_modens.dtype)

        for iens in range(nems_modens):
            z_modens[iens] = np.dot(h, x_modens[iens])
            z_modens[iens] = nonlinear_h(z_modens[iens], h_type)

        zmean_modens = z_modens.mean(axis=0)
        zptb_modens = z_modens - zmean_modens
        zrsd_modens = obs - zmean_modens

    if svd_calc:                                                                    # singular value decomposition (for simulation)
        Rsqrt_inv = 1./np.sqrt(obs_r)                                               # R^(-1/2)
        YbRsqrtinv = zptb_modens * Rsqrt_inv                                        # (R^(-1/2)*H*Xp)^T = (Yp*H)^T*R^(-1/2)
        u, s, v = svd(YbRsqrtinv,full_matrices=False,lapack_driver='gesvd')         # u, v: orthogonal matrix, s: singular values
        sp = (nems_modens-1) + (s**2)
        painv =  (u*(1./sp)).dot(u.T)

        kfgain = np.dot(xptb_modens.T, np.dot(painv, YbRsqrtinv*Rsqrt_inv))         # kalman gain
        mod_xmean = xmean + np.dot(kfgain, zrsd_modens)                                 # analysis mean
    
    else:                                                                           # eigenvalue decompostion (for operational mode)
        Rinv = 1./obs_r                                                             # R^(-1)
        YbRinv = zptb_modens*Rinv                                                   # Yp^T*R^(-1)
        a = np.dot(YbRinv, zptb_modens.T)                                           # Yp^T*R^(-1)*Yp
        evals, evecs = np.linalg.eigh(a)                                            # evals: eigenvalue, evecs: eigenvector

        b = (nems_modens-1) + evals
        painv =  np.dot(evecs*(1./b), evecs.T)

        kfgain = np.dot(xptb_modens.T, np.dot(painv, YbRinv))                       # kalman gain
        mod_xmean = xmean + np.dot(kfgain, zrsd_modens)                                 # analysis mean

        reducedgain = np.dot(xptb_modens.T, evecs)*(1.-np.sqrt((nems_modens-1)/b))*(1./evals)   # reduced kalman gain (modified kalman gain)
        reducedgain = np.dot(reducedgain, np.dot(evecs.T, YbRinv))

    # modulated ensemble mean
    xmean = mod_xmean

    if noLoc:
        # ETKF method
        Rinv = (1./obs_r)*np.eye(nobs)
        C = np.dot(zptb,Rinv)                              # (nem, nobs)
        sqrt_pa, pa = symsqrtinv_psd((nems-1)*np.eye(nems)+np.dot(C,zptb.T))
        Wa = np.sqrt(nems-1)*sqrt_pa
        xptb = np.dot(Wa.T,xptb)                         # (nem, ndim)

    else:
        if mpi:
            xptb_T = np.empty((ndim,nems), float)
            tmp_xptb = np.zeros(nems, dtype=xptb.dtype)
            n = myrank
            if n < ndim:
                rho = obs_rloc[n, :]
                if use_zloc:
                    rho = locmtx_pert[n, :]
                    xptb_star = xptb * np.sqrt(rho)[None, :]
                    x_star = xmean + xptb_star

                    Z_star = np.empty((nems, nobs), dtype=xptb.dtype)
                    for iens in range(nems):
                        y = h @ x_star[iens]
                        Z_star[iens] = nonlinear_h(y, h_type)

                    ymean_mod = h @ xmean               # use analysis mean from GETKF

                    zptb_star  = Z_star - ymean_mod

                    Rinv = 1.0 / obs_r
                    C = zptb_star * Rinv
                    A = (nems - 1) * np.eye(nems) + np.dot(C, zptb_star.T)
                    
                    sqrt_pa, _ = symsqrtinv_psd(A)
                    Wa = np.sqrt(nems - 1) * sqrt_pa 
                    tmp_xptb[:] = np.dot(Wa.T, xptb_star[:, n])
                else:
                    weight = obs_rloc[n,:]/obs_r
                    C = zptb * weight                                                  # zptb: (Yb)^T
                    sqrt_pa, pa = symsqrtinv_psd((nems-1)*np.eye(nems)+np.dot(C,zptb.T))
                    Wa = np.sqrt(nems-1)*sqrt_pa
                    tmp_xptb = np.dot(Wa.T, xptb[:,n])

            comm.Allgather(tmp_xptb, xptb_T)
            xptb = xptb_T.T
        else:
        # LETKF (implicit R-loc)
            for n in range(ndim):
                Rinv = np.diag(obs_rloc[n,:]/obs_r)
                C = np.dot(zptb, Rinv)
                sqrt_pa, pa = symsqrtinv_psd((nems-1)*np.eye(nems)+np.dot(C,zptb.T))
                Wa = np.sqrt(nems-1)*sqrt_pa
                xptb[:,n] = np.dot(Wa.T, xptb[:,n])

    return xmean, xptb, 1.
