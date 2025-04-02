import numpy as np
import pandas as pd
from pytransit.utils import de
import matplotlib.pyplot as plt
import emcee
import batman
import astropy.constants as aconst
import radvel
from scipy.stats import binned_statistic
from .priors import PriorSet, UP, NP, JP, FP
from .likelihood import ll_normal_ev_py
from . import stats_help
from . import rvutils
from . import mcmc_help

def v_cov(rv):
    """
    Calculate the v_cov velocity coverage metric

    INPUT:
        rv: rv values

    OUTPUT:
        v_cov: coverage fraction of the rv values

    NOTES:
        See Equation 5 in Fernandez et al. 2017

    EXAMPLE:
        v_cov(df_bin.rv1), v_cov(df_bin.rv2)
    """
    rv = np.array(rv)
    rv = sorted(rv)
    rv = rv - np.mean(rv)
    N = len(rv)
    rv_span = np.max(rv) - np.min(rv)
    rv_min = np.min(rv)
    #print(rv_span)
    diff = np.sum(np.diff(rv)**2)
    #print(diff)
    v_cov = (N/(N-1))*(1 - diff/(rv_span**2) )
    #sum = 0
    #for i in range(len(rv)-1):
    #    #print(i)
    #    sum = sum + ((rv[i+1] -  rv_min)/rv_span - (rv[i] -  rv_min)/rv_span )**2
    #v_cov = (N/(N-1))*(1 - sum)
    return v_cov



class LPFunction2(object):
    """
    Log-Likelihood function class for 2 RV data streams
       
    NOTES:
        Based on hpprvi's class, see: https://github.com/hpparvi/exo_tutorials
    """
    def __init__(self,inp,file_priors):
        """
        INPUT:
            x - time values in BJD
            y - y values in m/s
            yerr - yerr values in m/s
            file_priors - prior file name
        """
        self.data = {"time"   : inp['time'],  
                     "rv1"    : inp['rv1'],   
                     "rv1_err": inp['rv1_err'],
                     "rv2"    : inp['rv2'],
                     "rv2_err": inp['rv2_err']}
        # Setting priors
        self.ps_all = priorset_from_file(file_priors) # all priors
        self.ps_fixed = PriorSet(np.array(self.ps_all.priors)[np.array(self.ps_all.fixed)]) # fixed priorset
        self.ps_vary  = PriorSet(np.array(self.ps_all.priors)[~np.array(self.ps_all.fixed)]) # varying priorset
        self.ps_fixed_dict = {key: val for key, val in zip(self.ps_fixed.labels,self.ps_fixed.args1)}
        print('Reading in priorfile from {}'.format(file_priors))
        print(self.ps_all.df)
        
    def get_jump_parameter_index(self,lab):
        """
        Get the index of a given label
        """
        return np.where(np.array(self.ps_vary.labels)==lab)[0][0]
    
    def get_jump_parameter_value(self,pv,lab):
        """
        Get the current value in the argument list 'pv' that has label 'lab'
        """
        # First check if we are actually varying it
        if lab in self.ps_vary.labels:
            return pv[self.get_jump_parameter_index(lab)]
        else:
            # We are not varying it
            return self.ps_fixed_dict[lab]

    def compute_rv_model(self,pv,times1=None):
        """
        Compute the RV model

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps 
        
        OUTPUT:
            rv - the rv model evaluated at 'times' if supplied, otherwise 
                      defaults to original data timestamps
        """
        if times1 is None: times1 = self.data["time"]
        tp      = self.get_jump_parameter_value(pv,'tp_p1')
        P       = self.get_jump_parameter_value(pv,'P_p1')
        gamma   = self.get_jump_parameter_value(pv,'gamma')
        K       = self.get_jump_parameter_value(pv,'K_p1')
        q       = self.get_jump_parameter_value(pv,'q')
        e       = self.get_jump_parameter_value(pv,'ecc_p1')
        w       = self.get_jump_parameter_value(pv,'omega_p1')
        rv1 = get_rv_curve_peri(times1,P=P,tp=tp,e=e,omega=w,K=K)
        rv2 = rv1*(-1/q)
        self.rv1 = rv1 + gamma
        self.rv2 = rv2 + gamma
        return self.rv1, self.rv2
        
    def compute_total_model(self,pv,times1=None):
        """
        Computes the full RM model (including RM and RV and CB)

        INPUT:
            pv    - a list of parameters (only parameters that are being varied)
            times - times (optional), array of timestamps 
        
        OUTPUT:
            rm - the rm model evaluated at 'times' if supplied, otherwise 
                      defaults to original data timestamps

        NOTES:
            see compute_rm_model(), compute_rv_model()
        """
        rv1, rv2 = self.compute_rv_model(pv,times1=times1) 
        return rv1, rv2
                    
    def __call__(self,pv):
        """
        Return the log likelihood

        INPUT:
            pv - the input list of varying parameters
        """
        if any(pv < self.ps_vary.pmins) or any(pv>self.ps_vary.pmaxs):
            return -np.inf

        ###############
        # Prepare data and model and error for ingestion into likelihood
        #y_data = self.data['y']
        y1, y2 = self.compute_total_model(pv)
        # jitter in quadrature
        jitter1 = self.get_jump_parameter_value(pv,'sigma_rv1')
        jitter2 = self.get_jump_parameter_value(pv,'sigma_rv2')
        error1 = np.sqrt(self.data['rv1_err']**2.+jitter1**2.)
        error2 = np.sqrt(self.data['rv2_err']**2.+jitter2**2.)
        ###############

        # Return the log-likelihood
        log_of_priors = self.ps_vary.c_log_prior(pv)
        # Calculate log likelihood
        #log_of_model  = ll_normal_ev_py(y_data, y_model, error)
        log_of_model1  = ll_normal_ev_py(self.data["rv1"], y1, error1)
        log_of_model2  = ll_normal_ev_py(self.data["rv2"], y2, error2)
        log_ln = log_of_priors + log_of_model1 + log_of_model2 
        return log_ln


class RVBinFit(object):
    """
    A class that does RV fitting of binaries

    NOTES:
        - Needs to have LPFunction defined
    """
    def __init__(self,LPFunction):
        self.lpf = LPFunction

    def minimize_AMOEBA(self):
        centers = np.array(self.lpf.ps_vary.centers)

        def neg_lpf(pv):
            return -1.*self.lpf(pv)
        self.min_pv = minimize(neg_lpf,centers,method='Nelder-Mead',tol=1e-9,
                                   options={'maxiter': 100000, 'maxfev': 10000, 'disp': True}).x

    def minimize_PyDE(self,npop=100,de_iter=200,mc_iter=1000,mcmc=True,threads=8,maximize=True,plot_priors=True,sample_ball=False,k=None,n=None):
        """
        Minimize using the PyDE optimizer

        """
        centers = np.array(self.lpf.ps_vary.centers)
        print("Running PyDE Optimizer")
        self.de = de.DiffEvol(self.lpf, self.lpf.ps_vary.bounds, npop, maximize=maximize) # we want to maximize the likelihood
        self.min_pv, self.min_pv_lnval = self.de.optimize(ngen=de_iter)
        print("Optimized using PyDE")
        print("Final parameters:")
        self.print_param_diagnostics(self.min_pv)
        #self.lpf.ps.plot_all(figsize=(6,4),pv=self.min_pv)
        print("LogPost value:",-1*self.min_pv_lnval)
        self.lnl_max  = -1*self.min_pv_lnval-self.lpf.ps_vary.c_log_prior(self.min_pv)
        print("LnL value:",self.lnl_max)
        print("Log priors",self.lpf.ps_vary.c_log_prior(self.min_pv))
        if k is not None and n is not None:
            print("BIC:",stats_help.bic_from_likelihood(self.lnl_max,k,n))
            print("AIC:",stats_help.aic(k,self.lnl_max))
        if mcmc:
            print("Running MCMC")
            self.sampler = emcee.EnsembleSampler(npop, self.lpf.ps_vary.ndim, self.lpf,threads=threads)

            #pb = ipywidgets.IntProgress(max=mc_iter/50)
            #display(pb)
            #val = 0
            print("MCMC iterations=",mc_iter)
            for i,c in enumerate(self.sampler.sample(self.de.population,iterations=mc_iter)):
                print(i,end=" ")
                #if i%50 == 0:
                    #val+=50.
                    #pb.value += 1
            print("Finished MCMC")
            self.min_pv_mcmc = self.get_mean_values_mcmc_posteriors().medvals.values

    def get_mean_values_mcmc_posteriors(self,flatchain=None):
        """
        Get the mean values from the posteriors

            flatchain - if not passed, then will default using the full flatchain (will likely include burnin)

        EXAMPLE:
        """
        if flatchain is None:
            flatchain = self.sampler.flatchain
            print('No flatchain passed, defaulting to using full chains')
        df_list = [rvutils.get_mean_values_for_posterior(flatchain[:,i],label,description) for i,label,description in zip(range(len(self.lpf.ps_vary.descriptions)),self.lpf.ps_vary.labels,self.lpf.ps_vary.descriptions)]
        return pd.concat(df_list)

    def print_param_diagnostics(self,pv):
        """
        A function to print nice parameter diagnostics.
        """
        self.df_diagnostics = pd.DataFrame(zip(self.lpf.ps_vary.labels,self.lpf.ps_vary.centers,self.lpf.ps_vary.bounds[:,0],self.lpf.ps_vary.bounds[:,1],pv,self.lpf.ps_vary.centers-pv),columns=["labels","centers","lower","upper","pv","center_dist"])
        print(self.df_diagnostics.to_string())
        return self.df_diagnostics

    def plot_fit(self,pv=None,times=None):
        """
        Plot the model curve for a given set of parameters pv

        INPUT:
            pv - an array containing a sample draw of the parameters defined in self.lpf.ps_vary
               - will default to best-fit parameters if none are supplied
        """
        if pv is None:
            print('Plotting curve with best-fit values')
            pv = self.min_pv
        x = self.lpf.data['x']
        y = self.lpf.data['y']
        jitter = self.lpf.get_jump_parameter_value(pv,'sigma_rv')
        yerr = np.sqrt(self.lpf.data['error']**2.+jitter**2.)
        model_obs = self.lpf.compute_total_model(pv)
        residuals = y-model_obs
            
        model = self.lpf.compute_total_model(pv)
        residuals = y-model

        # Plot
        nrows = 2
        self.fig, self.ax = plt.subplots(nrows=nrows,sharex=True,figsize=(10,6),gridspec_kw={'height_ratios': [5, 2]})
        self.ax[0].errorbar(x,y,yerr=yerr,elinewidth=1,lw=0,alpha=1,capsize=5,mew=1,marker="o",barsabove=True,markersize=8,label="Data")
        if times is not None:
            model = self.lpf.compute_total_model(pv,times=times)
            self.ax[0].plot(times,model,label="Model",color='crimson')
        else:
            self.ax[0].plot(x,model_obs,label="Model",color='crimson')
            
        self.ax[1].errorbar(x,residuals,yerr=yerr,elinewidth=1,lw=0,alpha=1,capsize=5,mew=1,marker="o",barsabove=True,markersize=8,
                            label="Residuals, std="+str(np.std(residuals)))
        for xx in self.ax:
            xx.minorticks_on()
            xx.legend(loc='lower left',fontsize=9)
            xx.set_ylabel("RV [m/s]",labelpad=2)
        self.ax[-1].set_xlabel("Time (BJD)",labelpad=2)
        self.ax[0].set_title("RM Effect")
        self.fig.subplots_adjust(wspace=0.05,hspace=0.05)

    def plot_mcmc_fit(self,times=None):
        df = self.get_mean_values_mcmc_posteriors()
        print('Plotting curve with best-fit mcmc values')
        self.plot_fit(pv=df.medvals.values)

def read_priors(priorname):
    """
    Read a prior file as in juliet.py style

    OUTPUT:
        priors - prior dictionary
        n_params - number of parameters

    EXAMPLE:
        P, numpriors = read_priors('../data/priors.dat')
    """
    fin = open(priorname)
    priors = {}
    n_transit = 0
    n_rv = 0
    n_params = 0
    numbering_transit = np.array([])
    numbering_rv = np.array([])
    while True:
        line = fin.readline()
        if line != '':
            if line[0] != '#':
                line = line.split('#')[0] # remove things after comment
                out = line.split()
                parameter,prior_name,vals = out
                parameter = parameter.split()[0]
                prior_name = prior_name.split()[0]
                vals = vals.split()[0]
                priors[parameter] = {}
                pvector = parameter.split('_')
                # Check if parameter/planet is from a transiting planet:
                if pvector[0] == 'r1' or pvector[0] == 'p':
                    pnumber = int(pvector[1][1:])
                    numbering_transit = np.append(numbering_transit,pnumber)
                    n_transit += 1
                # Check if parameter/planet is from a RV planet:
                if pvector[0] == 'K':
                    pnumber = int(pvector[1][1:])
                    numbering_rv = np.append(numbering_rv,pnumber)
                    n_rv += 1
                if prior_name.lower() == 'fixed':
                    priors[parameter]['type'] = prior_name.lower()
                    priors[parameter]['value'] = np.double(vals)
                    priors[parameter]['cvalue'] = np.double(vals)
                else:
                    n_params += 1
                    priors[parameter]['type'] = prior_name.lower()
                    if priors[parameter]['type'] != 'truncatednormal':
                        v1,v2 = vals.split(',')
                        priors[parameter]['value'] = [np.double(v1),np.double(v2)]
                    else:
                        v1,v2,v3,v4 = vals.split(',')
                        priors[parameter]['value'] = [np.double(v1),np.double(v2),np.double(v3),np.double(v4)]
                    priors[parameter]['cvalue'] = 0.
        else:
            break
    #return priors, n_transit, n_rv, numbering_transit.astype('int'), numbering_rv.astype('int'), n_params
    return priors, n_params

def priordict_to_priorset(priordict,verbose=True):
    """
    Get a PriorSet from prior diectionary

    EXAMPLE:
        P, numpriors = readpriors('../data/priors.dat')
        ps = priordict_to_priorset(priors)
        ps.df
    """
    priors = []
    for key in priordict.keys():
        inp = priordict[key]
        if verbose: print(key)
        val = inp['value']
        if inp['type'] == 'normal':
            outp = NP(val[0],val[1],key,key,priortype='model')
        elif inp['type'] == 'uniform':
            outp = UP(val[0],val[1],key,key,priortype='model')
        elif inp['type'] == 'fixed':
            outp = FP(val,key,key,priortype='model')
        else:
            print('Error, ptype {} not supported'.format(inp['type']))
        priors.append(outp)
    return PriorSet(priors)

def priorset_from_file(filename,verbose=False):
    """
    Get a PriorSet() from a filename
    """
    priordict, num_priors = read_priors(filename)
    return priordict_to_priorset(priordict,verbose)

def true_anomaly(time,T0,P,aRs,inc,ecc,omega):
    """
    Uses the batman function to get the true anomaly. Note that some 

    INPUT:
        time - in days
        T0 - in days
        P - in days
        aRs - in a/R*
        inc - in deg
        ecc - eccentricity
        omega - omega in deg

    OUTPUT:
        True anomaly in radians
    """
    # Some of the values here are just dummy values (limb dark etc.) to allow us to get the true anomaly
    params = batman.TransitParams()
    params.t0 = T0                           #time of inferior conjunction
    params.per = P                           #orbital period
    params.rp = 0.1                          #planet radius (in units of stellar radii)
    params.a = aRs                           #semi-major axis (in units of stellar radii)
    params.inc = inc                         #orbital inclination (in degrees)
    params.ecc = ecc                         #eccentricity
    params.w = omega                         #longitude of periastron (in degrees)
    params.u = [0.3,0.3]                     #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"           #limb darkening model
    m = batman.TransitModel(params, time)    #initializes model
    return m.get_true_anomaly()

def transit_time_from_ephem(tobs,P,T0,verbose=False):
    """
    Get difference between observed transit time and the 
    
    INPUT:
        tobs - observed transit time
        P - period in days
        T0 - ephem transit midpoint
        
    OUTPUT:
        T_expected in days
        
    EXAMPLE:
        P = 3.336649
        T0 = 2456340.72559
        x = [2456340.72559,2458819.8585888706,2458839.8782432866]
        y = np.copy(x)
        yerr = [0.00010,0.0002467413,0.0004495690]
        model = get_expected_transit_time_from_ephem(x,P,T0,verbose=True)

        fig, ax = plt.subplots(dpi=200)
        ax.errorbar(x,y,yerr,marker='o',lw=0,mew=0.5,capsize=4,elinewidth=0.5,)
    """
    n = np.round((tobs-T0)/P)
    if verbose:
        print('n={}'.format(n))
    T_expected = P*n+T0
    return T_expected


def get_rv_curve_peri(times_jd,P,tp,e,omega,K):
    """
    A function to calculate an RV curve as a function of time

    INPUT:
        times_jd - times in bjd
        P - period in days
        tp - T_periastron 
        e - eccentricity
        omega - omega in degrees
        K - semi-amplitude in km/s

    OUTPUT:
        RV curve in km/s
    """
    rvs = radvel.kepler.rv_drive(times_jd,[P,tp,e,np.deg2rad(omega),K])
    return rvs

def get_rv_curve(times_jd,P,tc,e,omega,K):
    """
    A function to calculate an RV curve as a function of time

    INPUT:
        times_jd - times in bjd
        P - period in days
        tc - transit center
        e - eccentricity
        omega - omega in degrees
        K - semi-amplitude in m/s

    OUTPUT:
        RV curve in m/s
    """
    t_peri = radvel.orbit.timetrans_to_timeperi(tc=tc,per=P,ecc=e,omega=np.deg2rad(omega))
    rvs = radvel.kepler.rv_drive(times_jd,[P,t_peri,e,np.deg2rad(omega),K])
    return rvs

def u1_u2_from_q1_q2(q1,q2):
    u1, u2 = 2.*np.sqrt(q1)*q2, np.sqrt(q1)*(1.-2*q2)
    return u1, u2

def b_from_aRs_and_i(aRs,i):
    return aRs*np.cos(np.deg2rad(i))


def calc_aRs_from_rprs_b_tdur_P(rprs,b,tdur,P):
    """

    INPUT:
        rprs
        b
        tdur - in days
        P - in days

    OUTPUT:
        a/R*

    See equation 8 in Seager et al. 2002
    """
    aa = (1+rprs)**2.
    bb = b*b*(1-np.sin(tdur*np.pi/P))
    cc = (np.sin(tdur*np.pi/P))**2.
    return np.sqrt((aa - bb)/cc)

def i_from_aRs_and_b(aRs,b):
    return np.rad2deg(np.arccos(b/aRs))#aRs*np.cos(np.deg2rad(i))


