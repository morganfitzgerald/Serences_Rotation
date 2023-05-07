import numpy as np
import SD_functions as SDF
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

# constant
pi=np.pi
r2d,d2r = 180/pi,pi/180

n_bns=91
bns = np.linspace(-90,90,n_bns)
overlap=3

n_step = 1000
s_0 = np.linspace(0,180,n_step)

norm_pdf = lambda x: x/np.mean(x)/2/pi
ori_wrap = lambda x: SDF.wrap(x*2)/2 # wraps -90,90

doVM_loss = SDF.rss_fun(SDF.Sd_vm) # fitting DoG

# def get_labs(n_gen):
#     if n_gen==3:
#         return ('Stim SD','Resp SD','No SD')
#     elif n_gen==4:
#         return ('Stim SD','Resp SD','StimResp SD','No SD')

def sawtooth(p,x):
    """
    sawtooth(p,x_deg):
    
    2 parameter sawtooth function.
    x0,y0=p
    
    x is expected to lie in range [-pi,pi]
    """
    x0,y0=p
    x = x/2
    if type(x0) is np.ndarray:
        y0[x0<0]*=-1
        x0[x0<0]*=-1
    else:
        if x0<0:
            x0,y0=x0*-1,y0*-1
#     assert x0>=0, 'flip values'
    m0 = y0/x0
    m1 = -2*y0/(pi/2-2*x0)
    b0 = m0*(pi/2-x0)+y0
    b1 = m1*x0-y0
    b3 = -b1
    b4 = -b0

    y_hat = (np.abs(x)<=x0)*(x*m0+0) +\
            ((x>x0)&(np.abs(x)<pi/2-x0))*(x*m1+b3) +\
            ((x<-x0)&(np.abs(x)<pi/2-x0))*(x*m1+b1) +\
            (x<=(-pi/2+x0))*(x*m0+b0) +\
            (x>=(pi/2-x0))*(x*m0+b4)

    return y_hat


def gen_seq_from_pdf(s_0,pdf,n_stim,vis=0):
    """
    gen_seq_from_pdf(s_0,pdf,n_stim,vis=0)
    
    Generate stimulus sequences using arbitary PDFs representing d_ori.
    s_0       | [0,180] samples of pdf
    pdf       | evaluated at values of s_0
    n_stim    | (scalar) number of stimuli to generate
    vis       | (flag)  (0), 1- visualize d_ori distribution of generated sequence
   
    """
    
    this_cdf = np.cumsum(pdf)/np.sum(pdf)
    rand_draws = np.random.random_sample(n_stim)
    this_d_ind = np.argmin(np.abs(np.expand_dims(this_cdf,1)-rand_draws),0)
    this_d = s_0[this_d_ind]-90 # want to be 
    stim_ac = np.mod(np.cumsum(this_d),180) # stim sequence (no starting value)
    if vis:
        plt.hist(this_d,s_0-90,density=0)
        plt.plot(s_0-90,ac_fun*n_trial/pi,'r')
        SDF.d_plot(0,0,-90)
    return stim_ac

def gauss(x,mu=0,sig=1): return 1/np.sqrt(2*pi*sig**2)*np.exp(-((x-mu)**2)/(2*sig**2)) # for convolution

def vonMises_ori(mu,kappa): 
    """
    used to generate noisy encoding
    scales vonMises to run with values expected [0,180]
    need to go to (-pi,pi)
    
    """

    mu_deg = mu/90*pi-pi
    samples = np.random.vonmises(mu_deg,kappa)
    samples_deg = (samples+pi)*90/pi
    return samples_deg

# likelihood function for encoding
def c(s_0,mu=0,kappa=5):
    return scipy.stats.vonmises(kappa,mu/90*pi).pdf(s_0*d2r*2)
# prior
def cf(s_0,mu=0,kappa=5,p_same=0.9): ## eq 6
    return p_same*c(s_0,mu,kappa) + (1-p_same)*(1/2/pi)

def cb_fun_kappa(stim,kappa=8):
    return kappa*(1+np.cos(stim*d2r*2)**2)

def cb_fun_bias(stim,amp=10): 
    return amp*(np.sin(stim*d2r*2*2))

def adapt_fun(d_ori,amp,width=1):
    return SDF.Sd_vm((-amp,width),d_ori)

def run_simulation(n_stim=20000,kappa=8,do_cb=1,do_oblique=1,
                   do_stim_autocorr=0,cb_bias_mag=10,p_same=0.8,
                   adapt_amp=0): 
    """
    INPUTS:
    n_stim     | int
    kappa      | scaler, used as baseline value (always higher by x1.5)
    do_cb      | int (flag), 0- no CB, 1- sine based CB, 2- sawtooth based CB
    do_oblique | int (flag), 0- no oblique bias, 1- include cosine modulation of kappa
    do_stim_autocorr | int (flag), 0- none, 1- pos autocorr, 2- neg autocorr
    
    bias_mag   | bias mag used by cb function
    adapt_amp  | (float, flag) used to introduce a repulsive bias from previous stim at encoding.

    
    
    return ((stim, response, E)
    stim       (n_stim)
    response,E (3, n_stim) corresponding to {stim, response, and no} serial dependence
    """
    
    kappa_prior = kappa*1.5  # uniform precision
    #### generate stimulus sequence ####
    if do_stim_autocorr==1:
        ac_fun = cf(s_0,90,1,.7)
        stim = gen_seq_from_pdf(s_0,ac_fun,n_stim,0)
    elif do_stim_autocorr==2:
        ac_fun = cf(s_0,0,1,.7)
        stim = gen_seq_from_pdf(s_0,ac_fun,n_stim,0)
    elif do_stim_autocorr==0: # default
        stim = np.linspace(0,180,n_stim)
        np.random.shuffle(stim)

    ####  kappa ####    
    if do_oblique:
        kappas = cb_fun_kappa(stim,kappa)
        stim_enc_cb = vonMises_ori(stim,kappas)
    else:
        kappas = np.ones(n_stim)*kappa_prior
        stim_enc_cb = vonMises_ori(stim,kappas)

    #### mu #####
    if do_cb==1:
        stim_enc_cb += cb_fun_bias(stim,cb_bias_mag)
    elif do_cb==2:
        cb_fun_sawtooth_bias = lambda x: sawtooth((5*d2r,cb_bias_mag),x)
        stim_enc_cb += cb_fun_sawtooth_bias(stim*d2r-pi)

    #### stim adaptation ####
    if adapt_amp:
        d_ori = SDF.get_nb(-1,stim,1,ori_wrap)*d2r*2
        this_adapt = adapt_fun(d_ori,adapt_amp)
        stim_enc_cb +=this_adapt

    stim_enc_cb = ori_wrap(stim_enc_cb-90)+90

    ### joint biasing ####
    inc_joint,n_resp_types = 0,3

    resp_all_cb = np.zeros((n_resp_types,n_stim)) # stim\resp bias
    resp_all_cb[-1] = stim_enc_cb # no bias condition

    #### Bayesian integration  ####
    for i in range(n_stim):

        # establish priors based on previous stim/response
        priors = [] # stim, resp, (both)
        priors.append(cf(s_0,stim[i-1],kappa=kappa_prior,p_same=p_same)) # just use default kappa for prior 
        priors.append(cf(s_0,resp_all_cb[1,i-1],kappa=kappa_prior,p_same=p_same))

        # get encoding likelihood (von Mises) given mu and kappa generated already
        this_encode = c(s_0,stim_enc_cb[i],kappa=kappas[i])

        # multiply likelihood and prior to get Bayesian posterior estimate
        posteriors = [norm_pdf(prior*this_encode) for prior in priors]

        # single trial estimates = Maximum a Posteriori
        for j in range(n_resp_types-1):
            resp_all_cb[j,i] = s_0[np.argmax(posteriors[j])] # could use circ_mean

    E_all = ori_wrap(resp_all_cb-stim) # assumes in range [0,180]

    return stim,resp_all_cb,E_all


def correct_cb(stim,resp,E,c_fun = 'fourier',n_param=6,mode='All',
               n_subj=0,n_trial=0,subjs=None):
    """
    Corrects cardinal or any other circular response bias. 
    
    INPUTS:
    stim   | [0, 180] (n_stim)
    resp   | [0, 180] (n_sim, n_stim)
    E      | [-90,90] (n_sim, n_stim)
    c_fun  | (str)
                fourier - many_sine_cos function. flexible # of parameters
                poly - fit polynomial
                sawtooth - 4 parameter sawtooth function
                
    n_param | (Scalar, int)
    mode    | (str) controls what is returned
                'All'       - return resp_corrected,E_corrected
                'resp'      - stim,resp_corrected, E_corrected
                'E'         - stim,resp,E_corrected
                'fit_fun'   - fit_mdl,sliding_bias
                
                
    n_subj  | (int, flag) [0] if included, number of subjects to run correction over
    n_trial | (int, flag) [0] if included, number of trials/subject to run correction over
    subjs   | (str, listlike) [None] if included, subject IDs to iterate correction over
    """

    if resp.ndim==1:
        resp = np.expand_dims(resp,0)
        E = np.expand_dims(E,0)
    E_rad = E*d2r # now [-pi,pi]
    stim_rad = stim*d2r*2-pi #[-pi,pi]
    if subjs is not None:
        u_subj = np.unique(subjs)
        n_subj = len(u_subj)
        n_trial = 1

    n_stim,n_stim_total = resp.shape
    resp_corrected = np.zeros_like(resp)

    if n_subj: # run per subject mode...
        assert n_trial, 'Need to set n_trial too.'
        # assert mode != 'fit_fun', 'Not setup'
        fit_mdl = []
        for si in range(n_subj):
            if subjs is not None:
                inds = subjs==u_subj[si]
            else:
                st_ind,end_ind = si*n_trial,(si+1)*n_trial  
                inds = (np.arange(n_stim_total)>=st_ind)&(np.arange(n_stim_total)<end_ind)
            
            _,c_resp,c_E,this_mdl = correct_cb(stim[inds],resp[:,inds],E[:,inds],
                c_fun=c_fun,n_param=n_param,mode='all',n_subj=0)
            resp_corrected[:,inds] = c_resp # correct resp for given subj...
            fit_mdl.append(this_mdl)

        # clip
        if subjs is None:
            stim = stim[:end_ind]
            resp = resp[:,:end_ind]
            resp_corrected = resp_corrected[:,:end_ind]
            E = E[:,:end_ind]
    else:
        
        rss_fit = 1
        if c_fun=='poly':
            assert 0, 'not setup right!'
            for i in range(n_stim):
                pfit = np.polyfit(stim,E[i],n_param) 
                fit_mdl = lambda stim: np.poly1d(pfit)(stim)
                rss_fit = 0

        elif c_fun=='fourier':
            this_fun  = SDF.many_sine_cos
            x0 = (0,)*n_param
        elif c_fun=='sine':
            this_fun = lambda a,x : a*np.sin(x*2)
            x0 = (0,)
            # assert n_param==1, 'invalid # of parameters. sine takes 1...'
        elif c_fun=='sawtooth':
            this_fun = sawtooth
            x0 = (5*d2r,0)
        else:
            assert 0, f'mode {c_fun} not set up.'

        if rss_fit :
            correction_fun = SDF.rss_fun(this_fun)
            for i in range(n_stim): # fit and correct seperately for each simulation
                this_fit = scipy.optimize.minimize(correction_fun,x0,(stim_rad,E_rad[i])) 
                fit_mdl = lambda stim: this_fun(this_fit.x,stim)*r2d
                mdl_pred = fit_mdl(stim_rad)
                resp_corrected[i] = ori_wrap(resp[i]-mdl_pred) # 
   
    E_corrected = ori_wrap(resp_corrected-stim)
    if mode=='all':
        return stim,resp_corrected, E_corrected, fit_mdl

    elif mode=='resp':
        return stim,resp_corrected, E_corrected
    elif mode=='E':
        return stim,resp,E_corrected
    elif mode=='fit_fun':
        sliding_bias = np.zeros((n_stim+1,n_bns)) # history indpendent, [E (stim, resp, none)], fit fun,,,
        sliding_bias[0] = SDF.do_bining(bns+90,overlap,stim,E[-1])
        for i in range(n_stim):
            sliding_bias[i+1] = SDF.do_bining(bns+90,overlap,stim,E_corrected[i])
        return fit_mdl,sliding_bias

def summarize_sim(stim,resp,E,nb_run = (-1,0),n_subj=30,n_trial = 360,get_vis=0,labs = None,
    fit_typ = 'DoVM',do_boot=0,subjs=None,subj_shuffle=0):
    """
    Returns parameterized estimates of bias from stim and resp/E. 

    stim   | [0, 180]
    resp   | [0, 180] 
    nb_run | list like, (-1)- influence of previous trial; (0)- shuffle; (1)- influence of future trial
    n_subj | for power analysis. If 1 just do one fit of all data...
    n_trial| per subject
    get_vis| return sliding avg bias 
    labs   | labels corresponding to rows of resp/E
    fit_typ| options: DoVM (default), DoG.
    do_boot| opt to bootstrap trials within or across participants
    subjs  | list like, subject IDs.
    subj_shuffle | do shuffling within subjects.
    """
    if resp.ndim==1:
        resp = np.expand_dims(resp,0)
    if E.ndim==1:
        E = np.expand_dims(E,0)
    
    n_stim_total = len(stim)
    n_gen = resp.shape[0]
    if labs is None:
        if n_gen==3:
            labs = ('Stim SD','Resp SD','No SD')
        elif n_gen==4:
            labs = ('Stim SD','Resp SD','StimResp SD','No SD')
    else:
        assert n_gen==len(labs),'invalid labs'

    if do_boot==0: assert n_stim_total >= (n_subj*n_trial), 'not enough trials'
        
    if subjs is not None:
        u_subj = np.unique(subjs)
#         n_subj = len(u_subj)
        # n_trial = 1 # not-necessary?

    if subj_shuffle:
         assert subjs is not None, 'cannot do this mode w/o subjs'
    n_nb = len(nb_run)
    sd_bias_all = np.zeros((n_nb,2,n_gen,n_subj,n_bns)) # (sort stim/resp), (gen stim/resp/neither)
    fits_all_struct = pd.DataFrame()
    
    # iterate over different nb
    for si in range(n_subj):
        subj=si
        if do_boot:
            inds = np.random.choice(n_stim_total,n_trial,0)
        else:
            if subjs is not None:
                subj = u_subj[si]
                inds = (subjs==subj)
            else:
                # if not boot and no subjs, just grab groups in order
                st_ind,end_ind = si*n_trial,(si+1)*n_trial
                if n_subj ==1: st_ind,end_ind = 0,len(stim)
                inds = (np.arange(n_stim_total)>=st_ind)&(np.arange(n_stim_total)<end_ind)
                
#         # hemmed in here, have to do by subject if term is included...
#         if subjs is not None:
#             subj = u_subj[si]
#             inds = (subjs==subj)
#         else:
#             subj=si
#             if do_boot:
#                 inds = np.random.choice(n_stim_total,n_trial,0)
#             else:
#                 st_ind,end_ind = si*n_trial,(si+1)*n_trial
#                 if n_subj ==1: st_ind,end_ind = 0,len(stim)
#                 inds = (np.arange(n_stim_total)>=st_ind)&(np.arange(n_stim_total)<end_ind)


        for nbi,nb in enumerate(nb_run): 
            if nb==0:
                if subj_shuffle: # if we want to shuffle within subjects...
                    new_order = np.zeros_like(stim).astype(int)
                    for subj in u_subj:
                        these_inds = np.where(subjs==subj)[0]
                        these_inds_shuf = np.random.choice(these_inds,len(these_inds),replace=False)
                        new_order[these_inds] = these_inds_shuf
#                         new_order.append(these_inds_shuf)
#                     new_order = np.concatenate(new_order)
                else:
                    new_order = np.random.choice(n_stim_total,n_stim_total)
                stim_use = stim[new_order]
                resp_use = resp[:,new_order]
                E_use = E[:,new_order]
                nb_use = -1
            else:
                stim_use=stim
                resp_use=resp
                E_use=E
                nb_use=nb

            d_ori = SDF.get_nb(nb_use,stim_use,1,ori_wrap)

            for i in range(n_gen): # stim bias/ resp bias/ none .. etc.
                d_resp = ori_wrap(SDF.get_nb(nb_use,resp_use[i],0)-stim_use)
            
                if get_vis:
                    sd_bias_all[nbi,0,i,si] = SDF.do_bining(bns,overlap,d_ori[inds],
                                                        E_use[i,inds]*d2r*2,'circ_mean')/d2r/2
                    sd_bias_all[nbi,1,i,si] = SDF.do_bining(bns,overlap,d_resp[inds],
                                                        E_use[i,inds]*d2r*2,'circ_mean')/d2r/2

                fit_ori = fit_history_fun(d_ori[inds]*d2r*2,E_use[i,inds],want_full=1,typ=fit_typ)
                fit_resp = fit_history_fun(d_resp[inds]*d2r*2,E_use[i,inds],want_full=1,typ=fit_typ)

                fit_joint = fit_history_fun(d_ori[inds]*d2r*2,E_use[i,inds],want_full=1,typ=fit_typ,
                    x2=d_resp[inds]*d2r*2)

                dicts = []
                dicts.append({'subj':subj,'nb':nb,'trueSD':labs[i],'dStim':'d_ori','amp':fit_ori.x[0]
                              ,'width':fit_ori.x[1],'success':fit_ori.success})
                dicts.append({'subj':subj,'nb':nb,'trueSD':labs[i],'dStim':'d_resp','amp':fit_resp.x[0],
                              'width':fit_resp.x[1],'success':fit_resp.success})
                dicts.append({'subj':subj,'nb':nb,'trueSD':labs[i],'dStim':'joint_ori','amp':fit_joint.x[0],
                              'width':fit_joint.x[1],
                              'success':fit_resp.success})
                dicts.append({'subj':subj,'nb':nb,'trueSD':labs[i],'dStim':'joint_resp','amp':fit_joint.x[2],
                              'width':fit_joint.x[3],
                              'success':fit_resp.success})                 
                for s in dicts:
                    fits_all_struct = fits_all_struct.append(s,ignore_index=1)

    if get_vis:
        return fits_all_struct, sd_bias_all
    else:
        return fits_all_struct

def fit_history_fun(x,y,typ='DoVM',want_full=0,x2=None):
    """
    Parameterize history bias. 

    x       | (-pi,pi)
    y       | (-90,90)
    type    | options: DoVM (default), DoG.
    want_full | (flag) 0 (default) - return fit coefficients, 1 - return full fitting object
    x2      | (-pi,pi) additional vector for "joint" fitting

    """
    jointFit=0
    if x2 is not None:
        jointFit=1

    assert np.all(np.abs(x)<=pi) and np.any(np.abs(x)>pi/2), 'x expected to range from [-pi,pi]'
    assert ~np.any(np.abs(y)>90) and np.any(np.abs(y)>(pi)), 'y expecteed to be in range [-90,90]'

    if typ == 'DoVM':

        this_fun = SDF.rss_fun(SDF.Sd_vm) # loss function
        x0 = (0,1) # mu, k
        bnds= ((-15,15),(0.001,100))
        if jointFit:
            this_fun = SDF.rss_fun(SDF.many_VM)

    elif typ =='DoG':
        this_fun = SDF.rss_fun(SDF.DoG) # loss function
        x0 = (0,1) # mu, k
        bnds= ((-15,15),(1,10))
        if jointFit:
            this_fun = SDF.rss_fun(SDF.many_DoG)
    else:
        assert 0, 'bad input'

    if jointFit:
        x0,bnds = x0*2, bnds*2
        x = np.stack((x,x2))

    this_fit = scipy.optimize.minimize(this_fun,x0,args=(x,y),bounds=bnds)
    if want_full:
        return this_fit
    else:
        return this_fit.x


### visualization ###
pal = ['k','c'] # palette
ALPHAS = np.array([0.05,.01,.001])
def tt2stars(t_test):
    t,p = t_test
    if p<ALPHAS[2]: return '***'
    if p<ALPHAS[1]: return '**'
    if p<ALPHAS[0]: return '*'
    else:           return ''

def vis_bias(bias_all,nb_use,stats=None,
             labs = None,yl=10.5,ann=(),set_fs = None):
    """
    Visualize biases of model.
    
    bias_all    |  (n_nb,n_sort_E,n_gen,n_subj,n_bin) sliding average bias.
                |  Corresponds to second output 'sd_bias_all' from RM.summarize_stim  
    nb_use      |  also input to sd_bias all. used for labeling plots. must be same length as (n_nb)
    stats       |  structure with parameterization of bias. Used to include stats. 'fits_all_struct' from RM.summarize_stim
    labs        |  labels for different responses. must be same length as (n_gen)
    yl          |  shared symmetric ylimit for all plots
    ann         |  listlike Boolean used for annotating figure. Specific to simulations. If included, must be 5 elements long
    set_fs      |  (width, height). Total figure size.
    """

    n_nb,n_sort_E,n_gen,n_subj,n_bin = bias_all.shape
    if labs is None:
        if n_gen==3:
            labs = ('Stim SD','Resp SD','No SD')
        elif n_gen==4:
            labs = ('Stim SD','Resp SD','StimResp SD','No SD')
    else:
        assert n_gen==len(labs),'invalid labs'

    if set_fs is None:
        fs = (2.5*n_gen,4*n_nb)
    else: 
        fs = set_fs
    plt.figure(figsize=fs)
    assert n_nb==len(nb_use),'need valid nb_use'
    ii=0
    for nbi,nb in enumerate(nb_use):
        for ngi,ng in enumerate(labs):
            ii+=1
            plt.subplot(n_nb,n_gen,ii)
            SDF.sem_plot(bns,bias_all[nbi,0,ngi],color=pal[0])
            SDF.sem_plot(bns,bias_all[nbi,1,ngi],color=pal[1])
            SDF.d_plot(1,yl,-90)

            if nb==0: plt.title(f'shuffle | {ng}')
            else: plt.title(f'nb:{nb} | {ng}')

            if ngi==0: 
                plt.ylabel('Bias (deg)')
                if yl>10:  plt.yticks([-8,-4,0,4,8])
            else: plt.yticks([])

            if (nbi+1)==len(nb_use):
                plt.xlabel('$\Delta \\theta$ (deg)')
            else:
                plt.xticks([])

            if stats is not None:
                inds = (stats.nb==nb)&(stats.trueSD==ng)
                vals = (stats[inds&(stats.dStim=='d_ori')].amp.values,
                        stats[inds&(stats.dStim=='d_resp')].amp.values)
                
                t0 = scipy.stats.ttest_1samp(vals[0],0)
                t1 = scipy.stats.ttest_1samp(vals[1],0)
                t2 = scipy.stats.ttest_rel(*vals)

                plt.annotate(tt2stars(t0),(60,yl-2),color='k')
                plt.annotate(tt2stars(t1),(60,yl-3),color='c')
                sta=''
                if t2[1]<ALPHAS[0]:
                    sta = ['S','R'][t2[0]<0]
                plt.annotate(sta + tt2stars(t2),(60,yl-4),color='m')
  
                if (len(ann)==5)&(ngi==(len(labs)-1))&(nbi==0):
                    use_corrected_E,use_corrected_resp,do_stim_autocorr,do_cb,do_oblique = ann
                    plt.annotate('%s E Corrected' %(('No','Yes')[use_corrected_E]),(-85,yl-2))
                    plt.annotate('%s resp Corrected' %(('No','Yes')[use_corrected_resp]),(-85,yl-4))
                    plt.annotate('%s AutoCorr' %(('Not','(+)','(-)')[do_stim_autocorr]),(-85,-yl+2))
                    plt.annotate('%s CB %s Oblique' %(('No','Yes','Saw')[do_cb],('No','Yes')[do_oblique]),(-85,-yl+4))
    plt.tight_layout()