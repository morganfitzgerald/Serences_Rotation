import matplotlib.pyplot as plt
import numpy as np
from scipy.special.cython_special import iv
pi=np.pi
k = np.sqrt(2)/np.exp(-0.5)
pm = np.array((-1,1))

# wrapping
def wrapRad(d): # wraps to +/- pi
    d[np.abs(d)>pi]-= 2*pi*np.sign(d[np.abs(d)>pi])
    return d
    
def wrap(d): # wraps to +/-180
    d[np.abs(d)>180]-= 360*np.sign(d[np.abs(d)>180])
    return d

def circ_vector(x,ax):
    return np.mean(np.exp(x*1j),ax) 
def circ_mean(x,ax=0): # assumes full circle in radians
    return np.angle(circ_vector(x,ax))
def circ_var(x,ax=0): # assumes full circle in radians
    return 1-np.abs(circ_vector(x,ax))

def I(d): return d
    
# - parameterizations
#- Von Mises
Bessel0 = lambda x: iv(0,x)
def d_vm(w,x):
    return w*np.sin(x)*np.exp(w*np.cos(x))/(Bessel0(float(w)))
def Sd_vm(p,x): # scaled von mises
    a,w=p
    unscaled = d_vm(w,x)
    peak_val = 2*np.arctan(np.sqrt(np.sqrt(4*w**2+1)-2*w))
    denom = d_vm(w,peak_val)
    return a*unscaled/denom
    

#- Gaussian
def DoG(p,x_min_deg):
    a,w = p[0],p[1]
    y = x_min_deg*a*w*k*np.exp(-(w*x_min_deg)**2)
    return y
def min_fun_dog(p,vals):
    x_min,y = vals
    bias = DoG(p,x_min)
    return np.sqrt(np.sum(wrap(y - bias)**2))

def many_sine_cos(p,x):
    # this is the best one, essentially a fourier decomp/ steerable filter.
    # p should be even number, all initialized at (0,)
    # x should range [0, 2pi] or [pi, pi]
    y_hat = np.zeros_like(x)
    for p_ind,amp in enumerate(p):
        if np.mod(p_ind,2)==0: # cos
            y_hat += amp*np.cos((p_ind//2+1)*x)
        else:
            y_hat += amp*np.sin((p_ind//2+1)*x)
    return y_hat


# 'many' functions, allow fitting of an arbitrary number of DVM or DOG functions
def many_VM(p,x):
    n_vm = len(p)//2
    n_trials = x.shape[1]
    assert n_vm == x.shape[0]
    
    y=np.zeros(n_trials)
    for di in range(n_vm):
        y+= Sd_vm(p[(di*2):(di*2)+2],x[di])
    return y


def many_DoG(p,x):
    n_dog = len(p)//2
    n_trials = x.shape[1]
    assert n_dog == x.shape[0]
    
    y=np.zeros(n_trials)
    for di in range(n_dog):
        y+= DoG(p[(di*2):(di*2)+2],x[di])
    return y
# def min_fun_many_dog(p,vals):
#     x_min,y = vals
#     bias = many_DoG(p,x_min)
#     return np.sqrt(np.mean(wrap(y - bias )**2))

# account for cardinal biases.
def sine5(amps,x): 
    # two sines + global sine (single phase) + offsets for each half
    this_out = amps[0]*np.sin(x*2)*(x<0) +amps[1]*np.sin(x*2)*(x>0) +amps[2]*np.sin(x)*(x>0)+amps[3]*np.sin(x)*(x<0)+amps[4]*np.sin(x*4)#+amps[4]*(x<0)+amps[5]*(x>0) 
#     this_out = amps[0]*np.sin(x*2) + amps[1]*np.sin(x*4) + amps[2]*np.sin(x)#+amps[4]*(x<0)+amps[5]*(x>0) 
    return this_out

#- general loss function, should be able to replace most 'min_fun' using this, just need to be in same format
def rss_fun(fun):
    def loss_fun(params,x,y):
        return np.sqrt(np.mean(wrapRad(y-fun(params,x))**2))
    return loss_fun

def rss_fun_bias(fun):
    def loss_fun(params,x,y):
        return np.sqrt(np.mean(wrapRad(y-fun(params[:-1],x)-params[-1])**2))
    return loss_fun

def rss_fun_l2(fun,lam,inds_penalty=2):
    # inds_penalty allows to only penalize amplitude parameters
    def loss_fun(params,x,y):
        return np.sum((y-fun(params,x))**2)+lam*np.sum(params[::inds_penalty]**2) 
    return loss_fun


#- NB helper functions...
# here negative numbers imply looking at past or influence of past trials

def get_nb(nb,vals,want_diff=1,wrap_fun=wrapRad):
    """
    nb          - number of trials back, -1 corresponds to previous trial, +1 future
    vals        - variable to be sorted
    want_diff   - flag, 0- returns shifted values, (1)- returns current - shifted 
    wrap_fun    - wrapping function applied to difference calculation. default is wrapRad,
                    SDF.I is convenience function for identity
    """
    assert nb!=0, 'nb must be positive or negative'
    if nb>1:
        print('warning! You probably meant to put in a negative number?')
        
    if want_diff:
        if nb<0:
            return np.concatenate((np.zeros(-nb),wrap_fun(vals[:nb] - vals[-nb:]))) # prev - current
        elif nb>0:
            return np.concatenate((wrap_fun(vals[nb:] - vals[:-nb]),np.zeros(nb))) # future - current
    else:
        if nb<0:
            return np.concatenate((np.zeros(-nb),vals[:nb])) # prev
        elif nb>0:
            return np.concatenate((vals[nb:],np.zeros(nb))) # future
        
# visualzation



fun_dict = {'mean':np.mean,'sd':np.std,'circ_mean':circ_mean,'circ_var':circ_var}
def do_bining(bns,overlap,grouping_var,var,want_var = 'mean'):

    assert want_var in fun_dict.keys()
    this_fun = fun_dict[want_var]
    n_bns=len(bns)
    grouper = np.zeros(len(bns))

    out = np.zeros(len(bns))*np.nan
    for i in range(n_bns):
        if i<overlap:
            these_ind = (grouping_var<=bns[i+overlap]) | (grouping_var>=bns[i-overlap])
        elif i>(n_bns-overlap-1):
            these_ind = (grouping_var>=bns[i-overlap]) | (grouping_var<=bns[i+overlap-n_bns])
        else:
            these_ind = (grouping_var>=bns[i-overlap])&(grouping_var<=bns[i+overlap]) # need to figure out 
        if ~np.any(these_ind):
            continue
        out[i] = this_fun(var[these_ind])
    return out


def sem_plot(x,y,axs=0,within_E=0,do_line=0,outline=0,do_errorbar=0,**args):
    # x is assumed to be examples x len(y)

    n_ex,n_points =  y.shape

    if n_points!=len(x):
        y=y.T
        n_ex,n_points =  y.shape
    assert n_points==len(x), 'x not correct shape'
    if np.any(np.isnan(y)): 
        print('ignoring nan values!')
        n_ex = np.sum(~np.isnan(y),0)
    m_y = np.nanmean(y,0)
    
    if within_E:
        y_use = (y.T-np.mean(y,1)).T
        s_y = np.nanstd(y_use,0)/np.sqrt(n_ex)
    else:
        s_y = np.nanstd(y,0)/np.sqrt(n_ex)
    if axs==0:
        if do_errorbar:
            plt.errorbar(x,m_y,s_y,**args)
            
        elif outline:
            plt.plot(x,m_y+s_y,**args)
            plt.plot(x,m_y-s_y,**args)
        else:
            plt.fill_between(x,m_y-s_y,m_y+s_y,**args)
            if do_line:
                plt.plot(x,m_y,'k')
    else:
        if outline:
            axs.plot(x,m_y+s_y,**args)
            axs.plot(x,m_y-s_y,**args)
        else:
            axs.fill_between(x,m_y-s_y,m_y+s_y,**args)
            if do_line:
                axs.plot(x,m_y,'k')
                
def d_plot(s=0,yl=0,xl=-180):
    if np.abs(xl)>90:
        space =90
    else:
        space =45
        
    if xl<0:
        xl = pm*np.abs(xl)
    else:
        xl = [0,xl]

    
    plt.xlim(xl)
    plt.xticks(np.arange(xl[0],xl[1]+1,space))

    if s:
        plt.plot(xl,[0,0],'k--')
    if yl:
        plt.plot([0,0],(-yl,yl),'b--')
        plt.ylim((-yl,yl))            
        
def circ_corr(x, y):
    """ calculate correlation coefficient between two circular variables
    Using Fisher & Lee circular correlation formula (code adapted from Ed Vul)
    x, y are both in radians [0,2pi]
    
    """
    if np.any(x>90): # assume both variable range from [0,180]
        x = x/90*pi
        y = y/90*pi
    if np.any(x<0) or np.any(x>2*pi) or np.any(y<0) or np.any(y>2*pi):
        raise ValueError('x and y values must be between 0-2pi')
    n = np.size(x);
    assert(np.size(y)==n)
    A = np.sum(np.cos(x)*np.cos(y));
    B = np.sum(np.sin(x)*np.sin(y));
    C = np.sum(np.cos(x)*np.sin(y));
    D = np.sum(np.sin(x)*np.cos(y));
    E = np.sum(np.cos(2*x));
    Fl = np.sum(np.sin(2*x));
    G = np.sum(np.cos(2*y));
    H = np.sum(np.sin(2*y));
    corr_coef = 4*(A*B-C*D) / np.sqrt((np.power(n,2) - np.power(E,2) - np.power(Fl,2))*(np.power(n,2) - np.power(G,2) - np.power(H,2)));
    return corr_coef
