# Author:  Rufus Fraanje
# Email:   p.r.fraanje@hhs.nl
# Licence: GNU General Public License (GNU GPLv3)
# Creation date: 2019-03-06
# Last modified: 2020-01-07

import numpy as np
from scipy.linalg import expm, logm
from scipy.signal import tf2ss, ss2tf, lsim2
# also see: https://docs.scipy.org/doc/scipy/reference/signal.html

# Symbols (variations may be made for clarity, depending on context,
# e.g. ud to stress that the input is in discrete-time):
#
# tc          - continuous time instants
# td          - discrete time instants (1-D array)
# h           - sampling time
# u           - input signal (1-D array for single channel signals, for
#                             multichannels 2-D array where each column
#                             is at one sample instant)
# y           - output signal (idem)
# x0          - initial state
# x           - state sequence
# NS          - number of samples
# ni          - number of inputs
# no          - number of outputs
# ns          - number of states
# Ac,Bc,Cc,Dc - continuous time state-space models
#    Ac       - state-transition matrix (ns x nx) (A in Computer-Controlled Systems)
#    Bc       - input matrix (ns x ni) (B in Computer-Controlled Systems)
#    Cc       - output matrix (no x ns) (C in Computer-Controlled Systems)
#    Dc       - direct feedthrough matrix (no x ni) (D in Computer-Controlled Systems)
# Ad,Bd,Cd,Dd - discrete time state-space models
#    Ad       - state-transition matrix (ns x nx) (Φ (Phi) in Computer-Controlled Systems)
#    Bd       - input matrix (ns x ni) (Γ (Gamma) in Computer-Controlled Systems)
#    Cd       - output matrix (no x ns) (C in Computer-Controlled Systems)
#    Dd       - direct feedthrough matrix (no x ni) (D in Computer-Controlled Systems)

def spike(NS=100,at_sample=1):
    signal = np.zeros((1,NS))
    signal[0,at_sample] = 1.
    return signal

def random_impulses(av_samples_per_spike=10,NS=100,nchan=1):
    signal = np.zeros((NS,nchan))
    for i in range(nchan):
        signal[:,i] = np.floor(np.random.randint(0,av_samples_per_spike+1,NS)/av_samples_per_spike)
    return signal

def c2d_zoh_AB(Ac,Bc,h):
    AB = np.hstack((Ac,Bc))
    ns,ni = Bc.shape
    lower_lines = np.zeros((ni,ns+ni))
    ABext = np.vstack((AB,lower_lines))*h
    ABext_zoh = expm(ABext)
    Ad = ABext_zoh[0:ns,0:ns]
    Bd = ABext_zoh[0:ns,ns:ns+ni]
    return Ad,Bd

def d2c_zoh_AB(Ad,Bd,h):
    AB_zoh = np.hstack((Ad,Bd))
    ns,ni = Bd.shape
    lower_lines = np.hstack((np.zeros((ni,ns)),np.eye(ni)))
    ABext_zoh = np.vstack((AB_zoh,lower_lines))
    ABext = 1/h * logm(ABext_zoh)
    Ac = ABext[0:ns,0:ns]
    Bc = ABext[0:ns,ns:ns+ni]
    return Ac,Bc

def c2d_zoh(Ac,Bc,Cc,Dc,h):
    Ad,Bd = c2d_zoh_AB(Ac,Bc,h)
    Cd = Cc.copy()
    Dd = Dc.copy()
    return Ad,Bd,Cd,Dd

def d2c_zoh(Ad,Bd,Cd,Dd,h):
    Ac,Bc = d2c_zoh_AB(Ad,Bd,h)
    Cc = Cd.copy()
    Dc = Dd.copy()
    return Ac,Bc,Cc,Dc

def c2d_zoh_intersample(Ac,Bc,Cc,Dc,h,number_intersamples):
    """Convert continuous time state-space model to discrete-time and
    intersample discrete time, for use in sim_intersample."""
    ns = Ac.shape[0]
    no,ni = Dc.shape
    Ad,Bd = c2d_zoh_AB(Ac,Bc,h)
    Cd = np.zeros((number_intersamples*no,ns))
    Dd = np.zeros((number_intersamples*no,ni))
    for i in range(number_intersamples):
        if i==0:
            Cd[0:no,:] = Cc
            Dd[0:no,:] = Dc
        else:
            Ad_interstep,Bd_interstep = c2d_zoh_AB(Ac,Bc,i*h/number_intersamples)
            Cd[i*no:(i+1)*no,:] = Cc @ Ad_interstep
            Dd[i*no:(i+1)*no,:] = Cc @ Bd_interstep + Dc
    return Ad,Bd,Cd,Dd

def c2d_pole(lambda_i,h):
    """Convert poles from continuous to discrete time"""
    return np.exp(lambda_i*h)

def d2c_pole(lambda_i,h):
    """Convert poles from discrete to continuous time"""
    return 1/h*np.log(lambda_i)

def c2d_characteristic_equation(ac,h):
    """Convert characteristic polynomial from continuous to discrete
    time"""
    n = np.max(np.shape(ac))-1
    ac_adj = ac[1:].reshape((1,n))/ac[0]
    Ac = np.diag(np.ones((n-1)),-1)
    Ac[0:1,:] = -ac_adj
    Ad = expm(Ac*h)
    ad = np.poly(Ad)
    return ad

def d2c_characteristic_equation(ad,h):
    """Convert characteristic polynomial from discrete to continuous
    time"""
    n = np.max(np.shape(ad))-1
    ad_adj = ad[1:].reshape((1,n))/ad[0]
    Ad = np.diag(np.ones((n-1)),-1)
    Ad[0:1,:] = -ad_adj
    Ac = logm(Ad)/h
    ac = np.poly(Ac)
    return ac

def sim_continuous(A,B,C,D,u=None,t=None,x0=None,return_X=False):
    # use lsim2 (scipy.signal.lsim2) to simulate continuous-time linear state-space model
    # lsim2 uses on its turn the function scipy.integrate.odeint.
    # also see documentation of scipy.signal.lsim2
    T,yout,xout = lsim2((A,B,C,D), U=u, T=t, X0=x0)
    if return_X:
        return T,yout,xout
    else:
        return T,yout

def sim_state(A,B,u,x0=None):
    # Discrete-time forced state iteration:
    ns,ni=B.shape
    reshape_state = False
    if len(u.shape)==1: # u is a 1-dim signal
        u = np.reshape(u,(1,u.shape[0]))
        if ns==1: # make state a 1-dim array as input was
            reshape_state = True
    else:
        if u.shape[0]>u.shape[1]:
            Warning('u is not wide, u should have its samples on each column!')
    N=u.shape[1]
    if x0 is None: x0 = np.zeros((ns))
    x = np.zeros((ns,N))
    x[:,0] = x0
    for i in range(N-1):
        x[:,i+1] = A @ x[:,i] + B @ u[:,i]
    if reshape_state:
        x = np.reshape(x,(N))
    return x

def sim(A,B,C,D,u,x0=None,return_X=False):
    # Discrete-time forced state-space model iteration
    no,ni=D.shape
    ns=A.shape[0]
    reshape_output = False
    reshape_state = False
    if len(u.shape)==1: # u is a 1-dim signal
        u = np.reshape(u,(1,u.shape[0]))
        if no==1: # make output a 1-dim array as input was
            reshape_output = True
        if ns==1: # make state a 1-dim array as input was
            reshape_state = True
    else:
        if u.shape[0]>u.shape[1]:
            Warning('u is not wide, u should have its samples on each column!')
    N=u.shape[1]
    x = sim_state(A,B,u,x0)
    if reshape_state==1:
        x = np.reshape(x,(ns,N))
    y = C @ x + D @ u
    if reshape_output:
        y = np.reshape(y,(N))
    if return_X:
        if reshape_state:
            x = np.reshape(x,(N))
        return y,x
    else:
        return y

def sim_intersample(Ac,Bc,Cc,Dc,h,number_is,ud,td,x0=None):
    """ Intersample simulation of a continuous state space model
        (Ac,Bc,Cc,Dc) discretised with zero-order-hold with sampling
        time h. The parameter number_is determines the number of (equidistant) 
        intersamples. ud is the zero-order hold input, x0 is the initial
        state (None for zero initial state)."""
    Ad_is,Bd_is,Cd_is,Dd_is = c2d_zoh_intersample(Ac,Bc,Cc,Dc,h,number_is)
    NS = len(td)
    yd_is = sim(Ad_is,Bd_is,Cd_is,Dd_is,ud)
    td_is = np.repeat(td,number_is) + np.tile(h/number_is * np.array(range(number_is)), NS)
    yd_is = np.reshape(yd_is,(NS*number_is),order='F')
    return td_is,yd_is

def similarity_trans(A,B,C,T):
    """ Similarity transformation, gives 
        state-space representation matrices At,Bt,Ct
        of system with transformed state: 
          x_T = T x """
    Ti = np.linalg.inv(T)
    At = T @ A @ Ti
    Bt = T @ B
    Ct = C @ Ti
    return At,Bt,Ct

def ctrb(A,B):
    """ Returns controllability matrix:
        Wc = [B AB ... A^(n-1) B]
        """
    ns,ni = np.shape(B)
    Wc = np.zeros((ns,ns*ni));
    Wc[:,0:ni] = B
    for i in range(ns-1):
        Wc[:,(1+i)*ni:(2+i)*ni] =  A @ Wc[:,i*ni:(1+i)*ni]
    return Wc

def obsv(A,C):
    """ Returns observability matrix:
        Wo = [C
              CA
              ...
              CA^(n-1)]
        """
    no,ns = np.shape(C)
    Wo = np.zeros((ns*no,ns));
    Wo[0:no,:] = C
    for i in range(ns-1):
        Wo[(1+i)*no:(2+i)*no,:] =  Wo[i*no:(1+i)*no,:] @ A
    return Wo

def is_reachable(A,B):
    """Tests if state-space matrix pair (A,B) is reachable."""
    n = np.shape(A)[0]
    Wc = ctrb(A,B)
    r = np.linalg.matrix_rank(Wc)
    return r==n

def is_observable(A,C):
    """Tests if state-space matrix pair (A,C) is observable."""
    n = np.shape(A)[0]
    Wo = obsv(A,C)
    r = np.linalg.matrix_rank(Wo)
    return r==n

def is_stable(A,domain="discrete"):
    """Tests if state-transition matrix A is stable for discrete time
    (default, domain="discrete") or continuous time
    (domain="continuous")."""
    lambda_i = np.linalg.eig(A)[0]
    if (domain == "discrete") or (domain == "d"):
        return np.all(np.abs(lambda_i)<1.)
    elif (domain == "continuous") or (domain == "c"):
        return np.all(lambda_i.real<0.)

def canon_ctrl(A,B):
    """Find similarity transformation to transform to controller canonical form."""
    if not(is_reachable(A,B)):
        raise Exception("The state-space matrix pair (A,B) is not reachable.")
    n,ni = np.shape(B)
    if ni>1:
        raise NotImplementedError("System should have a single input.")
    a = np.poly(A)
    Actrl = np.diag(np.ones((n-1)),-1)
    Actrl[0,:] = -a[1:]/a[0]
    Bctrl = np.zeros((n,ni))
    Bctrl[0,0] = 1.

    Wc = ctrb(A,B)
    Wcctrl = ctrb(Actrl,Bctrl)
    #Tctrl = np.linalg.solve(Wc.T,Wcctrl.T).T
    Tctrl = Wcctrl @ np.linalg.pinv(Wc)

    # see section 2 in Feng Ding, Transformations between some special matrices,
    # Computers and Mathematics with Applications 59(2010), 2676-2694.
    # if method=="ctrl":
    #     T2inv[n-1:n,:] = np.hstack((np.array([[1.]]),np.zeros((1,n-1))))
    #     for i in range(n-1):
    #         T2inv[n-i-2:n-i-1,:] = T2inv[n-i-1:n-i,:] @ A
    return Tctrl

def canon_obsv(A,C):
    """Find similarity transformation to transform to observer canonical form."""
    if not(is_observable(A,C)):
        raise Exception("The state-space matrix pair (A,C) is not observable.")
    no,n = np.shape(C)
    if no>1:
        raise NotImplementedError("System should have a single output.")
    a = np.poly(A)
    Aobsv = np.diag(np.ones((n-1)),+1)
    Aobsv[:,0] = -a[1:]/a[0]
    Cobsv = np.zeros((no,n))
    Cobsv[0,0] = 1.

    Wo = obsv(A,C)
    Woobsv = obsv(Aobsv,Cobsv)
    Tobsv = np.linalg.pinv(Woobsv) @ Wo
    return Tobsv

def canon(A,B,C,method='ctrl'):
    if method=='ctrl':
        T = canon_ctrl(A,B)
    elif method=='obsv':
        T = canon_obsv(A,C)
    else:
        raise NotImplementedError("Unknown method.")
    At,Bt,Ct = similarity_trans(A,B,C,T)
    return At,Bt,Ct

def place(A,B,pole_vec):
    """Calculates the state-feedback matrix L such that the poles are at
    the specified locations in the vector pole_vec."""
    n,ni = np.shape(B)
    if ni>1:
        raise Exception("This pole-placement method only works for single-input systems")
    char_poly_desired = np.poly(pole_vec) # desired characteristic polynomial
    char_poly_actual = np.poly(A)         # characteristic polynomial of matrix A
    Tctrl = canon_ctrl(A,B) # this computes the matrix \tilde{W}_c W_c^(-1)
    L = (char_poly_desired[1:]-char_poly_actual[1:]).reshape((ni,n)) @ Tctrl
    return L

def dc_gain(A,B,C,D,domain="discrete"):
    """Computes the DC gain of the discrete (domain = "discrete") or continuous  (domain = "continuous")
    state-space models."""
    n = np.shape(A)[0]
    if (domain == "discrete") or (domain == "d"):
        return C @ np.linalg.inv(np.eye(n)-A) @ B + D
    elif (domain == "continuous") or (domain == "c"):
        return -C @ np.linalg.inv(A) @ B + D


# for following functions, also see the answer on stackexchange:
# https://math.stackexchange.com/questions/2933407/formula-for-interconnection-of-ltis-in-state-space-form

def ss_series(A1,B1,C1,D1,A2,B2,C2,D2):
    """Series interconnection of two state-space models:
        --> (A1,B1,C1,D1) --> (A2,B2,C2,D2) -->
        state of series connection is: [ x1 ]
                                       [ x2 ]
        """
    n1 = np.shape(A1)[0]
    n2 = np.shape(A2)[0]
    # note that A @ B is matrix multiplication in numpy, this is the
    # same as np.dot(A,B).
    A = np.vstack( (np.hstack( (A1,np.zeros((n1,n2))) ), np.hstack( (B2@C1,A2) )  ) )
    B = np.vstack( (B1,B2@D1) )
    C = np.hstack( (D2@C1, C2) )
    D = D2@D1
    return (A,B,C,D)

def ss_parallel(A1,B1,C1,D1,A2,B2,C2,D2,sign=None):
    """Parallel interconnection of two state-space models:
                            +
        ---> (A1,B1,C1,D1) -->sum ---->
         |                     ^sign
         |-> (A2,B2,C2,D2) ----|

         sign is a list of signs, if None, than all signs are positive 

        state of parallel connection is: [ x1 ]
                                         [ x2 ]
        """
    no = np.shape(D2)[0]
    if not(sign): sign = np.ones((no)) 
    signMat = np.diag(sign)
    n1 = np.shape(A1)[0]
    n2 = np.shape(A2)[0]
    A = np.vstack( (np.hstack( (A1,np.zeros((n1,n2)) )), np.hstack( (np.zeros((n2,n1)),A2) )  ) )
    B = np.vstack( (B1,B2) )
    C = np.hstack( (C1, signMat@C2) )
    D = D1+signMat@D2
    return (A,B,C,D)

def ss_feedback(A1,B1,C1,D1,A2,B2,C2,D2,sign=None):
    """Feedback interconnection of two state-space models:

        ---> sum --> (A1,B1,C1,D1) ---->
         sign ^                     |
              |-- (A2,B2,C2,D2) <---|

        sign is a list of signs, if None, than all signs are negative 

        state of feedback connection is: [ x1 ]
                                         [ x2 ]
        """
    no,ni = np.shape(D1)
    if not(sign): sign = -np.ones((ni))
    signMat = np.diag(sign)
    DDinv = np.linalg.pinv(np.eye(no)-(D1@signMat)@D2)
    Acl11 = A1+(B1@signMat)@D2@DDinv@C1
    Acl12 = (B1@signMat)@C2+(B1@signMat)@D2@DDinv*(D1@signMat)@C2
    Bcl1 = B1 +(B1@signMat)@D2@DDinv@D1
    Acl21 = B2@DDinv@C1
    Acl22 = A2+B2@DDinv@(D1@signMat)@C2
    Bcl2 = B2@DDinv@D1
    Ccl1 = DDinv@C1
    Ccl2 = DDinv@(D1@signMat)@C2
    Acl = np.vstack(  (np.hstack((Acl11,Acl12)),np.hstack((Acl21,Acl22))) )
    Bcl = np.vstack( (Bcl1,Bcl2) )
    Ccl = np.hstack( (Ccl1,Ccl2) )
    Dcl = DDinv@D1
    return (Acl,Bcl,Ccl,Dcl)

# for nicer printing of polynomials:
def pretty_polynomial(poly,symbol='q',decimals=2,tol=1e-10):
    """Returns a string with nicely formated polynomial with
    coefficients given by the vector poly."""
    str_poly = ""
    n = len(poly)
    for i in range(n):
        if abs(poly[i])>tol: # skip very small numbers
            if abs(poly[i]-1.)<tol: # don't print multiplication by 1.
                str_poly += f"{symbol}^{n-i-1}"
            else:
                str_poly += f"{np.round(poly[i],decimals)}{symbol}^{n-i-1}"
            if i<n-1:
                if poly[i+1]>0:
                    str_poly += "+"
    return str_poly
