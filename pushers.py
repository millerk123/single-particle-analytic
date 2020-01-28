import numpy as np
import sys
dt = 0.14
rqm = -1.0
a0 = 50.0
omega0 = 1.0
phi0 = np.pi/2
p10 = 0.0
t_final = 300.0
# Available pushers: boris, vay, cary, fullrot, euler

def main():

    global dudt, a0, p10

    pushers = {
        "boris" : dudt_boris,
        "vay" : dudt_vay,
        "cary" : dudt_cary,
        "fullrot" : dudt_fullrot,
        "euler" : dudt_euler,
        }

    args = sys.argv
    dudt = pushers[args[1]]

    if len(args)>2:
        if len(args)!=4:
            print("Must have both a0 and p10 as inputs, no more")
            return
        a0 = float(args[2])
        p10 = float(args[3])

    n_p = 51
    x = np.zeros(n_p)
    p = np.zeros((3,n_p))

    #----Set up initial conditions----
    # Multiple particles
    x[:] = np.linspace(-np.pi,np.pi,n_p)
    p[0,:] = p10

    # Single particle
    # x[0] = 0.0
    # p[0,:] = p10

    # Analytic correction to initial momentum from Boris pusher
    p[1,:] = a0*dt/2 - np.sqrt( ( np.sqrt( np.square( a0*dt*p10 )
                                  + np.square( 1 + np.square(p10) ) )
                                  - 1 - np.square(p10) ) / 2 )

    n_steps = np.ceil(t_final/dt).astype(int)

    # Set up diagnostic arrays
    diag_x = np.zeros((2,n_p,n_steps+1))
    diag_p = np.zeros((3,n_p,n_steps+1))

    diag_x[0,:,0] = x
    diag_p[:,:,0] = p

    for n in np.arange(n_steps):

        diag_x[:,:,n+1], diag_p[:,:,n+1] = adv_dep( diag_x[:,:,n], diag_p[:,:,n], n )
    
    diag_gamma = np.sqrt( 1 + np.sum( np.square(diag_p), axis=0) )

    np.savez( 'single-part-'+dudt.__name__[5:], x=diag_x, p=diag_p, gamma=diag_gamma,
                dt=dt, n_steps=n_steps, a0=a0, phi0=phi0, p10=p10 )

# Assumes all arrays are of shape [dim,part]
# Except x, which is shape [part] since we don't need to keep track of other 2 dimensions

def e( x, n ):

    global a0

    ef = np.zeros( (3,x.size) )

    ef[1,:] = a0 * omega0 * np.sin( omega0*( x - n*dt ) + phi0 )

    return ef

def b( x, n ):

    global a0

    bf = np.zeros( (3,x.size) )

    bf[2,:] = a0 * omega0 * np.sin( omega0*( x - n*dt ) + phi0 )

    return bf

def dudt_boris( p_in, ep, bp, n ):

    tem = 0.5 * dt / rqm

    ep = ep * tem
    utemp = p_in + ep

    gam_tem = tem / np.sqrt( 1.0 + np.sum( np.square(utemp), axis=0 ) )

    bp = bp * gam_tem

    p = utemp + np.cross(utemp,bp,axis=0)

    bp = bp * 2.0 / ( 1.0 + np.sum( np.square(bp), axis=0 ) )

    utemp = utemp + np.cross(p,bp,axis=0)

    p = utemp + ep

    return p

def dudt_vay( p_in, ep, bp, n ):

    tem = 0.5 * dt / rqm

    ep = ep * tem
    bp = bp * tem
    rgamma = 1.0 / np.sqrt( 1.0 + np.sum( np.square(p_in), axis=0 ) )

    temp_vec = p_in * rgamma
    bpsq = np.sum( np.square(bp), axis=0 )

    utemp = p_in + ep + np.cross(temp_vec,bp,axis=0)

    temp_vec = utemp + ep

    ustar = np.sum( temp_vec*bp, axis=0 )
    sigma = 1.0 + np.sum( np.square(temp_vec), axis=0 ) - bpsq

    rgam = 1.0 / np.sqrt( 0.5 * ( sigma + np.sqrt( np.square(sigma)
                      + 4.0 * ( bpsq + np.square(ustar) ) ) ) )

    bp = bp * rgam

    spar = 1.0 / ( 1.0 + np.sum( np.square(bp), axis=0 ) )
    uptp = np.sum( temp_vec*bp, axis=0 )

    p = spar * ( temp_vec + uptp * bp + np.cross(temp_vec,bp,axis=0) )

    return p

def dudt_cary( p_in, ep, bp, n ):

    tem = 0.5 * dt / rqm

    ep = ep * tem
    bp = bp * tem
    utemp = p_in + ep

    gam_minus_sq = 1.0 + np.sum( np.square(utemp), axis=0 )
    bpsq = np.sum( np.square(bp), axis=0 )
    bdotusq = np.square( np.sum( bp*utemp, axis=0 ) )
    gamma = np.sqrt( 0.5 * ( gam_minus_sq - bpsq
        + np.sqrt( np.square( gam_minus_sq - bpsq ) + 4.0 *( bpsq + bdotusq ) ) ) )

    bp = bp / gamma

    p = utemp + np.cross(utemp,bp,axis=0)

    bp = bp * 2.0 / ( 1.0 + np.sum( np.square(bp), axis=0 ) )

    utemp = utemp + np.cross(p,bp,axis=0)

    p = utemp + ep

    return p

def dudt_fullrot( p_in, ep, bp, n ):

    tem = 0.5 * dt / rqm

    ep = ep * tem
    utemp = p_in + ep

    tem_gam = tem / np.sqrt( 1.0 + np.sum( np.square(utemp), axis=0 ) )

    bp = bp * tem_gam

    bnorm = np.sqrt( np.sum( np.square(bp), axis=0 ) )
    t = np.ones_like( bnorm )
    inds = bnorm>0
    t[inds] = np.tan( bnorm[inds] ) / bnorm[inds]
    bp = bp * t

    p = utemp + np.cross(utemp,bp,axis=0)

    bp = bp * 2.0 / ( 1.0 + np.sum( np.square(bp), axis=0 ) )

    utemp = utemp + np.cross(p,bp,axis=0)

    p = utemp + ep

    return p

def dudt_euler( p_in, ep, bp, n ):

    tem = 0.5 * dt / rqm

    ep = ep * tem
    utemp = p_in + ep

    gam_tem = dt / ( rqm * np.sqrt( 1.0 + np.sum( np.square(utemp), axis=0 ) ) )

    bp = bp * gam_tem

    bnorm = np.sqrt( np.sum( np.square(bp), axis=0 ) )

    s = np.sin( bnorm / 2.0 )
    a = np.cos( bnorm / 2.0 )
    inds = bnorm>0
    s[inds] = - s[inds] / bnorm[inds]
    s[~inds] = -1

    b = bp[0,:] * s
    c = bp[1,:] * s
    d = bp[2,:] * s

    r11 = a*a+b*b-c*c-d*d;  r21=2*(b*c-a*d);      r31=2*(b*d+a*c)
    r12 = 2*(b*c+a*d);      r22=a*a+c*c-b*b-d*d;  r32=2*(c*d-a*b)
    r13 = 2*(b*d-a*c);      r23=2*(c*d+a*b);      r33=a*a+d*d-b*b-c*c

    p = np.zeros_like(p_in)

    p[0,:] = r11 * utemp[0,:] + r21 * utemp[1,:] + r31 * utemp[2,:]
    p[1,:] = r12 * utemp[0,:] + r22 * utemp[1,:] + r32 * utemp[2,:]
    p[2,:] = r13 * utemp[0,:] + r23 * utemp[1,:] + r33 * utemp[2,:]

    p = p + ep

    return p

def adv_dep( x, p_in, n ):

    p = dudt( p_in, e(x[0,:],n), b(x[0,:],n), n )

    rgamma = 1.0 / np.sqrt( 1.0 + np.sum( np.square(p), axis=0 ) )

    x = x + p[0:2,:] * rgamma * dt

    return x, p


if __name__ == "__main__":
    main()

