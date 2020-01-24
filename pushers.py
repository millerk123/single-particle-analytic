import numpy as np
dt = 0.014
rqm = -1.0
a0 = 5.0
omega0 = 1.0
phi0 = np.pi/2

def main():

    global dudt

    n_p = 1
    x = np.zeros(n_p)
    p = np.zeros((3,n_p))

    #----Set up initial conditions----
    # Multiple particles
    # x[:] = np.linspace(-0.1,0.1,n_p)
    # p[0,:] = 0

    # Single particle
    x[0] = 0.0
    p[0,:] = 30.0

    # Analytic correction to initial momentum from Boris pusher
    p[1,:] = a0*dt/2 - np.sqrt( ( np.sqrt( np.square( a0*dt*p[0,:] )
                                  + np.square( 1 + np.square( p[0,:] ) ) )
                                  - 1 - np.square(p[0,:]) ) / 2 )

    t_final = 300.0
    n_steps = np.ceil(t_final/dt).astype(int)

    # Set up diagnostic arrays
    diag_x = np.zeros((2,n_p,n_steps+1))
    diag_p = np.zeros((3,n_p,n_steps+1))

    diag_x[0,:,0] = x
    diag_p[:,:,0] = p
    dudt = dudt_boris

    for n in np.arange(n_steps):

        diag_x[:,:,n+1], diag_p[:,:,n+1] = adv_dep( diag_x[:,:,n], diag_p[:,:,n], n )
    
    diag_gamma = np.sqrt( np.sum( np.square(diag_p), axis=0) )

    np.savez( 'single-part', x=diag_x, p=diag_p, gamma=diag_gamma, dt=dt, n_steps=n_steps )

# Assumes all arrays are of shape [dim,part]
# Except x, which is shape [part] since we don't need to keep track of other 2 dimensions

def e( x, n ):

    ef = np.zeros( (3,x.size) )

    ef[1,:] = a0 * omega0 * np.sin( omega0*( x - n*dt ) + phi0 )

    return ef

def b( x, n ):

    bf = np.zeros( (3,x.size) )

    bf[2,:] = a0 * omega0 * np.sin( omega0*( x - n*dt ) + phi0 )

    return bf

def dudt_boris( p, ep, bp, n ):

    tem = 0.5 * dt / rqm

    ep = ep * tem
    utemp = p + ep

    u2 = np.sum( np.square(utemp), axis=0 )
    gamma = np.sqrt( u2 + 1.0 )
    gam_tem = tem / gamma

    bp = bp * gam_tem

    p = utemp + np.cross(utemp,bp,axis=0)

    bp = bp * 2.0 / ( 1.0 + np.sum( np.square(bp), axis=0 ) )

    utemp = utemp + np.cross(p,bp,axis=0)

    p = utemp + ep

    return p

def adv_dep( x, p, n ):

    p = dudt( p, e(x[0,:],n), b(x[0,:],n), n )

    rgamma = 1.0 / np.sqrt( 1.0 + np.sum( np.square(p), axis=0 ) )

    x = x + p[0:2,:] * rgamma * dt

    return x, p


if __name__ == "__main__":
    main()

