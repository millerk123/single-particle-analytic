import numpy as np
import sys
from scipy import optimize
dt = 0.14
rqm = -1.0
a0 = 1.0
omega0 = 53.6488
phi0 = np.pi/2
p10 = 1.0
t_final = 3000.0
v_ph = 1.0007
# Available pushers: boris, vay, cary, fullrot, euler, petri

def main():

    global dudt, a0, p10, t_final

    pushers = {
        "boris" : dudt_boris,
        "vay" : dudt_vay,
        "cary" : dudt_cary,
        "fullrot" : dudt_fullrot,
        "euler" : dudt_euler,
        "petri" : dudt_petri,
        }

    args = sys.argv
    dudt = pushers[args[1]]

    if len(args)>2:
        a0 = float(args[2])
        if len(args)>3:
            p10 = float(args[3])
            if len(args)>4:
                t_final = float(args[4])

    n_p = 51
    x = np.zeros(n_p)
    p = np.zeros((3,n_p))

    #----Set up initial conditions for multiple particles----
    x[:] = np.linspace(-np.pi,np.pi,n_p)

    # Get initial conditions
    p[0,:] = 10

    n_steps = np.ceil(t_final/dt).astype(int)

    # Set up diagnostic arrays
    diag_x = np.zeros((2,n_p,n_steps+1))
    diag_p = np.zeros((3,n_p,n_steps+1))

    diag_x[0,:,0] = x
    diag_x[1,:,0] = 1.5
    diag_p[:,:,0] = p

    for n in np.arange(n_steps):

        diag_x[:,:,n+1], diag_p[:,:,n+1] = adv_dep( diag_x[:,:,n], diag_p[:,:,n], n )

    diag_gamma = np.sqrt( 1 + np.sum( np.square(diag_p), axis=0) )

    np.savez( 'betatron-{}-a0-{}-p10-{}'.format(dudt.__name__[5:],a0,p10),
                x=diag_x, p=diag_p, gamma=diag_gamma,
                dt=dt, n_steps=n_steps, a0=a0, phi0=phi0, p10=p10 )

# Assumes all arrays are of shape [dim,part]
# Except x, which is shape [part] since we don't need to keep track of other 2 dimensions

def e( x, y, n ):

    global a0

    ef = np.zeros( (3,x.size) )

    ef[1,:] = a0 * omega0 * np.sin( omega0*( x/v_ph - n*dt ) + phi0 ) + y / 2.0

    return ef

def b( x, n ):

    global a0

    bf = np.zeros( (3,x.size) )

    bf[2,:] = a0 * omega0 * np.sin( omega0*( x/v_ph - n*dt ) + phi0 )

    return bf

def dudt_boris( p_in, ep, bp ):

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

def dudt_petri( p_in, ep, bp ):

    def lorentz_rotate2( sign ):

        # Initialize variables
        ndir2_rot = np.empty_like(ep)
        u_tmp = np.empty_like(ep)
        u_out = np.empty_like(u4)

        ndir1 = bp / b_amp
        ndir2 = ep / e_amp

        c_theta = ndir1[0,:]
        s_theta = np.sqrt( np.sum( np.square(ndir1[1:,:]), axis=0 ) )
        s_theta[inds] = 1 # Avoid dividing by 0

        # We always have c_phi = 0
        s_phi = ndir1[2,:] / s_theta
        s_theta[inds] = 0

        ndir2_rot[0,:] =  c_theta * ndir2[0,:] + s_theta * s_phi * ndir2[2,:]
        ndir2_rot[1,:] = -s_theta * ndir2[0,:] + c_theta * s_phi * ndir2[2,:]
        ndir2_rot[2,:] = -s_phi * ndir2[1,:]

        if (sign==1):

            # Forward transform
            c = ndir2_rot[1,:]
            s = ndir2_rot[2,:]

            u_tmp[0,:] =  u4[1,:]
            u_tmp[1,:] =  s * u4[2,:] + c * u4[3,:]
            u_tmp[2,:] = -c * u4[2,:] + s * u4[3,:]

            u_out[0,:] = u4[0,:]
            u_out[1,:] = c_theta * u_tmp[0,:] - s_theta * u_tmp[1,:]
            u_out[2,:] = -s_phi * u_tmp[2,:]
            u_out[3,:] = s_theta * s_phi * u_tmp[0,:] + c_theta * s_phi * u_tmp[1,:]

        else:

            # Backward transform
            u_tmp[0,:] =  c_theta * u4[1,:] + s_theta * s_phi * u4[3,:]
            u_tmp[1,:] = -s_theta * u4[1,:] + c_theta * s_phi * u4[3,:]
            u_tmp[2,:] = -s_phi * u4[2,:]

            c = ndir2_rot[1,:]
            s = ndir2_rot[2,:]

            u_out[0,:] = u4[0,:]
            u_out[1,:] = u_tmp[0,:]
            u_out[2,:] = s * u_tmp[1,:] - c * u_tmp[2,:]
            u_out[3,:] = c * u_tmp[1,:] + s * u_tmp[2,:]

        return u_out

    def proper_dt_light():

        dtau0 = dt / u4_tmp[0,:]
        n = 0
        nmax = 100

        while (n<nmax):

            t0 = dtau0
            t1 = t0 * wb * dtau0 / 2.0
            t2 = t1 * wb * dtau0 / 3.0
            f = u4_tmp[0,:]*t0 + u4_tmp[3,:]*t1 + (u4_tmp[0,:]-u4_tmp[2,:])*t2 - dt

            t1 = t1 * 2.0 / dtau0
            t2 = t2 * 3.0 / dtau0
            df = u4_tmp[0,:] + u4_tmp[3,:]*t1 + (u4_tmp[0,:]-u4_tmp[2,:])*t2

            dtau1 = dtau0 - f / df
            n = n + 1

            if ( np.all( np.abs(dtau1 - dtau0) / dtau0 < 1e-12 ) ):
                break
            else:
                dtau0 = dtau1

        return dtau1

    e_amp = np.sqrt( np.sum( np.square(ep), axis=0 ) )
    b_amp = np.sqrt( np.sum( np.square(bp), axis=0 ) )
    inds = e_amp==0
    e_amp[inds] = 1; b_amp[inds] = 1 # Just to avoid dividing by zero

    # 4-vector of momentum
    u4 = np.zeros((4,p_in.shape[1]))
    u4[1:,:] = p_in
    u4[0,:] = np.sqrt( 1.0 + np.sum( np.square(p_in), axis=0 ) )

    # Always assume light-like fields
    # Rotate the coordinate system so that B is along x1, and E is along x3
    u4_tmp = lorentz_rotate2( sign=-1 )

    wb = b_amp / rqm
    wb[inds] = 0

    # Calculate the proper time step
    dtau = proper_dt_light()

    # Advance 4-vector of momentum
    s = wb * dtau
    c = np.square(s) / 2
    u4[0,:] = u4_tmp[0,:] + (u4_tmp[0,:] - u4_tmp[2,:]) * c + u4_tmp[3,:] * s
    u4[1,:] = u4_tmp[1,:]
    u4[2,:] = u4_tmp[2,:] + (u4_tmp[0,:] - u4_tmp[2,:]) * c + u4_tmp[3,:] * s
    u4[3,:] = u4_tmp[3,:] + (u4_tmp[0,:] - u4_tmp[2,:]) * s

    # Rotate the coordinate system back to the simulation frame
    u4_tmp = lorentz_rotate2( sign=1 )

    # If E = B = 0, then we just return the original momentum
    u4_tmp[1:,inds] = p_in[:,inds]

    return u4_tmp[1:,:]

def dudt_vay( p_in, ep, bp ):

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

def dudt_cary( p_in, ep, bp ):

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

def dudt_fullrot( p_in, ep, bp ):

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

def dudt_euler( p_in, ep, bp ):

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

    p = dudt( p_in, e(x[0,:],x[1,:],n), b(x[0,:],n) )

    rgamma = 1.0 / np.sqrt( 1.0 + np.sum( np.square(p), axis=0 ) )

    x = x + p[0:2,:] * rgamma * dt

    return x, p


if __name__ == "__main__":
    main()

