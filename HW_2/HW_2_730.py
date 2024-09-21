import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
import astropy.units as u
import astropy.constants as const
from integrator import euler_cromer
from plotting import plot_trajectories_2d

import numpy as np

def total_energy(w, m):
    """
    Find the total kinetic energy K and the total potential energy U
    given the array m of masses and the array w of position and velocity
    components in 3D.
    """
    N = int(w.size / 6)  # Now 6 components per body (x, y, z, vx, vy, vz)
    v_x = w[3*N: 4*N]
    v_y = w[4*N: 5*N]
    v_z = w[5*N: 6*N]
    v2 = v_x**2 + v_y**2 + v_z**2
    K = 0.5 * np.dot(m, v2)

    U = 0.0
    for j in range(N-1):
        for k in range(j+1, N):
            x_kj = w[k] - w[j]
            y_kj = w[N+k] - w[N+j]
            z_kj = w[2*N+k] - w[2*N+j]
            r = np.sqrt(x_kj**2 + y_kj**2 + z_kj**2)
            U = U - m[j]*m[k]/r

    return K, U




#part c
def main(initial_conditions):
    N = initial_conditions['N']
    m = np.array(initial_conditions['masses'], dtype=np.float64)

    # Construct w0 from initial conditions
    w0 = np.zeros(6 * N)
    w0[:N] = initial_conditions['x']
    w0[N:2 * N] = initial_conditions['y']
    w0[2 * N:3 * N] = initial_conditions['z']
    w0[3 * N:4 * N] = initial_conditions['vx']
    w0[4 * N:5 * N] = initial_conditions['vy']
    w0[5 * N:] = initial_conditions['vz']

    dT = initial_conditions['dT']
    tp = initial_conditions['tp']

    w, T = euler_cromer(dT, tp, w0, m)

    plot_trajectories_2d(w, N)

    return w, T




if __name__ == "__main__":
    # Example initial conditions for a 3D three-body system
    N = 2

    # noinspection PyDictCreation
    IC = {
        'N': 2,
        'masses': np.ones(N, dtype=np.float64),  # units of solar masses
        'x': np.zeros(N, dtype=np.float64),  # units of AU
        'y': np.zeros(N, dtype=np.float64),
        'z': np.zeros(N, dtype=np.float64),
        'vx': np.zeros(N, dtype=np.float64),  # units of AU/year
        'vy': np.zeros(N, dtype=np.float64),
        'vz': np.zeros(N, dtype=np.float64),
        'dT': .01,  # units of years
        'tp': 30  # units of years
    }
    IC['y'] = np.array([10, -10])

    #calculate the velocity needed to have a circular orbit
    G = 4 * np.pi ** 2  # AU^3 / (M_sun * year^2)
    M = 1  # Total mass of the system in solar masses (1 + 1)
    r = 10  # Orbital radius in AU
    v0 = np.sqrt(G * M / (4 * r))

    T = 2 * np.pi * r / v0
    IC['T'] = T
    IC['vx'] = np.array([-v0, v0])


    #units of stars are in solar masses, units of distances are in AU, units of time are in years
    #make dimensionsless, so far we have 1 solar mass, 1 AU, 1 year
    #from keplers third law T^2 = 4pi^2 a^3 / GM
    #In units of solar masses, AU, and years, G = 4pi^2

    # so to make G = 1, we divide by mass by  4pi^2
    mass_conversion = 4*np.pi**2
    IC['masses'] = (IC['masses'])*mass_conversion

    W,T = main(IC)
    n_out_tot = T.size
    E = np.zeros(n_out_tot)
    for j in range(n_out_tot):
        w = W[j, :]
        K, U = total_energy(w, IC['masses'])
        E[j] = K + U

    #get the relative error
    E_error = abs(E - E[0])/E[0]

    fig,ax = plt.subplots()
    ax.plot(T, abs(E - E[0])/E[0])
    ax.set_xlabel('Time')

    #get energy error from beginning to end
    print(E_error[0], E_error[-1])
    plt.show()




# will need to come up with a stratgey to use an adaptive time step






