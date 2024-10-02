import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
import astropy.units as u
import astropy.constants as const
from integrator import euler_cromer
from integrator import evolve_system
from plotting import plot_trajectories_2d
import time
import pickle

import numpy as np


def t_dyn(IC):
    m = np.array(IC['masses'], dtype=np.float64)
    r = np.sqrt(IC['x'] ** 2 + IC['y'] ** 2 + IC['z'] ** 2)
    r = np.average(r)
    return np.sqrt(r ** 3 / m.sum())

def circular_orbit_velocity(IC):
    M = IC['masses'][0]
    r = np.sqrt(IC['x'] ** 2 + IC['y'] ** 2 + IC['z'] ** 2)
    r = r[0]
    return np.sqrt( M / (4* r))


#part c
def run_simulation(initial_conditions,load=False):
    """
    Run the simulation for the given initial conditions
    save and load the trajectory
    :param initial_conditions:
    :param load:
    :return:
    """
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
    tol = 1e-8

    if load == True:
        try:
            with open(f'N_{N}_dT_{dT}_tp_{tp}.pkl', 'rb') as f:
                w,T = pickle.load(f)
        except:
            print('File not found, running simulation')
            w, T,dt = evolve_system(dT, tp, w0, m, tol)
    else:
        w, T,dt = evolve_system(dT, tp, w0, m, tol)

    #save the trajectory
    #store w, T in a pickle file
    with open(f'N_{N}_dT_{dT}_tp_{tp}.pkl', 'wb') as f:
        pickle.dump([w,T], f)

    plot_trajectories_2d(w, N,T,initial_conditions)

    plt.show()

    return w, T


IC_list = [
    {
        'N': 2,
        'masses': np.ones(2, dtype=np.float64),
        'x': np.zeros(2, dtype=np.float64),
        'y': np.array([10, -10]),
        'z': np.zeros(2, dtype=np.float64),
        'vx': np.zeros(2, dtype=np.float64),
        'vy': np.zeros(2, dtype=np.float64),
        'vz': np.zeros(2, dtype=np.float64),
        'dT': 1,
        'tp': 30
    },
    {
        'N': 5,
        'masses': np.ones(5, dtype=np.float64),
        'x': np.array((-1,9,-11,4,-1), dtype=np.float64),
        'y': np.array((9,-1,-11,-1,4), dtype=np.float64),
        'z': np.array((-1,-1,4,-6,4), dtype=np.float64),
        'vx': np.array((-0.7,0.3,0.8,-0.7,0.3), dtype=np.float64),
        'vy': np.array((0.1,1.1,-0.4,0.1,-0.9), dtype=np.float64),
        'vz': np.zeros(5, dtype=np.float64),
        'dT': 0.5,
        'tp': 20
    }
]

if __name__ == "__main__":
    # compute time to run code
    start = time.time()
    #units of stars are in solar masses, units of distances are in AU, units of time are in years
    #make dimensionsless, so far we have 1 solar mass, 1 AU, 1 year
    #from keplers third law T^2 = 4pi^2 a^3 / GM
    #In units of solar masses, AU, and years, G = 4pi^2
    # so to make G = 1, we multiply by mass by  4pi^2
    #run simulation for each set of initial conditions
    for IC in IC_list:
        t_d = t_dyn(IC)
        IC['tp'] = 3 * t_d
        IC['dT'] = t_d / 2000
        IC['masses'] = IC['masses'] * (4 * np.pi ** 2)
        #also run with lower v0

        if IC['N'] == 2:
            print(t_d)
            v0 = circular_orbit_velocity(IC)
            print('v0:', v0)
            IC['vx'] = np.array([v0, -v0], dtype=np.float64)
            T = 2 * np.pi / v0 * 10
            IC['tp'] = T
            run_simulation(IC)
            IC['vx'] = np.array([v0/2, -v0/2], dtype=np.float64)
            run_simulation(IC)
        else:
            run_simulation(IC)
    print('Time to run code:', time.time() - start)








