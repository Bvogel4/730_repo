import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
import astropy.units as u
import astropy.constants as const

@njit
def f_Nbody_3d(w, m):
    """
    Evaluates the evolution function for the gravitational N-body problem
    in 3D space.
    Input:
    w = the array of positions and velocities for N bodies:
    w = (x_0, ..., x_(N-1), y_0, ..., y_(N-1), z_0, ..., z_(N-1),
         v_x0, ..., v_x(N-1), v_y0, ..., v_y(N-1), v_z0, ..., v_z(N-1))
    m = the array of masses (in units with G=1) for the N bodies

    Output:
    f = the array of time derivatives of the elements of w.
    """

    N = int(len(w) / 6)
    A = np.zeros((N, N, 3))

    for j in range(N - 1):
        for k in range(j + 1, N):
            x_kj = w[k] - w[j]
            y_kj = w[N + k] - w[N + j]
            z_kj = w[2 * N + k] - w[2 * N + j]
            r_kj3 = (x_kj ** 2 + y_kj ** 2 + z_kj ** 2) ** 1.5
            q = np.array([x_kj, y_kj, z_kj]) / r_kj3
            A[j, k] = m[k] * q
            A[k, j] = -m[j] * q

    f = np.zeros(6 * N)
    for j in range(N):
        f[j] = w[3 * N + j]  # v_x
        f[N + j] = w[4 * N + j]  # v_y
        f[2 * N + j] = w[5 * N + j]  # v_z
        f[3 * N + j] = np.sum(A[j, :, 0])  # a_x
        f[4 * N + j] = np.sum(A[j, :, 1])  # a_y
        f[5 * N + j] = np.sum(A[j, :, 2])  # a_z

    return f
@njit
def euler_cromer(dT, tfinal, w0, m):
    """
    Performs the Euler-Cromer integration
    """
    N = int(len(w0) / 6)
    T = np.arange(0, tfinal, dT)
    w = np.zeros((len(T), len(w0)))
    w[0] = w0

    for i in range(1, len(T)):
        f = f_Nbody_3d(w[i - 1], m)
        w[i, 3 * N:] = w[i - 1, 3 * N:] + dT * f[3 * N:]  # Update velocities
        w[i, :3 * N] = w[i - 1, :3 * N] + dT * w[i, 3 * N:]  # Update positions

    return w, T


import numpy as np



def euler_cromer_step(dt, w, m):
    """
    Perform a single step of the Euler-Cromer method.
    """
    N = int(len(w) / 6)
    f = f_Nbody_3d(w, m)
    w[3 * N:] = w[3 * N:] + dt * f[3 * N:]  # Update velocities
    w[:3 * N] = w[:3 * N] + dt * w[3 * N:]  # Update positions

    return w


def evolve_timestep(dt, t_final, w0, m):
    """
    Use an adaptive timestep by taking two time steps of size dt/2 and dt
    if error is less than tolerance, keep the result, otherwise, reduce the step size
    interpolate the results to get the final result as an ordered array of times and positions
    """
    N = int(len(w0) / 6)
    n_steps = int(t_final / dt)
    w = np.zeros((n_steps, len(w0)))
    w[0] = w0
    #keep track of our timesteps
    t = 0
    T = np.arange(0, t_final, dt)
    i = 0


    while t < t_final:
        w_half = euler_cromer_step(dt / 2, w[i - 1], m)
        w_full = euler_cromer_step(dt, w_half, m)
        if np.linalg.norm(w_full - w_half) < 1e-10:
            t=t+dt
        else:
            dt = dt / 2



    return w
