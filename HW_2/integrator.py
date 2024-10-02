import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from mpl_toolkits.mplot3d import Axes3D
import astropy.units as u
import astropy.constants as const
@njit
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
    U = the array of time derivatives of the elements of w.
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

    u = np.zeros(6 * N)
    for j in range(N):
        u[j] = w[3 * N + j]  # v_x
        u[N + j] = w[4 * N + j]  # v_y
        u[2 * N + j] = w[5 * N + j]  # v_z
        u[3 * N + j] = np.sum(A[j, :, 0])  # a_x
        u[4 * N + j] = np.sum(A[j, :, 1])  # a_y
        u[5 * N + j] = np.sum(A[j, :, 2])  # a_z

    return u
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


@njit
def euler_cromer_step(dt, w, m):
    """
    Perform a single step of the Euler-Cromer method.
    v_{n+1} = v_n + dt * a_n
    x_{n+1} = x_n + dt * v_{n+1}
    """
    N = int(len(w) / 6)
    u = f_Nbody_3d(w, m)
    w[3 * N:] = w[3 * N:] + dt * u[3 * N:]  # Update velocities
    w[:3 * N] = w[:3 * N] + dt * w[3 * N:]  # Update positions

    return w


@njit
def euler_cromer_adaptive_step(dt, w, m, tol, dt_guess):
    """
    Perform one or more steps of the Euler-Cromer method with adaptive step size spanning dt.
    Returns the final state and the last used step size.

    We implement an adaptive step size algorithm that takes two half steps and compares the results to a full step.
    If the error is within the tolerance, we accept the smaller step. If the error is too large, we reduce the step
    size and repeat.
    """
    t = 0
    dt_current = min(dt_guess, dt)  # Ensure initial guess doesn't exceed target dt
    i = 0

    w0 = np.copy(w)
    w1 = np.copy(w)
    w2 = np.copy(w)

    while t < dt:
        #store the last step
        w0 = np.copy(w)
        t0 = t
        w1 = np.copy(w)
        w2 = np.copy(w)

        # Take a full step
        w1 = euler_cromer_step(dt_current, w1, m)

        # Take two half steps
        w2 = euler_cromer_step(dt_current / 2, w2, m)
        w2 = euler_cromer_step(dt_current / 2, w2, m)

        # Compute error
        error = np.abs(w1 - w2).max()

        #compute energy error for adaptive step size too between the beginning of the step and the end of the step
        #K0, U0 = total_energy(w0, m)
        #K1, U1 = total_energy(w2, m)
        #energy_error = np.abs(K0 + U0 - K1 - U1)#/np.abs(K0 + U0)
        #energy_error = 0
        #print(error,dt_current)
        #energy error should be a little more forgiving than the position error

        i = i + 1
        if error <= tol: #and energy_error < tol:
            # Accept the smaller step
            w = w2
            w1 = np.copy(w2)
            t += dt_current
            # Increase step size if error is much smaller than tolerance
            if error < tol / 10:
                dt_current = min(dt_current * 2, dt)
        else:
            # Reduce step size
            dt_current = dt_current / 2
            if dt_current < 1e-10:
                raise ValueError("Step size underflow")
        if i > 10**7:
            raise ValueError("Too many iterations")
        #also make sure dt does not get too small
        if dt_current < 1e-10:
            raise ValueError("Step size underflow")

    #check if time t is exactly dt within floating point error
    if np.isclose(t, dt, atol=1e-10):
        return w , dt_current
    elif t > dt:
        #use last stored step and time and take exact step to dt
        dt = dt - t0
        w = euler_cromer_step(dt, w0, m)
        return w, dt_current
    else:
        raise ValueError("Time t is less than dt")

@njit
def evolve_system(dt, t_final, w0, m,tol):
    """
    Evolve the system using the Euler-Cromer method with a function to adapt step size for each output time.
    """
    assert t_final > dt > 0 # Ensure valid input
    N = int(len(w0) / 6)
    n_steps = int(t_final / dt)+1
    w = np.zeros((n_steps, len(w0)))
    w[0] = w0
    dt_guess = dt
    t = np.arange(0, t_final, dt)
    dt_guesses = np.ones(n_steps) *np.nan



    for i in range(1, n_steps):
        #only save times for the specified time steps
        w[i], dt_guess = euler_cromer_adaptive_step(dt, w[i - 1], m, tol, dt_guess)
        dt_guesses[i] = dt_guess


    return w, t, dt_guesses
