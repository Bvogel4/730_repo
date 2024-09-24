import matplotlib.pyplot as plt
import numpy as np
from integrator import total_energy




plot_kwargs = {
    0: {'marker': 'o', 'color': 'black', 'linestyle': '-', 'facecolor': 'none',  },
    1: {'marker': 's', 'color': 'red', 'linestyle': '-', 'facecolor': 'none', },
    2: {'marker': '^', 'color': 'blue', 'linestyle': '-', 'facecolor': 'blue',  },
    3: {'marker': '^', 'color': 'purple', 'linestyle': '-', 'facecolor': 'none', },
    4: {'marker': 's', 'color': 'black', 'linestyle': '--', 'facecolor': 'black',  }
}


def plot_trajectories_2d(W, N, T,IC):
    # Use plt.subplots with constrained_layout for better spacing

    if N == 5:
        fig, ax = plt.subplots(2, 1, figsize=(10, 18), dpi=100,sharex=True)

        # Reset to matplotlib defaults
        plt.rcParams.update(plt.rcParamsDefault)

        for i in range(N):
            x = W[:, i]
            y = W[:, N + i]
            z = W[:, 2 * N + i]

            kwargs = plot_kwargs[i]

            ax[0].plot(x, y, label=f'Star {i + 1}', color=kwargs['color'], linestyle=kwargs['linestyle'])
            ax[1].plot(x, z, label=f'Star {i + 1}', color=kwargs['color'], linestyle=kwargs['linestyle'])
            #only plot every 10th point
            ax[0].scatter(x[::100], y[::100], **kwargs)
            ax[1].scatter(x[::100], z[::100], **kwargs)

        for a in ax:
            a.set_xlabel('X', fontsize=30)
            a.set_ylabel('Y' if a == ax[0] else 'Z', fontsize=30)
            a.legend(fontsize=20)
            a.set_aspect('equal')

    if N == 2:

        scatter_kwarg = {
            0: {'marker': 'o', 'color': 'black', 'linestyle': '--'},
            1: {'marker': 'o', 'color': 'red' , 'linestyle': 'dotted'}
        }
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        for i in range(N):
            x = W[:, i]
            y = W[:, N + i]

            kwargs = scatter_kwarg[i]

            ax.plot(x, y, label=f'Star {i + 1}', color = kwargs['color'], linestyle = kwargs['linestyle'],marker='None')
            #only plot every 10th point
            ax.scatter(x[::100], y[::100], color = kwargs['color'], marker = kwargs['marker'], linestyle = 'None')





    n_out_tot = T.size


    #create another fig for error in energy
    E = np.zeros(n_out_tot)*np.nan
    for j in range(n_out_tot):
        w = W[j, :]
        K, U = total_energy(w, IC['masses'])
        E[j] = K + U

    #get the relative error
    E_error = abs(E - E[0])/E[0]


    fig,ax = plt.subplots()
    ax.plot(T, abs(E - E[0])/E[0])
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Error in Energy')
    plt.show()


