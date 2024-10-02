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

markersize = 50

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories_2d(W, N, T, IC):
    if N == 5:
        fig, ax = plt.subplots(1, 2, figsize=(19, 10), dpi=300, sharey=True)
        plt.rcParams.update(plt.rcParamsDefault)

        for i in range(N):
            x = W[:, i]
            y = W[:, N + i]
            z = W[:, 2 * N + i]
            kwargs = plot_kwargs[i]
            print(f'x {x[0]}, y {y[0]}, z {z[0]}')

            # Plot lines
            line0, = ax[0].plot(x, y, color=kwargs['color'], linestyle=kwargs['linestyle'], label=f'Star {i + 1}')
            line1, = ax[1].plot(x, z, color=kwargs['color'], linestyle=kwargs['linestyle'])

            # Plot markers
            ax[0].scatter(x[0], y[0], s=markersize, color=kwargs['facecolor'],
                          marker=kwargs['marker'], edgecolors=kwargs['color'])
            ax[1].scatter(x[0], z[0], s=markersize, color=kwargs['facecolor'],
                          marker=kwargs['marker'], edgecolors=kwargs['color'])

            # Update line's legend handle to include both line and marker
            line0.set_markerfacecolor(kwargs['facecolor'])
            line0.set_markeredgecolor(kwargs['color'])
            line0.set_marker(kwargs['marker'])
            line0.set_markersize(np.sqrt(markersize))
            line0.set_markevery([0])  # To match the scatter plot

        for a in ax:

            a.set_ylabel('Y' if a == ax[0] else 'Z', fontsize=30)
            a.legend(fontsize=20)
            a.set_aspect('equal')
        ax[1].set_xlabel('X', fontsize=30)

    elif N == 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)

        for i in range(N):
            x = W[:, i]
            y = W[:, N + i]
            kwargs = plot_kwargs[i]

            # Plot line
            line, = ax.plot(x, y, color=kwargs['color'], linestyle=kwargs['linestyle'], label=f'Star {i + 1}')

            # Plot markers
            ax.scatter(x[0], y[0], s=markersize, color=kwargs['facecolor'],
                       marker=kwargs['marker'], edgecolors=kwargs['color'])

            # Update line's legend handle to include both line and marker
            line.set_markerfacecolor(kwargs['facecolor'])
            line.set_markeredgecolor(kwargs['color'])
            line.set_marker(kwargs['marker'])
            line.set_markersize(np.sqrt(markersize))
            line.set_markevery([0])  # To match the scatter plot

        ax.set_xlabel('X', fontsize=30)
        ax.set_ylabel('Y', fontsize=30)
        ax.legend(fontsize=20)
        ax.set_aspect('equal')

    #save plots
    fig.tight_layout()

    # Energy error plot
    n_out_tot = T.size
    E = np.zeros(n_out_tot) * np.nan
    for j in range(n_out_tot):
        w = W[j, :]
        K, U = total_energy(w, IC['masses'])
        E[j] = K + U

    E_error = abs(E - E[0]) / abs(E[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
    ax.plot(T, E_error)
    ax.set_xlabel('Time')
    ax.set_ylabel('Relative Error in Energy')
    ax.set_yscale('log')  # Use log scale for better visualization of small errors

    plt.show()



# anim = animate_trajectories(W, N, T, IC)
# plt.show()
# To save: anim.save('trajectory_animation.mp4', writer='ffmpeg', fps=30)

# Usage example:
# anim = animate_trajectories(W, N, T, IC)
# anim.save('trajectory_animation.mp4', writer='ffmpeg', fps=30)
# plt.show()
