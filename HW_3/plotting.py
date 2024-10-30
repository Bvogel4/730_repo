import numpy as np
from integrator import total_energy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider, Button
from matplotlib.widgets import TextBox
mpl.use('Qt5Agg')



plot_kwargs = {
    0: {'marker': 'o', 'color': 'black', 'linestyle': '-', 'facecolor': 'none',  },
    1: {'marker': 's', 'color': 'red', 'linestyle': '-', 'facecolor': 'none', },
    2: {'marker': '^', 'color': 'blue', 'linestyle': '-', 'facecolor': 'blue',  },
    3: {'marker': '^', 'color': 'purple', 'linestyle': '-', 'facecolor': 'none', },
    4: {'marker': 's', 'color': 'black', 'linestyle': '--', 'facecolor': 'black',  }
}

markersize = 50

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.widgets import TextBox


def plot_interactive_time_range(W, N, T):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 10))
    plt.subplots_adjust(bottom=0.35)  # Increased bottom margin for controls

    lines1 = []
    lines2 = []
    scatters1 = []
    scatters2 = []

    for i in range(N):
        x = W[:, i]
        y = W[:, N + i]
        z = W[:, 2 * N + i]
        kwargs = plot_kwargs[i]

        line1, = ax1.plot(x, y, color=kwargs['color'])
        line2, = ax2.plot(x, z, color=kwargs['color'])
        scatter1 = ax1.scatter(x[0], y[0], s=markersize, color=kwargs['facecolor'],
                               marker=kwargs['marker'], edgecolors=kwargs['color'], label=f'Star {i + 1}')
        scatter2 = ax2.scatter(x[0], z[0], s=markersize, color=kwargs['facecolor'],
                               marker=kwargs['marker'], edgecolors=kwargs['color'])

        lines1.append(line1)
        lines2.append(line2)
        scatters1.append(scatter1)
        scatters2.append(scatter2)

    #create legend for just scatter plots


    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.legend()

    # Create sliders for start and end times
    slider_start_ax = plt.axes([0.15, 0.2, 0.7, 0.03])
    slider_end_ax = plt.axes([0.15, 0.15, 0.7, 0.03])

    # Convert index values to actual time values
    time_min, time_max = T[0], T[-1]
    print(f'time min {time_min}, time max {time_max}')

    slider_start = Slider(slider_start_ax, 'Start Time', time_min, time_max,
                          valinit=time_min, valfmt='%.2f t_dyn')
    slider_end = Slider(slider_end_ax, 'End Time', time_min, time_max,
                        valinit=time_max, valfmt='%.2f t_dyn')

    # Add text boxes for precise input
    text_start_ax = plt.axes([0.15, 0.08, 0.2, 0.04])
    text_end_ax = plt.axes([0.65, 0.08, 0.2, 0.04])
    text_start = TextBox(text_start_ax, 'Start Time:', initial=f"{time_min:.2f}")
    text_end = TextBox(text_end_ax, 'End Time:', initial=f"{time_max:.2f}")

    def find_nearest_idx(array, value):
        return np.abs(array - float(value)).argmin()

    def update(val=None):
        start_time = slider_start.val
        end_time = slider_end.val

        # Find nearest indices in T array
        start_idx = find_nearest_idx(T, start_time)
        end_idx = find_nearest_idx(T, end_time)

        # Ensure end time is not before start time
        if end_idx < start_idx:
            end_idx = start_idx
            slider_end.set_val(T[end_idx])

        for i in range(N):
            lines1[i].set_data(W[start_idx:end_idx + 1, i],
                               W[start_idx:end_idx + 1, N + i])
            lines2[i].set_data(W[start_idx:end_idx + 1, i],
                               W[start_idx:end_idx + 1, 2 * N + i])
            scatters1[i].set_offsets(np.column_stack((W[end_idx, i],
                                                      W[end_idx, N + i])))
            scatters2[i].set_offsets(np.column_stack((W[end_idx, i],
                                                      W[end_idx, 2 * N + i])))

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw_idle()

    def on_text_start_submit(text):
        try:
            value = float(text)
            if time_min <= value <= time_max:
                slider_start.set_val(value)
            else:
                text_start.set_val(f"{slider_start.val:.2f}")
        except ValueError:
            text_start.set_val(f"{slider_start.val:.2f}")

    def on_text_end_submit(text):
        try:
            value = float(text)
            if time_min <= value <= time_max:
                slider_end.set_val(value)
            else:
                text_end.set_val(f"{slider_end.val:.2f}")
        except ValueError:
            text_end.set_val(f"{slider_end.val:.2f}")

    slider_start.on_changed(update)
    slider_end.on_changed(update)
    text_start.on_submit(on_text_start_submit)
    text_end.on_submit(on_text_end_submit)

    # Reset button
    reset_ax = plt.axes([0.45, 0.025, 0.1, 0.04])
    reset_button = Button(reset_ax, 'Reset')

    def reset(event):
        slider_start.reset()
        slider_end.reset()
        text_start.set_val(f"{time_min:.2f}")
        text_end.set_val(f"{time_max:.2f}")

    reset_button.on_clicked(reset)

    plt.show()


# Usage example:
# plot_interactive_time_range(W, N, T)




# Usage example:
# plot_interactive_time_range(W, N, T)




def plot_trajectories_2d(W, N, T, IC,t_dyn,dt):
    T = T / t_dyn # Convert time to dynamical time
    if N == 5:
        fig, ax = plt.subplots(1, 2, figsize=(19, 10), dpi=300, sharey=True)
        plt.rcParams.update(plt.rcParamsDefault)

        for i in range(N):
            x = W[:, i]
            y = W[:, N + i]
            z = W[:, 2 * N + i]
            kwargs = plot_kwargs[i]
            #print(f'x {x[0]}, y {y[0]}, z {z[0]}')

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
            #a.legend(fontsize=20)
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
        #ax.legend(fontsize=20)
        ax.set_aspect('equal')

    #save plots
    fig.tight_layout()

    # Energy error plot
    n_out_tot = T.size
    #E = np.zeros(n_out_tot) * np.nan

    K = np.zeros(n_out_tot) * np.nan
    U = np.zeros(n_out_tot) * np.nan
    for j in range(n_out_tot):
        w = W[j, :]
        K[j],U[j] = total_energy(w, IC['masses'])
    E = K + U

    E_error = abs(E - E[0]) / abs(E[0])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
    ax.plot(T, E_error)
    ax.set_xlabel('Time in dynamical time')
    ax.set_ylabel('Relative Error in Energy')
    ax.set_yscale('log')  # Use log scale for better visualization of small errors

    plt.show()

    #create figure showing 6 plots for different time ranges  (2 subplots for each time) x,y and x,z
    #split arrays into different time ranges

    plot_interactive_time_range(W, N, T)

    # create plot of -K/U
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), dpi=300)


    ax.plot(T, K/-U, label='-K/U',lw=0.5)

    ax.set_xlabel(f'Time in dynamical time: ')
    ax.legend()
    #ax.set_yscale('log')


    plt.suptitle(f'Average -K/U = {np.mean(K/-U):.2f}')
    plt.show()
