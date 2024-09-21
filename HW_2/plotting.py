import matplotlib.pyplot as plt

def plot_trajectories_3d(w, N):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(N):
        ax.plot(w[:, i], w[:, N + i], w[:, 2 * N + i], label=f'Body {i + 1}')

    # Plot the initial positions
    ax.scatter(w[0, :N], w[0, N:2 * N], w[0, 2 * N:3 * N], c='r', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectories of Bodies')
    ax.legend()
    plt.show()


def plot_trajectories_2d(w, N):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    for i in range(N):
        ax.plot(w[:, i], w[:, N + i], label=f'Body {i + 1}')

    # Plot the initial positions
    ax.scatter(w[0, :N], w[0, N:2 * N], c='r', s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Trajectories of Bodies')
    ax.legend()
    plt.show()
