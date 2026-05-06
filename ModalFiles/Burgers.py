import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

NU = 0.001
X_LEFT = -5.0
X_RIGHT =  5.0
T_END =  5.0
NX = 500
NT_PLOT = 40

x  = np.linspace(X_LEFT, X_RIGHT, NX)
dx = x[1] - x[0]
dt = 0.45 * min(dx / 1.0, dx**2 / (2 * NU))
NT = int(T_END / dt) + 1

u = np.exp(-((x - 1)**2) / 2) - np.exp(-((x + 1)**2) / 2)

plot_every = max(1, NT // NT_PLOT)
u_history  = []
t_history  = []

u_curr = u.copy()

for n in range(NT):
    if n % plot_every == 0:
        u_history.append(u_curr.copy())
        t_history.append(n * dt)

    u_pos = np.maximum(u_curr, 0)
    u_neg = np.minimum(u_curr, 0)

    conv = (u_pos * (u_curr - np.roll(u_curr,  1)) +
            u_neg * (np.roll(u_curr, -1) - u_curr)) / dx

    diff = NU * (np.roll(u_curr, -1) - 2*u_curr + np.roll(u_curr, 1)) / dx**2

    u_new = u_curr - dt * conv + dt * diff
    u_new[0]  = u[0]
    u_new[-1] = u[-1]
    u_curr = u_new

u_history.append(u_curr.copy())
t_history.append(NT * dt)

u_history = np.array(u_history)
t_history = np.array(t_history)

X_mesh, T_mesh = np.meshgrid(x, t_history)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    X_mesh, T_mesh, u_history,
    cmap='plasma'
)

ax.set_xlabel('x')
ax.set_ylabel('time t')
ax.set_zlabel('u(x,t)')
ax.set_title(f'1D Burgers Equation\n'
             f'ν = {NU},  Δx = {dx:.4f},  Δt = {dt:.6f},  CFL ≈ {dt/dx:.3f}')

ax.view_init(elev=28, azim=-55)
plt.tight_layout()
plt.show()


N_SLICES = 10
slice_ids = np.linspace(0, len(t_history) - 1, N_SLICES, dtype=int)

fig2, ax2 = plt.subplots(figsize=(11, 6))
cmap_slices = plt.cm.viridis
colors = cmap_slices(np.linspace(0, 1, N_SLICES))

for colour, idx in zip(colors, slice_ids):
    ax2.plot(x, u_history[idx], color=colour, linewidth=1.8,
             label=f't = {t_history[idx]:.2f}')

ax2.set_xlabel('x', fontsize=13)
ax2.set_ylabel('u(x, t)', fontsize=13)
ax2.set_title(
    '1D Burgers Equation Time Slices'
)

ax2.legend(loc='upper right', title='Time')
ax2.set_xlim(X_LEFT, X_RIGHT)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()