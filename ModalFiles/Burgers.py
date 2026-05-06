import numpy as np
import matplotlib.pyplot as plt


NU = 0.001
X_LEFT = -5.0
X_RIGHT =  5.0
T_END =  5.0
NX = 500
PLOT_SLICES = False

x  = np.linspace(X_LEFT, X_RIGHT, NX)
dx = x[1] - x[0]
dt = 0.45 * min(dx / 1.0, dx**2 / (2 * NU))
NT = int(T_END / dt) + 1

print(f'ν = {NU},  dx = {dx:.3f},  dt = {dt:.3f}')

u = np.exp(-((x - 1)**2) / 2) - np.exp(-((x + 1)**2) / 2)

u_history  = []
t_history  = []

u_curr = u.copy()

for n in range(NT):
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

if PLOT_SLICES:
    N_SLICES = 10
    slice_ids = np.linspace(0, len(t_history) - 1, N_SLICES, dtype=int)

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    cmap_slices = plt.cm.viridis
    colors = cmap_slices(np.linspace(0, 1, N_SLICES))

    for colour, idx in zip(colors, slice_ids):
        ax2.plot(x, u_history[idx], color=colour, label=f't = {t_history[idx]:.2f}')

    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x, t)')
    ax2.set_title('1D Burgers Equation Time Slices')

    ax2.legend(loc='upper right', title='Time')
    ax2.set_xlim(X_LEFT, X_RIGHT)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


#Hell reigns below
###################################################################

PLOT_AMPLITUDES = True
PLOTTING    = True
PLOT_MODES  = True
PRINT_XI    = True
N_MODES     = 7
TRAIN_FRAC  = 0.3

t = t_history
u = u_history
n = len(x)

print(f"U Shape: {np.shape(u_history)}")

modes = np.fft.fft(u, axis=1)

U_hat_filtered = np.zeros_like(modes, dtype=complex)
U_hat_filtered[:, 1:N_MODES+1] = modes[:, 1:N_MODES+1]
U_reconstructed = np.fft.ifft(U_hat_filtered, n=n, axis=1).real

a_complex = modes[:, 1:N_MODES+1]
a_real = a_complex.real
a_imag = a_complex.imag
a = np.hstack([a_real, a_imag])

if PLOT_MODES:
    for mode in range(N_MODES):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t, a[:, mode])
        ax.set_xlabel("Time")
        ax.set_ylabel("Mode amplitude")
        ax.set_title(f"Mode {mode}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

N_total = len(t)
N_train = int(N_total * TRAIN_FRAC)

a_train = a[:N_train]
t_train = t[:N_train]


def compute_time_derivative(a, dt):
    da = np.zeros_like(a)
    da[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
    da[0] = (a[1]  - a[0])   / dt
    da[-1] = (a[-1] - a[-2])  / dt
    return da


a_dot_train = compute_time_derivative(a_train, dt)
v_train = a_dot_train
v_dot_train = compute_time_derivative(v_train, dt)

X_train     = np.hstack([a_train, v_train])
X_dot_train = np.hstack([v_train, v_dot_train])


def build_library(X):
    n_cols  = X.shape[1]
    library = [X[:, i] for i in range(n_cols)]
    return np.column_stack(library)


if PLOTTING:
    fig1 = plt.figure(figsize=(11, 6))
    Xm, Tm = np.meshgrid(x, t)

    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(Xm, Tm, u, cmap='plasma')
    ax1.set_xlabel('x'); ax1.set_ylabel('t'); ax1.set_zlabel('u(x, t)')
    ax1.set_title('Burgers Equation Solution')

    ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(Xm, Tm, U_reconstructed, cmap='plasma')
    ax2.set_xlabel('x'); ax2.set_ylabel('t'); ax2.set_zlabel('u(x, t)')
    ax2.set_title(f'Reconstruction ({N_MODES} modes)')

    plt.tight_layout()
    plt.show()