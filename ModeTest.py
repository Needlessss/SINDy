import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PDE_FIND import STRidge
from pysindy.optimizers import FROLS

PLOTTING = False
PLOT_MODES = True
PRINT_XI = False
N_MODES = 5
METHOD = "FROLS"  # Options: STRidge, FROLS
TRAIN_FRAC = 0.3
NORMALISE = False


def solve_wave_equation(
    c=1.0,
    L=2.0,
    N=200,
    n_steps=1000,
    sigma=0.15,
    x0=0.0,
):
    dx = L / N
    dt = 0.4 * dx / c
    r = c * dt / dx

    x = np.linspace(-L/2, L/2, N + 1)
    t = np.arange(n_steps + 1) * dt

    u = np.zeros((n_steps + 1, N + 1))
    u[0] = np.exp(-0.5 * ((x - x0) / sigma) ** 2)

    u[1, 1:-1] = (
        u[0, 1:-1]
        + 0.5 * r**2 * (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2])
    )
    u[1, 0] = u[1, -1] = 0.0

    for n in range(1, n_steps):
        u[n+1, 1:-1] = (
            2*u[n, 1:-1]
            - u[n-1, 1:-1]
            + r**2 * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
        )
        u[n + 1, 0] = u[n + 1, 1]
        u[n + 1, -1] = u[n + 1, -2]

    u_fft = np.real(np.fft.rfft(u[:, :], axis=1))
    freqs = np.fft.rfftfreq(N, d=dx)

    return x, t, u, dx, dt, freqs, u_fft


x, t, u, dx, dt, freqs, modes = solve_wave_equation()

U_hat_filtered = np.zeros_like(modes)
U_hat_filtered[:, 1:N_MODES+1] = modes[:, 1:N_MODES+1]
U_reconstructed = np.real(np.fft.irfft(U_hat_filtered, n=len(x), axis=1))

a = np.real(modes[:, 1:N_MODES+1])

N_total = len(t)
N_train = int(N_total * TRAIN_FRAC)

a_train = a[:N_train]
t_train = t[:N_train]

a_maxabs = np.abs(a_train).max(axis=0)
a_maxabs[a_maxabs < 1e-10] = 1.0

if NORMALISE:
    a_norm = a / a_maxabs
    a_train_norm = a_train / a_maxabs
else:
    a_norm = a
    a_train_norm = a_train


def compute_time_derivative(a, dt):
    da = np.zeros_like(a)
    da[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
    da[0]    = (a[1] - a[0]) / dt
    da[-1]   = (a[-1] - a[-2]) / dt
    return da


a_dot_train = compute_time_derivative(a_train_norm, dt)
v_train = a_dot_train
v_dot_train = compute_time_derivative(v_train, dt)

X_train = np.hstack([a_train_norm, v_train])
X_dot_train = np.hstack([v_train, v_dot_train])



def build_library(X):
    Nt, n = X.shape
    library = []

    for i in range(n):
        library.append(X[:, i])

    return np.column_stack(library)


Theta_train = build_library(X_train)
print(f"Training library shape: {Theta_train.shape}")


if METHOD == "STRidge":
    lam   = 0
    tol   = 1e-1
    maxit = int(1e6)

    Xi = np.zeros((Theta_train.shape[1], X_dot_train.shape[1]))
    for i in range(X_dot_train.shape[1]):
        Xi[:, i] = STRidge(
            Theta_train,
            X_dot_train[:, i],
            lam,
            maxit,
            tol,
            normalize=0
        ).flatten()

if METHOD == "FROLS":
    opt = FROLS(max_iter=5, alpha=0, kappa=1e-5)
    opt.fit(Theta_train, X_dot_train)
    Xi = opt.coef_.T

if PRINT_XI:
    for i in range(2*N_MODES):
        formatted = [f"{num:.6f}" for num in Xi[:, i]]
        print(f"Equation {i}: {formatted}")

print(f"Active terms: {np.count_nonzero(Xi)}")

def sindy_rhs_ivp(t, X, Xi):
    Theta = build_library(X.reshape(1, -1))
    return (Theta @ Xi).flatten()


a0_norm = a_norm[0]
v0_norm = compute_time_derivative(a_norm, dt)[0]
X0 = np.concatenate([a0_norm, v0_norm])

t_span = (t[0], t[-1])
t_eval = t

sol = solve_ivp(
    fun=lambda t, X: sindy_rhs_ivp(t, X, Xi),
    t_span=t_span,
    y0=X0,
    t_eval=t_eval,
    method='RK45',
    rtol=1e-8,
    atol=1e-10
)

X_sim = sol.y.T

a_sim_norm = X_sim[:, :N_MODES]
if NORMALISE:
    a_sim = a_sim_norm * a_maxabs
else:
    a_sim = a_sim_norm


U_hat_sindy = np.zeros_like(modes)

U_hat_sindy[:, 1:N_MODES+1] = a_sim

U_sindy = np.real(np.fft.irfft(U_hat_sindy, n=len(x), axis=1))


if PLOT_MODES:
    for mode in range(N_MODES):
        fig, ax = plt.subplots(figsize=(8, 3))

        ax.plot(t, a[:, mode], color='black', label="True")
        ax.plot(t[:N_train], a_sim[:N_train, mode], '--', color='steelblue', label="SINDy (train)")
        ax.plot(t[N_train:], a_sim[N_train:, mode], '--', color='tomato', label="SINDy (forecast)")
        ax.axvline(t[N_train], color='gray', linestyle=':', linewidth=1)

        ax.set_xlabel("Time")
        ax.set_ylabel("Mode amplitude")
        ax.set_title(f"Mode {mode} — {METHOD}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

if PLOTTING:
    fig1 = plt.figure(figsize=(16, 6))
    ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
    Xm, Tm = np.meshgrid(x, t)
    ax1.plot_surface(Xm, Tm, u, cmap='viridis')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u')
    ax1.set_title('Travelling Wave Solution')

    ax2 = fig1.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(Xm, Tm, U_reconstructed, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u')
    ax2.set_title(f'Travelling Wave Solution Reconstruction ({N_MODES} modes)')

    ax2 = fig1.add_subplot(1, 3, 3, projection='3d')
    ax2.plot_surface(Xm, Tm, U_sindy, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u')
    ax2.set_title(f'Travelling Wave SINDy Modal Solution Reconstruction ({N_MODES} modes)')
    plt.tight_layout()
    plt.show()

error = np.mean((U_reconstructed - U_sindy) ** 2)
print("SINDy-Modal MSE:", error)

error = np.mean((u - U_reconstructed) ** 2)
print("Modal-True MSE:", error)

error = np.mean((u - U_sindy) ** 2)
print("SINDy-True MSE:", error)
