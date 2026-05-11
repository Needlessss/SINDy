import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pysindy.optimizers import FROLS

#Disable this if you aren't certain ill conditioned matrices won't cause you issues
################################################################################
import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)
################################################################################


PLOT_AMPLITUDES = False
PLOTTING = True
PLOT_MODES = True
PRINT_XI = True
N_MODES = 4
METHOD = "FROLS"
TRAIN_FRAC = 0.3


def sin_ic(x, L):
    return np.sin((2 * np.pi * x / L))


def solve_wave_equation(
    c=1.0,
    L=2.0,
    N=200,
    n_steps=1000,
    sigma=0.15,
    x0=0.3,
    ic=None,
):
    dx = L / N
    dt = 0.4 * dx / c
    r  = c * dt / dx

    x = np.linspace(-L/2, L/2, N + 1)
    t = np.arange(n_steps + 1) * dt

    u = np.zeros((n_steps + 1, N + 1))
    if ic is None:
        u[0] = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    else:
        u[0] = ic(x, L)

    u[1, 1:-1] = (
        u[0, 1:-1]
        + 0.5 * r**2 * (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2])
    )
    u[1, 0] = (
        u[0, 0]
        + 0.5 * r**2 * (u[0, 1] - 2*u[0, 0] + u[0, -2])
    )
    u[1, -1] = u[1, 0]

    for n in range(1, n_steps):
        u[n+1, 1:-1] = (
            2*u[n, 1:-1]
            - u[n-1, 1:-1]
            + r**2 * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
        )
        u[n+1, 0] = (
            2*u[n, 0]
            - u[n-1, 0]
            + r**2 * (u[n, 1] - 2*u[n, 0] + u[n, -2])
        )
        u[n+1, -1] = u[n+1, 0]

    u_fft = np.fft.fft(u, axis=1)

    return x, t, u, dx, dt, u_fft


x, t, u, dx, dt, modes = solve_wave_equation()
n = len(x)

a = modes[:, 1:N_MODES+1]

U_hat_filtered = np.zeros_like(modes, dtype=complex)
U_hat_filtered[:, 1:N_MODES+1] = modes[:, 1:N_MODES+1]
U_hat_filtered[:, 0] = modes[:, 0]
U_hat_filtered[:, n-N_MODES:n] = np.conj(a[:, ::-1])
U_reconstructed = np.fft.ifft(U_hat_filtered, n=n, axis=1).real

N_total = len(t)
N_train = int(N_total * TRAIN_FRAC)

a_train = a[:N_train]
t_train = t[:N_train]


def compute_time_derivative(a, dt):
    da = np.zeros_like(a)
    da[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
    da[0] = (a[1]  - a[0]) / dt
    da[-1] = (a[-1] - a[-2]) / dt
    return da


a_dot_train = compute_time_derivative(a_train, dt)
v_train = a_dot_train
v_dot_train = compute_time_derivative(v_train, dt)

X_train = np.hstack([a_train, v_train])
X_dot_train = np.hstack([v_train, v_dot_train])


def build_library(X):
    return np.column_stack([X[:, i] for i in range(X.shape[1])])


Theta_train = build_library(X_train)

if METHOD == "FROLS":
    opt = FROLS(max_iter=1, alpha=0, kappa=3e-13)
    opt.fit(Theta_train, X_dot_train)
    Xi = opt.coef_.T.astype(complex)

if PRINT_XI:
    for i in range(2 * N_MODES):
        formatted = [f"{num:.6f}" for num in Xi[:, i]]
        print(f"Equation {i}: {formatted}")

print(f"Active terms: {np.count_nonzero(Xi)}")


def sindy_rhs_ivp(t, X, Xi):
    Theta = build_library(X.reshape(1, -1))
    return (Theta @ Xi).flatten()


a0 = a[0]
v0 = compute_time_derivative(a, dt)[0]
X0 = np.concatenate([a0, v0])

sol = solve_ivp(
    fun=lambda t, X: sindy_rhs_ivp(t, X, Xi),
    t_span=(t[0], t[-1]),
    y0=X0,
    t_eval=t,
    method='RK45',
)

X_sim = sol.y.T
a_sim = X_sim[:, :N_MODES]

U_hat_sindy = np.zeros_like(modes, dtype=complex)
U_hat_sindy[:, 1:N_MODES+1] = a_sim
U_hat_sindy[:, 0] = modes[:, 0]
U_hat_sindy[:, n-N_MODES:n] = np.conj(a_sim[:, ::-1])
U_sindy = np.fft.ifft(U_hat_sindy, n=n, axis=1).real

if PLOT_MODES:
    for k in range(N_MODES):
        fig, ax = plt.subplots(figsize=(9, 3))

        ax.plot(t, a[:, k].real,
                color="black", label="True (Re)")
        ax.plot(t, a_sim[:, k].real, "--",
                color="steelblue", label="SINDy (Re)")
        ax.plot(t, a[:, k].imag,
                color="gray", label="True (Im)", alpha=0.6)
        ax.plot(t, a_sim[:, k].imag, ":",
                color="salmon", label="SINDy (Im)", alpha=0.9)

        ax.axvline(t[N_train], color="k", linestyle=":",
                   alpha=0.5, label="Train | Test")

        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Mode {k+1})")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

if PLOT_AMPLITUDES:
    plt.imshow(
        modes.real[:, :1000].T,
        aspect='auto',
        origin='lower',
    )
    plt.xlabel("Time index")
    plt.ylabel("Mode number")
    plt.title("Fourier Modal Evolution (Real Component)")
    plt.colorbar()
    plt.show()

    plt.imshow(
        modes.imag[:, :1000].T,
        aspect='auto',
        origin='lower',
    )
    plt.xlabel("Time index")
    plt.ylabel("Mode number")
    plt.title("Fourier Modal Evolution (Imaginary Component)")
    plt.colorbar()
    plt.show()


if PLOTTING:
    fig1 = plt.figure(figsize=(16, 6))
    Xm, Tm = np.meshgrid(x, t)

    ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
    ax1.plot_surface(Xm, Tm, u, cmap='viridis')
    ax1.set_xlabel('x'); ax1.set_ylabel('t'); ax1.set_zlabel('u')
    ax1.set_title('Travelling Wave Solution')

    ax2 = fig1.add_subplot(1, 3, 2, projection='3d')
    ax2.plot_surface(Xm, Tm, U_reconstructed, cmap='viridis')
    ax2.set_xlabel('x'); ax2.set_ylabel('t'); ax2.set_zlabel('u')
    ax2.set_title(f'Reconstruction ({N_MODES} modes)')

    ax3 = fig1.add_subplot(1, 3, 3, projection='3d')
    ax3.plot_surface(Xm, Tm, U_sindy, cmap='viridis')
    ax3.set_xlabel('x'); ax3.set_ylabel('t'); ax3.set_zlabel('u')
    ax3.set_title(f'SINDy Modal Reconstruction ({N_MODES} modes)')

    plt.tight_layout()
    plt.show()

print("\nSINDy-Modal MSE:", np.mean((U_reconstructed - U_sindy) ** 2))
print("Modal-True MSE:", np.mean((u - U_reconstructed) ** 2))
print("SINDy-True MSE:", np.mean((u - U_sindy) ** 2))

x_test, t_test, u_test, dx_test, dt_test, modes_test = solve_wave_equation(ic=sin_ic)
n_test = len(x_test)

a_test = modes_test[:, 1:N_MODES+1]

a0_test = a_test[0]
v0_test = compute_time_derivative(a_test, dt_test)[0]
X0_test = np.concatenate([a0_test, v0_test])

sol_test = solve_ivp(
    fun=lambda t, X: sindy_rhs_ivp(t, X, Xi),
    t_span=(t_test[0], t_test[-1]),
    y0=X0_test,
    t_eval=t_test,
    method='RK45',
)

X_sim_test = sol_test.y.T
a_sim_test = X_sim_test[:, :N_MODES]

U_hat_sindy_test = np.zeros_like(modes_test, dtype=complex)
U_hat_sindy_test[:, 1:N_MODES+1] = a_sim_test
U_hat_sindy[:, 0] = modes_test[:, 0]
U_hat_sindy[:, n_test-N_MODES:n_test] = np.conj(a_sim_test[:, ::-1])
U_sindy_test = np.fft.ifft(U_hat_sindy_test, n=n_test, axis=1).real

print("\nSINDy-True MSE (test IC):", np.mean((u_test - U_sindy_test)**2))

if PLOTTING:
    fig1 = plt.figure(figsize=(14, 6))
    Xm_test, Tm_test = np.meshgrid(x_test, t_test)

    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(Xm_test, Tm_test, u_test, cmap='viridis')
    ax1.set_xlabel('x'); ax1.set_ylabel('t'); ax1.set_zlabel('u')
    ax1.set_title('Travelling Wave Solution')

    ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(Xm_test, Tm_test, U_sindy_test, cmap='viridis')
    ax2.set_xlabel('x'); ax2.set_ylabel('t'); ax2.set_zlabel('u')
    ax2.set_title(f'Reconstruction')

    plt.tight_layout()
    plt.show()