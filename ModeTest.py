import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from PDE_FIND import STRidge
from pysindy.optimizers import FROLS

#Disable this if you aren't certain ill conditioned matrices won't cause you issues
################################################################################
import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)
################################################################################


PLOTTING    = False
PLOT_MODES  = True
PRINT_XI    = False
N_MODES     = 7
METHOD      = "FROLS"   # Options: STRidge, FROLS
TRAIN_FRAC  = 0.3


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
    u[1, 0]  = (
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

    n_x   = len(x)
    u_fft = np.fft.fft(u, axis=1)
    freqs = np.fft.fftfreq(n_x, d=dx)

    return x, t, u, dx, dt, freqs, u_fft


x, t, u, dx, dt, freqs, modes = solve_wave_equation()
n = len(x)

U_hat_filtered = np.zeros_like(modes, dtype=complex)
U_hat_filtered[:, 1:N_MODES+1] = modes[:, 1:N_MODES+1]
U_reconstructed = np.fft.ifft(U_hat_filtered, n=n, axis=1).real

a_complex = modes[:, 1:N_MODES+1]
a_real = a_complex.real
a_imag = a_complex.imag
a = np.hstack([a_real, a_imag])

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


Theta_train = build_library(X_train)

if METHOD == "STRidge":
    lam   = 0
    tol   = 1e-1
    maxit = int(1e6)
    Xi    = np.zeros((Theta_train.shape[1], X_dot_train.shape[1]))
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
    opt = FROLS(max_iter=5, alpha=0, kappa=1e-13)
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
a_sim = X_sim[:, :2*N_MODES]
a_sim_real = a_sim[:, :N_MODES]
a_sim_imag = a_sim[:, N_MODES:]
a_sim_complex = a_sim_real + 1j * a_sim_imag

U_hat_sindy = np.zeros_like(modes, dtype=complex)
U_hat_sindy[:, 1:N_MODES+1] = a_sim_complex
U_sindy = np.fft.ifft(U_hat_sindy, n=n, axis=1).real

if PLOT_MODES:
    for mode in range(N_MODES):
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t,          a[:, mode],            color='black',     label="True")
        ax.plot(t[:N_train], a_sim_real[:N_train, mode], '--', color='steelblue', label="SINDy (train)")
        ax.plot(t[N_train:], a_sim_real[N_train:, mode], '--', color='tomato',    label="SINDy (forecast)")
        ax.axvline(t[N_train], color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mode amplitude")
        ax.set_title(f"Mode {mode} — {METHOD}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

if PLOTTING:
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
print("Modal-True  MSE:", np.mean((u - U_reconstructed) ** 2))
print("SINDy-True  MSE:", np.mean((u - U_sindy) ** 2))


#Test on different initial condition
def sin_ic(x, L):
    return np.sin(2 * np.pi * x / L)


x_test, t_test, u_test, dx_test, dt_test, freqs_test, modes_test = solve_wave_equation(ic=sin_ic)
n_test = len(x_test)

a_complex_test = modes_test[:, 1:N_MODES+1]
a_real_test = a_complex_test.real
a_imag_test = a_complex_test.imag
a_test = np.hstack([a_real_test, a_imag_test])

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
a_sim_test = X_sim_test[:, :2*N_MODES]
a_sim_real_test = a_sim_test[:, :N_MODES]
a_sim_imag_test = a_sim_test[:, N_MODES:]
a_sim_complex_test = a_sim_real_test + 1j * a_sim_imag_test

U_hat_sindy_test = np.zeros_like(modes_test, dtype=complex)
U_hat_sindy_test[:, 1:N_MODES+1] = a_sim_complex_test
U_sindy_test = np.fft.ifft(U_hat_sindy_test, n=len(x), axis=1).real

U_hat_true_test = np.zeros_like(modes_test, dtype=complex)
U_hat_true_test[:, 1:N_MODES+1] = modes_test[:, 1:N_MODES+1].copy()
U_reconstructed_test = np.fft.ifft(U_hat_true_test, n=n_test, axis=1).real

print("\nSINDy-True MSE (test IC):", np.mean((u_test - U_sindy_test)**2))
print("Modal-True MSE (test IC):", np.mean((u_test - U_reconstructed_test)**2))

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
