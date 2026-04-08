import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pysindy.optimizers import FROLS
from itertools import product
import numpy as np

#Disable this if you aren't certain ill conditioned matrices won't cause you issues
################################################################################
import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)
################################################################################


PRINT_XI    = False
N_MODES     = 4
METHOD      = "FROLS"
TRAIN_FRAC  = 0.3


def sin_ic(x, L):
    return np.sin(2 * np.pi * x / L)


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

    u_fft = np.fft.fft(u, axis=1)

    return x, t, u, dx, dt, u_fft


x, t, u, dx, dt, modes = solve_wave_equation()
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

vals = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subs = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22, 1e-23, 1e-24]
kappas = [x * y for x, y in product(vals, subs)]
errors = np.ones(np.shape(kappas))
terms = np.ones(np.shape(kappas))

for i, kappa_val in enumerate(kappas):
    try:
        if METHOD == "FROLS":
            opt = FROLS(max_iter=5, alpha=0, kappa=kappa_val)
            opt.fit(Theta_train, X_dot_train)
            Xi = opt.coef_.T

        if PRINT_XI:
            for i in range(2*N_MODES):
                formatted = [f"{num:.6f}" for num in Xi[:, i]]
                print(f"Equation {i}: {formatted}")


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

        terms[i] = np.count_nonzero(Xi)
        errors[i] = np.mean((U_reconstructed - U_sindy) ** 2)
        print(f"{(i/len(kappas))*100}% Complete")

    except:
        print("Exploded Probably")

min_index = np.argmin(errors)
print(f"Minimum Error: {errors[min_index]}")
print(f"Minimum Kappa: {kappas[min_index]}")
print(f"Terms for Min: {terms[min_index]}")

plt.plot(kappas, errors, 'o')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Kappa Value")
plt.ylabel("Error Value")
plt.title("Kappa Val vs Mode Reconstruction Error")
plt.grid(True, which="both")
plt.show()

