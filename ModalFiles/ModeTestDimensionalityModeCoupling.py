import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pysindy.optimizers import FROLS

import warnings
from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=LinAlgWarning)

PLOTTING = True
PLOT_MODES = True
PRINT_XI = True
n_modes = 10
TRAIN_FRAC = 0.3
L = 2
c = 1

FROLS_MAX_ITER = 1
FROLS_ALPHA = 0
FROLS_KAPPA = 0


def compute_time_derivative(a, dt):
    da = np.zeros_like(a)
    da[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
    da[0] = (a[1] - a[0]) / dt
    da[-1] = (a[-1] - a[-2]) / dt
    return da


def solve_wave_equation_2d(
        c=1.0, L=2.0, Nx=100, Ny=100, n_steps=1500, sigma=0.15,
):
    dx = L / Nx
    dt = 0.4 * dx / c
    r = c * dt / dx

    x = np.linspace(-L / 2, L / 2, Nx + 1)
    y = np.linspace(-L / 2, L / 2, Ny + 1)
    t = np.arange(n_steps + 1) * dt

    X, Y = np.meshgrid(x, y, indexing='ij')

    x0_disp, y0_disp = +0.2, +0.2
    u0 = np.exp(-((X - x0_disp) ** 2 + (Y - y0_disp) ** 2) / (2 * sigma ** 2))

    x0_vel, y0_vel = -0.1, -0.3
    v0 = np.exp(-((X - x0_vel) ** 2 + (Y - y0_vel) ** 2) / (2 * sigma ** 2))

    u = np.zeros((n_steps + 1, Nx + 1, Ny + 1))
    u[0] = u0

    lap_u0 = (
                     u0[2:, 1:-1] + u0[:-2, 1:-1] +
                     u0[1:-1, 2:] + u0[1:-1, :-2] -
                     4.0 * u0[1:-1, 1:-1]
             ) / dx ** 2

    u[1, 1:-1, 1:-1] = (
            u0[1:-1, 1:-1]
            + dt * v0[1:-1, 1:-1]
            + 0.5 * dt ** 2 * c ** 2 * lap_u0
    )
    u[1, 0, :] = u[1, -2, :];
    u[1, -1, :] = u[1, 1, :]
    u[1, :, 0] = u[1, :, -2];
    u[1, :, -1] = u[1, :, 1]

    for n in range(1, n_steps):
        u[n + 1, 1:-1, 1:-1] = (
                2.0 * u[n, 1:-1, 1:-1]
                - u[n - 1, 1:-1, 1:-1]
                + r ** 2 * (
                        u[n, 2:, 1:-1] + u[n, :-2, 1:-1] +
                        u[n, 1:-1, 2:] + u[n, 1:-1, :-2] -
                        4.0 * u[n, 1:-1, 1:-1]
                )
        )
        u[n + 1, 0, :] = u[n + 1, -2, :];
        u[n + 1, -1, :] = u[n + 1, 1, :]
        u[n + 1, :, 0] = u[n + 1, :, -2];
        u[n + 1, :, -1] = u[n + 1, :, 1]

    u_fft = np.fft.rfft2(u, axes=(1, 2))
    return x, y, t, u, u_fft, dx, dt


x, y, t, u, modes, dx, dt = solve_wave_equation_2d()
Nt = len(t)

Nx_grid = u.shape[1]
Ny_grid = u.shape[2]

energy = np.mean(np.abs(modes) ** 2, axis=0)
Nx_fft, Ny_fft = energy.shape
flat_energy = energy.flatten()
top_indices = np.argsort(flat_energy)[-n_modes:]
kx_idx, ky_idx = np.unravel_index(top_indices, (Nx_fft, Ny_fft))
selected_modes = list(zip(kx_idx, ky_idx))
selected_modes = [
    (kx, ky) for kx, ky in selected_modes
    if not (kx == 0 and ky == 0)
]
n_sel = len(selected_modes)


def physical_k_sq(kx, ky, Nx_grid, Ny_grid, L):
    kx_w = kx if kx <= Nx_grid // 2 else kx - Nx_grid
    ky_w = ky if ky <= Ny_grid // 2 else ky - Ny_grid
    return ((2 * np.pi / L) ** 2) * (kx_w ** 2 + ky_w ** 2)


U_hat_filtered = np.zeros_like(modes, dtype=complex)
for i, j in selected_modes:
    U_hat_filtered[:, i, j] = modes[:, i, j]
U_reconstructed = np.fft.irfft2(U_hat_filtered, s=(u.shape[1], u.shape[2]), axes=(1, 2)).real

N_train = int(Nt * TRAIN_FRAC)
t_train = t[:N_train]

print(f"Building stacked state matrix for {n_sel} modes  "
      f"→ {2 * n_sel}-dimensional complex joint system")

state_cols = []
dot_cols = []

for kx, ky in selected_modes:
    a = modes[:, kx, ky]
    v = compute_time_derivative(a, dt)
    v_d = compute_time_derivative(v, dt)

    state_cols.extend([a, v])
    dot_cols.extend([v, v_d])

X_full = np.column_stack(state_cols)
X_dot_full = np.column_stack(dot_cols)

Theta_train = X_full[:N_train]
X_dot_train = X_dot_full[:N_train]

print(np.shape(Theta_train))

print(f"Fitting single joint SINDy model  (library shape: {Theta_train.shape})")

opt = FROLS(max_iter=FROLS_MAX_ITER, alpha=FROLS_ALPHA, kappa=FROLS_KAPPA)
opt.fit(Theta_train, X_dot_train)

Xi_full = opt.coef_.T.astype(complex)

if PRINT_XI:
    labels = []
    for kx, ky in selected_modes:
        kx_w = kx if kx <= Nx_grid // 2 else kx - Nx_grid
        ky_w = ky if ky <= Ny_grid // 2 else ky - Ny_grid
        tag = f"({kx_w},{ky_w})"
        labels += [f"a{tag}", f"v{tag}"]

    col_w = max(len(l) for l in labels) + 8
    row_lbl = max(len(l) for l in labels)

    print(f"\nJoint coefficient matrix Xi_full  ({Xi_full.shape[0]}×{Xi_full.shape[1]})")

    header = " " * (row_lbl + 3) + "".join(l.rjust(col_w) for l in labels)
    print(header)
    print(" " * (row_lbl + 3) + "-" * (col_w * len(labels)))

    for row_idx, row_label in enumerate(labels):
        row_vals = "".join(f"{Xi_full[row_idx, c]:>{col_w}.2f}" for c in range(len(labels)))
        print(f"{row_label:>{row_lbl}}  |{row_vals}")

    print("\nPer-mode diagonal blocks (recovered ω²):")
    for idx, (kx, ky) in enumerate(selected_modes):
        kx_w = kx if kx <= Nx_grid // 2 else kx - Nx_grid
        ky_w = ky if ky <= Ny_grid // 2 else ky - Ny_grid
        kx_phys = (2 * np.pi / L) * kx_w
        ky_phys = (2 * np.pi / L) * ky_w
        omega_sq_expected = c ** 2 * (kx_phys ** 2 + ky_phys ** 2)
        print(
            f"  mode ({kx_w:+d},{ky_w:+d})  "
            f"ω²_expected={omega_sq_expected:.4f}"
        )

print("\nSimulating joint stacked system …")

X0_full = X_full[0]


def rhs_full(t_val, state, Xi=Xi_full):
    return (state.reshape(1, -1) @ Xi).flatten()


sol = solve_ivp(
    rhs_full,
    t_span=(t[0], t[-1]),
    y0=X0_full,
    t_eval=t,
    method='RK45'
)

sim_full = sol.y.T
a_sim_complex = np.column_stack([sim_full[:, 2 * i] for i in range(n_sel)])

U_hat_sindy = np.zeros_like(modes, dtype=complex)
for idx, (i, j) in enumerate(selected_modes):
    U_hat_sindy[:, i, j] = a_sim_complex[:, idx]

U_sindy = np.fft.irfft2(U_hat_sindy, s=(u.shape[1], u.shape[2]), axes=(1, 2)).real

if PLOTTING:
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    for i in range(Nt):
        if i % 200 == 0 or i == (Nt - 1):
            fig, axes = plt.subplots(3, 1, figsize=(8, 14))
            for ax, field, title in zip(
                    axes,
                    [u[i], U_reconstructed[i], U_sindy[i]],
                    [f"True (step {i})",
                     f"Modal reconstruct (step {i})",
                     f"SINDy joint forecast (step {i})"],
            ):
                im = ax.imshow(
                    field.T, origin="lower",
                    extent=[x[0], x[-1], y[0], y[-1]],
                    cmap="Blues_r",
                )
                ax.set(xlabel="x", ylabel="y", title=title)
                fig.colorbar(im, ax=ax, label="Amplitude")
            plt.tight_layout()
            plt.show()

if PLOT_MODES:
    for idx, (kx, ky) in enumerate(selected_modes):
        fig, ax = plt.subplots(figsize=(9, 3))
        a_true = modes[:, kx, ky].real
        ax.plot(t, a_true, color='black', lw=1.5, label="True (real part)")
        ax.plot(t[:N_train], a_sim_complex[:N_train, idx].real, '--', color='steelblue', lw=1.5,
                label="SINDy joint (train)")
        ax.plot(t[N_train:], a_sim_complex[N_train:, idx].real, '--', color='tomato', lw=1.5,
                label="SINDy joint (forecast)")
        ax.axvline(t[N_train], color='gray', linestyle=':', lw=1)
        kx_w = kx if kx <= Nx_grid // 2 else kx - Nx_grid
        ky_w = ky if ky <= Ny_grid // 2 else ky - Ny_grid
        ax.set(xlabel="Time", ylabel="Amplitude (real part)",
               title=f"Mode (kx_phys={kx_w}, ky_phys={ky_w})  [idx=({kx},{ky})]")
        ax.legend();
        ax.grid(True)
        plt.tight_layout();
        plt.show()

print("\nSINDy-Modal MSE :", np.mean((U_reconstructed - U_sindy) ** 2))
print("Modal-True  MSE :", np.mean((u - U_reconstructed) ** 2))
print("SINDy-True  MSE :", np.mean((u - U_sindy) ** 2))