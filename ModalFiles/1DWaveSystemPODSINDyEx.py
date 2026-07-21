import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pysindy.optimizers import FROLS

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)

N_MODES = 9
FROLS_MAX_ITER = 6
FROLS_ALPHA = 1e-5
KAPPA_SCALE = 1e-3
GENUINE_TOL = np.inf

PLOT_MODES = True
PLOT_EIGS = True
PLOTTING = True
PRINT_SUMMARY = True

C, L, N, N_STEPS = 1.0, 2.0, 200, 1000

def solve_wave_equation(c=C, L=L, N=N, n_steps=N_STEPS, sigma=0.15, x0=0.3, ic=None):
    dx = L / N
    dt = 0.4 * dx / c
    r = c * dt / dx
    x = np.linspace(-L / 2, L / 2, N + 1)
    t = np.arange(n_steps + 1) * dt
    u = np.zeros((n_steps + 1, N + 1))
    if ic is None:
        u[0] = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    else:
        u[0] = ic(x, L)
    u[1, 1:-1] = u[0, 1:-1] + 0.5 * r ** 2 * (u[0, 2:] - 2 * u[0, 1:-1] + u[0, :-2])
    u[1, 0] = u[0, 0] + 0.5 * r ** 2 * (u[0, 1] - 2 * u[0, 0] + u[0, -2])
    u[1, -1] = u[1, 0]
    for n in range(1, n_steps):
        u[n + 1, 1:-1] = 2 * u[n, 1:-1] - u[n - 1, 1:-1] + r ** 2 * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])
        u[n + 1, 0] = 2 * u[n, 0] - u[n - 1, 0] + r ** 2 * (u[n, 1] - 2 * u[n, 0] + u[n, -2])
        u[n + 1, -1] = u[n + 1, 0]
    return x, t, u, dx, dt


def fourier_ic(n, parity):
    def ic(x, L):
        arg = 2 * np.pi * n * x / L
        return np.cos(arg) if parity == "cos" else np.sin(arg)
    return ic


def compute_time_derivative(a, dt):
    da = np.zeros_like(a)
    da[1:-1] = (a[2:] - a[:-2]) / (2 * dt)
    da[0] = (a[1] - a[0]) / dt
    da[-1] = (a[-1] - a[-2]) / dt
    return da


def build_library(X):
    return np.column_stack([X[:, i] for i in range(X.shape[1])])


def make_frols(Theta, max_iter=FROLS_MAX_ITER, alpha_reg=FROLS_ALPHA, kappa_scale=KAPPA_SCALE):
    col_norms = np.linalg.norm(Theta, 2, axis=0)
    col_norms[col_norms == 0] = 1.0
    cond = np.linalg.cond(Theta / col_norms)
    kappa = kappa_scale / cond if cond > 0 else 0
    opt = FROLS(max_iter=max_iter, alpha=alpha_reg, kappa=kappa, normalize_columns=True)
    return opt, cond, kappa


train_specs = [(n, "cos") for n in range(1, 6)] + [(n, "sin") for n in range(1, 5)]
if PRINT_SUMMARY:
    print(f"Training ICs: {train_specs}")

sims = [solve_wave_equation(ic=fourier_ic(n, p)) for n, p in train_specs]
x, t, _, dx, dt = sims[0]
U_list = [s[2] for s in sims]

print(np.shape(U_list))
U_stacked = np.vstack(U_list)
u_mean = U_stacked.mean(axis=0)
print(np.shape(U_stacked))
_, S, Vt = np.linalg.svd(U_stacked - u_mean, full_matrices=False)
phi = Vt[:N_MODES, :]

a_list, v_list, vdot_list = [], [], []
for U_i in U_list:
    a_i = (U_i - u_mean) @ phi.T
    v_i = compute_time_derivative(a_i, dt)
    vdot_i = compute_time_derivative(v_i, dt)
    a_list.append(a_i)
    v_list.append(v_i)
    vdot_list.append(vdot_i)

X1_parts, X2_parts = [], []
for a_i, v_i in zip(a_list, v_list):
    state_i = np.hstack([a_i, v_i])
    X1_parts.append(state_i[:-1, :].T)
    X2_parts.append(state_i[1:, :].T)
X1_dmd, X2_dmd = np.hstack(X1_parts), np.hstack(X2_parts)

A_tilde = X2_dmd @ np.linalg.pinv(X1_dmd)
evals, V = np.linalg.eig(A_tilde)
V_inv = np.linalg.pinv(V)


def find_true_conjugate(V, k, candidates, tol=1e-6):
    vk_conj = np.conj(V[:, k])
    best_j, best_score = None, 0.0
    for j in candidates:
        if j == k:
            continue
        num = np.abs(np.vdot(V[:, j], vk_conj))
        denom = np.linalg.norm(V[:, j]) * np.linalg.norm(vk_conj)
        score = num / denom if denom > 0 else 0.0
        if score > best_score:
            best_j, best_score = j, score
    return best_j if best_score > 1 - tol else None


canonical = []
secondary_of = {}
candidates = list(range(len(evals)))
represented = set()
for k in candidates:
    if k in represented:
        continue
    j = find_true_conjugate(V, k, candidates)
    if j is None:
        canonical.append(k)
        represented.add(k)
    else:
        primary, secondary = (k, j) if evals[k].imag >= evals[j].imag else (j, k)
        canonical.append(primary)
        secondary_of[secondary] = primary
        represented.add(primary)
        represented.add(secondary)
canonical = sorted(canonical)

if PRINT_SUMMARY:
    print(f"\nNon-Conjugate Modes: {canonical}")
    print(f"Conjugate Modes: {secondary_of}")


def pack_ri(z, keep_idx):
    real_mask = np.all(np.abs(z[:, keep_idx].imag) < 1e-6 * np.maximum(np.abs(z[:, keep_idx].real), 1e-12), axis=0)
    cols, labels = [], []
    for pos, k in enumerate(keep_idx):
        cols.append(z[:, k].real[:, None]); labels.append((k, "Re"))
        if not real_mask[pos]:
            cols.append(z[:, k].imag[:, None]); labels.append((k, "Im"))
    return np.column_stack(cols), labels


z_list, zdot_list = [], []
for a_i, v_i, vdot_i in zip(a_list, v_list, vdot_list):
    state_i = np.hstack([a_i, v_i])
    statedot_i = np.hstack([v_i, vdot_i])
    z_list.append((V_inv @ state_i.T).T)
    zdot_list.append((V_inv @ statedot_i.T).T)

Z_train, col_labels = pack_ri(np.vstack(z_list), canonical)
Zdot_train, _ = pack_ri(np.vstack(zdot_list), canonical)

opt_z, cond_z, kappa_z = make_frols(Z_train)
opt_z.fit(Z_train, Zdot_train)
Xi_z = opt_z.coef_.T

if PRINT_SUMMARY:
    print(f"\nZ_train shape={Z_train.shape}, cond={cond_z:.3e}, kappa={kappa_z:.3e}")
    print("Discovered system in the rotated basis (nonzero terms only):")
    znames = [f"{part}(z{k})" for k, part in col_labels]
    for row, (k, part) in enumerate(col_labels):
        nz = np.nonzero(np.abs(Xi_z[:, row]) > 1e-8)[0]
        terms = " + ".join(f"{Xi_z[j,row]:.5f}*{znames[j]}" for j in nz)
        print(f"  d[{part}(z{k})]/dt = {terms if terms else '0'}")
    print(f"Active terms: {np.count_nonzero(np.abs(Xi_z) > 1e-8)} / {Xi_z.size}")


def rhs_z(t_val, y, Xi, degree_labels):
    Theta_row = build_library(y.reshape(1, -1))
    return (Theta_row @ Xi).flatten()


N_TRUSTED = int(np.sum(S > 1e-6 * S[0]))
if PRINT_SUMMARY:
    print(f"\nTrusted POD directions: {N_TRUSTED} / {N_MODES}")


def predict(u0_field, t_eval):
    a0 = (u0_field - u_mean) @ phi.T
    a0[N_TRUSTED:] = 0.0
    v0 = np.zeros_like(a0)
    state0 = np.concatenate([a0, v0])
    z0_full = V_inv @ state0
    y0 = []
    for k, part in col_labels:
        y0.append(z0_full[k].real if part == "Re" else z0_full[k].imag)
    y0 = np.array(y0)

    sol = solve_ivp(rhs_z, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval,
                     args=(Xi_z, col_labels), method="RK45")

    z_sim = np.zeros((len(t_eval), 2 * N_MODES), dtype=complex)
    for row, (k, part) in enumerate(col_labels):
        if part == "Re":
            z_sim[:, k] += sol.y[row]
        else:
            z_sim[:, k] += 1j * sol.y[row]
    for secondary, primary in secondary_of.items():
        z_sim[:, secondary] = np.conj(z_sim[:, primary])
    state_sim = (V @ z_sim.T).T
    a_sim = state_sim[:, :N_MODES].real
    U_pred = a_sim @ phi + u_mean
    return U_pred, a_sim


def var_captured(true, pred):
    return 1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)


results = {}


def combo_ic(x, L):
    return (0.5 * np.cos(2*np.pi*1*x/L) + 0.3 * np.sin(2*np.pi*2*x/L)
            + 0.4 * np.cos(2*np.pi*3*x/L) - 0.2 * np.sin(2*np.pi*4*x/L)
            + 0.6 * np.cos(2*np.pi*5*x/L))


def sin5_ic(x, L):
    return np.sin(2*np.pi*5*x/L)


def gaussian_ic(x, L):
    return np.exp(-0.5 * ((x - 0.3) / 0.15) ** 2)


test_cases = [("combo", combo_ic),
              ("sin(5πx)", sin5_ic),
              ("Gaussian pulse", gaussian_ic)]

for name, ic_fn in test_cases:
    xt, tt, u_true, dxt, dtt = solve_wave_equation(ic=ic_fn)
    U_pred, a_pred = predict(u_true[0], tt)
    U_pod_only = ((u_true - u_mean) @ phi.T) @ phi + u_mean
    a_true_all = (u_true - u_mean) @ phi.T
    a_trusted_only = a_true_all.copy()
    a_trusted_only[:, N_TRUSTED:] = 0.0
    U_pod_trusted = a_trusted_only @ phi + u_mean
    vc_dyn = var_captured(u_true, U_pred)
    vc_pod = var_captured(u_true, U_pod_only)
    vc_pod_trusted = var_captured(u_true, U_pod_trusted)
    mse_dyn = np.mean((u_true - U_pred) ** 2)
    mse_pod = np.mean((u_true - U_pod_only) ** 2)
    results[name] = dict(u_true=u_true, U_pred=U_pred, U_pod_only=U_pod_only,
                          x=xt, t=tt, vc_dyn=vc_dyn, vc_pod=vc_pod,
                          vc_pod_trusted=vc_pod_trusted, mse_dyn=mse_dyn, mse_pod=mse_pod)
    if PRINT_SUMMARY:
        print(f"\nTest IC: {name}")
        print(f"Variance captured: {vc_dyn:.4f}")
        print(f"MSE: SINDy={mse_dyn:.3e}, Pure POD Modes={mse_pod:.3e}")


if PLOT_MODES:
    for name, r in results.items():
        n_plot = N_MODES
        fig, axes = plt.subplots(n_plot, 1, figsize=(9, 2.2*n_plot), sharex=True)
        a_true = (r["u_true"] - u_mean) @ phi.T
        _, a_pred_only = predict(r["u_true"][0], r["t"])
        for k in range(n_plot):
            axes[k].plot(r["t"], a_true[:, k], color="black", label="True")
            axes[k].plot(r["t"], a_pred_only[:, k], "--", color="steelblue", label="SINDy Reconstruction")
            axes[k].set_ylabel(f"a{k}")
            axes[k].grid(True, alpha=0.3)
        axes[0].set_title(name)
        axes[0].legend(ncol=2, fontsize=8)
        axes[-1].set_xlabel("Time")
        plt.tight_layout(); plt.show()

if PLOTTING:
    for name, r in results.items():
        fig = plt.figure(figsize=(14, 5))
        Xm, Tm = np.meshgrid(r["x"], r["t"])
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(Xm, Tm, r["u_true"], cmap="viridis")
        ax1.set_title(f"True: {name}")
        ax1.set_xlabel("x"); ax1.set_ylabel("t")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(Xm, Tm, r["U_pred"], cmap="viridis")
        ax2.set_title(f"SINDy Reconstruction")
        ax2.set_xlabel("x"); ax2.set_ylabel("t")
        plt.tight_layout(); plt.show()