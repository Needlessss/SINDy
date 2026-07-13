import warnings
from itertools import combinations_with_replacement

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.integrate import solve_ivp
from pysindy.optimizers import FROLS
from scipy.linalg import LinAlgWarning

warnings.filterwarnings("ignore", category=LinAlgWarning)

N_MODES = 10

MAX_ACTIVE = 4
RDC_THRESHOLD = 0.9
ENERGY_CAPTURE = 0.995
POLY_DEGREE_DYNAMICS = 2
POLY_DEGREE_MANIFOLD = 4
FROLS_MAX_ITER = 20
FROLS_ALPHA = 1e-5
KAPPA_SCALE = 5e-3
BLOWUP_THRESHOLD = 1e3

PLOT_SLICES = True
PLOT_MODES = False
PLOTTING = True
PRINT_SUMMARY = True

RNG_SEED = 0


def CFLcondition(u, dx, CFLcoe):
    toll = 1e-10
    dtmax = CFLcoe * dx / np.maximum(np.abs(u), toll)
    dt = np.min(dtmax)
    return dt


def PeriodicBoundary(u, uL, uR):
    nc = len(u) - 2
    u[0] = u[nc]
    u[nc + 1] = u[1]
    return u


def GodunovFlux(u):
    toll = 1e-10
    F = np.zeros(len(u) - 1)

    for i in range(len(u) - 1):
        if u[i] > u[i + 1]:  # shock
            s = 0.5 * (u[i] + u[i + 1])
            u0 = u[i] if s > 0 else u[i + 1]
        else:  # rarefaction
            if u[i] > toll:
                u0 = u[i]
            elif u[i + 1] < -toll:
                u0 = u[i + 1]
            else:
                u0 = 0.0
        F[i] = 0.5 * u0 ** 2

    return F


L = 1000
x0 = -50
nc = 1000
dx = L / nc

x = np.linspace(x0 - dx / 2, x0 + L + dx / 2, nc + 2)
u = np.ones(nc + 2)

NX = len(x)
tend = 1000
CFLcoe = 0.9
t = 0.0

testname = "bump"
uL, uR = 0, 0
u = np.e ** -(((x - (x0 + L / 2)) ** 2) / (2 * ((L / 6) ** 2)))


u_history = [u.copy()]
t_history = [t]

while t < tend:
    u = PeriodicBoundary(u, uL, uR)
    dt = CFLcondition(u, dx, CFLcoe)
    dt = min(dt, tend - t)
    F = GodunovFlux(u)
    u[1:nc + 1] = u[1:nc + 1] - dt / dx * (F[1:nc + 1] - F[0:nc])
    t += dt
    u_history.append(u.copy())
    t_history.append(t)

u = np.array(u_history)
t = np.array(t_history)

if PLOT_SLICES:
    N_SLICES = 5
    slice_ids = np.linspace(0, len(t_history) - 1, N_SLICES, dtype=int)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, N_SLICES))
    for colour, idx in zip(colors, slice_ids):
        ax.plot(x, u_history[idx], color=colour, label=f't = {t_history[idx]:.2f}')
    ax.set_xlabel('x'); ax.set_ylabel('u(x, t)')
    ax.set_title('Inviscid Burgers Equation Time Slices')
    ax.legend(loc='upper right', title='Time')
    ax.set_xlim(x[0], x[-1]); ax.grid(True)
    plt.tight_layout(); plt.show()

u_mean = u.mean(axis=0)
u_centred = u - u_mean

_, S, Vt = np.linalg.svd(u_centred, full_matrices=False)
phi = Vt[:N_MODES, :]          # (N_MODES, NX) spatial POD modes
a = u_centred @ phi.T          # (Nt, N_MODES) POD temporal coefficients
U_reconstructed = a @ phi + u_mean

a_train = a
t_train = t
Nt = len(t_train)

X1 = a_train[:-1, :].T
X2 = a_train[1:, :].T
A_tilde = X2 @ np.linalg.pinv(X1)

evals, V = np.linalg.eig(A_tilde)
V_inv = np.linalg.pinv(V)

alpha_train = (V_inv @ a_train.T).T
Phi_dmd = phi.T @ V


def rdc(x, y, k=20, s=1.0 / 6.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    n = x.shape[0]

    try:
        cx = np.column_stack([rankdata(col) for col in x.T]) / n
        cy = np.column_stack([rankdata(col) for col in y.T]) / n

        O = np.ones((n, 1))
        X = np.column_stack([cx, O])
        Y = np.column_stack([cy, O])

        Rx = (s / X.shape[1]) * rng.standard_normal((X.shape[1], k))
        Ry = (s / Y.shape[1]) * rng.standard_normal((Y.shape[1], k))

        fX = np.column_stack([np.cos(X @ Rx), np.sin(X @ Rx)])
        fY = np.column_stack([np.cos(Y @ Ry), np.sin(Y @ Ry)])

        C = np.cov(np.hstack([fX, fY]).T)
        kk = fX.shape[1]
        Cxx = C[:kk, :kk] + 1e-8 * np.eye(kk)
        Cyy = C[kk:, kk:] + 1e-8 * np.eye(kk)
        Cxy = C[:kk, kk:]
        Cyx = C[kk:, :kk]

        eigs = np.linalg.eigvals(np.linalg.pinv(Cxx) @ Cxy @ np.linalg.pinv(Cyy) @ Cyx)
        eigs = np.clip(eigs.real, 0.0, 1.0)
        return float(np.sqrt(eigs.max())) if eigs.size else 0.0
    except np.linalg.LinAlgError:
        return 0.0


def mode_features(alpha_k):
    if np.allclose(alpha_k.imag, 0.0, atol=1e-10):
        return alpha_k.real.reshape(-1, 1)
    return np.column_stack([alpha_k.real, alpha_k.imag])


def find_conjugate_partner(evals, k, rel_tol=1e-4):
    ek = evals[k]
    if abs(ek.imag) < rel_tol * max(abs(ek.real), 1e-12):
        return None
    target = np.conj(ek)
    for j in range(len(evals)):
        if j != k and abs(evals[j] - target) < rel_tol * max(abs(target), 1e-12):
            return j
    return None


canonical = []
secondary_of = {}
represented = set()
for k in range(N_MODES):
    if k in represented:
        continue
    j = find_conjugate_partner(evals, k)
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

energy = np.mean(np.abs(alpha_train) ** 2, axis=0)
total_energy = energy.sum()
order = [k for k in np.argsort(energy)[::-1] if k in canonical]

rng = np.random.default_rng(RNG_SEED)
active = []
rdc_report = []
for k in order:
    fk = mode_features(alpha_train[:, k])
    if active:
        scores = {j: rdc(fk, mode_features(alpha_train[:, j]), rng=rng) for j in active}
        max_r = max(scores.values())
    else:
        scores, max_r = {}, 0.0
    rdc_report.append((k, scores, max_r))
    if max_r < RDC_THRESHOLD:
        active.append(k)
    if len(active) >= MAX_ACTIVE:
        break
    if energy[active].sum() / total_energy >= ENERGY_CAPTURE:
        break

active = sorted(active)
manifold_targets = sorted(set(canonical) - set(active))
partners = set(secondary_of.keys())

if PRINT_SUMMARY:
    print(f"Non-Conjugate Modes: {canonical}")
    print(f"Active Modes: {active}")
    print(f"Conjugate Mode Partners: {secondary_of}")
    print(f"Manifold-Regression Targets: {manifold_targets}")
    print(f"Active Mode Energy Fraction: {100 * energy[active].sum() / total_energy:.2f}%")


def build_library(state, degree):
    if state.ndim == 1:
        state = state.reshape(1, -1)
    Nt_, n = state.shape
    cols = [np.ones((Nt_, 1))]
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n), d):
            term = np.ones(Nt_)
            for c in combo:
                term = term * state[:, c]
            cols.append(term[:, np.newaxis])
    return np.column_stack(cols)


def pack_state_matrix(alpha_mat, idx):
    cols = []
    for k in idx:
        cols.append(alpha_mat[:, k].real[:, None])
        cols.append(alpha_mat[:, k].imag[:, None])
    return np.column_stack(cols)


def unpack_complex(y, idx):
    out = {}
    for i, k in enumerate(idx):
        out[k] = y[..., 2 * i] + 1j * y[..., 2 * i + 1]
    return out


def make_frols(Theta, max_iter=FROLS_MAX_ITER, alpha_reg=FROLS_ALPHA, kappa_scale=KAPPA_SCALE):
    col_norms = np.linalg.norm(Theta, 2, axis=0)
    col_norms[col_norms == 0] = 1.0
    cond = np.linalg.cond(Theta / col_norms)
    kappa = kappa_scale / cond if cond > 0 else 1e-16
    opt = FROLS(max_iter=max_iter, alpha=alpha_reg, kappa=kappa, normalize_columns=True)
    return opt, cond, kappa


xhat_train = pack_state_matrix(alpha_train, active)
n_active_real = xhat_train.shape[1]

if manifold_targets:
    Theta_manifold = build_library(xhat_train, degree=POLY_DEGREE_MANIFOLD)
    Y_manifold = pack_state_matrix(alpha_train, manifold_targets)

    opt_manifold, cond_manifold, kappa_manifold = make_frols(Theta_manifold)
    opt_manifold.fit(Theta_manifold, Y_manifold)
    Xi_manifold = opt_manifold.coef_.T

    if PRINT_SUMMARY:
        print(f"\nTheta Manifold Shape: {Theta_manifold.shape}")
        for i, m in enumerate(manifold_targets):
            nnz_re = int(np.sum(np.abs(Xi_manifold[:, 2 * i]) > 1e-10))
            nnz_im = int(np.sum(np.abs(Xi_manifold[:, 2 * i + 1]) > 1e-10))
            print(f"alpha_{m}: {nnz_re} terms (Re), {nnz_im} terms (Im)")
else:
    Theta_manifold, Xi_manifold = None, None
    if PRINT_SUMMARY:
        print("\nManifold Regression Skipped")

xhat_dot_train = np.gradient(xhat_train, t_train, axis=0)

Theta_dynamics = build_library(xhat_train, degree=POLY_DEGREE_DYNAMICS)
opt_dynamics, cond_dynamics, kappa_dynamics = make_frols(Theta_dynamics)
opt_dynamics.fit(Theta_dynamics, xhat_dot_train)
Xi_dynamics = opt_dynamics.coef_.T

if PRINT_SUMMARY:
    print(f"\nTheta Active Mode Dynamics Shape: {Theta_dynamics.shape}")
    for i, k in enumerate(active):
        nnz_re = int(np.sum(np.abs(Xi_dynamics[:, 2 * i]) > 1e-10))
        nnz_im = int(np.sum(np.abs(Xi_dynamics[:, 2 * i + 1]) > 1e-10))
        print(f"d(alpha_{k})/dt: {nnz_re} terms (Re), {nnz_im} terms (Im)")


def reduced_rhs(t_val, y, Xi_dyn, degree):
    Theta_row = build_library(y.reshape(1, -1), degree=degree)
    return (Theta_row @ Xi_dyn).flatten()


def make_blowup_event(threshold):
    def event(t_val, y, *args):
        return threshold - np.linalg.norm(y)
    event.terminal = True
    event.direction = -1
    return event


xhat0 = xhat_train[0, :]

sol = solve_ivp(
    fun=reduced_rhs,
    t_span=(t_train[0], t_train[-1]),
    y0=xhat0,
    t_eval=t_train,
    args=(Xi_dynamics, POLY_DEGREE_DYNAMICS),
    method='RK45',
    events=make_blowup_event(BLOWUP_THRESHOLD),
)

xhat_sim = np.tile(sol.y[:, -1], (len(t_train), 1))
xhat_sim[:sol.y.shape[1]] = sol.y.T

if sol.status != 0 or sol.y.shape[1] < len(t_train):
    print(f"Integration terminated early at t={sol.t[-1]:.3f} (of {t_train[-1]:.3f})")

alpha_sim = np.zeros((len(t_train), N_MODES), dtype=complex)

active_vals = unpack_complex(xhat_sim, active)
for k, val in active_vals.items():
    alpha_sim[:, k] = val

if manifold_targets:
    Theta_manifold_sim = build_library(xhat_sim, degree=POLY_DEGREE_MANIFOLD)
    Y_manifold_sim = Theta_manifold_sim @ Xi_manifold
    manifold_vals = unpack_complex(Y_manifold_sim, manifold_targets)
    for m, val in manifold_vals.items():
        alpha_sim[:, m] = val

for secondary, primary in secondary_of.items():
    alpha_sim[:, secondary] = np.conj(alpha_sim[:, primary])

a_sim_complex = (V @ alpha_sim.T).T
imag_residual = np.max(np.abs(a_sim_complex.imag))
if PRINT_SUMMARY:
    print(f"\nMax imaginary term: {imag_residual:.3e} (should be about 0)")
a_sim = a_sim_complex.real

U_sindy = a_sim @ phi + u_mean

if PLOT_SLICES:
    N_SLICES = 5
    slice_ids = np.linspace(0, len(t_history) - 1, N_SLICES, dtype=int)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, N_SLICES))
    for colour, idx in zip(colors, slice_ids):
        ax.plot(x, U_sindy[idx], color=colour, label=f't = {t[idx]:.2f}')
    ax.set_xlabel('x'); ax.set_ylabel('u(x, t)')
    ax.set_title('SINDy Reconstruction Time Slices')
    ax.legend(loc='upper right', title='Time')
    ax.set_xlim(x[0], x[-1]); ax.grid(True)
    plt.tight_layout(); plt.show()

if PLOT_MODES:
    for k in range(N_MODES):
        fig, ax = plt.subplots(figsize=(9, 3))
        tag = "active" if k in active else ("partner" if k in partners else "manifold")
        ax.plot(t, a[:, k], color="black", label="True (POD)")
        ax.plot(t, a_sim[:, k], "--", color="steelblue", label=f"Reduced model ({tag})")
        ax.set_xlabel("Time"); ax.set_ylabel("Amplitude")
        ax.set_title(f"POD mode {k + 1} ({tag})")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout(); plt.show()

azim_val = -15
if PLOTTING:
    Xm, Tm = np.meshgrid(x, t_history)
    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.view_init(azim=azim_val)
    ax1.plot_surface(Xm, Tm, u, cmap="plasma")
    ax1.set_title("True Solution"); ax1.set_xlabel("x"); ax1.set_ylabel("t")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.view_init(azim=azim_val)
    ax2.plot_surface(Xm, Tm, U_reconstructed, cmap="plasma")
    ax2.set_title(f"True ({N_MODES} POD Modes)"); ax2.set_xlabel("x"); ax2.set_ylabel("t")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.view_init(azim=azim_val)
    ax3.plot_surface(Xm, Tm, U_sindy, cmap="plasma")
    ax3.set_title(f"Manifold+SINDy ({len(active)} active DOF)")
    ax3.set_xlabel("x"); ax3.set_ylabel("t")

    plt.tight_layout(); plt.show()

print("\nSINDy-Modal MSE:", np.mean((U_reconstructed - U_sindy) ** 2))
print("Modal-True MSE:", np.mean((u - U_reconstructed) ** 2))
print("SINDy-True MSE:", np.mean((u - U_sindy) ** 2))