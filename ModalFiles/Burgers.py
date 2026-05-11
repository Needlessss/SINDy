import numpy as np
import matplotlib.pyplot as plt
from pysindy.optimizers import FROLS
from scipy.integrate import solve_ivp

#Disable this if you aren't certain ill conditioned matrices won't cause you issues
################################################################################
import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=LinAlgWarning)
################################################################################

N_MODES    = 10
TRAIN_FRAC = 0.5

PLOT_SLICES     = True
PLOT_MODES      = False
PLOTTING        = True
PLOT_AMPLITUDES = True

METHOD = "FROLS"

def CFLcondition(u, dx, CFLcoe):
    toll = 1e-10
    dtmax = CFLcoe * dx / np.maximum(np.abs(u), toll)
    dt = np.min(dtmax)
    return dt


def PeriodicBoundary(u):
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
            if s > 0:
                u0 = u[i]
            else:
                u0 = u[i + 1]

        else:  # rarefaction
            if u[i] > toll:
                u0 = u[i]
            elif u[i + 1] < -toll:
                u0 = u[i + 1]
            else:
                u0 = 0.0

        F[i] = 0.5 * u0**2

    return F


L = 1000
x0 = -50
nc = 1000
dx = L / nc

x = np.linspace(x0 - dx/2, x0 + L + dx/2, nc + 2)
u = np.zeros(nc + 2)

NX = len(x)
tend = 1500
CFLcoe = 0.9
t = 0.0

itest = 1

if itest == 1:
    testname = "Right Rarefaction"
    uL, uR = 1, 2
elif itest == 2:
    testname = "Left Rarefaction"
    uL, uR = -2, -1
elif itest == 3:
    testname = "Centred Rarefaction"
    uL, uR = -1, 2
elif itest == 4:
    testname = "Left Travelling Shock"
    uL, uR = -1, -2
elif itest == 5:
    testname = "Right Travelling Shock"
    uL, uR = 3, 1
elif itest == 6:
    testname = "Steady Shock"
    uL, uR = 1, -1
else:
    raise ValueError("Invalid test case")

print(testname)

u[x <= x0 + L/2] = uL
u[x > x0 + L/2] = uR

u_history = [u.copy()]
t_history = [t]

while t < tend:

    u = PeriodicBoundary(u)

    dt = CFLcondition(u, dx, CFLcoe)
    dt = min(dt, tend - t)

    F = GodunovFlux(u)

    u[1:nc+1] = u[1:nc+1] - dt/dx * (F[1:nc+1] - F[0:nc])
    t += dt

    u_history.append(u.copy())
    t_history.append(t)

u_history = np.array(u_history)
t_history = np.array(t_history)

if PLOT_SLICES:
    N_SLICES  = 5
    slice_ids = np.linspace(0, len(t_history) - 1, N_SLICES, dtype=int)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors  = plt.cm.viridis(np.linspace(0, 1, N_SLICES))

    for colour, idx in zip(colors, slice_ids):
        ax.plot(x, u_history[idx], color=colour,
                label=f't = {t_history[idx]:.2f}')

    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title('1D Burgers Equation — Time Slices (Periodic BC)')
    ax.legend(loc='upper right', title='Time')
    ax.set_xlim(x[0], x[-1])
    ax.grid(True)
    plt.tight_layout()
    plt.show()

modes = np.fft.fft(u_history, axis=1)

if PLOT_AMPLITUDES:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(modes.real[:, :50].T, aspect='auto', origin='lower')
    axes[0].set_xlabel("Time index")
    axes[0].set_ylabel("Mode number")
    axes[0].set_title("Fourier Modal Evolution (Real)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(modes.imag[:, :50].T, aspect='auto', origin='lower')
    axes[1].set_xlabel("Time index")
    axes[1].set_ylabel("Mode number")
    axes[1].set_title("Fourier Modal Evolution (Imaginary)")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

a = modes[:, 1:N_MODES+1]

k_full     = 2 * np.pi * np.fft.fftfreq(NX, d=dx)
k_positive = k_full[1:N_MODES+1]

N_total = len(t_history)
N_train = int(TRAIN_FRAC * N_total)

a_train = a[:N_train]
t_train = t_history[:N_train]


def compute_fourier_nonlinearity(a_complex_slice, a0_slice, k_positive, NX, dx):
    L = NX * dx
    N_time, K = a_complex_slice.shape
    nonlinear = np.zeros_like(a_complex_slice, dtype=complex)

    for t_idx in range(N_time):
        modes_dict = {0: a0_slice[t_idx]}
        for j in range(K):
            val = a_complex_slice[t_idx, j]
            modes_dict[j+1] = val
            modes_dict[-(j+1)] = np.conj(val)

        for j in range(K):
            k_idx = j + 1
            s = 0.0 + 0.0j
            for m in range(-K, K + 1):
                km = k_idx - m
                if m in modes_dict and km in modes_dict:
                    s += modes_dict[m] * modes_dict[km]
            #nonlinear[t_idx, j] = -1j * s / NX
            nonlinear[t_idx, j] = s

    return nonlinear


def build_global_library(a_complex_slice, a0_slice, k_positive, NX, dx):
    nonlinear = compute_fourier_nonlinearity(
        a_complex_slice, a0_slice, k_positive, NX, dx
    )
    columns, labels = [], []
    for k in range(len(k_positive)):
        columns.append(nonlinear[:, k])
        labels.append(f"NL_k{k+1}")

        #columns += [diffusion[:, k], nonlinear[:, k]]
        #labels  += [f"diff_k{k+1}", f"NL_k{k+1}"]

    return np.column_stack(columns), labels


a_dot_train = np.gradient(a_train, t_train, axis=0)

a0_all   = modes[:, 0]
a0_train = a0_all[:N_train]
Theta_train, lib_labels = build_global_library(
    a_train, a0_train, k_positive, NX, dx
)

print(f"\nLibrary shape : {Theta_train.shape}  ({len(lib_labels)} terms × {N_train} samples)")
print(f"Target  shape : {a_dot_train.shape}  ({N_MODES} complex outputs)\n")

if METHOD == "FROLS":
    opt = FROLS(max_iter=1, alpha=0, kappa=0)
    opt.fit(Theta_train, a_dot_train)

    Xi = opt.coef_.T

print(f"Xi shape: {Xi.shape}")
print(f"\n{'Output':<22} {'Selected library terms'}")
print("-" * 70)

output_labels = [f"dâ_k{k+1}/dt" for k in range(N_MODES)]

for col_idx, out_label in enumerate(output_labels):
    selected = [
        f"{lib_labels[i]} ({Xi[i, col_idx]:.6g})"
        for i in range(len(lib_labels))
        if Xi[i, col_idx] != 0
    ]

    print(f"  {out_label:<20}  {', '.join(selected) if selected else '(none)'}")


def sindy_rhs_ivp(t_val, X, Xi, k_positive, a0_const, NX, dx):
    a_complex = X.reshape(1, -1)
    a0        = np.array([a0_const])
    Theta, _  = build_global_library(a_complex, a0, k_positive, NX, dx)
    return (Theta @ Xi).flatten()


a0_const = modes[0, 0]
X0 = a[0]

sol = solve_ivp(
    fun=lambda t_val, X: sindy_rhs_ivp(t_val, X, Xi, k_positive, a0_const, NX, dx),
    t_span=(t_history[0], t_history[-1]),
    y0=X0,
    t_eval=t_history,
    method="RK45",
)

a_sim_complex = sol.y.T

U_hat_sindy = np.zeros((len(t_history), NX), dtype=complex)
U_hat_sindy[:, 1:N_MODES+1] = a_sim_complex
U_hat_sindy[:, 0]  = modes[:, 0]
U_hat_sindy[:, NX-N_MODES:NX]  = np.conj(a_sim_complex[:, ::-1])
U_sindy = np.fft.ifft(U_hat_sindy, axis=1).real

U_hat_filtered = np.zeros((len(t_history), NX), dtype=complex)
U_hat_filtered[:, 1:N_MODES+1] = a
U_hat_filtered[:, 0]  = modes[:, 0]
U_hat_filtered[:, NX-N_MODES:NX] = np.conj(a[:, ::-1])
U_reconstructed = np.fft.ifft(U_hat_filtered, axis=1).real

if PLOT_MODES:
    for k in range(N_MODES):
        fig, ax = plt.subplots(figsize=(9, 3))

        ax.plot(t_history, a[:, k].real,
                color="black",    label="True (Re)")
        ax.plot(t_history, a_sim_complex[:, k].real, "--",
                color="steelblue", label="SINDy (Re)")
        ax.plot(t_history, a[:, k].imag,
                color="gray",     label="True (Im)", alpha=0.6)
        ax.plot(t_history, a_sim_complex[:, k].imag, ":",
                color="salmon",   label="SINDy (Im)", alpha=0.9)

        ax.axvline(t_history[N_train], color="k", linestyle=":",
                   alpha=0.5, label="Train | Test")

        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Mode {k+1} (k = {k_positive[k]:.3f})")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.show()

azim_val = 15

if PLOTTING:
    Xm, Tm = np.meshgrid(x, t_history)

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.view_init(azim=azim_val)
    ax1.plot_surface(Xm, Tm, u_history, cmap="plasma")
    ax1.set_title("True Solution")
    ax1.set_xlabel("x"); ax1.set_ylabel("t")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.view_init(azim=azim_val)
    ax2.plot_surface(Xm, Tm, U_reconstructed, cmap="plasma")
    ax2.set_title(f"True ({N_MODES} Fourier Modes)")
    ax2.set_xlabel("x"); ax2.set_ylabel("t")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.view_init(azim=azim_val)
    ax3.plot_surface(Xm, Tm, U_sindy, cmap="plasma")
    ax3.set_title("SINDy Reconstruction")
    ax3.set_xlabel("x"); ax3.set_ylabel("t")

    plt.tight_layout()
    plt.show()

print("\nSINDy-Modal MSE:", np.mean((U_reconstructed - U_sindy) ** 2))
print("Modal-True  MSE:", np.mean((u - U_reconstructed) ** 2))
print("SINDy-True  MSE:", np.mean((u - U_sindy) ** 2))