import numpy as np
import matplotlib.pyplot as plt
import PDE_FIND
from scipy.integrate import solve_ivp

space_res = [64, 128, 256, 512]
time_err_list = []
res_vals = np.zeros(np.shape(space_res))
for i in range(len(res_vals)):
    res_vals[i] = 60/space_res[i]
for scale_val in space_res:
    class KdVSolver:
        def __init__(self, L=60, N=256):
            self.L = L
            self.N = N
            self.x = np.linspace(0, L, N, endpoint=False)
            self.dx = L / N
            self.k = 2 * np.pi * np.fft.fftfreq(N, d=self.dx)

        def soliton(self, x, c, x0):
            return 2 * c / np.cosh(np.sqrt(c) * (x - x0)) ** 2

        def two_solitons(self, c1=0.5, c2=0.2, sep=20):
            x1 = self.L / 3
            x2 = x1 + sep
            return self.soliton(self.x, c1, x1) + self.soliton(self.x, c2, x2)

        def solve(self, u0, dt, T, save_every=1):
            nt = int(T / dt)
            u = u0.copy()
            times = [0]
            solutions = [u.copy()]

            L_op_half = np.exp(-1j * self.k ** 3 * dt / 2)

            def nonlinear(u_):
                u_hat_ = np.fft.fft(u_)
                ux_hat = 1j * self.k * u_hat_
                ux = np.real(np.fft.ifft(ux_hat))
                return 6 * u_ * ux

            for n in range(nt):
                u_hat = np.fft.fft(u)
                u_hat = L_op_half * u_hat
                u = np.real(np.fft.ifft(u_hat))

                k1 = nonlinear(u)
                k2 = nonlinear(u + 0.5 * dt * k1)
                k3 = nonlinear(u + 0.5 * dt * k2)
                k4 = nonlinear(u + dt * k3)
                u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

                u_hat = np.fft.fft(u)
                u_hat = L_op_half * u_hat
                u = np.real(np.fft.ifft(u_hat))

                if (n + 1) % save_every == 0:
                    times.append((n + 1) * dt)
                    solutions.append(u.copy())

            return np.array(times), np.array(solutions)


    LAM = 0
    TOL = 1e-4
    TRAIN_FRAC = 0.8

    solver = KdVSolver(L=60, N=scale_val)
    u0 = solver.two_solitons(c1=0.5, c2=0.2, sep=18)

    times, sols = solver.solve(u0, dt=0.001, T=2)
    x = solver.x
    dt = times[1] - times[0]
    k = solver.k
    N = len(x)

    U_list = np.array(sols)
    t_list = np.array(times)
    n_t = len(t_list)

    U_dot = np.gradient(U_list, dt, axis=0)

    U_x   = np.zeros_like(U_list)
    U_xx  = np.zeros_like(U_list)
    U_xxx = np.zeros_like(U_list)

    for i in range(n_t):
        U_hat = np.fft.fft(U_list[i])
        U_x[i]   = np.real(np.fft.ifft(1j * k * U_hat))
        U_xx[i]  = np.real(np.fft.ifft((1j * k) ** 2 * U_hat))
        U_xxx[i] = np.real(np.fft.ifft((1j * k) ** 3 * U_hat))

    n_train = int(n_t * TRAIN_FRAC)
    train_idx = np.arange(n_train)
    test_idx  = np.arange(n_train, n_t)

    n_features = 6

    def build_library(U, Ux, Uxx, Uxxx):
        return np.column_stack([Ux, Uxx, Uxxx, U * Ux, U * Uxx, U * Uxxx])

    Theta_full = build_library(U_list, U_x, U_xx, U_xxx)
    n_t_full, n_x = U_list.shape
    Theta_3d = Theta_full.reshape(n_t_full, n_x, n_features)
    Theta_train = build_library(
        U_list[train_idx].reshape(-1),
        U_x[train_idx].reshape(-1),
        U_xx[train_idx].reshape(-1),
        U_xxx[train_idx].reshape(-1),
    )

    def flatten_library(idx):
        rows = []
        for i in idx:
            rows.append(build_library(U_list[i], U_x[i], U_xx[i], U_xxx[i]))
        return np.vstack(rows)
    X_train = flatten_library(train_idx)
    y_train = U_dot[train_idx].reshape(-1, 1)

    print(f"Training STRidge  lam={LAM}, tol={TOL} ...")
    w = np.real(PDE_FIND.STRidge(X_train, y_train, LAM, maxit=10000, tol=TOL, normalize=0))
    print(f"Learned weights: {w.flatten()}")

    def rhs_pde(U_flat, w):
        u_hat  = np.fft.fft(U_flat)
        u_x    = np.real(np.fft.ifft(1j * k * u_hat))
        u_xx   = np.real(np.fft.ifft((1j * k) ** 2 * u_hat))
        u_xxx  = np.real(np.fft.ifft((1j * k) ** 3 * u_hat))
        Theta  = np.column_stack([u_x, u_xx, u_xxx, U_flat * u_x,
                                   U_flat * u_xx, U_flat * u_xxx])
        return Theta @ w.flatten()

    def rhs_wrapper(t, U_flat):
        return rhs_pde(U_flat, w)

    t_test_local = t_list[test_idx] - t_list[test_idx[0]]
    sol = solve_ivp(
        rhs_wrapper,
        (0, t_test_local[-1]),
        U_list[test_idx[0]].copy(),
        t_eval=t_test_local,
        method="RK45",
        rtol=1e-6, atol=1e-8,
    )

    pred = sol.y.T
    true = U_list[test_idx]

    test_mse = np.mean((pred - true) ** 2)
    print(f"\nTest MSE: {test_mse:.6e}")

    labels = ['u_x', 'u_xx', 'u_xxx', 'u*u_x', 'u*u_xx', 'u*u_xxx']
    coef   = w.flatten()

    terms = []
    for label, c in zip(labels, coef):
        if np.abs(c) > TOL:
            sign = "+ " if (c > 0 and terms) else ""
            terms.append(f"{sign}{c:.4f}*{label}")

    time_err_list.append(test_mse)
    print(f"time val {scale_val} done")
plt.plot(res_vals, time_err_list, marker='o')

plt.yscale("log")

plt.xlabel("Spatial Resolution (dx)")
plt.ylabel("Error Value")
plt.title("Data Spatial Resolution vs Model Error")

plt.grid(True, which="both")

plt.show()