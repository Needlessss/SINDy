import numpy as np
import matplotlib.pyplot as plt
import itertools
import PDE_FIND
from sklearn.model_selection import KFold
from scipy.integrate import solve_ivp
import time


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

            if (n+1) % save_every == 0:
                times.append((n+1)*dt)
                solutions.append(u.copy())

        return np.array(times), np.array(solutions)


solver = KdVSolver(L=60, N=512)

c1, c2 = 0.5, 0.2
u0 = solver.two_solitons(c1=c1, c2=c2, sep=18)

dt = 0.001
T = 10

times, sols = solver.solve(u0, dt, T)
x = solver.x
dt = times[1]-times[0]
N = len(x)


fig = plt.figure(figsize=(6, 5))
ax3 = plt.subplot(1, 1, 1, projection='3d')
X_sub = x
T_sub = times
Xm, Tm = np.meshgrid(X_sub, T_sub)
surf = ax3.plot_surface(Xm, Tm, sols, cmap='viridis')
ax3.set_xlabel('x')
ax3.set_ylabel('t')
ax3.set_zlabel('u')
ax3.set_title('KdV Solution Two Soliton Case')
plt.tight_layout()
plt.show()

U_list = np.array(sols)
x = np.array(solver.x)
h = solver.dx
t_list = np.array(times)

U_dot = np.gradient(U_list, dt, axis=0)

U_x = np.zeros_like(U_list)
U_xx = np.zeros_like(U_list)
U_xxx = np.zeros_like(U_list)

k = solver.k

for i in range(len(t_list)):
    U = U_list[i, :]

    U_hat = np.fft.fft(U)

    Ux_hat = 1j * k * U_hat
    U_x[i, :] = np.real(np.fft.ifft(Ux_hat))

    Uxx_hat = (1j * k) ** 2 * U_hat
    U_xx[i, :] = np.real(np.fft.ifft(Uxx_hat))

    Uxxx_hat = (1j * k) ** 3 * U_hat
    U_xxx[i, :] = np.real(np.fft.ifft(Uxxx_hat))


candidate_library = np.column_stack([
    U_x.flatten(),
    U_xx.flatten(),
    U_xxx.flatten(),
    (U_list*U_x).flatten(),
    (U_list * U_xx).flatten(),
    (U_list * U_xxx).flatten(),
])

U_dot_flat = U_dot.flatten()

maxit = 10000
normalize = 2


lambdas = [0, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
tolerances = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]

#lambdas = [0]
#tolerances = [1e-4]

k_folds = 5

kf = KFold(n_splits=k_folds, shuffle=False)
time_indices = np.arange(len(t_list))

best_mse = float('inf')
best_params = {'lam': None, 'tol': None}

start = time.time()
print(f"Starting {k_folds}-fold Cross-Validation...")
plot = True
for lam, tol in itertools.product(lambdas, tolerances):
    fold_mses = []
    for train_times, val_times in kf.split(time_indices):
        n_t, n_x = U_list.shape
        n_features = candidate_library.shape[1]

        Theta = candidate_library.reshape(n_t, n_x, n_features)

        X_train = Theta[train_times].reshape(-1, n_features)
        y_train = U_dot[train_times].reshape(-1, 1)

        w = np.real(PDE_FIND.STRidge(X_train, y_train, lam, maxit=1000, tol=tol, normalize=normalize))

        def rhs_pde(U, w):
            u_x = np.zeros(U.shape)
            u_xx = np.zeros(U.shape)
            u_xxx = np.zeros(U.shape)

            u_hat = np.fft.fft(U)

            ux_hat = 1j * k * u_hat
            u_x[:] = np.real(np.fft.ifft(ux_hat))

            uxx_hat = (1j * k) ** 2 * u_hat
            u_xx[:] = np.real(np.fft.ifft(uxx_hat))

            uxxx_hat = (1j * k) ** 3 * u_hat
            u_xxx[:] = np.real(np.fft.ifft(uxxx_hat))

            Theta = np.column_stack([
                u_x,
                u_xx,
                u_xxx,
                U*u_x,
                U*u_xx,
                U*u_xxx
            ])
            Theta = Theta.T
            return_matrix = np.zeros(np.shape(Theta[0]))
            for i in range(len(w)):
                return_matrix += w[i][0] * Theta[i]
            return return_matrix


        def rhs_wrapper(t, U_flat):
            return rhs_pde(U_flat, w)


        sol = solve_ivp(
            rhs_wrapper,
            (0, len(val_times) * dt),
            U_list[val_times[0]].flatten(),
            t_eval=t_list[val_times] - t_list[val_times[0]],
            method="RK45"
        )

        traj_arr = sol.y.T.reshape(-1, N)
        err = np.mean((traj_arr - U_list[val_times]) ** 2)
        fold_mses.append(err)

        if plot:
            X, T = np.meshgrid(x, t_list[val_times])
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, T, traj_arr, cmap="viridis")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            ax.set_zlabel("u(x,t)")
            ax.set_title(f"est lam {lam}, tol {tol}")
            plt.tight_layout()
            plt.show()
            plot = False

    avg_mse = np.mean(fold_mses)
    print(f"Lambda: {lam} | Tol: {tol} | Avg MSE: {avg_mse:.6e}")

    if avg_mse < best_mse:
        best_mse = avg_mse
        best_params['lam'] = lam
        best_params['tol'] = tol

print(f"Cross validation Time: {time.time()-start}")

print("-" * 100)
print(f"Best Parameters: {best_params}")
print(f"Lowest MSE: {best_mse:.6e}")

lam = best_params['lam']
tol = best_params['tol']

coef = PDE_FIND.STRidge(candidate_library, U_dot_flat.reshape(-1, 1), lam, maxit, tol, normalize=normalize)
coef = np.real(coef.flatten())
print('-' * 100)
print(f"Discovered coefficients (l={lam}, t={tol}:")
labels = ['u_x', 'u_xx', 'u_xxx', 'u*u_x', 'u*u_xx', 'u*u_xxx']
print("\nDiscovered PDE: du/dt = \n", end="")
terms = []
for i, (label, c) in enumerate(zip(labels, coef)):
    if np.abs(c) > tol:
        if c > 0 and terms:
            terms.append(f"+ {c:.4f}*{label}")
        else:
            terms.append(f"{c:.4f}*{label}")

if terms:
    print("Equation: du/dt = " + " ".join(terms))
else:
    print("No significant terms found")

print(f"\nFull coefficient vector: {coef}")
print('-' * 100)
