import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import UnivariateSpline
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import solve_ivp

def STRidge(X0, y, lam, maxit, tol, normalize=0, print_results=False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n, d = X0.shape
    X = np.zeros((n, d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d, 1))
        for i in range(0, d):
            Mreg[i] = 1.0 / (np.linalg.norm(X0[:, i], normalize))
            X[:, i] = Mreg[i] * X0[:, i]
    else:
        X = X0

    # Get the standard ridge esitmate
    if lam != 0:
        w = np.linalg.lstsq(X.T.dot(X) + lam * np.eye(d), X.T.dot(y), rcond=None)[0]
    else:
        w = np.linalg.lstsq(X, y, rcond=None)[0]
    num_relevant = d
    biginds = np.where(abs(w) > tol)[0]

    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where(abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]

        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                # if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else:
                break
        biginds = new_biginds

        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0:
            w[biginds] = \
            np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam * np.eye(len(biginds)), X[:, biginds].T.dot(y),
                            rcond=None)[0]
        else:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if len(biginds)>0: w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    if normalize != 0:
        return np.multiply(Mreg, w)
    else:
        return w

noise_vals = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
mse_vals = []

GREG = 1000

h = 0.01
x = np.arange(-10, 10 + h, h)
N = len(x)


def f(u):
    return 4 * u * (1 - u ** 2)


diagonals = []
diagonals.append(-2 * np.ones(N))
diagonals.append(np.ones(N - 1))
diagonals.append(np.ones(N - 1))
A = sp.sparse.diags(diagonals, [0, 1, -1], shape=(N, N), format="lil")

A[0, 1] = 2
A[-1, -2] = 2
A = A.tocsr()

k = 0.001

I = sp.sparse.identity(N)
M_left = I - (k / (h ** 2)) * A

solver = spla.factorized(M_left.tocsc())

U = np.exp(-x ** 2)

t = 0.0
U_list = [U.copy()]
t_list = [t]

for n in range(GREG):
    rhs = U + k * f(U)
    U_new = solver(rhs)
    t += k
    U = U_new
    U_list.append(U.copy())
    t_list.append(t)

U_list_temp = np.array(U_list)
True_U = U_list_temp.copy()
t_list = np.array(t_list)

for noise_val in noise_vals:
    LAM = 1e5
    TOL = 0.05

    noise = np.random.normal(0, noise_val, size=U_list_temp.shape)
    U_list = U_list_temp + noise


    print(f"Generated trajectory shape: {U_list.shape}")

    X, T = np.meshgrid(x, t_list)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, T, U_list, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("u(x,t)")
    ax.set_title(f"Numerical solution to $u_t = u_{{xx}} + 4u(1-u^2)$")
    plt.tight_layout()
    plt.show()

    U_dot = np.zeros_like(U_list)
    U_dot[1:-1, :] = (U_list[2:, :] - U_list[:-2, :]) / (2 * k)
    U_dot[0, :] = (U_list[1, :] - U_list[0, :]) / k
    U_dot[-1, :] = (U_list[-1, :] - U_list[-2, :]) / k

    print("1")

    window_t = 31
    window_x = 31

    polyorder = 3

    U_dot = savgol_filter(
        U_list,
        window_length=window_t,
        polyorder=polyorder,
        deriv=1,
        delta=k,
        axis=0,
        mode='interp'
    )

    U_x = savgol_filter(
        U_list,
        window_length=window_x,
        polyorder=polyorder,
        deriv=1,
        delta=h,
        axis=1,
        mode='interp'
    )

    U_xx = savgol_filter(
        U_list,
        window_length=window_x,
        polyorder=polyorder,
        deriv=2,
        delta=h,
        axis=1,
        mode='interp'
    )
    """
    for i in range(len(t_list)):
        U = U_list[i, :]
        U_x[i, 1:-1] = (U[2:] - U[:-2]) / (2 * h)
        U_x[i, 0] = (U[1] - U[0]) / h
        U_x[i, -1] = (U[-1] - U[-2]) / h

        U_xx[i, :] = (A @ U) / h ** 2
    """
    U_0 = np.ones_like(U_list)
    U_sq = U_list ** 2
    U_cu = U_list ** 3

    candidate_library_train = np.column_stack([
        U_0.flatten(),
        U_list.flatten(),
        U_sq.flatten(),
        U_cu.flatten(),
        U_x.flatten(),
        U_xx.flatten(),
    ])

    U_dot_train = U_dot.flatten()

    print("2")

    coef = STRidge(candidate_library_train, U_dot_train.reshape(-1, 1), LAM, 1000, TOL, normalize=0)
    coef = coef.flatten().real

    coef_zero = np.zeros_like(coef)
    res_idx = np.where(abs(coef) >= TOL,
                        True,
                        False)
    for i in range(len(res_idx)):
        if res_idx[i]:
            coef_zero[i] = coef[i]

    coef = coef_zero.copy()

    print("\nDiscovered coefficients:")
    labels = ['1', 'u', 'u²', 'u³', 'u_x', 'u_xx']
    terms = []
    for label, c in zip(labels, coef):
        if np.abs(c) > 1e-6:
            print(f"{label}: {c:.6f}")
            sign = "+ " if (c > 0 and terms) else ""
            terms.append(f"{sign}{c:.4f}*{label}")

    print("\n" + "─" * 70)
    if terms:
        print("Discovered PDE: du/dt = " + " ".join(terms))
    else:
        print("No significant terms found")
    print(f"Full coefficient vector: {coef}")
    print("─" * 70)

    def compute_derivatives(U_vec):
        u_xx = (A @ U_vec) / h ** 2
        u_x = np.zeros_like(U_vec)
        u_x[1:-1] = (U_vec[2:] - U_vec[:-2]) / (2 * h)
        u_x[0] = (U_vec[1] - U_vec[0]) / h
        u_x[-1] = (U_vec[-1] - U_vec[-2]) / h
        return u_x, u_xx


    def rhs_pde(t, U_vec):
        u_x, u_xx = compute_derivatives(U_vec)

        features = np.column_stack([
            np.ones(N),
            U_vec,
            U_vec ** 2,
            U_vec ** 3,
            u_x,
            u_xx,
        ])

        return features @ coef

    t_eval = t_list

    print("3")

    sol = solve_ivp(
        rhs_pde,
        (t_list[0], t_list[-1]),
        True_U[0],
        t_eval=t_eval,
        method="Radau",
    )

    U_sindy = sol.y.T

    print("True trajectory shape:", True_U.shape)
    print("SINDy trajectory shape:", U_sindy.shape)

    try:
        mse = np.mean((True_U - U_sindy) ** 2)
        rmse = np.sqrt(mse)

        relative_l2 = np.linalg.norm(True_U - U_sindy) / np.linalg.norm(True_U)
    except:
        mse = 1
        rmse = 1

    mse_vals.append(mse)

plt.plot(noise_vals, mse_vals, marker='o')

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Noise Function S.D.")
plt.ylabel("SINDy-True Solution MSE")
plt.title("Data Noise S.D. vs SINDy Model Error with Savgol Filter")

plt.grid(True, which="both")

plt.show()
