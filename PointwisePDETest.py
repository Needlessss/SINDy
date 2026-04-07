from sklearn.model_selection import KFold
import itertools
import numpy as np
import scipy.sparse.linalg as spla
from scipy.interpolate import UnivariateSpline
import scipy as sp
import matplotlib.pyplot as plt


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
    if len(biginds) > 0: w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

    if normalize != 0:
        return np.multiply(Mreg, w)
    else:
        return w


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
# U = np.sin(x)
noise = np.random.normal(0, 0.001, size=U.shape)
# U += noise

t = 0.0
U_list = [U.copy()]
t_list = [t]

for n in range(1000):
    rhs = U + k * f(U)
    U_new = solver(rhs)

    t += k

    diff = np.linalg.norm(U_new - U, ord=np.inf)

    U = U_new

    U_list.append(U.copy())
    t_list.append(t)

U_list = np.array(U_list)
t_list = np.array(t_list)
# U_list = U_list + noise

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

# Time derivative matrix:

U_dot = np.zeros_like(U_list)
U_dot[1:-1, :] = (U_list[2:, :] - U_list[:-2, :]) / (2 * k)
U_dot[0, :] = (U_list[1, :] - U_list[0, :]) / k
U_dot[-1, :] = (U_list[-1, :] - U_list[-2, :]) / k

U_x = np.zeros_like(U_list)
U_xx = np.zeros_like(U_list)

for i in range(len(t_list)):
    U = U_list[i, :]
    U_x[i, 1:-1] = (U[2:] - U[:-2]) / (2 * h)
    U_x[i, 0] = (U[1] - U[0]) / h
    U_x[i, -1] = (U[-1] - U[-2]) / h

    U_xx[i, :] = (A @ U) / h ** 2

U_0 = np.ones_like(U_list)
U_sq = U_list ** 2
U_cu = U_list ** 3

candidate_library = np.column_stack([
    U_0.flatten(),  # 1
    U_list.flatten(),  # u
    U_sq.flatten(),  # u²
    U_cu.flatten(),  # u³
    U_x.flatten(),  # u_x
    U_xx.flatten(),  # u_xx

])

U_dot_flat = U_dot.flatten()

print(f"Library shape: {candidate_library.shape}")
print(f"U_dot shape: {U_dot_flat.shape}")

lam = 30
maxit = 10000
tol = 0.01
normalize = 0

#lambdas = [0, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]  # Ridge penalties
#tolerances = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5]  # Sparsity thresholds

lambdas=[1]
tolerances=[0.1]

k_folds = 5

kf = KFold(n_splits=k_folds, shuffle=False)
time_indices = np.arange(len(t_list))

best_mse = float('inf')
best_params = {'lam': None, 'tol': None}

print(f"Starting {k_folds}-fold Cross-Validation...")

for lam, tol in itertools.product(lambdas, tolerances):
    fold_mses = []
    l = 0
    for train_times, val_times in kf.split(time_indices):
        # Split data
        n_t, n_x = U_list.shape
        n_features = candidate_library.shape[1]

        Theta = candidate_library.reshape(n_t, n_x, n_features)

        X_train = Theta[train_times].reshape(-1, n_features)
        y_train = U_dot[train_times].reshape(-1, 1)

        # Train model
        w = STRidge(X_train, y_train, lam, maxit=1000, tol=tol)

        # Validation design matrix and targets
        X_val = Theta[val_times].reshape(-1, n_features)
        y_val = U_dot[val_times].reshape(-1, 1)

        # Predicted time derivatives
        y_pred = X_val @ w

        # Mean-squared error on u_t
        mse = np.mean((y_val - y_pred) ** 2)

        fold_mses.append(mse)

        #print(w)

        """
        if l == 0:
            X, T = np.meshgrid(x, t_list[val_times])
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(X, T, traj_arr, cmap="viridis")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            ax.set_zlabel("u(x,t)")
            ax.set_title(f"est lam {lam}, tol {tol}")
            plt.tight_layout()
            plt.show()

            X, T = np.meshgrid(x, t_list[val_times])
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(X, T, U_list[val_times], cmap="viridis")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            ax.set_zlabel("u(x,t)")
            ax.set_title(f"true lam {lam}, tol {tol}")
            plt.tight_layout()
            plt.show()

        l += 1
        """

    avg_mse = np.mean(fold_mses)
    print(f"Lambda: {lam} | Tol: {tol} | Avg MSE: {avg_mse:.6e}")

    if avg_mse < best_mse:
        best_mse = avg_mse
        best_params['lam'] = lam
        best_params['tol'] = tol

print("-" * 30)
print(f"Best Parameters: {best_params}")
print(f"Lowest MSE: {best_mse:.6e}")

final_coef = STRidge(candidate_library, U_dot_flat.reshape(-1, 1),
                     best_params['lam'], 10000, best_params['tol'], normalize=0)
final_coef = final_coef.flatten().real
coef = final_coef

print("\nDiscovered coefficients:")
labels = ['1', 'u', 'u²', 'u³', 'u_x', 'u_xx']
print("\nDiscovered PDE: du/dt = \n", end="")
terms = []
for i, (label, c) in enumerate(zip(labels, coef)):
    if np.abs(c) > 1e-6:
        print(f"{label}: {c:.6f}")
        if c > 0 and terms:
            terms.append(f"+ {c:.4f}*{label}")
        else:
            terms.append(f"{c:.4f}*{label}")

if terms:
    print("Equation: du/dt = " + " ".join(terms))
else:
    print("No significant terms found")

print(f"\nFull coefficient vector: {coef}")

