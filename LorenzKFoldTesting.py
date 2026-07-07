from ngclearn.utils.feature_dictionaries.polynomialLibrary import PolynomialLibrary
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

def lorenz(t, xyz):
    sigma = 10.0
    rho = 28.0
    beta = 8 / 3.0
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])

#Numerically integrate true solution to obtain "sample" data
x0y0z0 = (-8, 7, 27)
t_eval = np.linspace(0, 100, 10000)
result = sp.integrate.solve_ivp(lorenz, (0,100), x0y0z0, method='RK45', t_eval=t_eval)
t = result.t
x, y, z = result.y
X = np.stack([x, y, z], axis=-1)

lambdas = [1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
k_folds = 5

kf = KFold(n_splits=k_folds, shuffle=False)
time_indices = np.arange(len(t_eval))

best_mse = float('inf')
best_params = {'lam': None, 'tol': None}

lib_creator = PolynomialLibrary(poly_order=2, include_bias=False)
feature_lib, feature_names = lib_creator.fit([X[:, i] for i in range(X.shape[1])])

dX = np.array(np.gradient(X, t.ravel(), axis=0))  # shape: (T, 3)

kf = KFold(n_splits=k_folds, shuffle=False)
time_indices = np.arange(len(t_eval))

lambda_mse = {}

dt = t[1] - t[0]

horizon_steps = 200

stride = 5
max_rollouts_per_fold = 50000

print(f"Starting {k_folds}-fold Cross-Validation (trajectory-based)...")

for lam in lambdas:
    fold_mses = []

    for fold_id, (train_times, val_times) in enumerate(kf.split(time_indices)):

        # ---- Split data ----
        Phi_train = feature_lib[train_times]
        Phi_val   = feature_lib[val_times]

        dX_train = dX[train_times]
        dX_val   = dX[val_times]

        # ---- Fit LASSO on each state dimension ----
        coeffs = []
        threshold = 0.1

        for dim in range(dX.shape[1]):
            lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=1_000_000)
            lasso.fit(Phi_train, dX_train[:, dim])

            coef = lasso.coef_
            coef = np.where(np.abs(coef) >= threshold, coef, 0.0)
            coeffs.append(coef)

        coeffs = np.array(coeffs)   # shape: (3, n_features)

        #Trajectory:

        def learned_lorenz(t, xyz):
            x, y, z = xyz
            features = [z, z * z, y, y * z, y * y, x, x * z, x * y, x * x]
            features = np.array(features)
            dxdt = np.dot(features, coeffs[0])
            dydt = np.dot(features, coeffs[1])
            dzdt = np.dot(features, coeffs[2])

            return np.array([dxdt, dydt, dzdt])


        val_times_sorted = np.sort(val_times)



        start_indices = val_times_sorted[
                        :-horizon_steps:stride
                        ]


        if len(start_indices) == 0:
            fold_mses.append(1e6)
            continue

        # Optional: cap number of rollouts for speed
        if len(start_indices) > max_rollouts_per_fold:
            start_indices = np.random.choice(
                start_indices,
                size=max_rollouts_per_fold,
                replace=False
            )

        rollout_errors = []

        for start_idx in start_indices:

            t0 = t[start_idx]
            t1 = t[start_idx + horizon_steps]
            t_window = t[start_idx:start_idx + horizon_steps + 1]

            x0_val = X[start_idx]

            sol = sp.integrate.solve_ivp(
                learned_lorenz,
                (t0, t1),
                x0_val,
                t_eval=t_window,
                method="RK45"
            )

            if not sol.success or sol.y.shape[1] != len(t_window):
                rollout_errors.append(1e3)
                continue

            X_pred = sol.y.T  # (K, 3)
            X_true = X[start_idx:start_idx + horizon_steps + 1]

            err = np.mean((X_true - X_pred) ** 2)
            rollout_errors.append(err)

        # ---- Fold short-horizon MSE ----
        mse = np.mean(rollout_errors)




        """
        #Pointwise:
        dX_val_pred = Phi_val @ coeffs.T   # (N_val, 3)

        # ---- Compute MSE for this fold ----
        mse = np.mean((dX_val - dX_val_pred) ** 2)
        



        t0 = t[val_times_sorted[0]]
        t1 = t[val_times_sorted[-1]]
        t_window = t[val_times_sorted]

        x0_val = X[val_times_sorted[0]]

        sol = sp.integrate.solve_ivp(
            learned_lorenz,
            (t0, t1),
            x0_val,
            t_eval=t_window,
            method="RK45"
        )

        if not sol.success or sol.y.shape[1] != len(t_window):
            # If integration fails, penalize this fold
            fold_mses.append(1e6)
            continue

        X_pred = sol.y.T                      # (N_val, 3)
        X_true = X[val_times_sorted]          # (N_val, 3)

        # ---- Fold MSE: full-trajectory error ----
        mse = np.mean((X_true - X_pred) ** 2)
        """


        fold_mses.append(mse)
    mean_mse = np.mean(fold_mses)
    lambda_mse[lam] = mean_mse

    print(f"λ = {lam:8g} | mean trajectory CV-MSE = {mean_mse:.6e}")


full_models = {}

def blowup_event(t, xyz):
    return 200.0 - np.linalg.norm(xyz)
blowup_event.terminal = True
blowup_event.direction = -1

for lam in lambdas:

    # ---- Fit on ALL data ----
    Phi_full = feature_lib
    dX_full  = dX

    coeffs = []
    threshold = 0.1

    for dim in range(dX_full.shape[1]):
        lasso = Lasso(alpha=lam, fit_intercept=False, max_iter=1000000)
        lasso.fit(Phi_full, dX_full[:, dim])

        coef = lasso.coef_
        coef = np.where(np.abs(coef) >= threshold, coef, 0.0)
        coeffs.append(coef)

    coeffs = np.array(coeffs)   # shape: (3, n_features)
    full_models[lam] = coeffs


    print("\n-------------------------------")
    print(f"λ = {lam}")
    print("-------------------------------")

    for eqn_idx in range(coeffs.shape[0]):
        terms = []
        for j in range(coeffs.shape[1]):
            c = coeffs[eqn_idx, j]
            if c != 0.0:
                terms.append(f"{c:+.5f}·{feature_names[j]}")

        if len(terms) == 0:
            eqn_str = "0"
        else:
            eqn_str = " ".join(terms)

        print(f"dX[{eqn_idx}]/dt = {eqn_str}")

    def learned_lorenz(t, xyz):
        x, y, z = xyz
        features = [z, z * z, y, y * z, y * y, x, x * z, x * y, x * x]
        features = np.array(features)
        # features, _ = lib_creator.fit(xyz)
        # features = features[0]
        dxdt = np.dot(features, coeffs[0])
        dydt = np.dot(features, coeffs[1])
        dzdt = np.dot(features, coeffs[2])

        return np.array([dxdt, dydt, dzdt])


    # Evaluate the learned system
    t_eval = np.linspace(0, 100, 10000)
    x0y0z0 = (-8, 7, 2)
    learned_sol = sp.integrate.solve_ivp(learned_lorenz, (0, 100), x0y0z0, method='RK45', t_eval=t_eval, events=blowup_event)

    # Plot the true and learned systems
    fig = plt.figure(figsize=(5, 5))
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.plot(learned_sol.y[0], learned_sol.y[1], learned_sol.y[2])
    ax2.set_title(f"Learned system, lam = {lam}")
    plt.tight_layout()
    plt.show()
