from ngclearn.utils.feature_dictionaries.polynomialLibrary import PolynomialLibrary
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

time_steps = [100000, 50000, 10000, 5000, 1000]
time_res = [0.001, 0.005, 0.01, 0.05, 0.1]
ranges = [5,10,50,100,500,1000, 5000,10000]
errs = []
dts = []
stride = 20
max_rollouts_per_fold = 50000



for range_val in time_steps:

    dt = 100 / range_val
    dts.append(dt)
    steps = int(100/dt)
    tau = 2.0
    horizon_steps = int(20)
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
    t_eval = np.linspace(0, 100, steps)
    result = sp.integrate.solve_ivp(lorenz, (0,100), x0y0z0, method='RK45', t_eval=t_eval)
    t = result.t
    x, y, z = result.y
    X = np.stack([x, y, z], axis=-1)

    #Create a library of polynomial candidate functions
    lib_creator = PolynomialLibrary(poly_order=2, include_bias=False)
    feature_lib, feature_names = lib_creator.fit([X[:, i] for i in range(X.shape[1])])

    #Take point-wise gradients to obtain the matrix X'
    dX = np.array(np.gradient(X, t.ravel(), axis=0))

    n_t = len(t_eval)
    n_train = int(n_t * 0.8)
    train_idx = np.arange(n_train)
    test_idx  = np.arange(n_train, n_t)


    Phi_train = feature_lib[train_idx]
    Phi_val = feature_lib[test_idx]

    dX_train = dX[train_idx]
    dX_val = dX[test_idx]


    #Run LASSO regression on the problem
    alpha = 0.001
    threshold = 0.02
    coeffs = []

    for dim in range(dX_train.shape[1]):
        lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000000)
        lasso.fit(Phi_train, dX_train[:, dim])
        coef = lasso.coef_
        coef = np.where(np.abs(coef) >= threshold, coef, 0.0)
        coeffs.append(coef)
    coeffs = np.array(coeffs)


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


    val_times_sorted = np.sort(test_idx)

    start_indices = val_times_sorted[
                    :-horizon_steps:stride
                    ]


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
            rollout_errors.append(0)
            continue

        X_pred = sol.y.T
        X_true = X[start_idx:start_idx + horizon_steps + 1]

        err = np.mean((X_true - X_pred) ** 2)
        rollout_errors.append(err)

    mse = np.mean(rollout_errors)
    errs.append(mse)
plt.plot(dts, errs, marker='o')

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Time Step Size (dt)")
plt.ylabel("SINDy Reconstruction vs True Data Short-Term MSE")
plt.title("Data Time Step Size vs SINDy Model Error")

plt.grid(True, which="both")

plt.show()

print(dts)
print(errs)
