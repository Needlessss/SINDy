from ngclearn.utils.feature_dictionaries.polynomialLibrary import PolynomialLibrary
import numpy as np
import scipy as sp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time


def lorenz(t, xyz):
    sigma = 10.0
    rho = 28.0
    beta = 8 / 3.0
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])


def run_sindy_experiment(n_points):
    """Run SINDy experiment with specified number of time points"""
    start_time = time.time()

    # Numerically integrate true solution to obtain "sample" data
    x0y0z0 = (-8, 7, 27)
    t_eval = np.linspace(0, 100, n_points)
    result = sp.integrate.solve_ivp(lorenz, (0, 100), x0y0z0, method='RK45', t_eval=t_eval)
    t = result.t
    x, y, z = result.y
    X = np.stack([x, y, z], axis=-1)

    # Create a library of polynomial candidate functions
    lib_creator = PolynomialLibrary(poly_order=2, include_bias=False)
    feature_lib, feature_names = lib_creator.fit([X[:, i] for i in range(X.shape[1])])

    # Take point-wise gradients to obtain the matrix X'
    dX = np.array(np.gradient(X, t.ravel(), axis=0))

    # Perform least-squares linear regression with thresholding
    threshold = 0.02
    coeffs = []

    for dim in range(dX.shape[1]):
        # Compute the first least squares solution
        coef = np.linalg.lstsq(feature_lib, dX[:, dim][:, None], rcond=None)[0]

        # Regression loop
        for i in range(1000):
            coef_pre = jnp.array(coef)
            coef_zero = jnp.zeros_like(coef)

            # Create a mask over all values in ξ where values above the threshold are True
            res_idx = jnp.where(jnp.abs(coef) >= threshold, True, False)
            res_mask = res_idx.T[0]

            # Remove the features in the library covered by the mask
            res_lib = feature_lib[:, res_mask]

            # Compute a new ξ using the new candidate library with the mask applied
            coef_new = np.linalg.lstsq(res_lib, dX[:, dim][:, None], rcond=None)[0]

            # Update all the coef values with the new ξ values where the mask is True
            coef = coef_zero.at[res_mask].set(coef_new)

            # Break if convergence is achieved
            if np.allclose(coef_pre, coef):
                break

        # Set all values in ξ below the threshold to 0
        coeff = jnp.where(jnp.abs(coef) >= threshold, coef, 0.)
        coeffs.append(coeff.T[0])

    coeffs = np.array(coeffs)

    # Function to model the learned system
    def learned_lorenz(t, xyz):
        x, y, z = xyz
        features = [z, z * z, y, y * z, y * y, x, x * z, x * y, x * x]
        features = np.array(features)
        dxdt = np.dot(features, coeffs[0])
        dydt = np.dot(features, coeffs[1])
        dzdt = np.dot(features, coeffs[2])
        return np.array([dxdt, dydt, dzdt])

    # Calculate vector field error on a coarse grid for efficiency
    x_vals = np.linspace(-20, 20, 8)
    y_vals = np.linspace(-30, 30, 8)
    z_vals = np.linspace(0, 50, 8)
    Xg, Yg, Zg = np.meshgrid(x_vals, y_vals, z_vals)

    errors = []
    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            for k in range(Xg.shape[2]):
                pt = np.array([Xg[i, j, k], Yg[i, j, k], Zg[i, j, k]])
                dtrue = lorenz(0, pt)
                dlearn = learned_lorenz(0, pt)
                err = np.linalg.norm(dtrue - dlearn)
                errors.append(err)

    errors = np.array(errors)
    mean_error = np.mean(errors)
    computation_time = time.time() - start_time

    return mean_error, computation_time, coeffs


def main():
    # Test different time series resolutions
    resolutions = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000]

    results = {
        'resolutions': [],
        'errors': [],
        'computation_times': [],
        'coefficients': []
    }

    print("Testing different time series resolutions...")
    print("Resolution | Mean Error | Computation Time (s)")
    print("-" * 45)

    for n_points in resolutions:
        try:
            error, comp_time, coeffs = run_sindy_experiment(n_points)

            results['resolutions'].append(n_points)
            results['errors'].append(error)
            results['computation_times'].append(comp_time)
            results['coefficients'].append(coeffs)

            print(f"{n_points:8d} | {error:10.6f} | {comp_time:15.3f}")

        except Exception as e:
            print(f"Failed at resolution {n_points}: {e}")
            continue

    # Convert to numpy arrays for easier manipulation
    resolutions = np.array(results['resolutions'])
    errors = np.array(results['errors'])
    computation_times = np.array(results['computation_times'])

    # Create comprehensive plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Error vs Resolution
    ax1.loglog(resolutions, errors, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Time Points')
    ax1.set_ylabel('Mean Vector Field Error')
    ax1.set_title('Error vs Time Series Resolution')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Computation Time vs Resolution
    ax2.loglog(resolutions, computation_times, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Time Points')
    ax2.set_ylabel('Computation Time (seconds)')
    ax2.set_title('Computation Time vs Resolution')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Error vs Computation Time (Efficiency Plot)
    ax3.loglog(computation_times, errors, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Computation Time (seconds)')
    ax3.set_ylabel('Mean Vector Field Error')
    ax3.set_title('Error vs Computation Time (Efficiency)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Normalized metrics
    normalized_times = computation_times / np.max(computation_times)
    normalized_errors = errors / np.max(errors)

    ax4.semilogx(resolutions, normalized_errors, 'b-', label='Normalized Error', linewidth=2)
    ax4.semilogx(resolutions, normalized_times, 'r-', label='Normalized Time', linewidth=2)
    ax4.set_xlabel('Number of Time Points')
    ax4.set_ylabel('Normalized Value')
    ax4.set_title('Normalized Error and Time vs Resolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    # Find optimal resolution (lowest error per computation time)
    efficiency = errors / computation_times
    optimal_idx = np.argmin(efficiency)

    print(f"Best resolution (lowest error/time ratio): {resolutions[optimal_idx]} points")
    print(f"  - Error: {errors[optimal_idx]:.6f}")
    print(f"  - Time: {computation_times[optimal_idx]:.3f} seconds")
    print(f"  - Efficiency ratio: {efficiency[optimal_idx]:.8f}")

    print(f"\nLowest error achieved: {np.min(errors):.6f} at {resolutions[np.argmin(errors)]} points")
    print(
        f"Fastest computation: {np.min(computation_times):.3f}s at {resolutions[np.argmin(computation_times)]} points")

    # Error reduction analysis
    error_improvement = (np.max(errors) - np.min(errors)) / np.max(errors) * 100
    time_increase = (np.max(computation_times) - np.min(computation_times)) / np.min(computation_times) * 100

    print(f"\nGoing from {resolutions[0]} to {resolutions[-1]} points:")
    print(f"  - Error improves by: {error_improvement:.1f}%")
    print(f"  - Computation time increases by: {time_increase:.1f}%")

    return results


if __name__ == "__main__":
    results = main()