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


def sindy_discover(X, dX, threshold, feature_lib, max_iter=1000):
    """
    Perform SINDy discovery for a given threshold
    Returns coefficients matrix and number of non-zero terms
    """
    coeffs = []
    total_nonzero = 0

    for dim in range(dX.shape[1]):
        # Compute the first least squares solution
        coef = np.linalg.lstsq(feature_lib, dX[:, dim][:, None], rcond=None)[0]

        # Regression loop with thresholding
        for i in range(max_iter):
            coef_pre = jnp.array(coef)
            coef_zero = jnp.zeros_like(coef)

            # Create mask for coefficients above threshold
            res_idx = jnp.where(jnp.abs(coef) >= threshold, True, False)
            res_mask = res_idx.T[0]

            # Remove features below threshold
            res_lib = feature_lib[:, res_mask]

            # Compute new coefficients
            if res_lib.shape[1] > 0:  # Make sure we have some features left
                coef_new = np.linalg.lstsq(res_lib, dX[:, dim][:, None], rcond=None)[0]
                coef = coef_zero.at[res_mask].set(coef_new)
            else:
                coef = coef_zero
                break

            # Check convergence
            if np.allclose(coef_pre, coef):
                break

        # Final thresholding
        coeff = jnp.where(jnp.abs(coef) >= threshold, coef, 0.)
        coeffs.append(coeff.T[0])
        total_nonzero += np.sum(np.abs(coeff) > 1e-10)

    return np.array(coeffs), int(total_nonzero)


def create_learned_system(coeffs, feature_names):
    """Create a learned system function from discovered coefficients"""

    def learned_lorenz(t, xyz):
        x, y, z = xyz
        # Create feature vector (matches the polynomial library structure)
        features = np.array([z, z * z, y, y * z, y * y, x, x * z, x * y, x * x])

        dxdt = np.dot(features, coeffs[0])
        dydt = np.dot(features, coeffs[1])
        dzdt = np.dot(features, coeffs[2])
        return np.array([dxdt, dydt, dzdt])

    return learned_lorenz


def calculate_vector_field_error(learned_system, grid_size=10):
    """Calculate MSE between true and learned vector fields"""
    x_vals = np.linspace(-20, 20, grid_size)
    y_vals = np.linspace(-30, 30, grid_size)
    z_vals = np.linspace(0, 50, grid_size)

    errors = []
    for x in x_vals:
        for y in y_vals:
            for z in z_vals:
                pt = np.array([x, y, z])
                dtrue = lorenz(0, pt)
                dlearn = learned_system(0, pt)
                error = np.linalg.norm(dtrue - dlearn) ** 2
                errors.append(error)

    return np.mean(errors)


def calculate_trajectory_error(learned_system, t_span=20, n_points=2000):
    """Calculate error in trajectory integration"""
    t_eval = np.linspace(0, t_span, n_points)
    x0y0z0 = (-8, 7, 27)

    # True solution
    true_sol = sp.integrate.solve_ivp(lorenz, (0, t_span), x0y0z0,
                                      method='RK45', t_eval=t_eval)

    # Learned solution
    try:
        learned_sol = sp.integrate.solve_ivp(learned_system, (0, t_span), x0y0z0,
                                             method='RK45', t_eval=t_eval)

        # Calculate MSE over trajectory
        if learned_sol.success and len(learned_sol.y[0]) == len(true_sol.y[0]):
            trajectory_error = np.mean((true_sol.y - learned_sol.y) ** 2)
        else:
            trajectory_error = float('inf')  # Integration failed
    except:
        trajectory_error = float('inf')

    return trajectory_error


# Generate training data
print("Generating training data...")
x0y0z0 = (-8, 7, 27)
t_eval = np.linspace(0, 100, 100000)
result = sp.integrate.solve_ivp(lorenz, (0, 100), x0y0z0, method='RK45', t_eval=t_eval)
t = result.t
x, y, z = result.y
X = np.stack([x, y, z], axis=-1)

# Create polynomial library
print("Creating feature library...")
lib_creator = PolynomialLibrary(poly_order=2, include_bias=False)
feature_lib, feature_names = lib_creator.fit([X[:, i] for i in range(X.shape[1])])
print(f"Feature library shape: {feature_lib.shape}")
print(f"Features: {feature_names}")

# Calculate derivatives
dX = np.array(np.gradient(X, t.ravel(), axis=0))

# Test different threshold values
thresholds = np.logspace(-3, 0, 20)  # From 0.001 to 1.0
print(f"\nTesting {len(thresholds)} threshold values...")

results = {
    'thresholds': [],
    'vector_field_errors': [],
    'trajectory_errors': [],
    'num_nonzero_terms': [],
    'computation_times': []
}

for i, threshold in enumerate(thresholds):
    print(f"Processing threshold {threshold:.4f} ({i + 1}/{len(thresholds)})")

    start_time = time.time()

    # Discover system with current threshold
    coeffs, num_nonzero = sindy_discover(X, dX, threshold, feature_lib)

    # Create learned system
    learned_system = create_learned_system(coeffs, feature_names)

    # Calculate errors
    vector_error = calculate_vector_field_error(learned_system, grid_size=8)
    trajectory_error = calculate_trajectory_error(learned_system, t_span=10, n_points=1000)

    comp_time = time.time() - start_time

    # Store results
    results['thresholds'].append(threshold)
    results['vector_field_errors'].append(vector_error)
    results['trajectory_errors'].append(trajectory_error)
    results['num_nonzero_terms'].append(num_nonzero)
    results['computation_times'].append(comp_time)

    print(f"  Vector field MSE: {vector_error:.6f}")
    print(f"  Trajectory MSE: {trajectory_error:.6f}")
    print(f"  Non-zero terms: {num_nonzero}")
    print(f"  Computation time: {comp_time:.3f}s")

# Convert to numpy arrays for easier plotting
for key in results:
    results[key] = np.array(results[key])

# Create comprehensive plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Vector field error vs threshold
ax1.loglog(results['thresholds'], results['vector_field_errors'], 'bo-', linewidth=2, markersize=6)
ax1.set_xlabel('Sparsity Threshold')
ax1.set_ylabel('Vector Field MSE')
ax1.set_title('Vector Field Error vs Sparsity Threshold')
ax1.grid(True, alpha=0.3)

# Plot 2: Trajectory error vs threshold
finite_traj_errors = np.where(np.isfinite(results['trajectory_errors']),
                              results['trajectory_errors'], np.nan)
ax2.loglog(results['thresholds'], finite_traj_errors, 'ro-', linewidth=2, markersize=6)
ax2.set_xlabel('Sparsity Threshold')
ax2.set_ylabel('Trajectory MSE')
ax2.set_title('Trajectory Error vs Sparsity Threshold')
ax2.grid(True, alpha=0.3)

# Plot 3: Number of terms vs threshold
ax3.semilogx(results['thresholds'], results['num_nonzero_terms'], 'go-', linewidth=2, markersize=6)
ax3.set_xlabel('Sparsity Threshold')
ax3.set_ylabel('Number of Non-zero Terms')
ax3.set_title('Model Complexity vs Sparsity Threshold')
ax3.grid(True, alpha=0.3)

# Plot 4: Error vs complexity trade-off
ax4.scatter(results['num_nonzero_terms'], results['vector_field_errors'],
            c=np.log10(results['thresholds']), cmap='viridis', s=60, alpha=0.7)
ax4.set_xlabel('Number of Non-zero Terms')
ax4.set_ylabel('Vector Field MSE')
ax4.set_title('Error vs Model Complexity')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)
cbar = plt.colorbar(ax4.collections[0], ax=ax4)
cbar.set_label('log₁₀(Threshold)')

plt.tight_layout()
plt.show()

# Find optimal threshold based on vector field error
min_error_idx = np.argmin(results['vector_field_errors'])
optimal_threshold = results['thresholds'][min_error_idx]
optimal_error = results['vector_field_errors'][min_error_idx]
optimal_terms = results['num_nonzero_terms'][min_error_idx]

print(f"\n{'=' * 60}")
print("ANALYSIS SUMMARY")
print(f"{'=' * 60}")
print(f"Optimal threshold (min vector field error): {optimal_threshold:.4f}")
print(f"Minimum vector field MSE: {optimal_error:.6f}")
print(f"Number of terms at optimum: {optimal_terms}")
print(f"Total terms in library: {feature_lib.shape[1]}")
print(f"Sparsity at optimum: {(feature_lib.shape[1] * 3 - optimal_terms) / (feature_lib.shape[1] * 3) * 100:.1f}%")

# Show the discovered equations for optimal threshold
print(f"\nDiscovered equations at optimal threshold ({optimal_threshold:.4f}):")
coeffs_opt, _ = sindy_discover(X, dX, optimal_threshold, feature_lib)
var_names = ['x', 'y', 'z']
for i in range(len(coeffs_opt)):
    output_string = f"d{var_names[i]}/dt = "
    terms = []
    for j in range(len(coeffs_opt[i])):
        if abs(coeffs_opt[i][j]) > 1e-10:
            terms.append(f"{coeffs_opt[i][j]:.4f}*{feature_names[j]}")
    output_string += " + ".join(terms) if terms else "0"
    print(output_string)