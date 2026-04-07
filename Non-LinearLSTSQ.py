from ngclearn.utils.feature_dictionaries.polynomialLibrary import PolynomialLibrary
import numpy as np
import scipy as sp
import jax.numpy as jnp
import matplotlib.pyplot as plt

def nonlinear(t, xy):
    x, y = xy
    dx_dt = -0.1 * x**3 + 2 * y**3
    dy_dt = -2 * x**3 - 0.1 * y**3
    return np.array([dx_dt, dy_dt])

#Numerically integrate true solution to obtain "sample" data
t_eval = np.linspace(0, 25, 100000)
x0y0 = (2, 0)
result = sp.integrate.solve_ivp(nonlinear, (0, 25), x0y0, method='RK45', t_eval=t_eval)
t = result.t
x, y = result.y
X = np.stack([x, y], axis=-1)

#Create a library of polynomial candidate functions
lib_creator = PolynomialLibrary(poly_order=3, include_bias=False)
feature_lib, feature_names = lib_creator.fit([X[:, i] for i in range(X.shape[1])])

#Take point-wise gradients to obtain the matrix X'
dX = np.array(np.gradient(X, t.ravel(), axis=0))

#Perform least-squares linear regression with thresholding
threshold = 0.02
coeffs = []

for dim in range(dX.shape[1]):
    #Compute the first least squares solution to Θ @ ξ = dX[i] for current i dimension, setting coef = ξ
    coef = np.linalg.lstsq(feature_lib, dX[:, dim][:, None], rcond=None)[0]

    #Regression loop
    for i in range(1000):
        coef_pre = jnp.array(coef)
        coef_zero = jnp.zeros_like(coef)

        #Create a mask over all values in ξ where values above the threshold are True, and below are False
        res_idx = jnp.where(jnp.abs(coef) >= threshold,
                            True,
                            False)
        res_mask = res_idx.T[0]

        #Remove the features in the library of candidate functions covered by the mask
        res_lib = feature_lib[:, res_mask]

        #Compute a new ξ using the new candidate library with the mask applied
        coef_new = np.linalg.lstsq(res_lib, dX[:, dim][:, None], rcond=None)[0]

        #Update all the coef values with the new ξ values where the mask is True
        coef = coef_zero.at[res_mask].set(coef_new)

        #Break if convergence is achieved
        if all(coef_pre == coef):
           break

    #Set all values in ξ below the threshold to 0
    coeff = jnp.where(jnp.abs(coef) >= threshold, coef, 0.)
    #Append ξ to the matrix Ξ
    coeffs.append(coeff.T[0])
coeffs = np.array(coeffs)

#Print the new system of equations
for i in range(len(coeffs)):
    output_string = ""
    for j in range(len(coeffs[i])):
        if coeffs[i][j] != 0.0:
            output_string += f"{coeffs[i][j]:.4f}*{feature_names[j]} | "
    print(f"Eqn {i}: {output_string}")

#Function to model the learned system
def learned_sys(t, xy):
    x, y = xy
    features = [y, y*y, y*y*y,
                x, x*y, x*y*y,
                x*x, x*x*y, x*x*x]
    features = np.array(features)
    dxdt = np.dot(features, coeffs[0])
    dydt = np.dot(features, coeffs[1])
    return np.array([dxdt, dydt])

#Evaluate the learned system
t_eval = np.linspace(0, 25, 100000)
learned_sol = sp.integrate.solve_ivp(learned_sys, (0, 25), x0y0, method='RK45', t_eval=t_eval)

#Create space for differential equations to be estimated and have vector fields computed on
xgrid = np.linspace(-3, 3, 20)
ygrid = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(xgrid, ygrid)
U_true, V_true = np.zeros(X.shape), np.zeros(Y.shape)
U_learned, V_learned = np.zeros(X.shape), np.zeros(Y.shape)

#Calculate mean squared error between vector fields
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        vec_true = nonlinear(0, [X[i, j], Y[i, j]])
        vec_learned = learned_sys(0, [X[i, j], Y[i, j]])
        U_true[i, j], V_true[i, j] = vec_true
        U_learned[i, j], V_learned[i, j] = vec_learned

mse = np.mean((U_true - U_learned)**2 + (V_true - V_learned)**2)
print(f"\nMean Squared Error between vector fields: {mse:.6f}")

#Plot the true and learned systems
fig1, axs1 = plt.subplots(1, 2, figsize=(14, 6))

axs1[0].plot(x, y, 'b')
axs1[0].set_title("True Trajectory")
axs1[0].set_xlabel("x")
axs1[0].set_ylabel("y")

axs1[1].plot(learned_sol.y[0], learned_sol.y[1], 'b')
axs1[1].set_title("Learned Trajectory")
axs1[1].set_xlabel("x")
axs1[1].set_ylabel("y")

plt.tight_layout()
plt.show()

#Plot true and learned vector fields
fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6))

axs2[0].quiver(X, Y, U_true, V_true, color='b')
axs2[0].set_title("True Vector Field")
axs2[0].set_xlabel("x")
axs2[0].set_ylabel("y")

axs2[1].quiver(X, Y, U_learned, V_learned, color='r')
axs2[1].set_title("Learned Vector Field")
axs2[1].set_xlabel("x")
axs2[1].set_ylabel("y")

plt.tight_layout()
plt.show()
