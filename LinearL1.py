from ngclearn.utils.feature_dictionaries.polynomialLibrary import PolynomialLibrary
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

A = np.array([
    [-0.1, -2.0, 0.0],
    [ 2.0, -0.1, 0.0],
    [ 0.0,  0.0, -0.3]
])

def linear_sys(t, xyz):
    return A @ xyz

#Numerically integrate true solution to obtain "sample" data
x0y0z0 = (2,0,1)
t_eval = np.linspace(0, 50, 100000)
result = sp.integrate.solve_ivp(linear_sys, (0,50), x0y0z0, method='RK45', t_eval=t_eval)
t = result.t
x, y, z = result.y
X = np.stack([x, y, z], axis=-1)

#Create a library of polynomial candidate functions
lib_creator = PolynomialLibrary(poly_order=2, include_bias=False)
feature_lib, feature_names = lib_creator.fit([X[:, i] for i in range(X.shape[1])])

#Take point-wise gradients to obtain the matrix X'
dX = np.array(np.gradient(X, t.ravel(), axis=0))

#Run LASSO regression on the problem
alpha = 0.002
threshold = 0.02
coeffs = []

for dim in range(dX.shape[1]):
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000000)
    lasso.fit(feature_lib, dX[:, dim])
    coef = lasso.coef_
    coef = np.where(np.abs(coef) >= threshold, coef, 0.0)
    coeffs.append(coef)
coeffs = np.array(coeffs)

#Print the new system of equations
for i in range(len(coeffs)):
    output_string = ""
    for j in range(len(coeffs[i])):
        if coeffs[i][j] != 0.0:
            output_string += f"{coeffs[i][j]:.4f}*{feature_names[j]} | "
    print(f"Eqn {i}: {output_string}")

#Function to model the learned system
def learned_linear(t, xyz):
    x, y, z = xyz
    features = [z,z*z,y,y*z,y*y,x,x*z,x*y,x*x]
    features = np.array(features)
    dxdt = np.dot(features, coeffs[0])
    dydt = np.dot(features, coeffs[1])
    dzdt = np.dot(features, coeffs[2])
    return np.array([dxdt, dydt, dzdt])

#Evaluate the learned system
t_eval = np.linspace(0, 50, 100000)
x0y0z0 = (2,0,1)
learned_sol = sp.integrate.solve_ivp(learned_linear, (0,50), x0y0z0, method='RK45', t_eval=t_eval)

#Plot the true and learned systems
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x, y, z)
ax1.set_title("True Lorenz system")
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(learned_sol.y[0], learned_sol.y[1], learned_sol.y[2])
ax2.set_title("Learned system")
plt.tight_layout()
plt.show()

#Create space for differential equations to be estimated and have vector fields computed on
x_vals = np.linspace(-20, 20, 12)
y_vals = np.linspace(-30, 30, 12)
z_vals = np.linspace(0, 50, 12)
Xg, Yg, Zg = np.meshgrid(x_vals, y_vals, z_vals)
U_true, V_true, W_true = np.zeros_like(Xg), np.zeros_like(Yg), np.zeros_like(Zg)
U_learn, V_learn, W_learn = np.zeros_like(Xg), np.zeros_like(Yg), np.zeros_like(Zg)

# Evaluate vector field on grid
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        for k in range(Xg.shape[2]):
            pt = np.array([Xg[i, j, k], Yg[i, j, k], Zg[i, j, k]])

            dtrue = linear_sys(0, pt)
            U_true[i, j, k], V_true[i, j, k], W_true[i, j, k] = dtrue

            dlearn = learned_linear(0, pt)
            U_learn[i, j, k], V_learn[i, j, k], W_learn[i, j, k] = dlearn

#Plot true and learned vector fields
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.quiver(Xg, Yg, Zg, U_true, V_true, W_true, length=2.0, normalize=True, color="blue", alpha=0.7)
ax1.set_title("True Lorenz Vector Field")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

ax2 = fig.add_subplot(122, projection='3d')
ax2.quiver(Xg, Yg, Zg, U_learn, V_learn, W_learn, length=2.0, normalize=True, color="red", alpha=0.7)
ax2.set_title("Learned Lorenz Vector Field")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

plt.tight_layout()
plt.show()

#Calculate mean squared error between vector fields
errors = []

for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        for k in range(Xg.shape[2]):
            pt = np.array([Xg[i, j, k], Yg[i, j, k], Zg[i, j, k]])

            dtrue = linear_sys(0, pt)
            dlearn = learned_linear(0, pt)

            err = np.linalg.norm(dtrue - dlearn)

            errors.append(err)

errors = np.array(errors)

print("Mean Absolute Error (vector field):", np.mean(errors))