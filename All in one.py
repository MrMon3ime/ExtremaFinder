import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_derivatives(f, variables):
    """Compute partial derivatives."""
    gradients = [sp.diff(f, var) for var in variables]
    return gradients

def find_critical_points(f, variables):
    """Find critical points by solving gradient = 0."""
    gradients = compute_derivatives(f, variables)
    solutions = sp.solve(gradients, variables, dict=True)
    return solutions, gradients

def classify_critical_points(f, variables, critical_points):
    """Classify critical points using the Hessian matrix."""
    hessian = sp.Matrix([[sp.diff(sp.diff(f, v1), v2) for v2 in variables] for v1 in variables])
    classifications = {}
    
    for point in critical_points:
        hessian_eval = hessian.subs(point)
        eigenvalues = hessian_eval.eigenvals()
        eigenvalues_list = list(eigenvalues.keys())
        
        if all(ev > 0 for ev in eigenvalues_list):
            classifications[tuple(point.values())] = "Local Min"
        elif all(ev < 0 for ev in eigenvalues_list):
            classifications[tuple(point.values())] = "Local Max"
        else:
            classifications[tuple(point.values())] = "Saddle Point"
    
    return classifications, hessian

def plot_function(f, variables):
    """Plot the function and critical points in 3D."""
    x, y = variables
    f_lambda = sp.lambdify((x, y), f, 'numpy')
    
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = f_lambda(X, Y)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    
    # Plot critical points
    critical_points, _ = find_critical_points(f, variables)
    for point in critical_points:
        x_val, y_val = float(point[x]), float(point[y])
        z_val = f_lambda(x_val, y_val)
        ax.scatter(x_val, y_val, z_val, color='red', s=100)
    
    plt.show()

x, y = sp.symbols('x y')
f_str = input("Enter a function of x and y: ")
f = sp.sympify(f_str)

critical_points, gradients = find_critical_points(f, (x, y))
classifications, hessian = classify_critical_points(f, (x, y), critical_points)

print("Function:", f)
print("Gradients:", gradients)
print("Critical Points:", critical_points)
print("Classifications:", classifications)
print("Hessian Matrix:", hessian)

plot_function(f, (x, y))
