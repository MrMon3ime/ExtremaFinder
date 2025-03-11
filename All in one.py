import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_derivatives(f, variables):
    """Compute partial derivatives."""
    return [sp.diff(f, var) for var in variables]

def find_critical_points(f, variables):
    """Find critical points by solving gradient = 0."""
    gradients = compute_derivatives(f, variables)
    solutions = sp.solve(gradients, variables, dict=True)
    return solutions, gradients

def classify_critical_points(f, variables, critical_points):
    """Classify critical points using the Hessian matrix."""
    classifications = {}
    
    if len(variables) == 1:  #hadi ki n7ot variable 1
        x = variables[0]
        second_derivative = sp.diff(f, x, x)
        
        for point in critical_points:
            second_derivative_eval = second_derivative.subs(point)
            classification = "Local Min" if second_derivative_eval > 0 else "Local Max" if second_derivative_eval < 0 else "Inflection Point"
            classifications[tuple(point.values())] = classification
            
        return classifications, second_derivative

    else:  # hadi fl cas nta3 2 variables
        hessian = sp.Matrix([[sp.diff(sp.diff(f, v1), v2) for v2 in variables] for v1 in variables])
        
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
    """Plot the function graph for one or two variables."""
    if len(variables) == 1:  #hna l cas ki nmed variable wa7d
        x = variables[0]
        f_lambda = sp.lambdify(x, f, 'numpy')
        
        X = np.linspace(-5, 5, 100)
        Y = f_lambda(X)
        
        plt.figure()
        plt.plot(X, Y, label=str(f))
        plt.xlabel(str(x))
        plt.ylabel("f(x)")
        plt.title("Graph of the Function")
        
        # ya7seb critical points
        critical_points, _ = find_critical_points(f, (x,))
        for point in critical_points:
            x_val = float(point[x])
            y_val = f_lambda(x_val)
            plt.scatter(x_val, y_val, color='red', s=100)

        plt.legend()
        plt.grid()
        plt.show()

    elif len(variables) == 2:  # hadi ll plot 3d
        x, y = variables
        f_lambda = sp.lambdify((x, y), f, 'numpy')

        X = np.linspace(-5, 5, 100)
        Y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(X, Y)
        Z = f_lambda(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

        # plot nta3 critical points
        critical_points, _ = find_critical_points(f, variables)
        for point in critical_points:
            x_val, y_val = float(point[x]), float(point[y])
            z_val = f_lambda(x_val, y_val)
            ax.scatter(x_val, y_val, z_val, color='red', s=100)

        plt.show()

# hna l user y7ot l function
f_str = input("Enter a function: ")
f = sp.sympify(f_str)

# hna y detecti ch7al mn variable fl function bach ya3ref kifach yedrosha
variables = sorted(f.free_symbols, key=lambda v: str(v))

if not variables:
    print("Error: No variables detected in the function.")
else:
    critical_points, gradients = find_critical_points(f, variables)
    classifications, hessian_or_second_derivative = classify_critical_points(f, variables, critical_points)

    print("Function:", f)
    print("Gradients:", gradients)
    print("Critical Points:", critical_points)
    print("Classifications:", classifications)
    print("Hessian Matrix / Second Derivative:", hessian_or_second_derivative)

    plot_function(f, variables)
