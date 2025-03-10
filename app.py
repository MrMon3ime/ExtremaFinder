from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import sympy as sp
from sympy import symbols, diff, solve, hessian

app = Flask(__name__, template_folder='templates')

def find_critical_points(f, variables):
    x, y = variables
    grad = [diff(f, var) for var in (x, y)]
    
    solutions = solve(grad, (x, y), dict=True)

    critical_points = []
    classifications = []
    
    hessian_matrix = hessian(f, (x, y))

    for solution in solutions:
        x_val = solution.get(x)
        y_val = solution.get(y)

        if x_val is not None and y_val is not None:
            critical_points.append((float(x_val), float(y_val)))

            H = hessian_matrix.subs(solution)
            det_H = H.det()
            if det_H > 0:
                if H[0, 0] > 0:
                    classifications.append((f"({x_val}, {y_val})", "Local Min"))
                else:
                    classifications.append((f"({x_val}, {y_val})", "Local Max"))
            elif det_H < 0:
                classifications.append((f"({x_val}, {y_val})", "Saddle Point"))
            else:
                classifications.append((f"({x_val}, {y_val})", "Indeterminate"))

    return critical_points, classifications, hessian_matrix



def plot_function(f, variables, critical_points):
    x, y = variables
    f_lambda = sp.lambdify((x, y), f, 'numpy')

    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    
    try:
        Z = f_lambda(X, Y)
    except Exception:
        return None

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)


    for point in critical_points:
        x_val, y_val = point
        z_val = f_lambda(x_val, y_val)
        ax.scatter(x_val, y_val, z_val, color='red', s=100)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("f(x, y)")
    ax.set_title("3D Plot of the Function")

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close(fig)
    return plot_url 


@app.route("/", methods=["GET", "POST"])
def home():
    plot_url = None
    function_str = None
    gradients = None
    critical_points = None
    hessian_matrix = None
    classifications = None

    if request.method == "POST":
        x, y = symbols("x y")

        try:
            function_str = request.form["function"]
            function = sp.sympify(function_str)

            gradients = [sp.diff(function, var) for var in (x, y)]

            critical_points, classifications, hessian_matrix = find_critical_points(function, (x, y))

            plot_url = plot_function(function, (x, y), critical_points)

            function_latex = sp.latex(function)
            gradients_latex = sp.latex(gradients)
            critical_points_latex = [f"({p[0]}, {p[1]})" for p in critical_points]
            hessian_latex = sp.latex(hessian_matrix)

            return render_template(
                "index.html",
                function_str=function_latex,
                gradients=gradients_latex,
                critical_points=critical_points_latex,
                classifications=classifications,
                hessian=hessian_latex,
                plot_url=plot_url
            )

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
