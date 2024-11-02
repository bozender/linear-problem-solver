import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

# Define variables
x, y = symbols('x y')

# Define constraints as inequalities
constraints = [
    (2*x + y <= 20),  # example constraint
    (4*x + 3*y <= 30),  # example constraint
    (x >= 0),  # x-axis constraint
    (y >= 0)   # y-axis constraint
]

# Define the objective function
objective = 3*x + 5*y  # example objective

# Define functions to calculate feasible region and solutions
def plot_constraints():
    plt.figure(figsize=(8, 8))
    x_vals = np.linspace(0, 20, 200)
    
    # Example plotting constraints
    plt.plot(x_vals, (20 - 2*x_vals), label=r'$2x + y \leq 20$')
    plt.plot(x_vals, (30 - 4*x_vals)/3, label=r'$4x + 3y \leq 30$')
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    # Fill feasible region (you may need to adjust to cover all constraints)
    plt.fill_between(x_vals, np.minimum((20 - 2*x_vals), (30 - 4*x_vals)/3), where=(x_vals >= 0), color='gray', alpha=0.3)

    # Adding labels, legend, and showing plot
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()

def find_optimal_points():
    # Find intersection points of lines, evaluate objective function
    # Example:
    intersections = [solve((2*x + y - 20, 4*x + 3*y - 30), (x, y))]
    for point in intersections:
        print(f'Intersection at: {point}, Objective: {objective.subs(point)}')

plot_constraints()
find_optimal_points()
