import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from shapely.geometry import Polygon
from parser import parse_problem  # Make sure the parser script is in the same directory

# Mathematical function: Solve constraint for y in terms of x
def calculate_constraint_points(constraints, x_range):
    x, y = symbols('x y')
    constraint_points = []

    for left_expr, operator, right_expr in constraints:
        left_eq = Eq(eval(left_expr), eval(right_expr))
        y_solution = solve(left_eq, y)
        
        if y_solution:
            y_func = lambda x_val: y_solution[0].subs(x, x_val)
            y_vals = [y_func(val) if y_func(val).is_real else np.nan for val in x_range]
            constraint_points.append((x_range, y_vals, operator, f"{left_expr} {operator} {right_expr}"))
        else:
            x_const = solve(left_eq, x)
            if x_const and x_const[0].is_real:
                constraint_points.append(([float(x_const[0])] * len(x_range), None, operator, f"{left_expr} {operator} {right_expr}"))

    return constraint_points

# Mathematical function: Determine plot range based on constraints
def determine_plot_range(constraints):
    x_vals, y_vals = [], []
    x, y = symbols('x y')

    for left_expr, _, right_expr in constraints:
        left_eq = Eq(eval(left_expr), eval(right_expr))
        y_solution = solve(left_eq, y)
        x_solution = solve(left_eq, x)

        if y_solution:
            x_range = np.linspace(-10, 10, 50)
            y_vals += [float(y_solution[0].subs(x, val)) for val in x_range if y_solution[0].subs(x, val).is_real]
        if x_solution and x_solution[0].is_real:
            x_vals.append(float(x_solution[0]))
    
    x_min, x_max = (min(x_vals, default=-10), max(x_vals, default=10))
    y_min, y_max = (min(y_vals, default=-10), max(y_vals, default=10))
    
    padding = 2
    return (x_min - padding, x_max + padding), (y_min - padding, y_max + padding)

# Drawing function: Draw constraints with shading on the plot
def draw_constraints(constraints, x_range, y_range):
    plt.figure(figsize=(8, 8))
    x_vals = np.linspace(x_range[0], x_range[1], 400)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(constraints)))
    constraint_points = calculate_constraint_points(constraints, x_vals)

    for i, (x_vals, y_vals, operator, label) in enumerate(constraint_points):
        color = colors[i]
        
        if y_vals:
            numeric_y_vals = [float(y) if y.is_real else np.nan for y in y_vals]
            plt.plot(x_vals, numeric_y_vals, label=label, color=color)
            if operator == '<=':
                plt.fill_between(x_vals, numeric_y_vals, y_range[0], color=color, alpha=0.2)
            elif operator == '>=':
                plt.fill_between(x_vals, numeric_y_vals, y_range[1], color=color, alpha=0.2)
        else:
            plt.axvline(x=x_vals[0], color=color, linestyle='--', label=label)
            if operator == '<=':
                plt.fill_betweenx(np.linspace(y_range[0], y_range[1], 400), x_vals[0], x_range[0], color=color, alpha=0.2)
            elif operator == '>=':
                plt.fill_betweenx(np.linspace(y_range[0], y_range[1], 400), x_vals[0], x_range[1], color=color, alpha=0.2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.annotate('', xy=(x_range[1], 0), xytext=(x_range[0], 0),
                 arrowprops=dict(arrowstyle='<->', color='black'))
    plt.annotate('', xy=(0, y_range[1]), xytext=(0, y_range[0]),
                 arrowprops=dict(arrowstyle='<->', color='black'))
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.legend()
    plt.grid(True)
    plt.title("Linear Programming Problem Solver")

# Function to find intersections of constraints in the feasible region
def find_feasible_region_vertices(constraints):
    x, y = symbols('x y')
    vertices = []

    for i in range(len(constraints)):
        left_expr1, op1, right_expr1 = constraints[i]
        eq1 = Eq(eval(left_expr1), eval(right_expr1))
        
        for j in range(i+1, len(constraints)):
            left_expr2, op2, right_expr2 = constraints[j]
            eq2 = Eq(eval(left_expr2), eval(right_expr2))
            
            solution = solve((eq1, eq2), (x, y))
            if solution and all(isinstance(val, (int, float)) for val in solution.values()):
                x_val, y_val = solution[x], solution[y]
                if x_val >= 0 and y_val >= 0:
                    vertices.append((float(x_val), float(y_val)))

    for left_expr, operator, right_expr in constraints:
        eq = Eq(eval(left_expr), eval(right_expr))

        y_intercept = solve(eq.subs(x, 0), y)
        if y_intercept and y_intercept[0] >= 0:
            vertices.append((0, float(y_intercept[0])))

        x_intercept = solve(eq.subs(y, 0), x)
        if x_intercept and x_intercept[0] >= 0:
            vertices.append((float(x_intercept[0]), 0))

    return vertices

# Function to shade the feasible region and return vertices
def shade_feasible_region(constraints):
    vertices = find_feasible_region_vertices(constraints)
    # Remove duplicate vertices by converting to a set and back to a list
    vertices = list(set(vertices))
    
    feasible_region = Polygon(vertices)
    
    if feasible_region.is_valid and feasible_region.area > 0:
        x_poly, y_poly = feasible_region.exterior.xy
        plt.fill(x_poly, y_poly, color="lightgreen", alpha=0.4, label="Feasible Region")
        
        # Mark vertices in red
        x_vertices, y_vertices = zip(*vertices)
        plt.scatter(x_vertices, y_vertices, color="red", s=50, zorder=5)

    return vertices


# Function to add a prominent text block at the bottom of the plot
def add_text_block(vertices):
    vertex_text = "Possible Optimal Solutions: " + ", ".join([f"({round(x, 2)}, {round(y, 2)})" for x, y in vertices])
    plt.figtext(
        0.5, 0.01, vertex_text, ha="center", fontsize=12, fontweight="bold",
        color="white", bbox={"facecolor": "black", "alpha": 0.8, "pad": 10}
    )


# Main function to execute the program
def main(json_file):
    objective_function, constraints = parse_problem(json_file)
    print("Objective Function:", objective_function)
    print("Constraints:", constraints)
    
    x_range, y_range = determine_plot_range(constraints)
    draw_constraints(constraints, x_range, y_range)
    vertices = shade_feasible_region(constraints)
    add_text_block(vertices)
    plt.show()

# Example usage
if __name__ == "__main__":
    main('linear-problem-solver/problem.json')  # Adjust path as needed
