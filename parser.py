import json

def parse_problem(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    objective_function = data['problem']['objectiveFunction']
    constraints = []

    for constraint in data['problem']['constraints']:
        left = constraint['left']
        right = constraint['right']
        operator = constraint['operator']
        constraints.append((left, operator, right))

    return objective_function, constraints

if __name__ == "__main__":
    objective, constraints = parse_problem('problem.json')
    print("Objective Function:", objective)
    print("Constraints:", constraints)
