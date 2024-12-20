import numpy as np

# Define the size of the environment
environment_rows = 11
environment_columns = 11

# Define rewards
rewards = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 3, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 3, 3, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 3, 1, 1, 0, 3, 3, 3],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Initialize the value function
V = np.zeros((environment_rows, environment_columns))
V += rewards  # Incorporate rewards into the initial values

# Define actions: up, right, down, left
actions = ['up', 'right', 'down', 'left']

# Determine if a location is a terminal state
def is_terminal_state(current_row_index, current_column_index):
    return rewards[current_row_index, current_column_index] != -1.

# Get the next location based on the chosen action
def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index, new_column_index = current_row_index, current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index

# Value iteration function
def value_iterations(V, threshold=0.01, discount_factor=0.9):
    number_iterations = 0
    while True:
        oldV = V.copy()
        for row_index in range(environment_rows):
            for column_index in range(environment_columns):
                if not is_terminal_state(row_index, column_index):
                    Q_values = np.zeros(4)
                    for action_index in range(len(actions)):
                        new_row_index, new_column_index = get_next_location(row_index, column_index, action_index)
                        Q_values[action_index] = rewards[new_row_index, new_column_index] + discount_factor * V[new_row_index, new_column_index]
                    V[row_index, column_index] = np.max(Q_values)
        number_iterations += 1
        if np.max(np.abs(oldV - V)) < threshold:
            break
    return V, number_iterations

V, number_iterations = value_iterations(V)
print("Value Function:")
print(V)
print("Number of Iterations:")
print(number_iterations)
