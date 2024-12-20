import numpy as np

# Define the shape of the environment (its states)
environment_rows = 11
environment_columns = 11

# Create a 3D numpy array to hold the current Q-values for each state and action pair
q_values = np.zeros((environment_rows, environment_columns, 4))

# Define actions
# Numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
actions = ['up', 'right', 'down', 'left']

# Create a 2D numpy array to hold the rewards for each state
rewards = np.full((environment_rows, environment_columns), -100.)
rewards[0, 5] = 100.  # Set the reward for the goal to 100

# Define aisle locations (white squares)
aisles = {i: [] for i in range(1, 10)}
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)] + [9]
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]

# Set the rewards for all aisle locations
for row_index in range(1, 10):
    for column_index in aisles[row_index]:
        rewards[row_index, column_index] = -1.

# Function to check if a location is a terminal state
def is_terminal_state(current_row_index, current_column_index):
    return rewards[current_row_index, current_column_index] != -1.

# Function to get a random non-terminal starting location
def get_starting_location():
    current_row_index, current_column_index = np.random.randint(environment_rows), np.random.randint(environment_columns)
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index, current_column_index = np.random.randint(environment_rows), np.random.randint(environment_columns)
    return current_row_index, current_column_index

# Function for epsilon-greedy policy
def get_next_action(current_row_index, current_column_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    else:
        return np.random.randint(4)

# Function to get the next location based on the action
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

# Function to find the shortest path from any location to the goal
def get_shortest_path(start_row_index, start_column_index):
    if is_terminal_state(start_row_index, start_column_index):
        return []
    else:
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        while not is_terminal_state(current_row_index, current_column_index):
            action_index = get_next_action(current_row_index, current_column_index, 1.)
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path

# Training parameters
epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9

# Training loop
for episode in range(1000):
    row_index, column_index = get_starting_location()
    while not is_terminal_state(row_index, column_index):
        action_index = get_next_action(row_index, column_index, epsilon)
        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(old_row_index, old_column_index, action_index)
        reward = rewards[row_index, column_index]
        old_q_value = q_values[old_row_index, old_column_index, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value
        q_values[old_row_index, old_column_index, action_index] += learning_rate * temporal_difference
    epsilon = max(0.1, epsilon * 0.99)  # Decay epsilon

print('Training complete!')

# Display shortest paths
print("Shortest path from (3, 9):", get_shortest_path(3, 9))
print("Shortest path from (5, 0):", get_shortest_path(5, 0))
print("Shortest path from (9, 5):", get_shortest_path(9, 5))

# Display an example of reversed shortest path
path = get_shortest_path(5, 2)
path.reverse()
print("Reversed path from (5, 2):", path)
