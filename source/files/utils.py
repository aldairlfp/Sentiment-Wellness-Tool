import random

def load_data(path, formatter):
    data = []
    with open(path, 'rb') as file:
        for line in file.readlines():
            can, parsed = formatter(line)
            if can:
                data.append(parsed)
            else:
                print(f"Error parsing file:{path} in index:{len(parsed) + 1}.")
                return False, None

    return True, data

# region Formatters

# region Encoding formatters
def save_list(path, data) -> bool:
    with open(path, 'w') as file:
        lines = ''
        for i, item in enumerate(data):
            line = ''
            
            for j, key in enumerate(item):
                if j < len(item) - 1:
                    line += str(key) + ','
                else:
                    line += str(key)
            if not i == len(data) - 1:
                lines += line + '\n'
            else:
                lines += line
            
        file.write(lines)

        return True

    return False

def save_lazy(path, data) -> bool:
    with open(path, 'a') as file:
        line = ''
        for i, item in enumerate(data):
            if i < len(data) - 1:
                line += str(item) + ','
            else:
                line += str(item) 
        file.write(line)

# endregion

# region Decoding formatters
def load_list(string) -> tuple[bool, list]:
    data = []
    for i, item in enumerate(string.split(',')):
        try:
            data.append(float(item))
        except ValueError:
            return False, data

    return True, data        

# endregion

# endregion

def generate_combinations(df, combination_size, num_combinations):
    # Ensure there are enough rows
    if df.shape[0] < combination_size:
        raise ValueError("DataFrame must have at least as many rows as the combination size.")

    combinations = []
    
    while len(combinations) < num_combinations:
        # Randomly sample 'combination_size' unique row indices
        selected_indices = random.sample(range(df.shape[0]), combination_size)
        
        # Create a combination of rows based on the selected indices
        combination = df.iloc[selected_indices].to_numpy().tolist()  # Convert to list of lists
        
        # Append the combination to the list
        combinations.append(combination)

    return combinations