
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
                if j < len(line) - 1:
                    line += key + ','
                else:
                    line += key
            if not i == len(data) - 1:
                line += '\n'
            else:
                line += str(item)
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
            data.append(int(item))
        except ValueError:
            return False, data

    return True, data        

# endregion

# endregion
