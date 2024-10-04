# new features
data = []

for item in features:
    temp = []

    for k in item:
        temp.append(item[k])

    data.append(temp)

data = np.array(data)

data