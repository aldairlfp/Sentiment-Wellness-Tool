conv_data = []
conv_labels = []

data_clustered = {}

for index, label in enumerate(list(kmeans.labels_)):
    if label in list(data_clustered.keys()):
        data_clustered[label].append(data[index])
    else:
        data_clustered[label] = [data[index]]
    
for count, key in enumerate(list(data_clustered.keys())):
    n_data = []
    for index, item in enumerate(data_clustered[key]):
        
        if len(n_data) == 128:
            conv_data.append(np.array(n_data))
            conv_labels.append(key)
            n_data = []
            if len(conv_data) == (count + 1) * 1000:
                break

        n_data.append(item)
        
        for i, item2 in enumerate(data_clustered[key]):
        
            if not i == index:
                n_data.append(item2)
        
            if len(n_data) == 128:
                conv_data.append(np.array(n_data))
                conv_labels.append(key)
                if len(conv_data) == (count + 1) * 1000:
                    break
                n_data = []
        
conv_data