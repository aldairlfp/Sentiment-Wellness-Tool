classes = []
idx_to_labels = {}
labels_to_idx = {}
p_labels = []

for label in labels:
    i = 0
    
    if label not in classes:
        classes.append(label)
        idx_to_labels[i] = label
        labels_to_idx[label] = i

    p_labels.append(labels_to_idx[label])
    
    i += 1

p_labels = np.array(p_labels)
p_labels