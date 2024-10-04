from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

for fold, (train_index, val_index) in enumerate(skf.split(data, p_labels)):
    print(f"Fold {fold + 1}/{3}")

    X_train, X_val = data[train_index], data[val_index]
    y_train, y_val = labels[train_index], labels[val_index]

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    # Create a new instance of the model for each fold
    model = create_model()
    
    # Fit the model on the training data
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Evaluate the model on the validation data
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy:.4f}")