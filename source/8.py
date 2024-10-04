X = transformed_data.values

Y = transformed_data.columns.values

# Grafica la distribución de datos (_ground truth_) usando matplotlib `scatter(axis-x, axis-y, color)`.
plt.scatter(X[:,0], X[:,1])