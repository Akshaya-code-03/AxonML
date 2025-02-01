from axonml import LinearRegression
import numpy as np

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Model training
model = LinearRegression()
model.fit(X, y)

# Prediction
predictions = model.predict(X)
print(predictions)