from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import pickle

with open("buffer_record.pkl", 'rb') as f:
    data = pickle.load(f)

data = np.array(data)

X = data[:, :-1]
Y = data[:, -1:]

X_train = X[:-1000]
Y_train = Y[:-1000]

X_test = X[-1000:]
Y_test = Y[-1000:]

# Initialize DecisionTreeRegressor
regressor = DecisionTreeRegressor()

# Train the model
regressor.fit(X_train, Y_train)

# Predictions on the test set
y_pred = regressor.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)


# Save the trained regressor
joblib.dump(regressor, 'regressor_model.pkl')

regressor = joblib.load('regressor_model.pkl')

# Predictions on the test set
y_pred = regressor.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(Y_test, y_pred)
print("Mean Squared Error:", mse)