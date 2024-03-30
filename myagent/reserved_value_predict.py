from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    X_train, X_test, Y_train, Y_test = prepare_data("./state_records.pkl")
    model = train(X_train, Y_train)
    eval(model, X_test, Y_test)
def prepare_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # print(f"Data Length: {len(data)}")

    data = np.array(data)

    X = data[:, :-1]
    Y = data[:, -1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, shuffle=False)
    return X_train, X_test, Y_train, Y_test

def train(X_train, Y_train):
    # Create a model and train it
    model = DecisionTreeRegressor()
    model.fit(X_train, Y_train)
    return model

def eval(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    print("Mean Squared Error:", mse)

if __name__ == '__main__':
    main()



