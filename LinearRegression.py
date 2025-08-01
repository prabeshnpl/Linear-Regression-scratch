import numpy as np
from sklearn.metrics import r2_score
import pandas as pd

class LinearRegression:
    """
    - 'X' should be pandas dataframe
    - 'y' should be np array
    - 'fit' using obj.fit(X, y)
    - 'predict' using obj.predict(X)
    - 'fit' has following parameters:
        1) X : features
        2) y : target feature
        3) itr : no. of iterations
        4) L : learning rate
    - 'fit' uses 'gradient_descent' function to fit the data
    """
    def __init__(self):
        self.m = []
        self.c = 0
        self.nrows = 0
        self.ncols = 0

    def gradient_descent(self, X , y, itr, L):
            
        for _ in range(itr):
            if _ % 50 == 0:
                # Print the M.S.E
                loss = np.mean((y - self.predict(X))**2) 
                print(f"Loss: {loss} for itrn {_}")

            m_descent = [0 for _ in range(self.ncols)]
            c_descent = 0
            
            # Calculate gradient_descent
            for i in range(self.nrows):   
                y_pred = sum(self.m[j] * X.iloc[i,j] for j in range(self.ncols)) + self.c
                # calculate derivative for every feature/column
                for col in range(self.ncols):
                    m_descent[col] += -(2/self.nrows) * X.iloc[i,col] * (y[i] - y_pred) 
                c_descent += -(2/self.nrows) * (y[i] - y_pred)

            # Update model parameters
            for i in range(self.ncols):
                self.m[i] -= m_descent[i] * L            
            self.c -= c_descent * L

    def fit(self, X, y, itr=300, L=0.00001):
        nrows, ncols = X.shape
        self.m = [0 for _ in range(ncols)]
        self.nrows = nrows
        self.ncols = ncols

        self.gradient_descent(X, y, itr, L)

    def predict(self, X):
        X_np = np.array(X, dtype=float)
        return np.dot(X_np, self.m) + self.c

np.random.seed(42)
X1 = np.random.uniform(0, 100, 500)
X2 = np.random.uniform(0, 100, 500)
X3 = np.random.uniform(0, 100, 500)
X4 = np.random.uniform(0, 100, 500)
X5 = np.random.uniform(0, 100, 500)

# Strong independent contributions
y = 5 * X1 + 7 * X2 - 2 * X3 + 5 * X4 - 5 * X5 + np.random.normal(0, 30, 500)

X = pd.DataFrame({
    'x1':X1,
    'x2':X2,
    'x3':X3,
    'x4':X4,
    'x5':X5,
})
print("\n Linear Regression algorithm for 5 feature and a target variable 'y' \n")
model = LinearRegression()
model.fit(X, y, itr=500)
y_pred = model.predict(X)

score = r2_score(y, y_pred)
print(f"Score:{score}")
print(f"Parameters : \nm =  {np.array(model.m,dtype='float')}")
print("c = ",model.c)