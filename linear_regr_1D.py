import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import  r2_score
# from mpl_toolkits.mplot3d import Axes3D

class LinearRegression1D:
    """
    -  This is linear regression algorithm that fits simple 1D feature that forms 2D geometry.
    -  fit() function that calls gradient descent to fit the 'X' variable.
    -  gradient_descent() applies some calculus calculation on 'X' with target variable 'y' to generate model - -  parameters 'm' and 'b'
    -  predict() function predicts the 'y' value on given 'X' using the formula 'y = m * x + b' 
    -  plot_3D() plots 'COST' vs 'M' vs 'b' in 3D graph 

    """

    def __init__(self):
        self.m = 0
        self.b = 0
        self.m_arr = [0]
        self.b_arr = [0]

    def plot_3D(self, X, y):
        M,B = np.meshgrid(self.m_arr, self.b_arr)
        cost_func = np.vectorize(lambda m, b: np.mean((y - m * X + b)**2))
        Z = cost_func(M, B)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # 111 refers to 1 row 1 column and this plot is no.1
        ax.plot_surface(M, B, Z, cmap='viridis' )
        ax.set_xlabel('m')
        ax.set_ylabel('b')
        ax.set_zlabel('Cost')
        plt.title("Cost Surface")
        plt.show()
    
    def gradient_descent(self, X, y, L, iterations=50):

        plt.ion() # For interactive visualization
        fig, axes = plt.subplots()
        
        for _ in range(iterations):        
            # Print the M.S.E
            loss = np.mean((y - self.predict(X))**2) 
            print(f"Loss: {loss}")  

            n = len(X)
            m_descent = 0 # derivative of m
            b_descent = 0 # derivative of m
            n = len(X)

            for i in range(n):

                # Convergence formula for gradient_descent
                m_descent += -(2/n) *  X[i] * ( y[i] - ( self.m * X[i] + self.b) ) # Sum of derivative of m
                b_descent += -(2/n) * ( y[i]- (self.m* X[i] + self.b) ) # Sum of derivative of b

                # Save intermediate parameter for visualization
                self.m_arr.append(self.m - m_descent * L) 
                self.b_arr.append(self.b - b_descent * L) 

                # Visualize interactive change in regression line
                if i % 50 == 0:
                    intermediate_m = self.m - m_descent * L
                    intermediate_b = self.b - b_descent * L

                    axes.clear()
                    axes.scatter(X, y, label='Actual data', alpha=0.7)
                    X_plot = np.linspace(40,95,100)
                    y_plot = [ intermediate_m * x + intermediate_b for x in X_plot]
                    axes.plot(X_plot, y_plot, color='red', linewidth=3, label='Regression line')
                    axes.legend()
                    plt.draw()
                    plt.pause(0.3)

            # Update the model parameter
            self.m -=  m_descent * L
            self.b -=  b_descent * L  

        plt.ioff() # Off interactive visualization
        plt.show()                       
        
    def fit(self, X, y, iter=2, learning_rate=0.0001):
        self.gradient_descent( X, y, learning_rate, iter)        

    def predict(self, X):
        return self.m * np.array(X, dtype=float) + self.b

X = np.random.randint(40,95,size=500)
y = X * 10 + np.random.normal(0,100, size=500)

model = LinearRegression1D()

model.fit(X,y)

y_pred = model.predict(X)

score = r2_score(y,y_pred)

print(f"M = {model.m}, b = {model.b}\nScore:{score}")

plt.scatter(X,y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Actual Data 1D feature")
plt.show() 
model.plot_3D(X, y)
