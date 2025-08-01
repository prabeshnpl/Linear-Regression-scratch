import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class LinearRegression2D:

    def __init__(self):
        self.m1 = 0
        self.m2 = 0
        self.b = 0
        self.m1_arr = [0]
        self.m2_arr = [0]
        self.b_arr = [0]

    def gradient_descent(self, X1, X2, y, itr, L):
        plt.ion() # For interactive visualization
        fig = plt.figure(figsize=(10, 7))  
        axes = fig.add_subplot(111, projection='3d')
        X1_mesh, X2_mesh = np.meshgrid(X1, X2)

        for _ in range(itr):

            # Print the M.S.E
            loss = np.mean((y - self.predict(X1, X2))**2) 
            print(f"Loss: {loss} for itrn {_}")

            m1_descent = m2_descent = b_descent = 0
            n = len(X1)
            
            for i in range(n):               

                # calculate derivative
                m1_descent += -(2/n) * X1[i] * ( y[i] - (self.m1 * X1[i] + self.m2 * X2[i] + self.b) ) 
                m2_descent += -(2/n) * X2[i] * ( y[i] - (self.m1 * X1[i] + self.m2 * X2[i] + self.b) ) 
                b_descent += -(2/n) * ( y[i] - (self.m1 * X1[i] + self.m2 * X2[i] + self.b) ) 

                # Save intermediate m1 and m2
                self.m1_arr.append(self.m1 - m1_descent * L )
                self.m2_arr.append(self.m2 - m2_descent * L )
                self.b_arr.append(self.b - b_descent * L )

                # Visualize interactive change in regression line
                if i % 25 == 0:
                    intermediate_m1 = self.m1 - m1_descent * L
                    intermediate_m2 = self.m2 - m2_descent * L
                    intermediate_b = self.b - b_descent * L

                    Y_mesh = intermediate_m1  * X1_mesh + intermediate_m2 * X2_mesh + intermediate_b

                    axes.clear()
                    axes.scatter(X1, X2, y, c='black', alpha=0.5, label='Actual Data')
                    axes.plot_surface(X1_mesh, X2_mesh, Y_mesh, alpha=0.6, cmap='viridis')
                    axes.set_xlabel('X1')
                    axes.set_ylabel('X2')
                    axes.set_zlabel('y')
                    axes.set_title(f'Regression Plane - Epoch {i}')
                    plt.draw()
                    plt.pause(0.1)
                
            # Update model parameters
            self.m1 -= m1_descent * L 
            self.m2 -= m2_descent * L 
            self.b -= b_descent * L 
        plt.ioff()
        plt.show()

    def fit(self, X1, X2, y, itr=2, L=0.0001):
        self.gradient_descent(X1, X2, y, itr, L)

    def predict(self, X1, X2):
        return self.m1 * np.array(X1, dtype=float) + self.m2 * np.array(X2, dtype=float) + self.b

    def plot_data_3D(self, X1, X2, y):
        # Plotting 3D
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d') # 1 row 1 col and 1st plot

        scatter = ax.scatter(X1, X2, y, c=y, cmap='viridis', alpha=0.7, s=40, label='Actual Data')
        X1, X2 = np.meshgrid(X1,X2)
        y_pred = self.m1 * X1 + self.m2 * X2 + self.b
        ax.plot_surface(X1, X2, y_pred, cmap='viridis', alpha=0.3, label='Regression Plane')

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('y')
        ax.set_title('Regression Plane')

        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='Target y')

        plt.legend()        
        plt.show()

    def plot_cost_function(self):
        M1, M2 = np.meshgrid(self.m1_arr, self.m2_arr)
        cost_func = np.vectorize(lambda m1, m2: np.mean((m1 * X1 + m2 * X2 + self.b)**2)) #letting b as constact
        Z = cost_func(M1, M2)
        
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(M1, M2, Z, cmap='viridis')

        ax.set_xlabel("M1")
        ax.set_ylabel("M2")
        ax.set_zlabel("Cost")
        plt.title("Cost Surface")
        plt.show()

np.random.seed(42)
X1 = np.random.uniform(0, 100, 500)
X2 = np.random.uniform(0, 100, 500)

# Strong independent contributions
y = 5 * X1 + 7 * X2 + np.random.normal(0, 30, 500)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d') # 1 row 1 col and 1st plot
ax.scatter(X1, X2, y, c=y, cmap='viridis', alpha=0.7, s=40, label='Actual Data')
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
plt.title("3D Scatter Plot of X1, X2, and y")
plt.show()

model = LinearRegression2D()
model.fit(X1, X2, y)
y_pred = model.predict(X1,X2)

score = r2_score(y, y_pred)
print(f"Score:{score}, m1={model.m1}, m2={model.m2}, b={model.b}")

model.plot_data_3D(X1, X2, y)
model.plot_cost_function()
