# 🧠 Custom Linear Regression from Scratch

This project demonstrates how **Linear Regression** works by implementing it from scratch using Python and NumPy. It includes interactive visualizations to help understand model training and the cost function surface.

## 📦 Project Structure

This project is divided into three parts:

### ✅ 1. Univariate Linear Regression (1D)
- Predict using one feature (e.g., `x` vs `y`)
- Visualizes the **fitting line** with animation
- Visualizes the **cost function (MSE) vs slope (m) and intercept (b)** as a 3D surface

### ✅ 2. Bivariate Linear Regression (2D)
- Predict using two features (e.g., `x₁`, `x₂` vs `y`)
- Animates the **fitting plane** being learned
- Visualizes the **cost surface with respect to m₁ and m₂**

### ✅ 3. Multivariate Linear Regression (nD)
- Generalized model for any number of features
- Uses gradient descent to optimize weights
- Trains with batch gradient descent
- Evaluates model using **R² score**

---

## 📊 Visualizations

- **Line and Plane Animation:** Observe how gradient descent updates weights to fit data
- **Cost Function Surfaces:**
  - For 1D: `J(m, b)`
  - For 2D: `J(m₁, m₂)`
- Helps visualize convexity of MSE cost

---

## 🛠 Features

- ✅ Implemented `LinearRegression` class from scratch
- ✅ Works for 1D, 2D, and nD data
- ✅ Visualizes convergence and training process

## 🧠 Learning Goals
- ✅ Understand gradient descent mechanics

- ✅ Visualize how cost minimization happens

- ✅ Build a model from raw math concepts (no libraries!)

- ✅ See where and how linear regression works—and where it doesn’t

---

## 🚀 Getting Started

1. Clone this repo
2. Install requirements:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
3. Run the relevant script:
   ```bash
   python linear_regression/LinearRegression.py
