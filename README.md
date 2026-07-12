# Linear Regression from Scratch using Gradient Descent

A machine learning project that implements Multiple Linear Regression from scratch using Gradient Descent optimization. This project is built without Scikit-Learn's regression libraries to demonstrate the underlying mathematics of linear modeling and convergence.

## Features
* Multiple Linear Regression implemented from scratch.
* Gradient Descent optimization loop.
* Feature scaling/normalization (Z-score normalization).
* Performance evaluation using Mean Squared Error (MSE) and R² Score.
* Interactive web interface using Streamlit.
* Unified plots showing Loss Curve and Actual vs Predicted values.

## Project Structure
```text
LinearRegression-GD/
├── House_Price_Predictor_LinearRegression.ipynb   # Google Colab Notebook (Model training & experiments)
├── app.py                                         # Streamlit Web App (Contains model, pipeline & UI)
└── README.md                                      # Documentation
```

## Mathematical Implementation

### 1. Loss Function (MSE)
The model measures error using Mean Squared Error:
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2$$

### 2. Gradient Descent Updates
Weights ($w$) and bias ($b$) are updated iteratively:
$$w = w - \alpha \frac{\partial J}{\partial w}$$
$$b = b - \alpha \frac{\partial J}{\partial b}$$

* $w$: Weights vector (Area, Bedrooms, Age parameters)
* $\alpha$: Learning Rate (step size)
* $J$: Cost function (MSE)

## How to Run

### 1. Install Dependencies
```bash
pip install streamlit numpy pandas matplotlib
```

### 2. Run the Streamlit Web App
```bash
streamlit run app.py
```

## Future Improvements
* Add Stochastic Gradient Descent (SGD).
* Implement L1/L2 Regularization (Lasso/Ridge).
* Support Polynomial Regression features.
