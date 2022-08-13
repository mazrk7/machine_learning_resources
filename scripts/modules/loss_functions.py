from sys import float_info

import numpy as np


# Define the logistic/sigmoid function
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# Define the prediction function y = 1 / (1 + np.exp(-X*theta))
# X.dot(theta) inputs to the sigmoid referred to as logits
def predict_prob(X, theta):
    logits = X.dot(theta)
    return sigmoid(logits)


# Binary cross entropy, equiv to NLL in this context
def bce(y, predictions):
    # Epsilon adjustment handles numerical errors by avoid underflow or overflow
    predictions = np.clip(predictions, float_info.epsilon, 1 - float_info.epsilon)
    # log P(Y=0 | x; theta)
    log_p0 = (1 - y) * np.log(1 - predictions + float_info.epsilon)
    # log P(Y=1 | x; theta)
    log_p1 = y * np.log(predictions + float_info.epsilon)

    return -np.mean(log_p0 + log_p1, axis=0)


# NOTE: This implementation may encounter numerical stability issues...
# Read into the log-sum-exp trick OR use a method like: sklearn.linear_model import LogisticRegression
def log_reg_loss(theta, X, y):
    # Size of batch
    B = X.shape[0]

    # Logistic regression model g(X * theta)
    predictions = predict_prob(X, theta)

    # NLL loss, 1/N sum [y*log(g(X*theta)) + (1-y)*log(1-g(X*theta))]
    nll = bce(y, predictions)

    # Partial derivative for GD
    error = predictions - y
    g = (1 / B) * X.T.dot(error)

    # Logistic regression loss, NLL (binary cross entropy is another interpretation)
    return nll, g


# Mean Squared Error (MSE) loss with gradient for optimization:
def lin_reg_loss(theta, X, y):
    # Size of batch
    B = X.shape[0]
    # Linear regression model X * theta
    predictions = X.dot(theta)
    # Residual error (X * theta) - y
    error = predictions - y
    # Loss function is MSE
    loss_f = np.mean(error ** 2)
    # Partial derivative for GD, X^T * ((X * theta) - y)
    g = (1 / B) * X.T.dot(error)

    return loss_f, g
