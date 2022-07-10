from sys import float_info

import numpy as np


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
