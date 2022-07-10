"""Learning algorithms and models."""

import matplotlib.pyplot as plt
import numpy as np

# Suppress scientific notation
np.set_printoptions(suppress=True)


from scripts.modules import data_utils


def analytical_ls_solution(X, y):
    # Analytical solution is (X^T*X)^-1 * X^T * y
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def gradient_descent(loss_func, theta_init, X, y, *args, **kwargs):
    """  Mini-batch GD for LS regression. Stochastic GD if batch_size=1.

    Args:
        loss_func: Loss function handle to optimize over using GD.
        theta_init: Initial parameters vector, of shape (n + 1).
        X: Design matrix (added bias units), shape (N, n + 1), where n is the feature dimensionality.
        y: Labels for regression problem, of shape (N, 1).
        opts: Options for total sweeps over data (max_epochs), and parameters, like learning rate and momentum.

    Returns:
        theta: Final weights solution converged to after `iterations`, of shape [n].
        trace: Arrays of loss and weight updates, of shape [iterations, -1].
    """

    # Default options
    max_epoch = kwargs['max_epoch'] if 'max_epoch' in kwargs else 200
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.1
    epsilon = kwargs['tolerance'] if 'tolerance' in kwargs else 1e-6

    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 10

    # Turn the data into batches
    X_batch, y_batch = data_utils.batchify(X, y, batch_size)
    num_batches = len(y_batch)
    print("%d batches of size %d\n" % (num_batches, batch_size))

    theta = theta_init

    trace = {}
    trace['loss'] = []
    trace['theta'] = []

    # Main loop:
    for epoch in range(1, max_epoch + 1):
        print("epoch %d\n" % epoch)
        for b in range(num_batches):
            X_b = X_batch[b]
            y_b = y_batch[b]
            # print("epoch %d batch %d\n" % (epoch, b))

            mse, gradient = loss_func(theta, X_b, y_b, *args)

            # Steepest descent update
            theta = theta - alpha * gradient

            # Storing the history of the parameters and loss values (MSE)
            trace['loss'].append(mse)
            trace['theta'].append(theta)

            # Terminating Condition is based on how close we are to minimum (gradient = 0)
            if np.linalg.norm(gradient) < epsilon:
                print("Gradient Descent has converged")
                break

        # Also break epochs loop
        if np.linalg.norm(gradient) < epsilon:
            break

    return theta, trace