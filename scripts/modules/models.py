"""Learning algorithms and models."""

import matplotlib.pyplot as plt
import numpy as np

# Suppress scientific notation

np.set_printoptions(suppress=True)

# If you used the scikit-learn dimensionality reduction models
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.decomposition import PCA


from modules import data_utils


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


def perform_lda(X, labels, C=2, plot_vec=True):
    """  Fisher's Linear Discriminant Analysis (LDA) on data from two classes (C=2).

    Note: you can generalize this implementation to multiple classes by finding a
    projection matrix W (rather than vector) that reduces dimensionality n inputs
    to a multidimensional projection z=W'*x (e.g. z of dimension C-1). Now we have
    a crude but quick way of achieving class-separability-preserving linear dimensionality
    reduction using the Fisher LDA objective as a measure of class separability.

    Args:
        X: Real-valued matrix of samples with shape [N, n], N for sample count and n for dimensionality.
        labels: Class labels per sample received as an [N, 1] column.
        C: Number classes, explicitly clarifying that we're doing binary classification here.
        plot_vec: If you want the option of directly plotting the linear projection vector over your input space.

    Returns:
        w: Fisher's LDA project vector, shape [n, 1].
        z: Scalar LDA projections of input samples, shape [N, 1].
    """

    # Estimate mean vectors and covariance matrices from samples
    # Note that reshape ensures my return mean vectors are of 2D shape (column vectors nx1)
    mu = np.array([np.mean(X[labels == i], axis=0).reshape(-1, 1) for i in range(C)])
    cov = np.array([np.cov(X[labels == i].T) for i in range(C)])

    # Determine between class and within class scatter matrix
    Sb = (mu[1] - mu[0]).dot((mu[1] - mu[0]).T)
    Sw = cov[0] + cov[1]

    # Regular eigenvector problem for matrix Sw^-1 Sb
    lambdas, U = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    # Get the indices from sorting lambdas in order of increasing value, with ::-1 slicing to then reverse order
    idx = lambdas.argsort()[::-1]
    # Extract corresponding sorted eigenvectors
    U = U[:, idx]
    # First eigenvector is now associated with the maximum eigenvalue, mean it is our LDA solution weight vector
    w = U[:, 0]

    # Scalar LDA projections in matrix form
    z = X.dot(w)

    # If using sklearn instead:
    # lda = LinearDiscriminantAnalysis()
    # X_fit = lda.fit(X, labels)  # Is a fitted estimator, not actual data to project
    # z = lda.transform(X)
    # w = X_fit.coef_[0]

    if plot_vec:
        # Get midpoint between class means
        mid_point = (mu[0] + mu[1]) / 2
        # Work out slope of Fisher's direction vector
        slope = w[1] / w[0]
        # Compute intercept through line segment between means
        c = mid_point[1] - slope * mid_point[0]

        xmax = np.max(X[:, 0])
        xmin = np.min(X[:, 0])
        x = np.linspace(xmin + 1, xmax + 1, 100)

        fig = plt.figure(figsize=(12, 12))

        x0 = X[labels == 0]
        x1 = X[labels == 1]
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(x0[:, 0], x0[:, 1], 'b.', x1[:, 0], x1[:, 1], 'r+')
        ax1.plot(x, slope * x + c, c='orange')
        ax1.set_xlabel(r"$x_1$", fontsize=18)
        ax1.set_ylabel(r"$x_2$", fontsize=18)
        ax1.legend(["Class 0", "Class 1", r"$w_{LDA}$"])

        ax2 = fig.add_subplot(2, 1, 2)
        z0 = z[labels == 0]
        z1 = z[labels == 1]
        ax2.plot(z0, np.zeros(len(z0)), 'b.', z1, np.zeros(len(z1)), 'r+')
        ax2.set_xlabel(r"$w_{LDA}^\intercal x$", fontsize=18)
        plt.show()

    return w, z


def perform_pca(X):
    """  Principal Component Analysis (PCA) on real-valued vector data.

    Args:
        X: Real-valued matrix of samples with shape [N, n], N for sample count and n for dimensionality.

    Returns:
        U: An orthogonal matrix [n, n] that contains the PCA projection vectors, ordered from first to last.
        D: A diagonal matrix [n, n] that contains the variance of each PC corresponding to the projection vectors.
        Z: PC projection matrix of the zero-mean input samples, shape [N, n].
    """

    # First derive sample-based estimates of mean vector and covariance matrix:
    mu = np.mean(X, axis=0)
    sigma = np.cov(X.T)

    # Mean-subtraction is a necessary assumption for PCA, so perform this to obtain zero-mean sample set
    C = X - mu

    # Get the eigenvectors (in U) and eigenvalues (in D) of the estimated covariance matrix
    lambdas, U = np.linalg.eig(sigma)
    # Get the indices from sorting lambdas in order of increasing value, with ::-1 slicing to then reverse order
    idx = lambdas.argsort()[::-1]
    # Extract corresponding sorted eigenvectors and eigenvalues
    U = U[:, idx]
    D = np.diag(lambdas[idx])

    # PC projections of zero-mean samples, U^Tx (x mean-centred), matrix over N is XU
    Z = C.dot(U)

    # If using sklearn instead:
    # pca = PCA(n_components=X.shape[1])  # n_components is how many PCs we'll keep... let's take all of them
    # X_fit = pca.fit(X)  # Is a fitted estimator, not actual data to project
    # Z = pca.transform(X)

    return U, D, Z
