"""Example of applying PCA on to data generated using Gaussian and uniform mixture models with different dimensions.
   PDF parameters selected first and then data is generated from a mixture of PDFs. Principal components are later
   generated using data, before plotting the Principal component axes.
"""

import matplotlib.pyplot as plt
import numpy as np

from modules import models, prob_utils

np.random.seed(7)

# No. of samples
N = 1000


def main():
    # ========== PCA applied to data from a Gaussian Mixture PDF ==========

    # Generate data  from an arbitrary GMM of 3D component distributions
    priors = np.array([0.2, 0.5, 0.3])  # Likelihood of each distribution to be selected
    # Determine number of mixture components
    C = len(priors)
    mu = np.array([[1, -10, 3],
                   [4, -2, 10],
                   [5, -5, 0]])  # Gaussian distributions means
    sigma = np.array([[[2, -.5, -.4],
                       [-.5, 6, -.2],
                       [-.4, -.2, 10]],
                      [[2, .5, .6],
                       [.5, 5, .8],
                       [.6, .8, 10]],
                      [[4, .5, -.9],
                       [.5, 7, .4],
                       [-.9, .4, 12]]])  # Gaussian distributions covariance matrices
    # Create PDF parameter structure
    gmm_params = prob_utils.GaussianMixturePDFParameters(priors, C, mu, np.transpose(sigma))
    gmm_params.print_pdf_params()
    # Determine dimensionality from mixture PDF parameters
    n = mu.shape[0]
    # Generate 3D matrix from a mixture of 3 Gaussians
    XT, y = prob_utils.generate_mixture_samples(N, n, gmm_params, False)
    # Transpose XT into shape [N, n] to fit into algorithm
    X_GMM = XT.T

    # Perform PCA on transposed GMM variable X
    _, _, Z = models.perform_pca(X_GMM)

    # Add back mean vector to PC projections if you want PCA reconstructions
    Z_GMM = Z + np.mean(X_GMM, axis=0)

    # Plot original data vs PCA reconstruction data
    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(211, projection='3d')
    ax1.scatter(X_GMM[:, 0], X_GMM[:, 1], X_GMM[:, 2], c=y)
    ax1.set_xlabel(r"$x_1$", fontsize=16)
    ax1.set_ylabel(r"$x_2$", fontsize=16)
    ax1.set_zlabel(r"$x_3$", fontsize=16)
    ax1.set_title("x ~ {}D GMM data".format(n), fontsize=20)

    ax2 = fig.add_subplot(212, projection='3d')
    ax2.scatter(Z_GMM[:, 0], Z_GMM[:, 1], Z_GMM[:, 2])
    ax2.set_xlabel(r"$z_1$", fontsize=16)
    ax2.set_ylabel(r"$z_2$", fontsize=16)
    ax2.set_zlabel(r"$z_3$", fontsize=16)
    ax2.set_title("PCA projections of {}D GMM data".format(n), fontsize=20)
    plt.show()

    # Let's see what it looks like only along the first two PCs
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(Z_GMM[:, 0], Z_GMM[:, 1])
    plt.xlabel(r"$z_1$", fontsize=16)
    plt.ylabel(r"$z_2$", fontsize=16)
    plt.title("PCA projections of {}D GMM to 2D space".format(n), fontsize=20)
    plt.show()

    # ========== PCA applied to data from a Uniform Mixture PDF ==========

    # Generate data  from an arbitrary UMM of 3D component distributions
    priors = np.array([0.2, 0.3, 0.5])  # Likelihood of each distribution to be selected
    # Determine number of mixture components
    C = len(priors)
    a = np.array([[1, -10, 10],
                  [5, 13, -5],
                  [1, -5, 5]])  # Uniform distributions lower endpoints (x & y axes)
    b = np.array([[5, -5, 20],
                  [8, 20, 0],
                  [2, 5, 10]])  # Uniform distributions higher endpoints (x & y axes)
    # Create PDF parameter structure
    umm_params = prob_utils.UniformMixturePDFParameters(priors, C, a, b)
    umm_params.print_pdf_params()
    # Determine dimensionality from mixture PDF parameters
    n = a.shape[0]
    # Generate 3D matrix from a mixture of 3 Uniforms
    XT, y = prob_utils.generate_mixture_samples(N, n, umm_params, False)
    # Transpose XT into shape [N, n] to fit into algorithm
    X_UMM = XT.T

    # Perform PCA on transposed UMM variable X
    _, _, Z = models.perform_pca(X_UMM)

    # Add back mean vector to PC projections if you want PCA reconstructions
    Z_UMM = Z + np.mean(X_UMM, axis=0)

    # Plot original data vs PCA reconstruction data
    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(211, projection='3d')
    ax1.scatter(X_UMM[:, 0], X_UMM[:, 1], X_UMM[:, 2], c=y)
    ax1.set_xlabel(r"$x_1$", fontsize=16)
    ax1.set_ylabel(r"$x_2$", fontsize=16)
    ax1.set_zlabel(r"$x_3$", fontsize=16)
    ax1.set_title("x ~ {}D UMM data".format(n), fontsize=20)

    ax2 = fig.add_subplot(212, projection='3d')
    ax2.scatter(Z_UMM[:, 0], Z_UMM[:, 1], Z_UMM[:, 2])
    ax2.set_xlabel(r"$z_1$", fontsize=16)
    ax2.set_ylabel(r"$z_2$", fontsize=16)
    ax2.set_zlabel(r"$z_3$", fontsize=16)
    ax2.set_title("PCA projections of {}D UMM data".format(n), fontsize=20)
    plt.show()

    # Let's see what it looks like only along the first two PCs
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(Z_UMM[:, 0], Z_UMM[:, 1])
    plt.xlabel(r"$z_1$", fontsize=16)
    plt.ylabel(r"$z_2$", fontsize=16)
    plt.title("PCA projections of {}D UMM to 2D space".format(n), fontsize=20)
    plt.show()


if __name__ == '__main__':
    main()
