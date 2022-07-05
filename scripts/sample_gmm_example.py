"""Example of sampling from a mixture of univariate, bivariate and multivariate Gaussians."""

import numpy as np

from modules import prob_utils

np.random.seed(7)

# Number of samples
N = 1000


def main():
    # 1D case
    priors = np.array([0.2, 0.5, 0.3])  # Likelihood of each distribution to be selected
    # Determine number of mixture components
    C = len(priors)
    mu = np.array([[1, -10, 3]])  # Gaussian distributions means
    sigma = np.array([[1, 2, 4]])  # Gaussian distributions variances
    # Create PDF parameter structure
    gmm_params = prob_utils.GaussianMixturePDFParameters(priors, C, mu, sigma)
    gmm_params.print_pdf_params()
    # Determine dimensionality from mixture PDF parameters
    n = mu.shape[0]
    # Generate 1D vector from a mixture of 3 Gaussians
    _, _ = prob_utils.generate_mixture_samples(N, n, gmm_params, True)

    # 2D case
    priors = np.array([0.2, 0.5, 0.3])  # Likelihood of each distribution to be selected
    # Determine number of mixture components
    C = len(priors)
    mu = np.array([[1, -10, 3],
                   [4, -2, 10]])  # Gaussian distributions means
    sigma = np.array([[[1, -1.5],
                       [-1.5, 3]],
                      [[2, 1.5],
                       [1.5, 8]],
                      [[4, 2.5],
                       [2.5, 4]]])  # Gaussian distributions covariance matrices
    # Create PDF parameter structure
    gmm_params = prob_utils.GaussianMixturePDFParameters(priors, C, mu, np.transpose(sigma))
    gmm_params.print_pdf_params()
    # Determine dimensionality from mixture PDF parameters
    n = mu.shape[0]
    # Generate 2D matrix from a mixture of 3 Gaussians
    _, _ = prob_utils.generate_mixture_samples(N, n, gmm_params, True)

    # 3D case
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
    _, _ = prob_utils.generate_mixture_samples(N, n, gmm_params, True)


if __name__ == '__main__':
    main()
