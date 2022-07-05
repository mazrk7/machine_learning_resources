"""Example of sampling from a univariate, bivariate and multivariate Gaussian PDF."""

import numpy as np

from modules import prob_utils

np.random.seed(7)

# Number of samples
N = 100


def main():
    # 1D case
    mu = np.array([5])
    sigma = np.array([4])
    # Create PDF parameter structure
    gauss_params = prob_utils.GaussianPDFParameters(mu, sigma)
    gauss_params.print_pdf_params()
    # Determine dimensionality from PDF parameters
    n = len(mu)
    # Generate N points of random samples from Gaussian PDF
    _ = prob_utils.generate_random_samples(N, n, gauss_params, True)

    # 2D case
    mu = np.array([[5], [1]])
    sigma = np.array([[1, -1.5],
                      [-1.5, 3]])
    gauss_params = prob_utils.GaussianPDFParameters(mu, sigma)
    gauss_params.print_pdf_params()
    # Determine dimensionality from PDF parameters
    n = len(mu)
    # Generate N points of random samples from Gaussian PDF
    _ = prob_utils.generate_random_samples(N, n, gauss_params, True)

    # 3D case
    mu = np.array([[1], [-10], [3]])  # Selected mean of x, y and z axis distributions
    sigma = np.array([[2, -.5, -.4],  # Selected covariance matrix for data
                      [-.5, 6, -.2],
                      [-.4, -.2, 10]])
    gauss_params = prob_utils.GaussianPDFParameters(mu, sigma)
    gauss_params.print_pdf_params()
    # Determine dimensionality from PDF parameters
    n = len(mu)
    # Generate N points of random samples from Gaussian PDF
    _ = prob_utils.generate_random_samples(N, n, gauss_params, True)


if __name__ == '__main__':
    main()
