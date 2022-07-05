"""Example of sampling from a univariate, bivariate and multivariate Uniform PDF."""

import numpy as np

from modules import prob_utils

np.random.seed(7)

# Number of samples
N = 100


def main():
    # 1D case
    a = np.array([1])  # Lower endpoint of the x axis
    b = np.array([5])  # Higher endpoint of the x axis
    # Create PDF parameter structure
    uniform_params = prob_utils.UniformPDFParameters(a, b)
    uniform_params.print_pdf_params()
    # Determine dimensionality from PDF parameters
    n = len(a)
    # Generate N points of random samples from Uniform PDF
    _ = prob_utils.generate_random_samples(N, n, uniform_params, True)

    # 2D case
    a = np.array([[1], [5]])  # Lower endpoints of the x and y axis
    b = np.array([[5], [10]])  # Higher endpoints of the x and y axis
    # Create PDF parameter structure
    uniform_params = prob_utils.UniformPDFParameters(a, b)
    uniform_params.print_pdf_params()
    # Determine dimensionality from PDF parameters
    n = len(a)
    # Generate N points of random samples from Uniform PDF
    _ = prob_utils.generate_random_samples(N, n, uniform_params, True)

    # 3D case
    a = np.array([[1], [5], [-10]])  # Lower endpoints of the x, y and z axis
    b = np.array([[5], [10], [-3]])  # Higher endpoints of the x, y and z axis
    # Create PDF parameter structure
    uniform_params = prob_utils.UniformPDFParameters(a, b)
    uniform_params.print_pdf_params()
    # Determine dimensionality from PDF parameters
    n = len(a)
    # Generate N points of random samples from Uniform PDF
    _ = prob_utils.generate_random_samples(N, n, uniform_params, True)


if __name__ == '__main__':
    main()
