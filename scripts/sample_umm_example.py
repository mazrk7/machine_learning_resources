"""Example of sampling from a mixture of univariate, bivariate and multivariate Uniforms."""

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
    a = np.array([[1, -10, 10]])  # Uniform distributions lower endpoints
    b = np.array([[5, -5, 20]])  # Uniform distributions higher endpoints
    # Create PDF parameter structure
    umm_params = prob_utils.UniformMixturePDFParameters(priors, C, a, b)
    umm_params.print_pdf_params()
    # Determine dimensionality from mixture PDF parameters
    n = a.shape[0]
    # Generate 1D vector from a mixture of 3 Uniforms
    _, _ = prob_utils.generate_mixture_samples(N, n, umm_params, True)

    # 2D case
    priors = np.array([0.2, 0.5, 0.3])  # Likelihood of each distribution to be selected
    # Determine number of mixture components
    C = len(priors)
    a = np.array([[1, -10, 10],
                  [5, 13, -5]])  # Uniform distributions lower endpoints (x & y axes)
    b = np.array([[5, -5, 20],
                  [8, 20, 0]])  # Uniform distributions higher endpoints (x & y axes)
    # Create PDF parameter structure
    umm_params = prob_utils.UniformMixturePDFParameters(priors, C, a, b)
    umm_params.print_pdf_params()
    # Determine dimensionality from mixture PDF parameters
    n = a.shape[0]
    # Generate 2D matrix from a mixture of 3 Uniforms
    _, _ = prob_utils.generate_mixture_samples(N, n, umm_params, True)

    # 3D case
    priors = np.array([0.2, 0.5, 0.3])  # Likelihood of each distribution to be selected
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
    _, _ = prob_utils.generate_mixture_samples(N, n, umm_params, True)


if __name__ == '__main__':
    main()
