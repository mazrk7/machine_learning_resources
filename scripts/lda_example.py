"""Example of applying LDA to data generated using Gaussian and uniform distributions with different dimensions.
   PDF parameters selected first and then data is generated from the multivariate PDFs. Projection vectors are later
   generated via LDA over the data, before plotting projection results.
"""

import numpy as np

from modules import models, prob_utils

np.random.seed(7)

# No. of samples in each category
N0 = 1000
N1 = 1300


def main():
    # ========== Gaussian distribution case ==========
    # Generate two datasets from two different Gaussian distributions/categories

    # 1st data category
    mu = np.array([[5], [1]])
    sigma = np.array([[1, -1.5],
                      [-1.5, 3]])
    # Create PDF parameter structure
    gauss_params = prob_utils.GaussianPDFParameters(mu, sigma)
    gauss_params.print_pdf_params()
    # Determine dimensionality from PDF parameters
    n = len(mu)
    # Generate N points of random samples from Gaussian PDF
    x0 = prob_utils.generate_random_samples(N0, n, gauss_params, False)

    # 2nd data category
    mu = np.array([[-1], [-3]])
    sigma = np.array([[5, -.5],
                      [-.5, 10]])
    # Create PDF parameter structure
    gauss_params = prob_utils.GaussianPDFParameters(mu, sigma)
    gauss_params.print_pdf_params()
    # Determine dimensionality from PDF parameters
    n = len(mu)
    # Generate N points of random samples from Gaussian PDF
    x1 = prob_utils.generate_random_samples(N1, n, gauss_params, False)

    # Transpose as that's how I expect the input shapes in my LDA function
    x = np.concatenate((x0.T, x1.T))

    labels = np.concatenate((np.zeros(N0), np.ones(N1)))

    _, _ = models.perform_lda(x, labels)

    # ========== Uniform distribution case ==========
    # Generate two datasets from two different uniform distributions categories

    # 1st data category
    a = np.array([[-5], [1]])  # Lower endpoints of the x and y
    b = np.array([[-1], [5]])  # Higher endpoints of the x and y
    # Create PDF parameter structure
    uniform_params = prob_utils.UniformPDFParameters(a, b)
    uniform_params.print_pdf_params()
    # Determine dimensionality from PDF parameters
    n = len(a)
    # Generate N points of random samples from Uniform PDF
    x0 = prob_utils.generate_random_samples(N0, n, uniform_params, False)

    # 2nd data category
    a = np.array([[1], [5]])  # Lower endpoints of the x and y
    b = np.array([[5], [10]])  # Higher endpoints of the x and y
    # Create PDF parameter structure
    uniform_params = prob_utils.UniformPDFParameters(a, b)
    uniform_params.print_pdf_params()
    # Determine dimensionality from PDF parameters
    n = len(a)
    # Generate N points of random samples from Uniform PDF
    x1 = prob_utils.generate_random_samples(N1, n, uniform_params, False)

    # Transpose as that's how I feed it into LDA
    x = np.concatenate((x0.T, x1.T))
    # Using same labels from before
    _, _ = models.perform_lda(x, labels)


if __name__ == '__main__':
    main()
