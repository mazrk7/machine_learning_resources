"""Utilities module for a variety of probability distributions."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig


class PDFParameters:
    """Structure for arbitrary PDF parameters."""

    def __init__(self, dist_type):
        """
        Args:
            dist_type: Type of PDF.
        """

        self.dist_type = dist_type

    def print_pdf_params(self):
        print("PDF has following parameters:")
        print(vars(self))


class UniformPDFParameters(PDFParameters):
    """Structure for Uniform PDF parameters."""

    def __init__(self, a, b):
        """
        Args:
            a: Lower endpoints of the n-dim axes, shape [n, 1].
            b: Higher endpoints of the n-dim axes, shape [n, 1].
        """

        super().__init__('Uniform')

        self.a = a
        self.b = b
        # Mean vector of shape [n, 1] for n-dimensionality
        self.mean = (a + b) / 2.
        # Scale array of shape [n, 1] that skews the volume to a parallelogram
        self.scale = (b - a) / 2.


class GaussianPDFParameters(PDFParameters):
    """Structure for Gaussian PDF parameters."""

    def __init__(self, mean, cov):
        """
        Args:
            mean: Mean vector, of shape [n, 1] for n-dimensionality.
            cov: Covariance matrix, of shape [n, n] for n-dimensionality. Note for n = 1,
                simply variance so shape [1, C].
        """

        super().__init__('Gaussian')

        self.mean = mean
        self.cov = cov


class UniformMixturePDFParameters(PDFParameters):
    """Structure for Uniform Mixture PDF parameters."""

    def __init__(self, priors, C, a, b):
        """
        Args:
            priors: Prior likelihoods of each distribution being selected.
            C: Number of mixture components.
            a: Each uniform component's n-dim lower endpoints n-dim, shape [n, C].
            b: Each uniform component's n-dim higher endpoints n-dim, shape [n, C].
        """

        super().__init__('UniformMixture')

        self.priors = priors
        self.C = C
        self.component_pdfs = np.ndarray(self.C, dtype=np.object)

        for c in range(C):
            self.component_pdfs[c] = UniformPDFParameters(a[:, [c]], b[:, [c]])


class GaussianMixturePDFParameters(PDFParameters):
    """Structure for Gaussian Mixture PDF parameters."""

    def __init__(self, priors, C, mean, cov):
        """
        Args:
            priors: Prior likelihoods of each distribution being selected.
            C: Number of mixture components.
            mean: Each Gaussian's mean vector, of shape [n, C] for n-dimensionality.
            cov: Each Gaussian's covariance matrix, of shape [n, n, C] for n-dimensionality. Note for n = 1,
                simply variance so shape [1, C].
        """

        super().__init__('GaussianMixture')

        self.priors = priors
        self.C = C
        self.component_pdfs = np.ndarray(self.C, dtype=np.object)

        for c in range(C):
            self.component_pdfs[c] = GaussianPDFParameters(mean[..., [c]], cov[..., c])


def generate_random_samples(N, n, pdf_params, visualize=False):
    """ Generates N vector-valued samples with dimensionality n
        according to the probability density function specified by pdf_params.

    Args:
        N: The number of samples to generate (scalar Python `int`).
        n: The input space dimension (scalar Python `int`).
        pdf_params: An object of type PDFParameters.
        visualize: Flag to visualize data (default False), if 0 < n <= 3

    Returns:
        x: Random samples drawn from PDF of shape [n, N].
    """

    if pdf_params.dist_type == 'Gaussian':
        if n > 1:
            l, u = eig(pdf_params.cov)
            scale = u * (l ** 0.5)
        else:
            scale = pdf_params.cov ** 0.5

        # z ~ N(0, I) are zero-mean identity-covariance Gaussian samples
        z = np.random.randn(n, N)
        # x ~ N(pdf.mean, pdf.cov)
        x = np.matmul(scale, z) + pdf_params.mean  # Matrix multiplication
    elif pdf_params.dist_type == 'Uniform':
        # z ~ Uniform[-1, 1] ^ n are zero-mean "unit-scale" uniformly distributed samples
        z = 2 * (np.random.rand(n, N) - 0.5)
        # x ~ Uniform(pdf.mean, pdf.scale)
        x = np.multiply(pdf_params.scale, z) + pdf_params.mean  # Element-wise multiplication
    else:
        print("PDF {} does not have a parameters type object!".format(pdf_params.dist_type))
        return

    if visualize and (0 < n <= 3):
        # Twice as wide figure as is tall
        fig = plt.figure(figsize=plt.figaspect(0.5))

        if n == 1:
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.scatter(z, np.zeros(N))
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.scatter(x, np.zeros(N))
        elif n == 2:
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.scatter(z[0, :], z[1, :])
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.scatter(x[0, :], x[1, :])
            ax1.set_ylabel("y-axis")
            ax2.set_ylabel("y-axis")
        else:
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax1.scatter(z[0, :], z[1, :], z[2, :])
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.scatter(x[0, :], x[1, :], x[2, :])
            ax1.set_ylabel("y-axis")
            ax2.set_ylabel("y-axis")
            ax1.set_zlabel("z-axis")
            ax2.set_zlabel("z-axis")

        ax1.set_title("z ~ Standard Shift and Scale")
        ax2.set_title("x ~ Specified Shift and Scale")
        ax1.set_xlabel("x-axis")
        ax2.set_xlabel("x-axis")

        plt.show()

    return x


def generate_mixture_samples(N, n, pdf_params, visualize=False):
    """ Generates N vector-valued samples with dimensionality n
        according to the mixture PDF specified by pdf_params.

    Args:
        N: The number of samples to generate (scalar Python `int`).
        n: The input space dimension (scalar Python `int`).
        pdf_params: An object of type PDFParameters.
        visualize: Flag to visualize data (default False), if 0 < n <= 3

    Returns:
        x: Random samples drawn from PDF of shape [n, N].
        y: Component labels associated with mixture PDF of shape [N].
    """

    # Output samples and labels
    x = np.zeros([n, N])
    y = np.zeros(N)

    # Decide randomly which samples will come from each component
    u = np.random.rand(N)
    thresholds = np.cumsum(pdf_params.priors)

    for c in range(pdf_params.C):
        c_ind = np.argwhere(u <= thresholds[c])[:, 0]  # Get randomly sampled indices for this component
        c_N = len(c_ind)  # No. of samples in this component
        y[c_ind] = c * np.ones(c_N)
        u[c_ind] = 1.1 * np.ones(c_N)  # Multiply by 1.1 to fail <= thresholds and thus not reuse samples
        x[:, c_ind] = generate_random_samples(c_N, n, pdf_params.component_pdfs[c], visualize=False)

    if visualize and (0 < n <= 3):
        fig = plt.figure()

        if n == 1:
            ax = fig.add_subplot(111)
            ax.scatter(x, np.zeros(N), c=y)
        elif n == 2:
            ax = fig.add_subplot(111)
            ax.scatter(x[0, :], x[1, :], c=y)
            ax.set_ylabel("y-axis")
        else:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x[0, :], x[1, :], x[2, :], c=y)
            ax.set_ylabel("y-axis")
            ax.set_zlabel("z-axis")

        ax.set_title("x ~ {}D data generated by a {}".format(n, pdf_params.dist_type))
        ax.set_xlabel("x-axis")

        plt.show()

    return x, y
