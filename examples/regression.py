import numpy as np
from gp_error.gaussianprocess import GPErrorVariance
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from gp_error.data import example_1d
import matplotlib.pyplot as plt


def example():
    seed = 123
    X, y, error_params = example_1d()


    # My GP
    gp_model = GPErrorVariance(x_covariance=error_params['x'], random_state=seed)
    gp_model.fit(X['train'], y['train'])
    mean, std = gp_model.predict(X['plot'], return_std=True)

    # Plot Figure

    fig, ax = plt.subplots(figsize=(10, 7))
    upper = y['plot'].ravel() + 1.9600 * std
    lower = y['plot'].ravel() - 1.9600 * std

    ax.plot(X['plot'], mean, linewidth=5, color='k', label='Predictions (GP)')
    ax.fill_between(X['plot'].ravel(), upper.ravel(), lower.ravel(),
                    color='red',
                    alpha=0.5, label='Standard Deviation')
    ax.scatter(X['train'], y['train'], s=100, color='r', label='Training Data')

    ax.legend(fontsize=14)
    ax.set_title('GP Regression with Error Variance')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.patch.set_visible(False)
    plt.show()

    # Their GP
    kernel = C() * RBF() + WhiteKernel()
    sk_gp_model = GaussianProcessRegressor(kernel=kernel, random_state=seed)
    sk_gp_model.fit(X['train'], y['train'])
    sk_mean, sk_std = sk_gp_model.predict(X['plot'], return_std=True)

    # Plot Figure

    fig, ax = plt.subplots(figsize=(10, 7))
    upper = y['plot'].ravel() + 1.9600 * sk_std
    lower = y['plot'].ravel() - 1.9600 * sk_std

    ax.plot(X['plot'], sk_mean, linewidth=5, color='k', label='Predictions (GP)')
    ax.fill_between(X['plot'].ravel(), upper.ravel(), lower.ravel(),
                    color='red',
                    alpha=0.5, label='Standard Deviation')
    ax.scatter(X['train'], y['train'], s=100, color='r', label='Training Data')

    ax.legend(fontsize=14)
    ax.set_title('Standard GP Regression')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.patch.set_visible(False)
    plt.show()

    return None

def main():

    example()

    pass

if __name__ == '__main__':
    main()