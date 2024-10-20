"""
CAD module main code.
"""

import numpy as np

def sigmoid(a):
    # Computes the sigmoid function
    # Input:
    # a - value for which the sigmoid function should be computed
    # Output:
    # s - output of the sigmoid function
    a = np.clip(a, -500, 500)
    s = 1 / (1 + np.exp(-a))
    return s


def lr_nll(X, Y, Theta):
    # Computes the negative log-likelihood (NLL) loss for the logistic
    # regression classifier.
    # Input:
    # X - the data matrix
    # Y - targets vector
    # Theta - parameters of the logistic regression model
    # Ouput:
    # L - the negative log-likelihood loss

    # compute the predicted probability by the logistic regression model
    p = sigmoid(X.dot(Theta))
    
    #Avoid getting numbers equal to 0 or 1
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1 - epsilon)
    
    # sum method
    L = -np.sum(Y*np.log(p) + (1-Y)*np.log(1-p))

    return L

def lr_agrad(X, Y, Theta):
    # Gradient of the negative log-likelihood for a logistic regression
    # classifier.
    # Input:
    # X - the data matrix
    # Y - targets vector
    # Theta - parameters of the logistic regression model
    # Example inputs:
    # X - training_x_ones.shape=(100, 1729)
    # Y - training_y[idx].shape=(100, 1)
    # Theta - Theta.shape=(1729, 1)
    # Ouput:
    # g - gradient of the negative log-likelihood loss
    #
    a = X.dot(Theta)
    p = sigmoid(a)
    g = np.sum((p - Y)*X, axis=0).reshape(1,-1)

    return g

def mypca(X):
    # Rotates the data X such that the dimensions of rotated data Xpca
    # are uncorrelated and sorted by variance.
    # Input:
    # X - Nxk feature matrix
    # Output:
    # X_pca - Nxk rotated feature matrix
    # U - kxk matrix of eigenvectors
    # Lambda - kx1 vector of eigenvalues
    # fraction_variance - kx1 vector which stores how much variance
    #                     is retained in the k components

    sigma_hat = np.cov(X, rowvar=False) # Calculate covariance matrix
    
    # eigenvalues
    w, v = np.linalg.eig(sigma_hat)
    ix = np.argsort(w)[::-1]    # Find ordering of eigenvalues
    w = w[ix]                   # Reorder eigenvalues
    v = v[:, ix]                # Reorder eigenvectors

    X_rotated = np.zeros_like(X)
    X_rotated = np.dot(X,v)
    
    #Return fraction of variance
    fraction_variance = np.zeros((X_rotated.shape[1],1))
    for i in np.arange(X_rotated.shape[1]):
        fraction_variance[i] = np.sum(w[:i+1])/np.sum(w)
    print("\nX_ROT:\n",X_rotated)
    return X_rotated, v, w, fraction_variance