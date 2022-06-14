"""
python implementation of Hilbert Schmidt Independence Criterion
hsic_gam implements the HSIC test using a Gamma approximation
Python 3.10

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B.,
& Smola, A. J. (2007). A kernel statistical test of independence.
In Advances in neural information processing systems (pp. 585-592).

Shoubo (shoubo.sub AT gmail.com)
09/11/2016

Inputs:
X 		n by dim_x matrix
Y 		n by dim_y matrix
alph 		level of test

Outputs:
testStat	test statistics
thresh		test threshold for level alpha test
"""

import numpy as np
from scipy.stats import gamma


def rbf_kernel(vec: np.ndarray):
    if len(vec.shape) == 1:
        vec = np.expand_dims(vec, 1)

    size = vec.shape[0]

    # Calculate squares of each element
    G = np.multiply(vec, vec)

    # Repeat them to get an n x n matrix so that
    # in each row, repeat that row's value n times
    Q = np.tile(G, (1, size))

    # Calculate the squared distance between elements
    H = (Q + Q.T) - 2 * np.dot(vec, vec.T)

	# Take the median of nonzero absolute distances as
	# the kernel width.
    dists = np.triu(H)
    dists = dists.reshape(size ** 2, 1)
    kernel_width = np.sqrt(np.median(dists[dists > 0]) / 2)

    rbf_gamma = 1 / (2 * (kernel_width ** 2))
    H = np.exp(-H * rbf_gamma)
    return H


def hsic_gam(X, Y, alph = 0.5):
	"""
	X, Y are numpy vectors with row - sample, col - dim
	alph is the significance level
	auto choose median to be the kernel width
	"""
	n = X.shape[0]

	K = rbf_kernel(X)
	L = rbf_kernel(Y)

	H = np.identity(n) - np.ones((n,n), dtype = float) / n

	Kc = np.dot(np.dot(H, K), H)
	Lc = np.dot(np.dot(H, L), H)

	testStat = np.sum(Kc.T * Lc) / n

	varHSIC = (Kc * Lc / 6)**2
	varHSIC = ( np.sum(varHSIC) - np.trace(varHSIC) ) / n / (n-1)
	varHSIC = varHSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

	K = K - np.diag(np.diag(K))
	L = L - np.diag(np.diag(L))

	bone = np.ones((n, 1), dtype = float)
	muX = np.dot(np.dot(bone.T, K), bone) / n / (n-1)
	muY = np.dot(np.dot(bone.T, L), bone) / n / (n-1)

	mHSIC = (1 + muX * muY - muX - muY) / n

	al = mHSIC**2 / varHSIC
	bet = varHSIC*n / mHSIC

	thresh = gamma.ppf(1-alph, al, scale=bet)[0][0]

	return testStat, thresh