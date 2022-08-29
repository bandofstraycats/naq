import numpy as np
from numpy import linalg as LA

def get_kernel(kernel_type, nS):

    # dot product in finite state space
    dot_product = np.eye(nS)

    # normalized kernel
    if kernel_type == 'ntk':
        def kappa_0(t):
            return (1./np.pi)*(np.pi - np.arccos(t))
        def kappa_1(t):
            return (1./np.pi)*(t*(np.pi - np.arccos(t)) + np.sqrt(1 - t**2))
        def h_ntk(t):
            return t*kappa_0(t) + kappa_1(t)
        def norm_h_ntk(t):
            return h_ntk(t) / h_ntk(1)
        kernel = norm_h_ntk(dot_product)
    elif kernel_type == 'exp':
        kernel = np.exp(dot_product) / np.exp(1)
    else: # linear kernel
        kernel = dot_product

    kernel /= nS
    np.testing.assert_array_almost_equal(kernel, np.transpose(kernel), decimal=2) # kernel is symmetric
    np.testing.assert_almost_equal(np.trace(kernel), 1, decimal=2) # unit trace

    w, v = LA.eig(kernel)
    print("Kernel eigenvalues")
    print(sorted(w, reverse=True))
    assert np.all(np.logical_or(w > 0, np.isclose(w, 0)))
    if np.any(np.isclose(w, 0)):
        print("Warning: Convergence might not be monotonic")
    kernel_norm = LA.norm(kernel, np.inf)
    print("Kernel norm")
    print(kernel_norm)
    assert kernel_norm <= 1

    return kernel
