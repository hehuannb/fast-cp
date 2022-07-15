#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:53:17 2022

@author: huan
"""

import numpy as np
from sklearn import preprocessing as skp


def rand_l1_matrix(In, R):
    """
    Randomly initialize a column normalized matrix
    """
    return skp.normalize(np.random.rand(In, R),
                         axis=0, norm='l1')


def random_factors(modeDim, R):
    """
    Randomly initialize column normalized matrices
    """
    return [rand_l1_matrix(m, R) for m in modeDim]


def listdiff(list1, list2):
    """
    returns the list of elements that are in list 1 but not in list2
    """
    if isinstance(list1, np.ndarray):
        list1 = list1.tolist()
    if isinstance(list2, np.ndarray):
        list2 = list2.tolist()
    s = set(list2)
    return [x for x in list1 if x not in s]

def tt_dimscheck(dims, N, M=None, exceptdims=False):
    """
    Checks whether the specified dimensions are valid in a
    tensor of N-dimension.
    If M is given, then it will also retuns an index for M multiplicands.
    If exceptdims == True, then it will compute for the
    dimensions not specified.
    """
    if exceptdims:
        dims = listdiff(range(N), dims)
    
    # check vals in between 0 and N-1
    if any(x < 0 or x >= N for x in dims):
        raise ValueError("invalid dimensions specified")
    if M is not None and M > N:
        raise ValueError("Cannot have more multiplicands than dimensions")
    if M is not None and M != N and M != len(dims):
        raise ValueError("invalid number of multiplicands")
    # number of dimensions in dims
    p = len(dims)
    sdims = []
    sdims.extend(dims)
    sdims.sort()

    # indices of the elements in the sorted array
    sidx = []
    # table that denotes whether the index is used
    table = np.ndarray([len(sdims)])
    table.fill(0)

    for i in range(0, len(sdims)):
        for j in range(0, len(dims)):
            if(sdims[i] == dims[j] and table[j] == 0):
                sidx.extend([j])
                table[j] = 1
                break
    if M is None:
        return sdims
    if(M == p):
        vidx = sidx
    else:
        vidx = sdims
    return (sdims, vidx)


'''
Compute Khatri-Rao product of matrices

The implementation is borrowed from tensorly
'''


def kr(matrices):
    """Khatri-Rao product of a list of matrices

        This can be seen as a column-wise kronecker product.

    Parameters
    ----------
    matrices : ndarray list
        list of matrices with the same number of columns, i.e.::

            for i in len(matrices):
                matrices[i].shape = (n_i, m)

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the product.

    Notes
    -----
    Mathematically:

    .. math::
         \\text{If every matrix } U_k \\text{ is of size } (I_k \\times R),\\\\
         \\text{Then } \\left(U_1 \\bigodot \\cdots \\bigodot U_n \\right) \\text{ is of size } (\\prod_{k=1}^n I_k \\times R)

    A more intuitive but slower implementation is::

        kr_product = np.zeros((n_rows, n_columns))
        for i in range(n_columns):
            cum_prod = matrices[0][:, i]  # Acuumulates the khatri-rao product of the i-th columns
            for matrix in matrices[1:]:
                cum_prod = np.einsum('i,j->ij', cum_prod, matrix[:, i]).ravel()
            # the i-th column corresponds to the kronecker product of all the i-th columns of all matrices:
            kr_product[:, i] = cum_prod

        return kr_product
    """
    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i+common_dim for i in target)
    operation = source+'->'+target+common_dim
    return np.einsum(operation, *matrices).reshape((-1, n_columns))



def khatri_rao(matrices, skip_matrix=None, reverse=False):
    """Khatri-Rao product of a list of matrices

        This can be seen as a column-wise kronecker product.
        (see [1]_ for more details).

        If one matrix only is given, that matrix is directly returned.

    Parameters
    ----------
    matrices : ndarray list
        list of matrices with the same number of columns, i.e.::

            for i in len(matrices):
                matrices[i].shape = (n_i, m)

    skip_matrix : None or int, optional, default is None
        if not None, index of a matrix to skip

    reverse : bool, optional
        if True, the order of the matrices is reversed

    Returns
    -------
    khatri_rao_product: matrix of shape ``(prod(n_i), m)``
        where ``prod(n_i) = prod([m.shape[0] for m in matrices])``
        i.e. the product of the number of rows of all the matrices in the product.

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications",
       SIAM REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    """
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

    # Khatri-rao of only one matrix: just return that matrix
    if len(matrices) == 1:
        return matrices[0]

    if reverse:
        matrices = matrices[::-1]

    return kr(matrices)

