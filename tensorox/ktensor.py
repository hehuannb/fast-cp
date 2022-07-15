#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:57:18 2022

@author: huan
"""

import numpy as np
from .utils import tt_dimscheck, khatri_rao



class Ktensor(object):
    '''
    A tensor stored as a decomposed Kruskal operator
    The code is the python implementation of the @ktensor folder in the
    MATLAB Tensor Toolbox

    Parameters
    ----------
    U: list of numpy.ndarrays
       Factor matrices from the tensor representation is created.
       All factor matrices ``U[i]`` must have the same number of columns,
       but different number of rows.
    lmbda: array_like of floats
       Weights for each dimension of the Kruskal operator.
    '''
    shape = None
    lmbda = None
    U = None
    R = 0

    def __init__(self, U, lmbda=None):
        """
        Tensor stored in decomposed form as a Kruskal operator.
        X = sum_r [lambda_r outer(a_r, b_r, c_r)].
        The columns of matrices A,B,C are the associated a_r, b_r, c_r
        """
        self.U = U
        self.shape = tuple([Ui.shape[0] for Ui in U])
        self.R = U[0].shape[1]
        self.lmbda = lmbda
        if lmbda is None:
            self.lmbda = np.ones(self.R)
        if not all([Ui.shape[1] == self.R for Ui in U]):
            raise ValueError('Dimension mismatch of factor matrices')
        if len(self.lmbda) != self.R:
            raise ValueError('Lambda mismatch with rank')

    def __str__(self):
        ret = "Kruskal tensor with size {0}\n".format(self.shape)
        ret += "Lambda: {0}\n".format(self.lmbda)
        for i in range(len(self.U)):
            ret += "U[{0}] = {1}\n".format(i, self.U[i])
        return ret

    def __get_type__(self):
        return "ktensor"


    def to_dt_arr(self):
        """
        Returns an ndarray that can be used directly to initialize
        a dense tensor
        """
        dtArr = np.dot(self.lmbda, khatri_rao(self.U).T)
        # reshape itself into something useful
        return dtArr.reshape(self.shape)

    def size(self):
        return self.shape

    def innerprod(self, Y):
        """
        Compute the inner product between this tensor and Y.
        If Y is a ktensor, the inner product is computed using inner products
        of the factor matrices.
        Otherwise, the inner product is computed using ttv with all of the
        columns of X's factor matrices
        """
        res = 0
        if isinstance(Y, Ktensor):
            M = np.outer(self.lmbda, Y.lmbda)
            for n in range(self.ndims()):
                M = np.multiply(M, np.inner(self.U[n].T, Y.U[n].T))
            res = np.sum(M)
        else:
            vecs = [{} for i in range(self.ndims())]
            for r in range(self.R):
                for n in range(self.ndims()):
                    vecs[n] = self.U[n][:, r]
                res = res + self.lmbda[r] * Y.ttv(vecs, range(self.ndims()))
        return res

    def ndims(self):
        return len(self.U)

    def norm(self):
        """
        Compute the Frobenius norm of the ktensor

        Returns
        -------
        norm : float
            Frobenius norm of the ktensor
        """
        coefMatrix = np.outer(self.lmbda, self.lmbda)
        for i in range(self.ndims()):
            coefMatrix = np.multiply(coefMatrix,
                                     np.dot(self.U[i].T, self.U[i]))
        return np.sqrt(np.sum(coefMatrix))

    def normalize(self, normtype=2):
        """"
        Normalize the column of each factor matrix U where the excess weight
        is absorbed by lambda. Also ensure lamda is positive.
        """
        for n in range(self.ndims()):
            self.normalize_mode(n, normtype)
        idx = np.count_nonzero(self.lmbda < 0)
        if idx > 0:
            for i in np.nonzero(self.lmbda < 0):
                self.U[0][:, i] = -1 * self.U[0][:, i]
                self.lmbda[i] = -1 * self.lmbda[i]

    def normalize_equal(self, normtype=2):
        self.normalize(normtype)  # Absorb the weights into lambda
        D = np.diag(np.power(self.lmbda, 1. / float(self.ndims())))

        # Now distribute evenly
        for u in range(len(self.U)):
            self.U[u] = np.dot(self.U[u], D)

        # Reset the lambda values to be all ones
        for i in range(len(self.lmbda)):
            self.lmbda[i] = 1

    def normalize_mode(self, mode, normtype):
        """Normalize the ith factor using the norm specified by normtype"""
        colNorm = np.apply_along_axis(np.linalg.norm, 0, self.U[mode],
                                      normtype)
        zeroNorm = np.where(colNorm == 0)[0]
        colNorm[zeroNorm] = 1
        self.lmbda = self.lmbda * colNorm
        self.U[mode] = self.U[mode] / colNorm[np.newaxis, :]


    def normalize_absorb(self, mode, normtype):
        """
        Normalize all the matrices using the norm specified by normtype and
        then absorb all the lambda magnitudes into the factors.
        """
        self.normalize(normtype)
        self.redistribute(mode)

    def fix_signs(self):
        """
        For each vector in each factor, the largest magnitude entries of K
        are positive provided that the sign on pairs of vectors in a rank-1
        component can be flipped.
        """
        for r in range(self.R):
            sgn = np.zeros(self.ndims())
            for n in range(self.ndims()):
                idx = np.argmax(np.abs(self.U[n][:, r]))
                sgn[n] = np.sign(self.U[n][idx, r])
            negidx = np.nonzero(sgn == -1)[0]
            nflip = int(2 * np.floor(len(negidx) / 2))
            for i in np.arange(nflip):
                n = negidx[i]
                self.U[n][:, r] = -self.U[n][:, r]
        return

    def permute(self, order):
        """
        Rearranges the dimensions of the ktensor so the order is
        specified by the vector order.
        """
        return Ktensor(self.U[order], self.lmbda)

    def redistribute(self, mode):
        """
        Distribute the lambda values to a specified mode.
        Lambda vector is set to all ones, and the mode n takes on the values
        """
        self.U[mode] = self.U[mode] * self.lmbda[np.newaxis, :]
        self.lmbda = np.ones(self.R)

    def sort_components(self):
        """
        Sort the ktensor components by magnitude, greatest to least.
        """
        sortidx = np.argsort(self.lmbda)[::-1]
        self.lmbda = self.lmbda[sortidx]
        # resort the U
        for i in range(self.ndims()):
            self.U[i] = self.U[i][:, sortidx]

    def ttv(self, v, dims):
        """
        Computes the product of the Kruskal tensor with the column vector along
        specified dimensions.

        Parameters
        ----------
        v - column vector
        dims - dimensions to multiply the product

        Returns
        -------
        out :
        """
        if not isinstance(v, list):
            v = [v]
        (dims, vidx) = tt_dimscheck(dims, self.ndims(), len(v))
        remdims = np.setdiff1d(range(self.ndims()), dims)
        # Collapse dimensions that are being multiplied out
        newlmbda = self.lmbda.reshape(self.R, 1)
        for i in range(len(dims)):
            newlmbda = newlmbda * np.dot(self.U[dims[i]].T, v[vidx[i]])
        if len(remdims) == 0:
            return np.sum(newlmbda)
        return Ktensor([self.U[j] for j in remdims], newlmbda.flatten())

    def copy(self):
        """
        Create a deep copy of the object
        """
        return Ktensor([self.U[n].copy() for n in range(len(self.U))], 
                       np.copy(self.lmbda))

    def _check_object(self, other):
        '''
        Check the other object to see if it is a ktensor or same shape
        '''
        if not isinstance(other, Ktensor):
            raise NotImplementedError("Can only handle same ktensors object")
        if self.shape != other.shape:
            raise ValueError("Shapes of the tensors do not match")

    # Mathematic and Logic functions
    def __add__(self, other):
        self._check_object(other)
        lambda_ = self.lmbda + other.lmbda
        U = [np.concatenate((self.U[m], other.U[m]),
                            axis=1) for m in range(self.ndims())]
        return Ktensor(U, lambda_)

    def __sub__(self, other):
        self._check_object(other)
        lambda_ = np.append(self.lmbda, -other.lmbda)
        U = [np.concatenate((self.U[m], other.U[m]),
                            axis=1) for m in range(self.ndims())]
        return Ktensor(U, lambda_)

    def __eq__(self, other):
        if other is None:
            return False
        self._check_object(other)
        if self.lmbda != other.lmbda:
            return False
        # if lambda is true, then continue onto the other components
        for m in range(self.ndims()):
            if np.min(np.equal(self.U[m], other.U[m])):
                return False
        return True
