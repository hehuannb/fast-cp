'''
Fast-CP
'''
import torch
import numpy as np
import random
import time
import numpy_groupies as npg
import collections
import pandas as pd
# import tqdm

from tensorox.decomposition import BaseCP
from tensorox.ktensor import Ktensor
from collections import defaultdict
from sklearn import preprocessing as skp
from string import ascii_letters as einchars
from operator import itemgetter


AUG_MIN = 1e-50
epsilon = 1e-8
beta_1 = 0.9
beta_2 = 0.99
gamma = .5


def reconstruct(factors):
    """Reconstruct the tensor from its CPD factors

    Parameters
    ----------
    factors: numpy.ndarray
        An object array contains the factor tensors

    Returnsp, q)
    if p == q:
    -------
    numpy.ndarray
        The reconstructed trensor
    """
    ndim = len(factors)
    input_subscripts = [einchars[i] + einchars[ndim] for i in range(ndim)]
    output_subscript = ''.join(einchars[i] for i in range(ndim))
    subscripts = ','.join(input_subscripts) + '->' + output_subscript
    return torch.einsum(subscripts, *factors)

def ls_feval(X, M):
    m = reconstruct(M.U)
    res= X.data-m
    return torch.sum(np.multiply(res,res))


def take_step(phi_k, psi_k, grad_k, epoch_step, beta_1, beta_2):
    phi_k = beta_1 * phi_k + (1 - beta_1)*grad_k
    phi_hat_k = phi_k / (1 - beta_1**(epoch_step + 1))

    psi_k = beta_2 * psi_k + (1 - beta_2) * np.square(grad_k)
    psi_hat_k = psi_k / (1 - beta_2**(epoch_step+1))

    return phi_k, phi_hat_k, psi_k, psi_hat_k

def grad_weight(subs, shape, batch_size):
    num_nonzeros = batch_size
    num_zeros = batch_size
    total = 1
    for i in shape:
        total *= i
    w1 = (1.0 * len(subs)) / num_nonzeros
    w2 = (1.0 * total - len(subs)) / num_zeros
    return w1, w2


def project_nonnegative_A(U, threshold=0):
    U[U < threshold] = 0
    return U

class spAGCP_torch(BaseCP):
    def __init__(self, r, leny):
        self.rank = r
        self.num_mode = 3
        self.lenofsub = leny
        

    def obj_eval(self, nonzero, samples, batch, X):
        cache_f = [self.model.U[i][samples[:, i], :] for i in range(self.num_mode)]
        kp = torch.einsum('ij,ij,ij ->ij', *cache_f)
        z = torch.sum(kp,1)
        
        ll = torch.sum(z[5*batch:] * z[5*batch:]) + torch.sum((self.vals[nonzero]- z[0:5*batch])*(self.vals[nonzero]- z[0:5*batch]))
        # ll =  torch.sum((self.vals[nonzero]- z)*(self.vals[nonzero]- z))
        return ll


    def cal_grad(self, X, batch, data_type='gauss', samplemethod='uniform'):
        if samplemethod=='uniform':
            indices =[]
            for i in range(batch):
                indices.append([random.randint(0, self.shape[m]-1) for m in range(self.num_mode)])
            indices = np.vstack(indices)
            xval = np.array([X[tuple(indices[n])] for n in range(batch)])
            # xval = np.array([X[indices[n][0], indices[n][1], indices[n][2]] for n in range(batch)])
        else:
            nonzero = np.random.randint(0, self.lenofsub, size=batch)
            nonzero_index =self.subs[nonzero]
            zero_sample = []
            for i in range(batch):
                zero_sample.append([random.randint(0, self.shape[m]-1) for m in range(self.num_mode)])
            zero_sample = np.vstack(zero_sample)
            
            # while len(zero_sample) < batch:
            #     sample = []
            #     for nn in self.shape:
            #         q = np.random.randint(0, nn)
            #         sample.append(q)
            #     if (sample== self.subs).all(axis = 1).any():
            #         zero_sample.append(sample)
            indices = np.vstack((nonzero_index, zero_sample))
            xval =self.vals[nonzero]
        cache_f = [self.model.U[i][indices[:, i], :] for i in range(self.num_mode)]
        kp = torch.einsum('ij,ij,ij ->ij', *cache_f)
        z = torch.sum(kp, 1)

        if data_type=='gauss':
            t = self.w1 *(z[0:batch]-xval) 
        elif data_type=='apr':
            t =self.w1 * (1 - np.divide(xval, z[0:batch] + 1e-10))
        else:
            t = self.w1 * (1 / (z[0:batch]) -xval/ (z[0:batch]))
        t = torch.reshape(t, (batch,1))
        if samplemethod=='stratified':
            if data_type=='gauss':
                ones = self.w2 * z[batch:]
            elif data_type=='apr':
                ones = self.w2 * np.ones((batch, 1))
            else:
                ones = self.w2 /( z[batch:])
            t = np.append(t, ones)
            t.shape = (2 * batch, 1)
        # print(t.shape)
        
        weight =[np.einsum('ij,ij,ij ->ij', t, *itemgetter(*self.allmodesset.difference({i}))(cache_f)) 
                 for i in range(self.num_mode)]
        grad = []
        for m in range(self.num_mode):
            x = indices[:,m]
            w = weight[m]
            grad_mat = np.zeros((self.shape[m], self.rank))
            labels2, levels2 = pd.factorize(x)
            x2 = labels2
            gg = npg.aggregate(x2, w, axis=0)
            grad_mat[levels2, :] = gg
            grad.append(grad_mat)
        return grad

    def epsalg(self, v, nn, k, X):
        """Epsilon Algorithm"""
        Y = []
        Y.append(v)
        z = np.zeros(v.shape)
        jj = 0
        while jj < 2 * k and jj < nn - 1:
            y = Y[jj] - X[jj]
            s = np.dot(y.T, y)
            z += np.divide(y, s)
            jj += 1
            Y.append(z)
            z = X[jj - 1]
        X = [Y[n] for n in range(len(Y))]  # this is equal to X = Y
        y = X[-1]
        if nn <= 2 * k and jj % 2 == 1:
            y = np.divide(y, np.dot(y.T, y))
        return y, X

    def fit(self, X, subs, vals, **kwargs):
        self.model = kwargs.pop('init', None)
        self.subs = subs
        self.vals = vals
        batch = kwargs.pop('batch',1000)
        maxepoch = kwargs.pop('epoch', 100)
        inner = kwargs.pop('inner',1000)
        adam = kwargs.pop('adam',True)
        loss_type = kwargs.pop('loss','gauss')
        alpha = kwargs.pop('alpha', 1e-3)
        extrapolation = kwargs.pop('acceleration', True)
        sampling = kwargs.pop('sample', 'uniform')
        print("batch_size:", batch, 'maxepoch:', maxepoch, 'inner-iter:', inner, 'adam:',adam, 'lr:', alpha,'acceleration:', extrapolation)
        seed = 0
        factors = np.empty(self.num_mode, dtype=object)
        rstate = np.random.RandomState(seed)
        if loss_type == 'binary':
            for i in range(self.num_mode):
                factors[i] = torch.tensor(rstate.rand(X.shape[i], self.rank))
        else:
            for i in range(self.num_mode):
                factors[i] = torch.tensor(skp.normalize(np.random.rand(X.shape[i], self.rank),axis=0, norm='l1'))
        self.model = Ktensor(factors)
        
        self.num_mode = len(X.size())
        self.all_modes = list(range(self.num_mode))
        self.allmodesset = set(list(range(self.num_mode)))
        self.shape= X.shape
        self.epstable = defaultdict(list)
        self.oldg = defaultdict(list)
        total = 1
        for i in self.shape:
            total *= i
        self.w11 = (1.0 * total) / (5 * batch)
        self.w22 = (1.0 * total -self.lenofsub) / (5 * batch)
        # self.w33 = (1.0 * total)/batch
        if sampling=='uniform':
            self.w1 = total/batch
            
        else:
            self.w1, self.w2 = grad_weight(self.subs,X.size(),batch)
        phi = [np.zeros((X.size()[mode], self.rank)) for mode in self.all_modes]
        psi = [np.zeros((X.size()[mode], self.rank)) for mode in self.all_modes]
        iter_info = collections.OrderedDict(sorted({}.items(),
                                            key=lambda t: t[1]))
        
        nonzero = np.random.randint(0, self.lenofsub, size=5 * batch)
        nonzero_index = self.subs[nonzero]
        zero_sample =[]
        while len(zero_sample) < 5*batch:
            print(len(zero_sample))
            sample = []
            for nn in self.shape:
                q = np.random.randint(0, nn)
                sample.append(q)
            # print((sample== self.subs))
            # if (sample != self.subs).all(axis = 1).any():
            zero_sample.append(sample)
        zero_sample = np.vstack(zero_sample)
        fixed_indices = np.vstack((nonzero_index, zero_sample))
        # fixed_indices = np.vstack(nonzero_index)
        loss = self.obj_eval(nonzero,fixed_indices, batch, X)
        # loss = ls_feval(xd, self.model)
        print(loss)
        loss_true = 1
        loss_prev = loss
        iter_info[0] = {'time': 0, 'loss': loss, 'losst':loss_true}
        epsi = 0
        start = time.time()
        ii = 0
        for e in (range(maxepoch)):
            for i in(range(inner)):
                grad = self.cal_grad(X, batch, data_type=loss_type, samplemethod=sampling)
                if adam:
                    gg = []
                    for mode in self.all_modes:
                        phi_k, phi_hat, psi_k, psi_hat = take_step(phi[mode], psi[mode], grad[mode], ii, beta_1, beta_2)
                        phi[mode] = phi_k
                        psi[mode] = psi_k
                        gl = alpha * np.divide(phi_hat, (np.sqrt(psi_hat) + epsilon))
                        gg.append(gl)
                    self.model.U = [self.model.U[i] - gg[i] for i in range(self.num_mode)]
                else:
                    self.model.U = [self.model.U[i] - alpha * grad[i] for i in range(self.num_mode)]
                if loss_type=='apr' or loss_type=='binary':
                    self.model.U = [project_nonnegative_A(self.model.U[i]) for i in range(self.num_mode)]
                if extrapolation:
                    for mode in self.all_modes:
                        g, self.epstable[mode] = self.epsalg(grad[mode].reshape(self.shape[mode] * self.rank, 1), epsi, 3,
                                                             self.epstable[mode])
                        self.model.U[mode] -= 1e-6 * g.reshape(self.shape[mode], self.rank)
                    epsi += 1
                ii +=1
                if loss_type == 'apr' or loss_type == 'binary':
                    self.model.U = [project_nonnegative_A(self.model.U[i]) for i in range(self.num_mode)]
            loss = self.obj_eval(nonzero, fixed_indices, batch, X)
            # loss = ls_feval(xd, self.model)
            # loss=1
            print(loss)
            iter_info[e+1] = {'time': time.time()-start, 'loss': loss, 'losst':loss_true}

        return iter_info




