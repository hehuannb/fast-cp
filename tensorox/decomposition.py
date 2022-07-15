# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np


from .utils import random_factors
from .ktensor import Ktensor


class BaseCP(object):
    """
    The base class for the CANDECOMP/PARAFAC (CP) decomposition.
    The CP model is described as approximating X using a ktensor
    """
    __metaclass__ = ABCMeta
    model = None       # ktensor model
    rank = 0           # rank of the tensor


    def _init_model(self, xShape):
        if self.model is None:
            self.model = Ktensor(random_factors(xShape, 
                                                self.rank),
                                 np.ones(self.rank))

    @abstractmethod
    def fit(self, xData, **kwargs):
        """
        Find the best decomposition for X

        Parameters
        ----------
        X : input tensor of the class dtensor or sptensor

        Returns
        ----------
        dict: statistics from the fitting process
        """
        pass

    @abstractmethod
    def transform(self, xData, n, **kwargs):
        """
        Transform the tensor X along mode n
        """
        pass