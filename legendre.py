# -*- coding: utf-8 -*-
"""
This function is an equivalent of the matlab legendre function.

        x is always regarded as a row vector.
		
		Requirements:
		numpy >= 1.3
		scipy >= 0.8
"""
import numpy as np
from scipy import special

def legendre(n,xm):
    xm = np.atleast_1d(np.array(xm)).flatten() #flattened
    res = np.zeros((xm.shape[0], n + 1))

    for i in range(xm.shape[0]):
        # to row vector
        res[i] = special.lpmn(n, n, xm[i])[0].transpose()[-1]

    return res.transpose()