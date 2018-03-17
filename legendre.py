# -*- coding: utf-8 -*-
"""
This function is an equivalent of the matlab legendre function.
legendre(n,xm) computes the associated Legendre functions with degree n and order m = 0,1,...,n, evaluated for each element of xm. 
n should be a positive integer.
xm should be real value in the range −1 ≤ xm ≤ 1.

		Args:
			n: degree of Legendre function
			xm: input array
		
		Returns:
			numpy array: associated Legendre function

        Notes: xm is always regarded as a row vector. More information: https://www.mathworks.com/help/matlab/ref/legendre.html
		
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