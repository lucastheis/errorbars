"""
Two ways of computing error bars for within-subject data.

Variability introduced by subjects can inflate error bars. These methods get
rid of this variability.
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'numpydoc'
__version__ = '1.0.0'

import numpy as np


def loftus_mason_sem(values):
	"""
	Compute Loftus & Masson's (1994) standard error.

	A single error bar is computed for all conditions. The 3-SEM rule can be used
	to gage significance (i.e., if means are separated by 3 or more SEMs, the differences
	are significant at the p=0.05 level).

	Parameters
	----------
	values : np.ndarray
		MxN array where M is the number of conditions and N is the number of subjects

	Returns
	-------
	float
		Loftus & Masson standard error

	References
	----------
	- G. R. Loftus and M. E. J. Masson, Using confidence intervals in within-subject designs, 1994
	- V. H. Franz and G. R. Loftus, Standard errors and confidence intervals in within-subjects designs, 2012
	"""

	values = np.asarray(values)

	M = values.shape[0]
	N = values.shape[1]

	sem_diff = np.zeros([M, M])

	# compute standard errors of differences between methods
	for i in range(M):
		for j in range(i + 1, M):
			# differences in performance between methods i and j
			sem_diff[i, j] = np.std(values[i] - values[j], ddof=1) / np.sqrt(N)

	mask = np.triu(np.ones([M, M]), 1) > .5
	return np.sqrt(np.mean(np.square(sem_diff[mask]) / 2.))


def normalized_sem(values):
	"""
	Compute bias-corrected standard errors from normalized data.

	Parameters
	----------
	values : np.ndarray
		MxN array where M is the number of conditions and N is the number of subjects

	Returns
	-------
	np.ndarray
		Normalized and bias-corrected standard errors

	References
	----------
	- R. D. Morey, Confidence intervals from normalized data, 2005
	"""

	values = np.asarray(values)

	M = values.shape[0]
	N = values.shape[1]

	# error bars
	sem = np.std(values - np.mean(values, 0), axis=1, ddof=1) / np.sqrt(N)

	# bias correction
	sem = sem * np.sqrt(M / (M - 1))

	return sem
