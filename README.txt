An adaptive basin-hopping Markov-chain Monte Carlo algorithm for Bayesian optimisation
======================================================================================

This is the python (v2.7) implementation of the hoppMCMC algorithm aiming to identify and sample from the high-probability regions of a posterior distribution. The algorithm combines three strategies: (i) parallel MCMC, (ii) adaptive Gibbs sampling and (iii) simulated annealing. Overall, hoppMCMC resembles the basin-hopping algorithm implemented in the optimize module of scipy, but it is developed for a wide range of modelling approaches including stochastic models with or without time-delay.

Contents
--------

1) Prerequisites
2) Linux installation

1) Prerequisites
----------------

The hoppMCMC algorithm requires the following packages, which are not included in this package:

	numpy
	scipy
	mpi4py (MPI parallelisation)

The mpi4py package is required for parallelisation; however, it can be omitted.

2) Linux installation
---------------------

1) Easy way:

If you have pip installed, you can use the following command to download and install the package.
	pip install hoppMCMC

Alternatively, you can download the source code from PyPI and run pip on the latest version xxx.
	pip install hoppMCMC-xxx.tar.gz

2) Hard way:

If pip is not available, you can unpack the package contents and perform a manual install.
	tar -xvzf hoppMCMC-xxx.tar.gz
	cd hoppMCMC-xxx
	python setup.py install

This will install the package in the site-packages directory of your python distribution. If you do not have root privileges or you wish to install to a different directory, you can use the --prefix argument.

	python setup.py install --prefix=<dir>

In this case, please make sure that <dir> is in your PYTHONPATH, or you can add it with the following command.

In bash shell:
	export PYTHONPATH=<dir>:$PYTHONPATH
In c shell:
	setenv PYTHONPATH <dir>:$PYTHONPATH

Credits
-------

'modern-package-template' - http://pypi.python.org/pypi/modern-package-template

