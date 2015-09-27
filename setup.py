from setuptools import setup, find_packages
import sys, os

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()
NEWS = open(os.path.join(here, 'NEWS.txt')).read()


version = '0.1'

# http://packages.python.org/distribute/setuptools.html#declaring-dependencies
install_requires = [
    'numpy>=1.9',
    'scipy>=0.14',
    'mpi4py>=1.3'
]


setup(name='hoppMCMC',
    version=version,
    description="An adaptive basin-hopping Markov-chain Monte Carlo algorithm for Bayesian optimisation",
    long_description=README + '\n\n' + NEWS,
    # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    keywords=['global','optimisation','parameter','inference','model','selection','simulated','annealing','bifurcation','stochastic','dynamics','biochemical','reaction','parallel','adaptive','basin','hopping','mcmc','metropolis','hastings','gibbs','importance','sampling'],
    author='Kamil Erguler',
    author_email='k.erguler@cyi.ac.cy',
    url = 'https://github.com/kerguler/hoppMCMC',
    download_url = "https://github.com/kerguler/hoppMCMC/tarball/%s" %(version),
    license='GPLv3',
    packages=find_packages('src'),
    package_dir = {'': 'src'},include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    py_modules=['hoppMCMC/__init__']
)