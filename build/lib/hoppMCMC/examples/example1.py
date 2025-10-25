"""

Following code demostrates the use of the hoppMCMC algorithm
for jumping between the two basins of attraction of 100*(x^2-1)^2

"""

from hoppMCMC import hoppMCMC
from numpy import repeat
import pylab

## The following is the objective function to be optimised
def fitness(x):
    return 100.0*pow(pow(x[0],2)-1.0,2)

## The following will run the hoppMCMC algorithm
results = hoppMCMC(fitness,            # define the objective function
                   param = [0],        # begin with x=0
                   varmat = [[1e-14]], # assign a low initial proposal variation
                   gibbs = True,       # this is enforced for single-parameter runs
                   rangeT = [1,10],    # define the range of annealing temperature
                   model_comp = 10,    # tolerance for accepting hopp-steps
                   num_hopp = 50,      # run for 50 hopp-steps
                   num_adapt = 100,    # each hopp-step comprises 100 adaptation steps
                   num_chain = 3,      # run with 3 parallel chains
                   chain_length = 10)  # each chain is 10 iterations long

# This will plot the state of the chains at the end of each hopp-step
for n in range(len(results.parmats)):
    pylab.plot(repeat(n,3),results.parmats[n][:,1],'o',c=["black","white"][n%2])
pylab.show()
