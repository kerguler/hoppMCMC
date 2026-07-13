import numpy
import pandas
import seaborn

from matplotlib import pyplot as plt

import myMPI
from inferFun import ABCModel, inferABCModels

# ---------------------------------------------------------------------
# Synthetic observation data
# ---------------------------------------------------------------------

temperatures = numpy.array([15, 18, 21, 24, 27, 30, 33, 36, 39], dtype=float)
observed_rates = numpy.array([0.07508502, 0.15548009, 0.25742823, 0.31720884, 0.39438616, 0.43292063, 0.41474694, 0.33905526, 0.        ])

def briere_rate(temp, c, tmin, tmax):
    """
    Brière temperature response.

    The response is zero below tmin and above tmax.
    Between tmin and tmax, it rises and then falls near tmax.
    """
    temp = numpy.asarray(temp, dtype=float)
    rate = c * temp * (temp - tmin) * numpy.sqrt(numpy.maximum(tmax - temp, 0.0))
    rate[(temp <= tmin) | (temp >= tmax)] = 0.0
    return rate

def exponential_rate(temp, a, b):
    """
    Exponential temperature response.

    This model can increase with temperature but cannot represent a
    high-temperature decline.
    """
    temp = numpy.asarray(temp, dtype=float)
    return a * numpy.exp(b * temp)

# ---------------------------------------------------------------------
# Shared distance function helper
# ---------------------------------------------------------------------

def normalised_rmse(predicted, observed):
    """
    Normalised root mean squared error.

    The normalisation keeps the ABC threshold easier to interpret across
    examples.
    """
    predicted = numpy.asarray(predicted, dtype=float)
    observed = numpy.asarray(observed, dtype=float)
    scale = numpy.std(observed)
    if scale == 0:
        scale = 1.0
    return numpy.sqrt(numpy.mean((predicted - observed) ** 2)) / scale

# ---------------------------------------------------------------------
# Exponential model definition
# ---------------------------------------------------------------------

lower_exp = numpy.array([1e-5, 0.001])
upper_exp = numpy.array([0.05, 0.25])

def pargen_exponential():
    return numpy.random.uniform(lower_exp, upper_exp)

def score_exponential(par, verbose=False):
    a, b = par
    predicted = exponential_rate(
        temperatures,
        a=a,
        b=b
    )
    return normalised_rmse(predicted, observed_rates)

model_exponential = ABCModel(
    name="exponential",
    pargen=pargen_exponential,
    score=score_exponential,
    lower=lower_exp,
    upper=upper_exp,
    prior=0.5,
    inferpar=[0, 1],
    kernel=numpy.array([1e-4, 1e-4])
)

# ---------------------------------------------------------------------
# Brière model definition
# ---------------------------------------------------------------------

lower_briere = numpy.array([1e-6, 0.0, 30.0])
upper_briere = numpy.array([1e-3, 20.0, 45.0])

def pargen_briere():
    return numpy.random.uniform(lower_briere, upper_briere)

def score_briere(par, verbose=False):
    c, tmin, tmax = par
    # Invalid thermal limits are rejected by returning an infinite score.
    if tmin >= tmax:
        return numpy.inf
    predicted = briere_rate(
        temperatures,
        c=c,
        tmin=tmin,
        tmax=tmax
    )
    return normalised_rmse(predicted, observed_rates)

model_briere = ABCModel(
    name="briere",
    pargen=pargen_briere,
    score=score_briere,
    lower=lower_briere,
    upper=upper_briere,
    prior=0.5,
    inferpar=[0, 1, 2],
    kernel=numpy.array([1e-8, 0.5, 0.5])
)

# ---------------------------------------------------------------------
# Run multi-model ABC
# ---------------------------------------------------------------------

abc = inferABCModels(
    models=[
        model_exponential,
        model_briere
    ],
    epsseq=[
        1.5,
        1.0,
        0.5
    ],
    size=100,
    niter=10,
    adapt=2,
    retain=True,
    verbose=True
)

# ---------------------------------------------------------------------
# Summarise model posterior probabilities
# ---------------------------------------------------------------------

if myMPI.MPI_RANK == myMPI.MPI_MASTER:
    model_probs = abc.model_posterior()
    print("\nPosterior model probabilities")
    print(model_probs)
    # Optional: inspect the final particle population.
    particles = abc.result
    print("\nFinal particle matrix shape:")
    print(particles.shape)

if myMPI.MPI_RANK == myMPI.MPI_MASTER:
    mats = [abc.results[:abc.size,:], abc.result]
    p0 = abc.infocol["param0"]
    #
    for ipop,mat in enumerate(mats):
        plt.figure(figsize=(7, 5))
        # Observations
        plt.scatter(
            temperatures,
            observed_rates,
            s=50,
            color="black",
            label="observed",
            zorder=10
        )
        # Exponential posterior population
        model_id = abc.model_names.index("exponential")
        sub = mat[mat[:, abc.infocol["model"]] == model_id]
        for row in sub:
            par = row[p0:p0 + abc.parsizes[model_id]]
            y = exponential_rate(temperatures, par[0], par[1])
            plt.plot(
                temperatures,
                y,
                color="tab:blue",
                alpha=0.5
            )
        # Brière posterior population
        model_id = abc.model_names.index("briere")
        sub = mat[mat[:, abc.infocol["model"]] == model_id]
        for row in sub:
            par = row[p0:p0 + abc.parsizes[model_id]]
            y = briere_rate(temperatures, par[0], par[1], par[2])
            plt.plot(
                temperatures,
                y,
                color="tab:orange",
                alpha=0.5
            )
        #
        # Dummy lines for legend
        plt.plot([], [], color="tab:blue", label="exponential")
        plt.plot([], [], color="tab:orange", label="Brière")
        plt.xlabel("Temperature")
        plt.ylabel("Rate")
        plt.title(f"{["First","Final"][ipop]} ABC posterior population")
        plt.legend()
        plt.tight_layout()
        plt.show()