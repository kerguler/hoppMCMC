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

# ---------------------------------------------------------------------
# Distance function
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

def score_briere(par, verbose=False):
    c, tmin, tmax = par
    # Invalid thermal limits are rejected by returning an infinite score.
    if tmin >= tmax:
        return numpy.inf
    predicted = briere_rate(temperatures, c=c, tmin=tmin, tmax=tmax)
    return normalised_rmse(predicted, observed_rates)

# ---------------------------------------------------------------------
# Brière model definition
# ---------------------------------------------------------------------

model_briere = ABCModel(
    name="briere",
    pargen=lambda: numpy.array([2.5e-4, 10.0, 38.0]),
    score=score_briere,
    lower=numpy.array([1e-6, 0.0, 30.0]),
    upper = numpy.array([1e-3, 20.0, 45.0]),
    prior=0.5,
    inferpar=[0, 1, 2],
    kernel=numpy.array([1e-8, 0.5, 0.5])
)

# ---------------------------------------------------------------------
# Run single-model ABC
# ---------------------------------------------------------------------

abc = inferABCModels(
    models=[model_briere],
    epsseq=[0.5],
    size=100,
    niter=20,
    adapt=2,
    retain=True,
    verbose=True
)

# ---------------------------------------------------------------------
# Summarise model posterior probabilities
# ---------------------------------------------------------------------

if myMPI.MPI_RANK == myMPI.MPI_MASTER:
    plt.plot(abc.results[:,abc.infocol['iter']],
             abc.results[:,abc.infocol['param0']],
             '.')
    plt.show()

if myMPI.MPI_RANK == myMPI.MPI_MASTER:
    # Use abc.result, unless your class stores it as abc.results
    mat = abc.result
    param_start = abc.infocol["param0"]
    params = mat[:, param_start:]
    # Remove padded NaN columns, useful if inferABCModels stores variable-length parameters
    valid_cols = ~numpy.all(numpy.isnan(params), axis=0)
    params = params[:, valid_cols]
    df = pandas.DataFrame(
        params,
        columns=[f"par{i+1}" for i in range(params.shape[1])]
    )
    df["score"] = mat[:, abc.infocol["score"]]
    #
    g = seaborn.pairplot(
        df,
        vars=[c for c in df.columns if c.startswith("par")],
        corner=True,
        diag_kind="kde",
        plot_kws={
            "s": 35,
            "alpha": 0.65,
            "edgecolor": "none"
        },
        diag_kws={
            "fill": True,
            "alpha": 0.45
        }
    )
    g.figure.suptitle("ABC posterior parameter samples", y=1.02)
    g.figure.set_size_inches(8, 8)
    plt.tight_layout()
    plt.show()    

if myMPI.MPI_RANK == myMPI.MPI_MASTER:
    model_probs = abc.model_posterior()
    print("\nPosterior model probabilities")
    print(model_probs)
    # Optional: inspect the final particle population.
    particles = abc.result
    print("\nFinal particle matrix shape:")
    print(particles.shape)