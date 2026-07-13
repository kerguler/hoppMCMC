import numpy
import pandas

import myMPI as mympi

from dataclasses import dataclass

from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

from mpi4py import MPI
MPI_SIZE = MPI.COMM_WORLD.Get_size()
MPI_RANK = MPI.COMM_WORLD.Get_rank()
MPI_MASTER = 0
worker_indices = numpy.delete(numpy.arange(MPI_SIZE),MPI_MASTER)
print("Process %d of %d is running" %(MPI_RANK,MPI_SIZE),flush=True)

ALGO_HOPP = 0
ALGO_ABC = 1

@dataclass
class ABCModel:
    """
    Model specification for multi-model ABC-SMC.

    `ABCModel` stores the model-specific components needed by
    `inferABCModels`: a parameter generator, a distance/score function,
    parameter bounds, prior model probability, and optional proposal-kernel
    information.

    Each model may have a different number of parameters and a different
    subset of parameters selected for inference.

    Parameters
    ----------
    name : str
        Name of the model. This is used in summaries, printed output, and
        posterior model probability tables.

    pargen : callable
        Function used to generate one parameter vector from the model prior.

        The function should take no arguments and return a one-dimensional
        array-like object.

        Example
        -------
        >>> def pargen():
        ...     return numpy.random.uniform(lower, upper)

    score : callable
        ABC distance function.

        The function should take a parameter vector and return a scalar
        distance between simulated and observed data. Smaller values indicate
        better agreement with the observations.

        A parameter vector is accepted at threshold `eps` if the returned score
        is finite, non-negative, and smaller than `eps`.

        The function may optionally accept `verbose=False`.

        Example
        -------
        >>> def score(par, verbose=False):
        ...     y = model_prediction(par)
        ...     return numpy.sqrt(numpy.mean((y - yobs) ** 2))

    lower : numpy.ndarray
        Lower bounds of the parameter vector.

        Proposed particles outside these bounds are rejected before simulation.

    upper : numpy.ndarray
        Upper bounds of the parameter vector.

        Proposed particles outside these bounds are rejected before simulation.

    prior : float, default=1.0
        Prior probability or prior weight of the model.

        The values supplied for all models are normalised internally, so they
        do not need to sum to one.

    inferpar : array-like or None, default=None
        Indices of parameters to infer and perturb with the proposal kernel.

        If `None`, all parameters are inferred.

        Parameters not listed in `inferpar` remain part of the parameter vector
        but are not perturbed by the default kernel.

    kernel : array-like or None, default=None
        Initial proposal kernel for the inferred parameters.

        If one-dimensional, it is interpreted as the diagonal of a covariance
        matrix. If two-dimensional, it is interpreted as a full covariance
        matrix.

        If `None`, a small diagonal kernel is used.

    logprior : callable or None, default=None
        Optional log-prior density function for the model parameters.

        If supplied, it should take a full parameter vector and return the log
        prior density.

        If `None`, `inferABCModels` assumes independent uniform priors over
        the bounds defined by `lower` and `upper`.

    fun_kernel : callable or None, default=None
        Optional model-specific proposal kernel.

        If supplied, this replaces the default multivariate normal proposal for
        this model. The function should have the form:

        >>> fun_kernel(abc, model_id, param0, kernel)

        and should return a new proposed parameter vector.

    Notes
    -----
    `ABCModel` only describes a model. It does not run inference by itself.
    Inference is performed by `inferABCModels`.
    """
    name: str
    pargen: callable
    score: callable
    lower: numpy.ndarray
    upper: numpy.ndarray
    prior: float = 1.0
    inferpar: object = None
    kernel: object = None
    logprior: object = None
    fun_kernel: object = None

class inferABCModels:
    """
    Multi-model Approximate Bayesian Computation with Sequential Monte Carlo.

    `inferABCModels` performs ABC-SMC over a shared population of particles,
    where each particle contains both a model identity and a parameter vector.
    Models therefore compete for particles during inference.

    This class is intended for ABC model comparison as well as parameter
    inference. The posterior probability of a model is estimated by summing the
    final particle weights belonging to that model.

    Parameters
    ----------
    models : list of ABCModel
        Models to compare.

        Each model is defined by an `ABCModel` object containing its parameter
        generator, distance function, bounds, prior model probability, and
        optional kernel settings.

    epsseq : list of float
        Sequence of ABC distance thresholds.

        Particles are accepted when their distance is finite, non-negative,
        and smaller than the current threshold.

        Thresholds are used in order. Smaller thresholds usually impose a
        stricter match to the observations.

    size : int
        Total population size across all models.

        This is the total number of ABC particles, not the number of particles
        per model. For example, if `size=300` and two models are supplied, the
        final population contains 300 particles in total, distributed between
        the two models according to their posterior support.

    niter : int
        Number of ABC-SMC generations to perform at each threshold.

        After every `niter` generations, the next value from `epsseq` is used.

    adapt : int
        Frequency of proposal-kernel adaptation.

        If `adapt > 0`, each model-specific kernel is adapted every `adapt`
        generations using the empirical covariance of particles currently
        assigned to that model.

        If `adapt == 0`, kernels are not adapted.

    fmt : str, default="{:g}"
        Format string used for printed numerical output.

    jitter : float, default=1e-8
        Small diagonal variance added to covariance matrices to reduce
        numerical problems caused by singular or nearly singular proposal
        kernels.

        This is especially useful when the posterior is stiff in one or more
        parameter directions.

    mcmc : float, default=-1
        Controls the acceptance rule.

        If `mcmc < 0`, hard ABC acceptance is used:

        accepted if `score < eps`.

        If `mcmc > 0`, an MCMC-like acceptance rule is used in which proposals
        with lower scores are always accepted and proposals with higher scores
        may still be accepted with probability depending on `(old_score -
        new_score) / mcmc`, provided the new score is below `eps`.

    retain : bool, default=False
        If `False`, only the final particle population is kept in `result`.

        If `True`, all generations are retained in `results`, allowing the
        inference trajectory to be inspected after completion.

    verbose : bool, default=True
        If `True`, progress information, kernels, thresholds, and particle
        matrices are printed during inference.

    Attributes
    ----------
    result : numpy.ndarray
        Final ABC particle population.

        Each row corresponds to one particle. The columns are indexed by
        `infocol` and contain generation number, model identity, particle
        weight, simulation counts, ABC score, and parameter values.

    results : list or numpy.ndarray
        Retained generations, available when `retain=True`.

    model_names : list of str
        Names of the models supplied through `ABCModel`.

    model_priors : numpy.ndarray
        Normalised prior model probabilities.

    parsizes : list of int
        Number of parameters for each model.

    infocol : dict
        Column indices for the particle matrix.

        Typical entries are:

        - `"iter"`: generation index;
        - `"model"`: model index;
        - `"weight"`: particle weight;
        - `"fcount"`: failed simulation/proposal count;
        - `"trials"`: number of proposal trials;
        - `"score"`: ABC distance;
        - `"param0"`: first parameter column.

    Methods
    -------
    model_posterior()
        Return posterior model probabilities by summing final particle weights
        for each model.

    Notes
    -----
    The usual workflow is:

    >>> abc = inferABCModels(...)
    >>> if myMPI.MPI_RANK == myMPI.MPI_MASTER:
    ...     print(abc.model_posterior())

    When running with MPI, only the master process should perform
    post-processing and plotting.

    Examples
    --------
    Define two models:

    >>> model_exp = ABCModel(
    ...     name="exponential",
    ...     pargen=pargen_exponential,
    ...     score=score_exponential,
    ...     lower=lower_exp,
    ...     upper=upper_exp,
    ...     prior=0.5,
    ...     inferpar=[0, 1],
    ...     kernel=numpy.array([1e-4, 1e-4])
    ... )

    >>> model_briere = ABCModel(
    ...     name="briere",
    ...     pargen=pargen_briere,
    ...     score=score_briere,
    ...     lower=lower_briere,
    ...     upper=upper_briere,
    ...     prior=0.5,
    ...     inferpar=[0, 1, 2],
    ...     kernel=numpy.array([1e-8, 0.5, 0.5])
    ... )

    Run multi-model ABC-SMC:

    >>> abc = inferABCModels(
    ...     models=[model_exp, model_briere],
    ...     epsseq=[2.0, 1.0, 0.5],
    ...     size=300,
    ...     niter=2,
    ...     adapt=1
    ... )

    Summarise posterior model probabilities on the master process:

    >>> if myMPI.MPI_RANK == myMPI.MPI_MASTER:
    ...     print(abc.model_posterior())
    """

    def __init__(self,
                 models,
                 epsseq,
                 size,
                 niter,
                 adapt,
                 fmt="{:g}",
                 jitter=1e-8,
                 mcmc=-1,
                 retain=False,
                 verbose=True) -> None:
        """
        Initialise and run multi-model ABC-SMC.

        See `help(inferABCModels)` for the full description of arguments,
        attributes, and examples.
        """

        self.models = models
        self.nmodels = len(models)

        self.model_names = [m.name for m in models]

        model_priors = numpy.array([m.prior for m in models], dtype=float)
        self.model_priors = model_priors / numpy.sum(model_priors)

        # Determine parameter sizes.
        self.parsizes = []
        for m in self.models:
            pr = numpy.asarray(m.pargen(), dtype=float)
            self.parsizes.append(len(pr))

        self.max_parsize = max(self.parsizes)

        # Set default inferpar for each model.
        self.inferpars = []
        for k, m in enumerate(self.models):
            if m.inferpar is None:
                self.inferpars.append(numpy.arange(self.parsizes[k]))
            else:
                self.inferpars.append(numpy.asarray(m.inferpar, dtype=int))

        self.infersizes = [len(x) for x in self.inferpars]

        self.jitter = jitter if jitter>0.0 else 0.0
        self.fmt = fmt
        self.mcmc = mcmc
        self.check = self.checkMove if mcmc < 0.0 else self.checkMoveMCMC

        self.infocol = {
            "iter": 0,
            "model": 1,
            "weight": 2,
            "fcount": 3,
            "trials": 4,
            "score": 5,
            "param0": 6
        }

        self.size = int(size)
        self.niter = int(niter)
        self.adapt = int(adapt)
        self.verbose = verbose
        self.epsseq = list(epsseq)
        self.retain = retain

        self.result = None
        self.results = []

        mpi = mympi.mpi(self.function_master, self.function_slave)
        mpi.clean()

    def inferABC_kernel(self, model_id, param0, kernel):
        inferpar = self.inferpars[model_id]
        param1 = param0.copy()
        param1[inferpar] = numpy.random.multivariate_normal(mean=param1[inferpar], cov=kernel)
        return param1
    
    def get_param(self, row, model_id):
        p0 = self.infocol["param0"]
        psize = self.parsizes[model_id]
        return numpy.asarray(row[p0:p0 + psize], dtype=float)

    def in_bounds(self, model_id, pr):
        m = self.models[model_id]
        return numpy.all(pr >= m.lower) and numpy.all(pr <= m.upper)

    def logprior(self, model_id, pr):
        """
        Parameter prior density.

        If the model supplies logprior, use it.
        Otherwise assume independent uniform priors over lower/upper
        for the inferred parameters.

        This matters for model selection when models have different
        parameter dimensions or prior ranges.
        """

        m = self.models[model_id]

        if m.logprior is not None:
            return m.logprior(pr)
        if not self.in_bounds(model_id, pr):
            return -numpy.inf

        inferpar = self.inferpars[model_id]
        width = m.upper[inferpar] - m.lower[inferpar]

        if numpy.any(width <= 0):
            return -numpy.inf
        return -numpy.sum(numpy.log(width))

    def eval_score(self, model_id, pr):
        scorefun = self.models[model_id].score
        try:
            return scorefun(pr, verbose=False)
        except TypeError:
            return scorefun(pr)

    def checkMove(self, f0, f1, eps):
        return (
            not numpy.isnan(f1)
            and not numpy.isinf(f1)
            and f1 >= 0
            and f1 < eps
        )

    def checkMoveMCMC(self, f0, f1, eps):
        return (
            not numpy.isnan(f1)
            and not numpy.isinf(f1)
            and f1 >= 0
            and f1 < eps
            and (
                f1 < f0
                or numpy.log(numpy.random.random()) < (f0 - f1) / self.mcmc
            )
        )

    def sample_model_from_prior(self):
        return numpy.random.choice(numpy.arange(self.nmodels), p=self.model_priors)

    def sample_param(self, init, kernels, mat):
        if init:
            model_id = self.sample_model_from_prior()
            pr = numpy.asarray(self.models[model_id].pargen(), dtype=float)
            scr = numpy.inf
            pr_new = pr.copy()
        else:
            index_new = numpy.random.choice(numpy.arange(mat.shape[0]), size=1, p=mat[:, self.infocol["weight"]], replace=True)[0]
            model_id = int(mat[index_new, self.infocol["model"]])
            pr = self.get_param(mat[index_new, :], model_id)
            scr = mat[index_new, self.infocol["score"]]
            #
            m = self.models[model_id]
            if m.fun_kernel is None:
                pr_new = self.inferABC_kernel(model_id, pr, kernels[model_id])
            else:
                pr_new = m.fun_kernel(self, model_id, pr, kernels[model_id])
        return model_id, pr, scr, pr_new

    def log_kernel_density(self, model_id, x, mean, kernel):
        inferpar = self.inferpars[model_id]
        x_inf = numpy.asarray(x[inferpar], dtype=float)
        mean_inf = numpy.asarray(mean[inferpar], dtype=float)
        d = len(inferpar)

        kernel = numpy.asarray(kernel, dtype=float)
        if kernel.ndim == 0:
            kernel = numpy.array([[float(kernel)]], dtype=float)
        elif kernel.ndim == 1:
            kernel = numpy.diag(kernel)
        elif kernel.ndim != 2:
            raise ValueError(
                f"Kernel for model {model_id} ({self.model_names[model_id]}) "
                f"must be scalar, 1D, or 2D. Got shape {kernel.shape}."
            )
        if kernel.shape != (d, d):
            raise ValueError(
                f"Kernel dimension mismatch for model {model_id} "
                f"({self.model_names[model_id]}). "
                f"Expected {(d, d)}, got {kernel.shape}. "
                f"inferpar={inferpar}"
            )
        if not numpy.all(numpy.isfinite(x_inf)):
            raise ValueError(
                f"Non-finite x values for model {model_id} "
                f"({self.model_names[model_id]}): {x_inf}"
            )
        if not numpy.all(numpy.isfinite(mean_inf)):
            raise ValueError(
                f"Non-finite kernel mean for model {model_id} "
                f"({self.model_names[model_id]}): {mean_inf}"
            )
        if not numpy.all(numpy.isfinite(kernel)):
            raise ValueError(
                f"Non-finite kernel entries for model {model_id} "
                f"({self.model_names[model_id]}):\n{kernel}"
            )
        # Symmetrise, just in case floating-point noise introduced asymmetry.
        kernel = 0.5 * (kernel + kernel.T)
        diag = numpy.diag(kernel)
        if numpy.any(diag < 0):
            raise ValueError(
                f"Negative diagonal variance in kernel for model {model_id} "
                f"({self.model_names[model_id]}).\n"
                f"Diagonal={diag}\n"
                f"Kernel=\n{kernel}"
            )
        eigvals, eigvecs = numpy.linalg.eigh(kernel)
        max_eig = numpy.max(eigvals)
        # A purely absolute jitter may be too small if the largest variance is large.
        # This floor is both absolute and relative.
        eig_floor = max(self.jitter, self.jitter * max(1.0, max_eig))
        if numpy.any(eigvals <= eig_floor):
            bad = numpy.where(eigvals <= eig_floor)[0]
            if self.verbose:
                print(
                    f"Warning: regularising near-singular kernel for model {model_id} "
                    f"({self.model_names[model_id]}).",
                    flush=True
                )
                print(f"  parameter indices: {inferpar}", flush=True)
                print(f"  diagonal variances: {diag}", flush=True)
                print(f"  eigenvalues before regularisation: {eigvals}", flush=True)
                print(f"  eigenvalue floor: {eig_floor:g}", flush=True)
                #
                for b in bad:
                    direction = eigvecs[:, b]
                    order = numpy.argsort(numpy.abs(direction))[::-1]
                    print(
                        f"  problematic eigenvalue {b}: {eigvals[b]:g}",
                        flush=True
                    )
                    print("  strongest dimensions in this direction:", flush=True)
                    for k in order[:min(5, len(order))]:
                        print(
                            f"    kernel dim {k}, parameter index {inferpar[k]}, "
                            f"loading={direction[k]:g}, "
                            f"variance={diag[k]:g}",
                            flush=True
                        )
            # Regularise eigenvalues, not only diagonal entries.
            eigvals = numpy.maximum(eigvals, eig_floor)
            kernel = eigvecs @ numpy.diag(eigvals) @ eigvecs.T
            kernel = 0.5 * (kernel + kernel.T)
        # Final check by Cholesky. This is stricter and clearer than waiting for SciPy.
        try:
            numpy.linalg.cholesky(kernel)
        except numpy.linalg.LinAlgError as err:
            eigvals_final = numpy.linalg.eigvalsh(kernel)
            raise ValueError(
                f"Kernel is still not positive definite for model {model_id} "
                f"({self.model_names[model_id]}).\n"
                f"parameter indices={inferpar}\n"
                f"x={x_inf}\n"
                f"mean={mean_inf}\n"
                f"diagonal variances={numpy.diag(kernel)}\n"
                f"eigenvalues={eigvals_final}\n"
                f"kernel=\n{kernel}"
            ) from err
        try:
            value = multivariate_normal.logpdf(
                x_inf,
                mean=mean_inf,
                cov=kernel,
                allow_singular=False
            )
        except Exception as err:
            eigvals_final = numpy.linalg.eigvalsh(kernel)
            raise ValueError(
                f"multivariate_normal.logpdf failed for model {model_id} "
                f"({self.model_names[model_id]}).\n"
                f"parameter indices={inferpar}\n"
                f"x={x_inf}\n"
                f"mean={mean_inf}\n"
                f"diagonal variances={numpy.diag(kernel)}\n"
                f"eigenvalues={eigvals_final}\n"
                f"kernel=\n{kernel}"
            ) from err
        if not numpy.isfinite(value):
            eigvals_final = numpy.linalg.eigvalsh(kernel)
            raise ValueError(
                f"Non-finite kernel log density for model {model_id} "
                f"({self.model_names[model_id]}).\n"
                f"logpdf={value}\n"
                f"parameter indices={inferpar}\n"
                f"x={x_inf}\n"
                f"mean={mean_inf}\n"
                f"diagonal variances={numpy.diag(kernel)}\n"
                f"eigenvalues={eigvals_final}\n"
                f"kernel=\n{kernel}"
            )
        return value

    def calc_weights(self, mat_new, mat_old, kernels):
        """
        ABC-SMC weights for particles with model identity.

        For each new particle:

            weight ∝ model_prior * parameter_prior_density
                     / proposal_density_from_previous_population

        The proposal density includes only previous particles with the
        same model, because this version does not use cross-model jumps.
        """

        logw = numpy.full(self.size, -numpy.inf, dtype=float)
        old_weights = mat_old[:, self.infocol["weight"]]

        for i in range(self.size):
            model_id = int(mat_new[i, self.infocol["model"]])
            pr_new = self.get_param(mat_new[i, :], model_id)

            numerator = (numpy.log(self.model_priors[model_id]) + self.logprior(model_id, pr_new))

            terms = []
            for j in range(mat_old.shape[0]):
                old_model_id = int(mat_old[j, self.infocol["model"]])
                if old_model_id != model_id:
                    continue
                if old_weights[j] <= 0:
                    continue

                pr_old = self.get_param(mat_old[j, :], old_model_id)
                lk = self.log_kernel_density(model_id, pr_new, pr_old, kernels[model_id])

                terms.append(numpy.log(old_weights[j]) + lk)

            if len(terms) == 0:
                logw[i] = -numpy.inf
            else:
                logw[i] = numerator - logsumexp(terms)

        normaliser = logsumexp(logw)
        if numpy.isinf(normaliser):
            raise ValueError("All particle weights are zero.")

        weights = numpy.exp(logw - normaliser)
        return weights
    
    def calc_kernel_init(self):
        kernels = []
        for model_id, m in enumerate(self.models):
            inferpar = self.inferpars[model_id]
            d = len(inferpar)
            if m.kernel is None:
                tmp = numpy.repeat(1e-6, d)
            else:
                tmp = numpy.asarray(m.kernel, dtype=float)
            kernel = numpy.diag(tmp) if tmp.ndim == 1 else tmp
            kernels.append(kernel)
            self.print_kernel(model_id, kernel)
        return kernels
    
    def calc_kernel_adapt(self, mat, iter, kernels, kernel_base):
        if iter == 0 or self.adapt == 0 or iter % self.adapt > 0:
            return kernels

        new_kernels = [k.copy() for k in kernels]
        for model_id in range(self.nmodels):
            mask = mat[:, self.infocol["model"]] == model_id
            # Need enough particles to estimate covariance.
            if numpy.sum(mask) <= self.infersizes[model_id] + 1:
                continue

            inferpar = self.inferpars[model_id]
            cols = self.infocol["param0"] + inferpar
            x = mat[mask, :][:, cols]
            tmp = numpy.cov(x.T)
            if tmp.ndim == 0:
                tmp = numpy.array([[tmp]])

            # Keep diagonal variances above the initial kernel floor.
            base = kernel_base[model_id]
            a = numpy.diag_indices_from(tmp)
            tmp[a] = numpy.maximum(tmp[a], base[a])

            new_kernels[model_id] = tmp
            self.print_kernel(model_id, tmp, iter)

        return new_kernels
    
    def calc_mat_init(self, mpi, eps):
        ncol = self.infocol["param0"] + self.max_parsize
        mat = numpy.full((self.size, ncol), numpy.nan, dtype=numpy.float64)
        return self.take_a_step(mpi, mat, 0, [], eps)

    def take_a_step(self, mpi, mat, iter, kernels, eps):
        ncol = self.infocol["param0"] + self.max_parsize
        mat_new = numpy.full((self.size, ncol), numpy.nan, dtype=numpy.float64)

        mat_new[:, self.infocol["weight"]] = 0.0
        mat_new[:, self.infocol["fcount"]] = 0.0
        mat_new[:, self.infocol["trials"]] = 0.0

        jobs = [{
            "mat": mat,
            "index": index,
            "init": iter == 0,
            "kernels": kernels,
            "eps": eps
        } for index in numpy.arange(self.size)]

        while len(jobs) > 0:
            ret = mpi.exec(jobs, multiple=True)
            jobs = []

            for elm in ret:
                idx = elm["index"]

                mat_new[idx, self.infocol["trials"]] += 1
                trials = mat_new[idx, self.infocol["trials"]]

                if len(elm["pr_new"]):
                    model_id = int(elm["model"])
                    pr_new = numpy.asarray(elm["pr_new"], dtype=float)
                    psize = self.parsizes[model_id]

                    fcount = mat_new[idx, self.infocol["fcount"]]

                    mat_new[idx, self.infocol["iter"]] = iter
                    mat_new[idx, self.infocol["model"]] = model_id
                    mat_new[idx, self.infocol["weight"]] = 1.0
                    mat_new[idx, self.infocol["fcount"]] = fcount
                    mat_new[idx, self.infocol["trials"]] = trials
                    mat_new[idx, self.infocol["score"]] = elm["score"]

                    p0 = self.infocol["param0"]
                    mat_new[idx, p0:p0 + psize] = pr_new

                    self.print_fcount(fcount, trials, idx)

                else:
                    mat_new[idx, self.infocol["fcount"]] += 1

                    # If this particle has not yet been accepted, try again.
                    if mat_new[idx, self.infocol["weight"]] == 0:
                        ntrial = min(MPI_SIZE, int(trials) * 2)

                        for n in range(ntrial):
                            jobs.append({
                                "mat": mat,
                                "index": idx,
                                "init": iter == 0,
                                "kernels": kernels,
                                "eps": eps
                            })

        if iter == 0:
            mat_new[:, self.infocol["weight"]] = 1.0 / self.size
        else:
            mat_new[:, self.infocol["weight"]] = self.calc_weights(mat_new, mat, kernels)

        return mat_new

    def model_posterior(self, mat=None):
        if mat is None:
            mat = self.result

        if mat is None:
            raise ValueError("No ABC result available.")

        rows = []
        for model_id, name in enumerate(self.model_names):
            mask = mat[:, self.infocol["model"]] == model_id
            rows.append({
                "model_id": model_id,
                "model": name,
                "n_particles": int(numpy.sum(mask)),
                "posterior_probability": numpy.sum(
                    mat[mask, self.infocol["weight"]]
                )
            })

        out = pandas.DataFrame(rows)
        total = out["posterior_probability"].sum()
        if total > 0:
            out["posterior_probability"] /= total

        return out.sort_values("posterior_probability", ascending=False).reset_index(drop=True)
    
    def print_kernel(self, model_id, kernel, iter=0):
        if self.verbose:
            name = self.model_names[model_id]
            print(
                "kernel.sd: "
                + str(iter)
                + ","
                + name
                + ","
                + ",".join([
                    self.fmt.format(m)
                    for m in numpy.diagonal(kernel)
                ]),
                flush=True
            )

    def print_mat(self, mat, eps, iter=0):
        if self.verbose:
            print(
                "param.mat: "
                + self.fmt.format(eps)
                + ","
                + ("\nparam.mat: " + self.fmt.format(eps) + ",").join([
                    ",".join([
                        self.fmt.format(mm) if not numpy.isnan(mm) else "nan"
                        for mm in m
                    ])
                    for m in mat
                ]),
                flush=True
            )

    def print_eps(self, eps, iter=0):
        if self.verbose:
            print("EPS:", eps, flush=True)

    def print_fcount(self, fcount, trials, index):
        if self.verbose:
            print(
                "Matrix: %d (%s / %s)"
                % (index, self.fmt.format(fcount), self.fmt.format(trials)),
                flush=True
            )

    def calc_get_eps(self):
        eps = None
        if len(self.epsseq):
            eps = self.epsseq.pop(0)
            self.print_eps(eps)
        return eps
    
    def function_slave(self, mpi, cmd, opt={}):
        mat = cmd["mat"]
        index = cmd["index"]
        init = cmd["init"] if "init" in cmd else False
        kernels = cmd["kernels"]
        eps = cmd["eps"]

        while True:
            model_id, pr, scr, pr_new = self.sample_param(init, kernels, mat)
            if self.in_bounds(model_id, pr_new):
                break

        scr_new = self.eval_score(model_id, pr_new)

        if self.check(scr, scr_new, eps):
            return {
                "index": index,
                "model": model_id,
                "score": scr_new,
                "init": init,
                "pr_new": pr_new
            }

        return {
            "index": index,
            "model": model_id,
            "score": numpy.inf,
            "init": init,
            "pr_new": []
        }
    
    def function_master(self, mpi, opt={}):
        eps = self.calc_get_eps()

        mat = self.calc_mat_init(mpi, eps)

        kernels = self.calc_kernel_init()
        kernel_base = [k.copy() for k in kernels]

        iter = 0
        self.print_mat(mat, eps, iter)

        while True:
            iter += 1

            mat = self.take_a_step(mpi, mat, iter, kernels, eps)

            kernels = self.calc_kernel_adapt(mat, iter, kernels, kernel_base)

            self.print_mat(mat, eps, iter)
            if self.retain:
                self.results.append(mat)

            if iter % self.niter == 0:
                eps = self.calc_get_eps()

                if eps is None:
                    break

        self.result = mat
        self.results = numpy.vstack(self.results)
        return mat

# https://darrenjw.wordpress.com/2010/08/15/metropolis-hastings-mcmc-algorithms/
def mcmc(pr,fun,lower,upper,kernel,niter=1000,thin=10,sig=1.0,verbose=False):
    acc = 0
    pr = numpy.array(pr)
    prn = pr
    scrn = scr = fun(pr)
    mat = [[scr] + pr.tolist()]
    for n in numpy.arange(niter):
        while True:
            prn = kernel * numpy.random.randn(pr.shape[0]) + pr
            if all(prn>=lower) and all(prn<=upper):
                break
        scrn = fun(prn)
        if scrn==scrn and (scrn < scr or numpy.log(numpy.random.random()) < (scr-scrn)/sig):
            pr = prn
            scr = scrn
            acc += 1
        if n % thin == 0:
            if verbose:
                vec = [scr] + pr.tolist()
                mat.append(vec)
                print("%d,%g,%s" %(n,acc,",".join([str(tmp) for tmp in vec])),flush=True)
            acc = 0
    return(numpy.array(mat))

class inferABC:
    def __init__(self,
                 pargen,
                 score,
                 epsseq,
                 lower,
                 upper,
                 size,
                 niter,
                 adapt,
                 inferpar=[],
                 kernel=[],
                 fmt="{:g}",
                 fun_kernel=None,
                 mcmc=-1,
                 verbose=True) -> None:
        '''
        pargen: function to generate parameter samples from the prior
        score: distance function
        epsseq: a list of thresholds for each niter steps will be taken
        lower - upper: lower and upper bounds of parameter values (independent of infervar)
        size: individual parameter sets to sample
        niter: number of iterations per eps
        adapt: number of steps before the kernel adapts
        inferpar: indices of parameters to infer
        fmt: output format string (default: {:g})
        fun_kernel: (optional) kernel in the form of a function(abc,param0,kernel)
        mcmc: (optional) positive values indicate MCMC-like fitness checking with mcmc being a scale factor
        '''
        self.pargen = pargen
        self.parsize = len(self.pargen())
        self.inferpar = numpy.array(inferpar)
        if len(self.inferpar)==0:
            self.inferpar = numpy.arange(self.parsize)
        self.infersize = len(self.inferpar)
        #
        self.fmt = fmt
        #
        self.fun_kernel = self.inferABC_kernel
        if fun_kernel != None:
            self.fun_kernel = fun_kernel
        #
        self.mcmc = mcmc
        self.check = self.checkMove if mcmc < 0.0 else self.checkMoveMCMC
        #
        self.infocol = {
            'iter': 0,
            'weight': 1,
            'fcount': 2,
            'trials': 3,
            'score': 4,
            'param0': 5
        }
        #
        self.kernel_init = numpy.array(kernel) if len(kernel) else []
        #
        self.score = score
        self.size = size
        self.lower = lower
        self.upper = upper
        self.niter = niter
        self.adapt = int(adapt)
        self.verbose = verbose
        #
        self.epsseq = epsseq
        #
        self.result = None
        #
        mpi = mympi.mpi(self.function_master,self.function_slave)
        mpi.clean()
        #
    def inferABC_kernel(self,abc,param0,kernel):
        param1 = param0.copy()
        param1[abc.inferpar] = numpy.random.multivariate_normal(mean = param1[abc.inferpar], cov = kernel)
        return param1
        #
    def checkMove(self,f0,f1,eps):
        return not numpy.isnan(f1) and not numpy.isinf(f1) and (f1 >= 0) and (f1 < eps)
        #
    def checkMoveMCMC(self,f0,f1,eps):
        return not numpy.isnan(f1) and not numpy.isinf(f1) and (f1 >= 0) and (f1 < eps) and ((f1 < f0) or (numpy.log(numpy.random.random()) < (f0-f1)/self.mcmc))
        #
    def sample_param(self,init,kernel,mat):
        if init:
            scr = numpy.inf
            pr = pr_new = numpy.array(self.pargen())
        else:
            index_new = numpy.random.choice(numpy.arange(mat.shape[0]),size=1,p=mat[:,self.infocol['weight']],replace=True)[0]
            pr = mat[index_new,self.infocol['param0']:]
            scr = mat[index_new,self.infocol['score']]
            pr_new = self.fun_kernel(self,pr,kernel)
            #
        return pr, scr, pr_new
        #
    def function_slave(self, mpi, cmd, opt={}):
        mat = cmd['mat']
        index = cmd['index']
        init = cmd['init'] if 'init' in cmd else False
        kernel = cmd['kernel']
        eps = cmd['eps']
        #
        while True:
            pr, scr, pr_new = self.sample_param(init,kernel,mat)
            #
            if all(pr_new >= self.lower) and all(pr_new <= self.upper):
                break
        #
        if True:
                scr_new = self.score(pr_new,verbose=False)
                if self.check(scr, scr_new, eps):
                    return {
                        'index': index,
                        'score': scr_new,
                        'init': init,
                        'pr_new': pr_new
                    }
                return {
                    'index': index,
                    'score': numpy.inf,
                    'init': init,
                    'pr_new': []
                }
        #
    def print_kernel(self, kernel, iter=0):
        if self.verbose:
            print("kernel.sd: "+str(iter)+","+",".join([self.fmt.format(m) for m in numpy.diagonal(kernel)]),flush=True)
        #
    def print_mat(self, mat, eps, iter=0):
        if self.verbose:
            print("param.mat: "+(self.fmt.format(eps)+",")+("\nparam.mat: "+(self.fmt.format(eps)+",")).join([",".join([self.fmt.format(mm) for mm in m]) for m in mat]), flush=True)
        #
    def print_eps(self, eps, iter=0):
        if self.verbose:
            print("EPS:", eps, flush=True)
        #
    def print_fcount(self, fcount, trials, index):
        if self.verbose:
            print("Matrix: %d (%s / %s)" %(index,self.fmt.format(fcount),self.fmt.format(trials)), flush=True)
        #
    def calc_weights(self,mat,kernel):
        weights = numpy.ndarray(self.size,dtype=numpy.float64)
        for index in numpy.arange(len(mat)):
            pr = mat[index,self.infocol['param0']:]
            weight_scale = numpy.sum([v[self.infocol['weight']]*numpy.prod(norm.pdf(pr[self.inferpar],loc=v[self.infocol['param0']+self.inferpar],scale=numpy.diag(kernel)**0.5)) for v in mat])
            if numpy.isnan(weight_scale) or weight_scale == 0.0:
                weight = 0.0
            else:
                weight = 1.0 / weight_scale
            weights[index] = 1.0 if numpy.isinf(weight) else weight
        weights /= numpy.sum(weights)
        return weights
        #
    def calc_kernel_init(self):
        tmp = self.kernel_init if len(self.kernel_init) else numpy.repeat(1e-6,len(self.inferpar))
        kernel = numpy.diag(tmp) if len(tmp.shape)==1 else tmp
        self.print_kernel(kernel)
        return kernel
        #
    def calc_kernel_adapt(self,mat,iter,krnl,kernel_base):
        if iter == int(0) or (self.adapt == int(0)) or (iter % self.adapt > int(0)):
            return krnl
        #
        kernel = krnl.copy()
        tmp = numpy.cov(mat[:,self.infocol['param0']+self.inferpar].T)
        a = numpy.diag_indices_from(tmp)
        tmp[ tmp[a]<kernel_base[a], tmp[a]<kernel_base[a] ] = kernel_base[ tmp[a]<kernel_base[a], tmp[a]<kernel_base[a] ]
        kernel = tmp.copy()
        #
        self.print_kernel(kernel,iter)
        return kernel
        #
    def calc_mat_init(self,mpi,eps):
        mat = numpy.ndarray((self.size,self.infocol['param0']+self.parsize),dtype=numpy.float64)
        return self.take_a_step(mpi,mat,0,[],eps)
        #
    def calc_get_eps(self):
        eps = None
        if len(self.epsseq):
            eps = self.epsseq.pop(0)
            self.print_eps(eps)
        return eps
        #
    def take_a_step(self,mpi,mat,iter,kernel,eps):
        mat_new = numpy.zeros(mat.shape,dtype=numpy.float64)
        #
        jobs = [{
            'mat': mat,
            'index': index,
            'init': (iter == 0),
            'kernel': kernel,
            'eps': eps
        } for index in numpy.arange(self.size)]
        #
        while len(jobs) > 0:
            ret = mpi.exec(jobs,multiple=True)
            jobs = []
            for elm in ret:
                mat_new[elm['index'],self.infocol['trials']] += 1
                trials = mat_new[elm['index'],self.infocol['trials']]
                if len(elm['pr_new']):
                    fcount = mat_new[elm['index'],self.infocol['fcount']]
                    mat_new[elm['index'],:] = [
                        iter,
                        1.0,
                        fcount,
                        trials,
                        elm['score']
                        ] + elm['pr_new'].tolist()
                    self.print_fcount(fcount,trials,elm['index'])
                else:
                    mat_new[elm['index'],self.infocol['fcount']] += 1
                    if ((mat_new[elm['index'],self.infocol['weight']] == 0) and
                        (len([j for j in jobs if j['index']==elm['index']]) == 0)):
                        ntrial = min(MPI_SIZE, int(trials)*2)
                        for n in range(ntrial):
                            jobs.append({
                                'mat': mat,
                                'index': elm['index'],
                                'init': (iter == 0),
                                'kernel': kernel,
                                'eps': eps
                            })
        #
        mat_new[:,self.infocol['weight']] = 1.0/self.size if iter == 0 else self.calc_weights(mat,kernel)
        mat = mat_new.copy()
        return mat
        #
    def function_master(self, mpi, opt={}):
        eps = self.calc_get_eps()
        #
        mat = self.calc_mat_init(mpi,eps)
        #
        kernel = self.calc_kernel_init()
        kernel_base = numpy.array(kernel).copy()
        #
        iter = 0
        self.print_mat(mat,eps,iter)
        #
        while True:
            iter += 1
            #
            mat = self.take_a_step(mpi,mat,iter,kernel,eps)
            #
            kernel = self.calc_kernel_adapt(mat,iter,kernel,kernel_base)
            #
            self.print_mat(mat,eps,iter)
            #
            if (iter % self.niter) == 0:
                eps = self.calc_get_eps()
                if eps == None:
                    break
            #
        self.result = mat
        return mat

class hoppMCMC:
    def __init__(self,
                 pargen,
                 score,
                 epsseq,
                 lower,
                 upper,
                 size,
                 niter,
                 adapt,
                 inferpar=[],
                 kernel=[],
                 fmt="{:g}",
                 fun_kernel=None,
                 mcmc=-1,
                 verbose=True) -> None:
        '''
        pargen: function to generate parameter samples from the prior
        score: distance function
        epsseq: a list of thresholds for each niter steps will be taken
        lower - upper: lower and upper bounds of parameter values (independent of infervar)
        size: individual parameter sets to sample
        niter: number of iterations per eps
        adapt: number of steps before the kernel adapts
        inferpar: indices of parameters to infer
        fmt: output format string (default: {:g})
        fun_kernel: (optional) kernel in the form of a function(abc,param0,kernel)
        mcmc: (optional) positive values indicate MCMC-like fitness checking with mcmc being a scale factor
        '''
        self.pargen = pargen
        self.parsize = len(self.pargen())
        self.inferpar = numpy.array(inferpar)
        if len(self.inferpar)==0:
            self.inferpar = numpy.arange(self.parsize)
        self.infersize = len(self.inferpar)
        #
        self.fmt = fmt
        #
        self.fun_kernel = self.inferABC_kernel
        if fun_kernel != None:
            self.fun_kernel = fun_kernel
        #
        self.mcmc = mcmc
        self.check = self.checkMove if mcmc < 0.0 else self.checkMoveMCMC
        #
        self.infocol = {
            'iter': 0,
            'weight': 1,
            'fcount': 2,
            'trials': 3,
            'score': 4,
            'param0': 5
        }
        #
        self.kernel_init = numpy.array(kernel) if len(kernel) else []
        #
        self.score = score
        self.size = size
        self.lower = lower
        self.upper = upper
        self.niter = niter
        self.adapt = int(adapt)
        self.verbose = verbose
        #
        self.epsseq = epsseq
        #
        self.result = None
        #
        mpi = mympi.mpi(self.function_master,self.function_slave)
        mpi.clean()
        #
    def inferABC_kernel(self,abc,param0,kernel):
        param1 = param0.copy()
        param1[abc.inferpar] = numpy.random.multivariate_normal(mean = param1[abc.inferpar], cov = kernel)
        return param1
        #
    def checkMove(self,f0,f1,eps):
        return not numpy.isnan(f1) and not numpy.isinf(f1) and (f1 >= 0) and (f1 < eps)
        #
    def checkMoveMCMC(self,f0,f1,eps):
        return not numpy.isnan(f1) and not numpy.isinf(f1) and (f1 >= 0) and (f1 < eps) and ((f1 < f0) or (numpy.log(numpy.random.random()) < (f0-f1)/self.mcmc))
        #
    def sample_param(self,init,kernel,mat):
        if init:
            scr = numpy.inf
            pr = pr_new = numpy.array(self.pargen())
        else:
            index_new = numpy.random.choice(numpy.arange(mat.shape[0]),size=1,p=mat[:,self.infocol['weight']],replace=True)[0]
            pr = mat[index_new,self.infocol['param0']:]
            scr = mat[index_new,self.infocol['score']]
            pr_new = self.fun_kernel(self,pr,kernel)
            #
        return pr, scr, pr_new
        #
    def function_slave(self, mpi, cmd, opt={}):
        mat = cmd['mat']
        index = cmd['index']
        init = cmd['init'] if 'init' in cmd else False
        kernel = cmd['kernel']
        eps = cmd['eps']
        #
        while True:
            pr, scr, pr_new = self.sample_param(init,kernel,mat)
            #
            if all(pr_new >= self.lower) and all(pr_new <= self.upper):
                break
        #
        if True:
                scr_new = self.score(pr_new,verbose=False)
                if self.check(scr, scr_new, eps):
                    return {
                        'index': index,
                        'score': scr_new,
                        'init': init,
                        'pr_new': pr_new
                    }
                return {
                    'index': index,
                    'score': numpy.inf,
                    'init': init,
                    'pr_new': []
                }
        #
    def print_kernel(self, kernel, iter=0):
        if self.verbose:
            print("kernel.sd: "+str(iter)+","+",".join([self.fmt.format(m) for m in numpy.diagonal(kernel)]),flush=True)
        #
    def print_mat(self, mat, eps, iter=0):
        if self.verbose:
            print("param.mat: "+(self.fmt.format(eps)+",")+("\nparam.mat: "+(self.fmt.format(eps)+",")).join([",".join([self.fmt.format(mm) for mm in m]) for m in mat]), flush=True)
        #
    def print_eps(self, eps, iter=0):
        if self.verbose:
            print("EPS:", eps, flush=True)
        #
    def print_fcount(self, fcount, trials, index):
        if self.verbose:
            print("Matrix: %d (%s / %s)" %(index,self.fmt.format(fcount),self.fmt.format(trials)), flush=True)
        #
    def calc_weights(self,mat,kernel):
        weights = numpy.ndarray(self.size,dtype=numpy.float64)
        for index in numpy.arange(len(mat)):
            pr = mat[index,self.infocol['param0']:]
            weight_scale = numpy.sum([v[self.infocol['weight']]*numpy.prod(norm.pdf(pr[self.inferpar],loc=v[self.infocol['param0']+self.inferpar],scale=numpy.diag(kernel)**0.5)) for v in mat])
            if numpy.isnan(weight_scale) or weight_scale == 0.0:
                weight = 0.0
            else:
                weight = 1.0 / weight_scale
            weights[index] = 1.0 if numpy.isinf(weight) else weight
        weights /= numpy.sum(weights)
        return weights
        #
    def calc_kernel_init(self):
        tmp = self.kernel_init if len(self.kernel_init) else numpy.repeat(1e-6,len(self.inferpar))
        kernel = numpy.diag(tmp) if len(tmp.shape)==1 else tmp
        self.print_kernel(kernel)
        return kernel
        #
    def calc_kernel_adapt(self,mat,iter,krnl,kernel_base):
        if iter == int(0) or (self.adapt == int(0)) or (iter % self.adapt > int(0)):
            return krnl
        #
        kernel = krnl.copy()
        tmp = numpy.cov(mat[:,self.infocol['param0']+self.inferpar].T)
        a = numpy.diag_indices_from(tmp)
        tmp[ tmp[a]<kernel_base[a], tmp[a]<kernel_base[a] ] = kernel_base[ tmp[a]<kernel_base[a], tmp[a]<kernel_base[a] ]
        kernel = tmp.copy()
        #
        self.print_kernel(kernel,iter)
        return kernel
        #
    def calc_mat_init(self,mpi,eps):
        mat = numpy.ndarray((self.size,self.infocol['param0']+self.parsize),dtype=numpy.float64)
        return self.take_a_step(mpi,mat,0,[],eps)
        #
    def calc_get_eps(self):
        eps = None
        if len(self.epsseq):
            eps = self.epsseq.pop(0)
            self.print_eps(eps)
        return eps
        #
    def take_a_step(self,mpi,mat,iter,kernel,eps):
        mat_new = numpy.zeros(mat.shape,dtype=numpy.float64)
        #
        jobs = [{
            'mat': mat,
            'index': index,
            'init': (iter == 0),
            'kernel': kernel,
            'eps': eps
        } for index in numpy.arange(self.size)]
        #
        while len(jobs) > 0:
            ret = mpi.exec(jobs,multiple=True)
            jobs = []
            for elm in ret:
                mat_new[elm['index'],self.infocol['trials']] += 1
                trials = mat_new[elm['index'],self.infocol['trials']]
                if len(elm['pr_new']):
                    fcount = mat_new[elm['index'],self.infocol['fcount']]
                    mat_new[elm['index'],:] = [
                        iter,
                        1.0,
                        fcount,
                        trials,
                        elm['score']
                        ] + elm['pr_new'].tolist()
                    self.print_fcount(fcount,trials,elm['index'])
                else:
                    mat_new[elm['index'],self.infocol['fcount']] += 1
                    if ((mat_new[elm['index'],self.infocol['weight']] == 0) and
                        (len([j for j in jobs if j['index']==elm['index']]) == 0)):
                        ntrial = min(MPI_SIZE, int(trials)*2)
                        for n in range(ntrial):
                            jobs.append({
                                'mat': mat,
                                'index': elm['index'],
                                'init': (iter == 0),
                                'kernel': kernel,
                                'eps': eps
                            })
        #
        mat_new[:,self.infocol['weight']] = 1.0/self.size if iter == 0 else self.calc_weights(mat,kernel)
        mat = mat_new.copy()
        return mat
        #
    def function_master(self, mpi, opt={}):
        eps = self.calc_get_eps()
        #
        mat = self.calc_mat_init(mpi,eps)
        #
        kernel = self.calc_kernel_init()
        kernel_base = numpy.array(kernel).copy()
        #
        iter = 0
        self.print_mat(mat,eps,iter)
        #
        while True:
            iter += 1
            #
            mat = self.take_a_step(mpi,mat,iter,kernel,eps)
            #
            kernel = self.calc_kernel_adapt(mat,iter,kernel,kernel_base)
            #
            self.print_mat(mat,eps,iter)
            #
            if (iter % self.niter) == 0:
                eps = self.calc_get_eps()
                if eps == None:
                    break
            #
        self.result = mat
        return mat
