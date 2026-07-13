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

    Each particle contains both a model identity and a parameter vector.
    The total population is shared across models, so models compete for
    particles during inference.

    Parameters
    ----------
    models : list of ABCModel
        Models to compare.

    epsseq : list of float
        Sequence of ABC thresholds.

    size : int
        Total population size across all models, not per model.

    niter : int
        Number of ABC-SMC generations per threshold.

    adapt : int
        Frequency of model-specific kernel adaptation. If zero, kernels are
        not adapted.

    fmt : str, default="{:g}"
        Format string for printed numerical output.

    jitter : float, default=1e-8
        Small variance floor used to avoid zero or singular kernels.

    mcmc : float, default=-1
        If negative, hard ABC acceptance is used. If positive, an MCMC-like
        acceptance check is used.

    retain : bool, default=False
        If True, keep all generations in `results`.

    kernel_mode : {"diagonal", "full"}, default="diagonal"
        Determines how kernels are used for proposal and weight calculation.

        "diagonal":
            Only the diagonal of the kernel is used. This is closest to the
            original `inferABC` behaviour.

        "full":
            The full covariance matrix is used for proposal and density
            calculation.

    verbose : bool, default=True
        Print progress information.

    Attributes
    ----------
    result : numpy.ndarray
        Final particle matrix.

    results : list
        All retained generations if `retain=True`.

    model_names : list of str
        Model names.

    model_priors : numpy.ndarray
        Normalised prior model probabilities.

    parsizes : list of int
        Number of parameters for each model.

    infocol : dict
        Column indices for the particle matrix.
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
                 kernel_mode="diagonal",
                 verbose=True) -> None:

        self.models = models
        self.nmodels = len(models)

        if self.nmodels == 0:
            raise ValueError("At least one model must be supplied.")

        self.model_names = [m.name for m in self.models]

        model_priors = numpy.array([m.prior for m in self.models], dtype=float)

        if numpy.any(model_priors < 0):
            raise ValueError("Model prior probabilities must be non-negative.")

        if numpy.sum(model_priors) <= 0:
            raise ValueError("At least one model prior must be positive.")

        self.model_priors = model_priors / numpy.sum(model_priors)

        self.parsizes = []
        for m in self.models:
            pr = numpy.asarray(m.pargen(), dtype=float)
            self.parsizes.append(len(pr))

        self.max_parsize = max(self.parsizes)

        self.inferpars = []
        for model_id, m in enumerate(self.models):
            if m.inferpar is None:
                self.inferpars.append(numpy.arange(self.parsizes[model_id]))
            else:
                self.inferpars.append(numpy.asarray(m.inferpar, dtype=int))

        self.infersizes = [len(x) for x in self.inferpars]

        self.fmt = fmt
        self.jitter = float(jitter)
        self.mcmc = mcmc
        self.retain = retain
        self.verbose = verbose

        if kernel_mode not in ["diagonal", "full"]:
            raise ValueError("kernel_mode must be either 'diagonal' or 'full'.")

        self.kernel_mode = kernel_mode

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

        self.epsseq = list(epsseq)

        self.result = None
        self.results = []

        mpi = mympi.mpi(self.function_master, self.function_slave)
        mpi.clean()

    # ------------------------------------------------------------------
    # Kernel handling
    # ------------------------------------------------------------------

    def prepare_kernel(self, model_id, kernel):
        """
        Convert a scalar, diagonal vector, or matrix into a valid covariance
        matrix consistent with self.kernel_mode.
        """

        inferpar = self.inferpars[model_id]
        d = len(inferpar)

        kernel = numpy.asarray(kernel, dtype=float)

        if kernel.ndim == 0:
            kernel = numpy.eye(d) * float(kernel)

        elif kernel.ndim == 1:
            if len(kernel) != d:
                raise ValueError(
                    f"Kernel length mismatch for model {model_id} "
                    f"({self.model_names[model_id]}). "
                    f"Expected {d}, got {len(kernel)}."
                )
            kernel = numpy.diag(kernel)

        elif kernel.ndim == 2:
            if kernel.shape != (d, d):
                raise ValueError(
                    f"Kernel shape mismatch for model {model_id} "
                    f"({self.model_names[model_id]}). "
                    f"Expected {(d, d)}, got {kernel.shape}."
                )
            kernel = kernel.copy()

        else:
            raise ValueError(
                f"Kernel for model {model_id} ({self.model_names[model_id]}) "
                f"must be scalar, one-dimensional, or two-dimensional."
            )

        if self.kernel_mode == "diagonal":
            var = numpy.diag(kernel)
            var = numpy.maximum(var, self.jitter)
            kernel = numpy.diag(var)

        else:
            kernel = 0.5 * (kernel + kernel.T)

            eigvals, eigvecs = numpy.linalg.eigh(kernel)

            max_eig = numpy.max(eigvals)
            eig_floor = max(self.jitter, self.jitter * max(1.0, max_eig))

            eigvals = numpy.maximum(eigvals, eig_floor)

            kernel = eigvecs @ numpy.diag(eigvals) @ eigvecs.T
            kernel = 0.5 * (kernel + kernel.T)

        return kernel

    def inferABC_kernel(self, abc, model_id, param0, kernel):
        """
        Default proposal kernel.

        The proposal distribution is consistent with `kernel_mode`.
        """

        inferpar = abc.inferpars[model_id]

        param1 = param0.copy()
        kernel = abc.prepare_kernel(model_id, kernel)

        param1[inferpar] = numpy.random.multivariate_normal(
            mean=param1[inferpar],
            cov=kernel
        )

        return param1

    def log_kernel_density(self, model_id, x, mean, kernel):
        """
        Log density of the proposal kernel.

        This must match how proposals are generated in `inferABC_kernel`.
        """

        inferpar = self.inferpars[model_id]

        x_inf = numpy.asarray(x[inferpar], dtype=float)
        mean_inf = numpy.asarray(mean[inferpar], dtype=float)

        kernel = self.prepare_kernel(model_id, kernel)

        if self.kernel_mode == "diagonal":
            var = numpy.diag(kernel)
            var = numpy.maximum(var, self.jitter)

            return numpy.sum(
                norm.logpdf(
                    x_inf,
                    loc=mean_inf,
                    scale=numpy.sqrt(var)
                )
            )

        return multivariate_normal.logpdf(
            x_inf,
            mean=mean_inf,
            cov=kernel,
            allow_singular=False
        )

    # ------------------------------------------------------------------
    # Acceptance checks
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Parameter, prior, and score helpers
    # ------------------------------------------------------------------

    def get_param(self, row, model_id):
        p0 = self.infocol["param0"]
        psize = self.parsizes[model_id]
        return numpy.asarray(row[p0:p0 + psize], dtype=float)

    def in_bounds(self, model_id, pr):
        m = self.models[model_id]
        return numpy.all(pr >= m.lower) and numpy.all(pr <= m.upper)

    def logprior(self, model_id, pr):
        """
        Parameter log-prior density.

        If the model supplies `logprior`, use it. Otherwise assume
        independent uniform priors between lower and upper bounds for the
        inferred parameters.
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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_model_from_prior(self):
        return numpy.random.choice(
            numpy.arange(self.nmodels),
            p=self.model_priors
        )

    def sample_param(self, init, kernels, mat):
        if init:
            model_id = self.sample_model_from_prior()
            pr = numpy.asarray(self.models[model_id].pargen(), dtype=float)
            scr = numpy.inf
            pr_new = pr.copy()

        else:
            index_new = numpy.random.choice(
                numpy.arange(mat.shape[0]),
                size=1,
                p=mat[:, self.infocol["weight"]],
                replace=True
            )[0]

            model_id = int(mat[index_new, self.infocol["model"]])
            pr = self.get_param(mat[index_new, :], model_id)
            scr = mat[index_new, self.infocol["score"]]

            m = self.models[model_id]

            if m.fun_kernel is None:
                pr_new = self.inferABC_kernel(
                    self,
                    model_id,
                    pr,
                    kernels[model_id]
                )
            else:
                pr_new = m.fun_kernel(
                    self,
                    model_id,
                    pr,
                    kernels[model_id]
                )

        return model_id, pr, scr, pr_new

    # ------------------------------------------------------------------
    # MPI slave
    # ------------------------------------------------------------------

    def function_slave(self, mpi, cmd, opt={}):
        mat = cmd["mat"]
        index = cmd["index"]
        init = cmd["init"] if "init" in cmd else False
        kernels = cmd["kernels"]
        eps = cmd["eps"]

        while True:
            model_id, pr, scr, pr_new = self.sample_param(
                init,
                kernels,
                mat
            )

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

    # ------------------------------------------------------------------
    # Printing helpers
    # ------------------------------------------------------------------

    def print_kernel(self, model_id, kernel, iter=0):
        if self.verbose:
            print(
                "kernel.var: "
                + str(iter)
                + ","
                + self.model_names[model_id]
                + ","
                + ",".join([
                    self.fmt.format(m)
                    for m in numpy.diagonal(kernel)
                ]),
                flush=True
            )

    def print_mat(self, mat, eps, iter=0):
        if self.verbose:
            prefix = "param.mat: " + self.fmt.format(eps) + ","

            print(
                prefix
                + ("\n" + prefix).join([
                    ",".join([
                        self.fmt.format(mm) if numpy.isfinite(mm) else "nan"
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

    # ------------------------------------------------------------------
    # Kernels and thresholds
    # ------------------------------------------------------------------

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
            kernel = self.prepare_kernel(model_id, kernel)

            kernels.append(kernel)

            self.print_kernel(model_id, kernel)

        return kernels

    def calc_kernel_adapt(self, mat, iter, kernels, kernel_base):
        if iter == 0 or self.adapt == 0 or iter % self.adapt > 0:
            return kernels

        new_kernels = [k.copy() for k in kernels]

        for model_id in range(self.nmodels):
            mask = mat[:, self.infocol["model"]] == model_id

            if numpy.sum(mask) <= self.infersizes[model_id] + 1:
                continue

            inferpar = self.inferpars[model_id]
            cols = self.infocol["param0"] + inferpar

            x = mat[mask, :][:, cols]

            tmp = numpy.cov(x.T)

            if tmp.ndim == 0:
                tmp = numpy.array([[tmp]])

            if self.kernel_mode == "diagonal":
                tmp = numpy.diag(numpy.diag(tmp))

            base = kernel_base[model_id]

            if self.kernel_mode == "diagonal":
                base = numpy.diag(numpy.diag(base))

            a = numpy.diag_indices_from(tmp)

            tmp[a] = numpy.maximum(tmp[a], base[a])
            tmp[a] = numpy.maximum(tmp[a], self.jitter)

            tmp = self.prepare_kernel(model_id, tmp)

            new_kernels[model_id] = tmp
            self.print_kernel(model_id, tmp, iter)

        return new_kernels

    def calc_get_eps(self):
        eps = None

        if len(self.epsseq):
            eps = self.epsseq.pop(0)
            self.print_eps(eps)

        return eps

    # ------------------------------------------------------------------
    # Matrix initialisation
    # ------------------------------------------------------------------

    def calc_mat_init(self, mpi, eps):
        ncol = self.infocol["param0"] + self.max_parsize

        mat = numpy.full(
            (self.size, ncol),
            numpy.nan,
            dtype=numpy.float64
        )

        return self.take_a_step(mpi, mat, 0, [], eps)

    # ------------------------------------------------------------------
    # Weight calculation
    # ------------------------------------------------------------------

    def calc_weights(self, mat_new, mat_old, kernels):
        """
        ABC-SMC weights for the new particle population.

        For each new particle:

            weight ∝ model_prior × parameter_prior_density
                     / proposal_density

        The proposal density is the mixture of previous particles of the same
        model, using the previous generation weights and the same kernel used
        to generate the new particle.
        """

        logw = numpy.full(self.size, -numpy.inf, dtype=float)

        old_weights = mat_old[:, self.infocol["weight"]]

        for i in range(self.size):
            model_id = int(mat_new[i, self.infocol["model"]])
            pr_new = self.get_param(mat_new[i, :], model_id)

            log_model_prior = numpy.log(self.model_priors[model_id])
            log_param_prior = self.logprior(model_id, pr_new)

            numerator = log_model_prior + log_param_prior

            if not numpy.isfinite(numerator):
                logw[i] = -numpy.inf
                continue

            terms = []

            for j in range(mat_old.shape[0]):
                old_model_id = int(mat_old[j, self.infocol["model"]])

                if old_model_id != model_id:
                    continue

                if old_weights[j] <= 0:
                    continue

                pr_old = self.get_param(mat_old[j, :], old_model_id)

                lk = self.log_kernel_density(
                    model_id,
                    pr_new,
                    pr_old,
                    kernels[model_id]
                )

                if numpy.isfinite(lk):
                    terms.append(numpy.log(old_weights[j]) + lk)

            if len(terms) == 0:
                logw[i] = -numpy.inf
            else:
                denominator = logsumexp(terms)
                logw[i] = numerator - denominator

        normaliser = logsumexp(logw)

        if not numpy.isfinite(normaliser):
            raise ValueError(
                "All particle weights are zero or non-finite. "
                "Check kernels, priors, model particle counts, and bounds."
            )

        weights = numpy.exp(logw - normaliser)

        return weights

    # ------------------------------------------------------------------
    # One ABC-SMC generation
    # ------------------------------------------------------------------

    def take_a_step(self, mpi, mat, iter, kernels, eps):
        ncol = self.infocol["param0"] + self.max_parsize

        mat_new = numpy.full(
            (self.size, ncol),
            numpy.nan,
            dtype=numpy.float64
        )

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

        mpi_size = getattr(mympi, "MPI_SIZE", 1)

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
                    p0 = self.infocol["param0"]

                    fcount = mat_new[idx, self.infocol["fcount"]]

                    mat_new[idx, self.infocol["iter"]] = iter
                    mat_new[idx, self.infocol["model"]] = model_id
                    mat_new[idx, self.infocol["weight"]] = 1.0
                    mat_new[idx, self.infocol["fcount"]] = fcount
                    mat_new[idx, self.infocol["trials"]] = trials
                    mat_new[idx, self.infocol["score"]] = elm["score"]

                    mat_new[idx, p0:p0 + psize] = pr_new

                    self.print_fcount(fcount, trials, idx)

                else:
                    mat_new[idx, self.infocol["fcount"]] += 1

                    if mat_new[idx, self.infocol["weight"]] == 0:
                        ntrial = min(mpi_size, int(trials) * 2)

                        for _ in range(ntrial):
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
            mat_new[:, self.infocol["weight"]] = self.calc_weights(
                mat_new,
                mat,
                kernels
            )

        return mat_new

    # ------------------------------------------------------------------
    # MPI master
    # ------------------------------------------------------------------

    def function_master(self, mpi, opt={}):
        eps = self.calc_get_eps()

        mat = self.calc_mat_init(mpi, eps)

        kernels = self.calc_kernel_init()
        kernel_base = [k.copy() for k in kernels]

        iter = 0

        self.print_mat(mat, eps, iter)

        if self.retain:
            self.results.append(mat.copy())

        while True:
            iter += 1

            mat = self.take_a_step(mpi, mat, iter, kernels, eps)

            kernels = self.calc_kernel_adapt(
                mat,
                iter,
                kernels,
                kernel_base
            )

            self.print_mat(mat, eps, iter)

            if self.retain:
                self.results.append(mat.copy())

            if iter % self.niter == 0:
                eps = self.calc_get_eps()

                if eps is None:
                    break

        self.result = mat

        return mat

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def model_posterior(self, mat=None):
        """
        Summarise posterior model probabilities.

        The posterior probability of a model is calculated as the sum of final
        particle weights assigned to that model.
        """

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

        return out.sort_values(
            "posterior_probability",
            ascending=False
        ).reset_index(drop=True)

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
