import numpy
import mympi
from scipy.stats import norm
from scipy.stats import multivariate_normal

from mpi4py import MPI
MPI_SIZE = MPI.COMM_WORLD.Get_size()
MPI_RANK = MPI.COMM_WORLD.Get_rank()
MPI_MASTER = 0
worker_indices = numpy.delete(numpy.arange(MPI_SIZE),MPI_MASTER)
print("Process %d of %d is running" %(MPI_RANK,MPI_SIZE),flush=True)

ALGO_HOPP = 0
ALGO_ABC = 1

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
