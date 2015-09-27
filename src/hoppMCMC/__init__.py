##########################################################################
## hoppMCMC: adaptive basin-hopping Markov-chain Monte Carlo            ##
##           for Bayesian optimisation                                  ##
##                                                                      ##
## Copyright (C) 2015 Kamil Erguler (k.erguler@cyi.ac.cy)               ##
##                                                                      ##
## This program is free software: you can redistribute it and/or modify ##
## it under the terms of the GNU General Public License version 3 as    ##
## published by the Free Software Foundation.                           ##
##                                                                      ##
## This program is distributed in the hope that it will be useful,      ##
## but WITHOUT ANY WARRANTY; without even the implied warranty of       ##
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the         ##
## GNU General Public License, <http://www.gnu.org/licenses/>,          ##
## for more details.                                                    ##
##                                                                      ##
## Author: Kamil Erguler (k.erguler@cyi.ac.cy)                          ##
## Dates: Aug 5, 2015;                                                  ##
## Organisation: The Cyprus Institute                                   ##
##########################################################################

import os
import sys
import numpy
from struct import *
from scipy.stats import ttest_1samp as ttest

MPI_MASTER = 0
try:
    from mpi4py import MPI
    MPI_SIZE = MPI.COMM_WORLD.Get_size()
    MPI_RANK = MPI.COMM_WORLD.Get_rank()
    def Abort(str):
        print("ERROR: "+str)
        MPI.COMM_WORLD.Abort(1)
except:
    MPI_SIZE = 1
    MPI_RANK = 0
    def Abort(str):
        raise errorMCMC(str)

EPS_PULSE_MIN = 1e-7
EPS_PULSE_MAX = 1e7
EPS_VARMAT_MIN = 1e-7
EPS_VARMAT_MAX = 1e7

class errorMCMC(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class binfile():
    def __init__(self,fname,mode,rowsize=1):
        self.headsize = 4
        self.bitsize = 8
        self.fname = fname
        self.mode = mode
        self.rowsize = rowsize
        if self.mode=='r':
            try:
                self.f = open(self.fname,"rb")
            except IOError:
                Abort("File not found: "+self.fname)
            self.rowsize = unpack('<i',self.f.read(self.headsize))[0]
        elif self.mode=='w':
            try:
                self.f = open(self.fname,"w+b")
            except IOError:
                Abort("File not found: "+self.fname)
            tmp = self.f.read(self.headsize)
            if not tmp:
                self.f.write(pack('<i',self.rowsize))
            else:
                self.rowsize = unpack('<i',tmp)[0]
        else:
            Abort("Wrong i/o mode: "+self.mode)
        self.fmt = '<'+'d'*self.rowsize
        self.size = self.bitsize*self.rowsize
        print "Success: %s opened for %s size %d double rows" %(self.fname,"reading" if self.mode=='r' else "writing",self.rowsize)
    # ---
    def writeRow(self,row):
        self.f.write(pack(self.fmt,*row))
        self.f.flush()
    # ---
    def readRows(self):
        self.f.seek(0,os.SEEK_END)
        filesize = self.f.tell()
        self.f.seek(self.headsize)
        tmp=self.f.read(filesize-self.headsize)
        ret=numpy.array(unpack('<'+'d'*((filesize-self.headsize)/self.bitsize),tmp),dtype=numpy.float64).reshape((((filesize-self.headsize)/self.bitsize)/self.rowsize,self.rowsize))
        return ret
    # ---
    def close(self):
        self.f.close()
        print "Success: %s closed" %(self.fname if self.fname else "file")

class binstream():
    def __init__(self,filestream,rowsize=1):
        self.headsize = 4
        self.bitsize = 8
        self.f = filestream
        self.rowsize = rowsize
        self.f.seek(0)
        tmp = self.f.read(self.headsize)
        if not tmp:
            self.f.write(pack('<i',self.rowsize))
        else:
            self.rowsize = unpack('<i',tmp)[0]
        self.fmt = '<'+'d'*self.rowsize
        self.size = self.bitsize*self.rowsize
        print "Success: Connected to stream for size %d double rows" %(self.rowsize)
    # ---
    def writeRow(self,row):
        self.f.write(pack(self.fmt,*row))
        self.f.flush()
    # ---
    def readRows(self):
        self.f.seek(0,os.SEEK_END)
        filesize = self.f.tell()
        self.f.seek(self.headsize)
        tmp=self.f.read(filesize-self.headsize)
        ret=numpy.array(unpack('<'+'d'*((filesize-self.headsize)/self.bitsize),tmp),dtype=numpy.float64).reshape((((filesize-self.headsize)/self.bitsize)/self.rowsize,self.rowsize))
        return ret

def readFile(filename):
    a=binfile(filename,"r")
    b=a.readRows()
    a.close()
    return b
    
def parMin(parmat):
        prm = min(parmat[:,0])
        prmi = numpy.where(parmat[:,0]==prm)[0][0]
        return {'i':prmi,'f':prm}

def diagdot(mat,vec):
    for n in numpy.arange(mat.shape[0]):
        mat[n,n] *= vec[n]

def covariance(mat):
    return numpy.cov(mat,rowvar=False)

def coefVar(mat):
    mean0 = 1.0/numpy.mean(mat,0)
    return mean0*(numpy.cov(mat,rowvar=False).T*mean0).T

def sensVar(mat):
    mean0 = numpy.mean(mat,0)
    return mean0*(numpy.linalg.inv(numpy.cov(mat,rowvar=False)).T*mean0).T

cov = covariance
rnorm = numpy.random.multivariate_normal
determinant = numpy.linalg.det

def join(a,s):
    return s.join(["%.16g" %(x) for x in a])

def calcAUC(f,parmat,T):
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(parmat.T)
    except:
        print 'Warning: Problem encountered in calcAUC!'
        return {'res':None,'mean':0}
    wt = numpy.array([kde.evaluate(pr) for pr in parmat])
    wt /= numpy.min(wt)
    # Importance sampling for Monte Carlo integration:
    res = numpy.array([numpy.exp(-f[m]/T) / wt[m] for m in range(len(f))])
    return {'res':res,'mean':numpy.mean(res)}

def compareAUCs(parmats,groups,T):
    from scipy.stats import gaussian_kde
    ids = numpy.unique(groups)
    parmats = parmats[:,numpy.var(parmats,axis=0)!=0]
    parmat_list = [parmats[groups==n,:] for n in ids]
    try:
        kde = gaussian_kde(parmats[:,1:].T)
    except:
        print 'Warning: Problem encountered in compareAUC!'
        return {ids[g]:0 for g in range(len(ids))}
    wt_list = numpy.array([[kde.evaluate(pr) for pr in parmat[:,1:]] for parmat in parmat_list])
    mn = numpy.min(wt_list)
    wt_list /= mn
    # Importance sampling for Monte Carlo integration:
    favg_list = numpy.array([numpy.mean([numpy.exp(-parmat_list[n][m,0]/T) / wt_list[n][m] for m in range(parmat_list[n].shape[0])]) for n in range(len(parmat_list))])
    favg_list /= numpy.sum(favg_list)
    return {ids[g]:favg_list[g] for g in range(len(ids))}

def compareAUC(parmat0,parmat1,T):
    from scipy.stats import gaussian_kde
    parmats = numpy.vstack((parmat0,parmat1))
    parmat0cp = parmat0[:,numpy.var(parmats,axis=0)!=0].copy()
    parmat1cp = parmat1[:,numpy.var(parmats,axis=0)!=0].copy()
    parmats = parmats[:,numpy.var(parmats,axis=0)!=0]
    try:
        kde = gaussian_kde(parmats[:,1:].T)
    except:
        print 'Warning: Problem encountered in compareAUC!'
        return {'acc':0, 'favg0':0, 'favg1':0}
    wt0 = numpy.array([kde.evaluate(pr) for pr in parmat0cp[:,1:]])
    wt1 = numpy.array([kde.evaluate(pr) for pr in parmat1cp[:,1:]])
    mn = numpy.min([wt0,wt1])
    wt0 /= mn
    wt1 /= mn
    # Importance sampling for Monte Carlo integration:
    favg0 = numpy.mean([numpy.exp(-parmat0cp[m,0]/T) / wt0[m] for m in range(parmat0cp.shape[0])])
    favg1 = numpy.mean([numpy.exp(-parmat1cp[m,0]/T) / wt1[m] for m in range(parmat1cp.shape[0])])
    acc = (not numpy.isnan(favg0) and not numpy.isnan(favg1) and
            favg0 > 0 and favg1 >= 0 and
            (favg1 >= favg0 or
            (numpy.random.uniform() < (favg1/favg0))))
    return {'acc':acc, 'favg0':favg0, 'favg1':favg1}

def anneal_exp(y0,y1,steps):
    return y0*numpy.exp(-(numpy.arange(steps,dtype=numpy.float64)/(steps-1.0))*numpy.log(numpy.float64(y0)/y1))

def anneal_linear(y0,y1,steps):
    return numpy.append(numpy.arange(y0,y1,(y1-y0)/(steps-1),dtype=numpy.float64),y1)

def anneal_sigma(y0,y1,steps):
    return y0+(y1-y0)*(1.0-1.0/(1.0+numpy.exp(-(numpy.arange(steps)-(0.5*steps)))))

def anneal_sigmasoft(y0,y1,steps):
    return y0+(y1-y0)*(1.0-1.0/(1.0+numpy.exp(-12.5*(numpy.arange(steps)-(0.5*steps))/steps)))

def ldet(mat):
    try:
        det = determinant(mat)
    except:
        return -numpy.Inf
    if det==0:
        return -numpy.Inf
    else:
        return numpy.log(det)

def finalTest(fitFun,param,testnum=10):
    for n in range(testnum):
        f = fitFun(param)
        if not numpy.isnan(f) and not numpy.isinf(f):
            return f
    Abort("Incompatible parameter set (%g): %s" %(f,join(param,",")))
    
# --- Class: hoppMCMC
class hoppMCMC:
    def __init__(self,fitFun,param,varmat,gibbs=False,anneal=1,num_hopp=1,num_adapt=10,num_chain=10,chain_length=100,rangeT=[1,1000],model_comp=1.0,method='posterior',param_reset=False,inferpar=[],print_iter=0,print_step=0,label='',outfilename=''):
        """
        Adaptive Basin-Hopping MCMC Algorithm
        
        ***** args *****

        fitFun:
              objective function which takes a numpy array as the only argument

        param:
              
        """
        self.multi = {'cov': covariance,
                      'rnorm': numpy.random.multivariate_normal,
                      'det': numpy.linalg.det}
        self.single = {'cov': covariance,
                       'rnorm': numpy.random.normal,
                       'det': abs}
        self.stat = self.multi
        self.gibbs = gibbs
        self.num_hopp = num_hopp
        self.num_adapt = num_adapt
        self.num_chain = num_chain
        self.chain_length = chain_length
        self.anneal = numpy.array(anneal,dtype=numpy.float64,ndmin=1)
        self.rangeT = numpy.sort(rangeT)
        self.model_comp = model_comp
        self.param_reset = param_reset
        self.method = method
        if self.method not in ["posterior","anneal"]:
            if MPI_RANK==MPI_MASTER:
                Abort("Invalid method: %s" %(self.method))
        self.print_iter = print_iter
        self.print_step = print_step
        self.label = label
        # ---
        self.fitFun = fitFun
        self.param = numpy.array(param,dtype=numpy.float64,ndmin=1)
        f0 = finalTest(self.fitFun,self.param)
        self.parmat = numpy.array([[f0]+self.param.tolist() for n in range(self.num_chain)],dtype=numpy.float64)
        self.varmat = numpy.array(varmat,dtype=numpy.float64,ndmin=2)
        self.parmats = []
        # ---
        if len(inferpar):
            self.inferpar = numpy.array(inferpar,dtype=numpy.int32)
        else:
            self.inferpar = numpy.arange(len(self.param),dtype=numpy.int32)
        print "Parameters to infer: %s" %(join(self.inferpar,","))
        # ---
        self.rank_indices = [numpy.arange(i,self.num_chain,MPI_SIZE) for i in range(MPI_SIZE)]
        self.worker_indices = numpy.delete(range(MPI_SIZE),MPI_MASTER)
        # ---
        self.outfilename = outfilename
        self.outparmat = None
        self.outfinal = None
        if MPI_RANK==MPI_MASTER:
            if self.outfilename:
                self.outparmat = binfile(self.outfilename+'.parmat','w',self.parmat.shape[1]+3)
                self.outfinal = binfile(self.outfilename+'.final','w',4)
        # ---
        for hopp_step in range(self.num_hopp):
            if self.method=="anneal":
                self.anneal = anneal_sigmasoft(self.rangeT[0],self.rangeT[1],self.num_adapt)
            for adapt_step in range(self.num_adapt):
                self.runAdaptStep(hopp_step*self.num_adapt+adapt_step)
            if MPI_RANK == MPI_MASTER:
                if not self.outfinal:
                    for chain_id in range(self.num_chain):
                        print "param.mat.final.%s: %d,%d,%s" %(self.label,hopp_step,chain_id,join(self.parmat[chain_id,:],","))
                # ---
                test = {'acc':True, 'favg0':numpy.nan, 'favg1':numpy.nan} if len(self.parmats)==0 else compareAUC(self.parmats[-1][:,[0]+(1+self.inferpar).tolist()],self.parmat[:,[0]+(1+self.inferpar).tolist()],self.model_comp)
                if test['acc']:
                    self.parmats.append(self.parmat)
                else:
                    self.parmat = self.parmats[-1].copy()
                    self.param = self.parmat[parMin(self.parmat)['i'],1:].copy()
                # ---
                if self.outfinal:
                    self.outfinal.writeRow([hopp_step,test['acc'],test['favg0'],test['favg1']])
                else:
                    print "parMatAcc.final.%s: %d,%s" %(self.label,hopp_step,join([test['acc'],test['favg0'],test['favg1'],ldet(self.stat['cov'](self.parmat[:,1+self.inferpar]))],","))
                # ---
            if MPI_SIZE>1:
                self.parmat = MPI.COMM_WORLD.bcast(self.parmat, root=MPI_MASTER)
                self.param = MPI.COMM_WORLD.bcast(self.param, root=MPI_MASTER)
        # ---
        if MPI_SIZE>1:
            self.parmats = MPI.COMM_WORLD.bcast(self.parmats, root=MPI_MASTER)
            if self.outparmat:
                self.outparmat.close()
            if self.outfinal:
                self.outfinal.close()

    def runAdaptStep(self,adapt_step):
        if self.method=="anneal":
            if MPI_RANK == MPI_MASTER:
                pm = parMin(self.parmat)
                self.param = self.parmat[pm['i'],1:].copy()
                self.parmat = numpy.array([self.parmat[pm['i'],:].tolist() for n in range(self.num_chain)],dtype=numpy.float64)
            if MPI_SIZE>1:
                self.param = MPI.COMM_WORLD.bcast(self.param, root=MPI_MASTER)
                self.parmat = MPI.COMM_WORLD.bcast(self.parmat, root=MPI_MASTER)
        # ---
        for chain_id in self.rank_indices[MPI_RANK]:
            # ---
            # print "Passing parameter %s to chain %d" %(join(self.param,","),chain_id)
            # ---
            mcmc = chainMCMC(self.fitFun,
                             self.param if self.param_reset else self.parmat[chain_id,1:],
                             self.varmat,
                             gibbs=self.gibbs,
                             chain_id=chain_id,
                             pulsevar=1.0,
                             anneal=self.anneal[0],
                             accthr=0.5,
                             inferpar=self.inferpar,
                             varmat_change=0,
                             pulse_change=10,
                             pulse_change_ratio=2,
                             print_iter=self.print_iter,
                             label="%s.%d.%d" %(self.label,adapt_step,chain_id))
            for m in range(self.chain_length):
                mcmc.iterate()
            self.parmat[chain_id,:] = mcmc.getParam()
            # ---
        if MPI_RANK == MPI_MASTER:
            for worker in self.worker_indices:
                parmat = MPI.COMM_WORLD.recv(source=worker, tag=1)
                for chain_id in self.rank_indices[worker]:
                    self.parmat[chain_id,:] = parmat[chain_id,:]
            # ---
            self.varmat = numpy.array(self.stat['cov'](self.parmat[:,1+self.inferpar]),ndmin=2)
            self.varmat[numpy.abs(self.varmat)<EPS_VARMAT_MIN] = 1.0
            # ---
            if self.print_step and (adapt_step%self.print_step)==0:
                for chain_id in range(self.num_chain):
                    if self.outparmat:
                        tmp = [adapt_step,chain_id,self.anneal[0]]+self.parmat[chain_id,:].tolist()
                        self.outparmat.writeRow(tmp)
                    else:
                        print "param.mat.step.%s: %d,%d,%s" %(self.label,adapt_step,chain_id,join(self.parmat[chain_id,:],","))
                print "parMatAcc.step.%s: %d,%s" %(self.label,adapt_step,join([ldet(self.stat['cov'](self.parmat[:,1+self.inferpar])),self.anneal[0]],","))
            # ---
            if self.method=="anneal":
                if len(self.anneal)>1: self.anneal = self.anneal[1:]            
        else:
             MPI.COMM_WORLD.send(self.parmat, dest=MPI_MASTER, tag=1)
        # ---
        if MPI_SIZE>1:
            self.parmat = MPI.COMM_WORLD.bcast(self.parmat, root=MPI_MASTER)
            self.varmat = MPI.COMM_WORLD.bcast(self.varmat, root=MPI_MASTER)
            if self.method=="anneal":
                self.anneal = MPI.COMM_WORLD.bcast(self.anneal, root=MPI_MASTER)
        # ---

# --- Class: chainMCMC
class chainMCMC:
    def __init__(self,fitFun,parmat,varmat,gibbs=False,chain_id=0,pulsevar=1.0,anneal=1,accthr=0.5,inferpar=[],varmat_change=0,pulse_change=10,pulse_change_ratio=2,print_iter=0,label=''):
        self.multi = {'cov': covariance,
                      'rnorm': numpy.random.multivariate_normal,
                      'det': numpy.linalg.det}
        self.single = {'cov': covariance,
                       'rnorm': numpy.random.normal,
                       'det': abs}
        # ---
        self.chain_id = chain_id;
        self.fitFun = fitFun
        self.parmat = numpy.array(parmat,dtype=numpy.float64,ndmin=1)
        self.varmat = numpy.array(varmat,dtype=numpy.float64,ndmin=2)
        self.inferpar = numpy.array(inferpar,dtype=numpy.int32)
        self.anneal = numpy.array(anneal,dtype=numpy.float64,ndmin=1)
        self.accthr = numpy.array(accthr,dtype=numpy.float64,ndmin=1)
        # ---
        self.pulse_change_ratio = pulse_change_ratio
        self.pulse_nochange = numpy.float64(1)
        self.pulse_increase = numpy.float64(self.pulse_change_ratio)
        self.pulse_decrease = numpy.float64(1.0/self.pulse_change_ratio)
        self.pulse_change = pulse_change
        self.pulse_collect = max(1,self.pulse_change)
        self.varmat_change = varmat_change
        self.varmat_collect = max(1,self.varmat_change)
        # ---
        if self.parmat.ndim==1:
            f0 = finalTest(self.fitFun,self.parmat)
            self.parmat = numpy.array([[f0]+self.parmat.tolist() for i in range(self.varmat_collect)])
        elif self.parmat.shape[0]!=self.varmat_collect:
            Abort("Dimension mismatch in chainMCMC! parmat.shape[0]=%d collect=%d" %(self.parmat.shape[0],self.varmat_collect))
        if self.varmat.shape and self.inferpar.shape[0] != self.varmat.shape[0]:
            Abort("Dimension mismatch in chainMCMC! inferpar.shape[0]=%d varmat.shape[0]=%d" %(self.inferpar.shape[0],self.varmat.shape[0]))
        # ---
        self.gibbs = gibbs
        if self.gibbs:
            self.pulsevar = numpy.array(numpy.repeat(pulsevar,len(self.inferpar)),dtype=numpy.float64)
            self.acc_vecs = [numpy.repeat(False,self.pulse_collect) for n in range(len(self.inferpar))]
            self.iterate = self.iterateSingle
        else:
            self.pulsevar = pulsevar
            self.acc_vec = numpy.repeat(False,self.pulse_collect)
            self.iterate = self.iterateMulti
        if not self.gibbs and (self.parmat.shape[1]==2 or self.inferpar.shape[0]==1):
            Abort("Please set gibbs=True!")
        self.varmat = self.varmat*self.pulsevar
        # ---
        self.halfa = 0.025
        if self.pulse_change<25:
            self.halfa = 0.05
        self.print_iter = print_iter
        self.label = label
        self.step = 0
        self.index = 0
        self.index_acc = 0

    def getParam(self):
        return self.parmat[self.index,:]

    def getVarPar(self):
        return ldet(self.multi['cov'](self.parmat[:,1+self.inferpar]))

    def getVarVar(self):
        return ldet(self.varmat)

    def getAcc(self):
        if self.gibbs:
            return numpy.array([numpy.mean(acc_vec) for acc_vec in self.acc_vecs])
        else:
            return numpy.mean(self.acc_vec)

    def setParam(self,parmat):
        self.parmat[self.index,:] = numpy.array(parmat,dtype=numpy.float64,ndmin=1).copy()

    def newParamSingle(self,param,stdev):
        try:
            param1 = self.single['rnorm'](param,
                                          stdev)
        except:
            print "Warning: Failed to generate a new parameter set"
            param1 = numpy.copy(param)
        return param1
        
    def newParamMulti(self):
        try:
            param1 = self.multi['rnorm'](self.parmat[self.index,1:][self.inferpar],self.varmat*self.pulsevar)
        except numpy.linalg.linalg.LinAlgError:
            print "Warning: Failed to generate a new parameter set"
            param1 = numpy.copy(self.parmat[self.index,1:][self.inferpar])
        return param1
    
    def checkMove(self,f0,f1):
        acc = (not numpy.isnan(f1) and not numpy.isinf(f1) and
            f1 >= 0 and
            (f1 <= f0 or
            (numpy.log(numpy.random.uniform()) < (f0-f1)/self.anneal[0])))
        # --- f0 = 0.5*SS_0
        # --- f = 0.5*SS
        # --- sqrt(anneal) == st.dev.
        # --- 0.5*x^2/(T*s^2)
        # --- return exp(-0.5*SS/anneal)/exp(-0.5*SS_0/anneal)
        return(acc)

    def pulsevarUpdate(self,acc_vec):
        # --- Test if higher than accthr
        try:
            r = ttest(acc_vec,self.accthr)
        except ZeroDivisionError:
            if all(acc_vec)<=0:
                return self.pulse_decrease
            elif all(acc_vec)>=0:
                return self.pulse_increase
            else:
                return self.pulse_nochange
        if (r[0]>0 and r[1]<self.halfa): return self.pulse_increase
        # --- Test if lower than accthr
        r = ttest(acc_vec,max(0.01,self.accthr-0.15))
        if (r[0]<0 and r[1]<self.halfa): return self.pulse_decrease
        # --- Return default
        return self.pulse_nochange

    def iterateMulti(self):
        self.step += 1
        # ---
        acc = False
        f0 = self.parmat[self.index,0]
        param1 = numpy.copy(self.parmat[self.index,1:])
        param1[self.inferpar] = self.newParamMulti()
        f1 = self.fitFun(param1)
        acc = self.checkMove(f0,f1)
        # ---
        self.index_acc = (self.index_acc+1)%self.pulse_collect
        if acc:
            self.index = (self.index+1)%self.varmat_collect
        # ---
        self.acc_vec[self.index_acc] = acc
        if acc:
            self.parmat[self.index,0] = f1
            self.parmat[self.index,1:] = param1
        # ---
        if self.print_iter and (self.step%self.print_iter)==0:
            print "param.mat.chain.%s: %d,%d,%s" %(self.label,self.step,self.chain_id,join(self.parmat[self.index,:],","))
        # ---
        if self.step>1:
            # --- 
            if self.pulse_change and (self.step%self.pulse_change)==0:
                self.pulsevar = max(1e-7,self.pulsevar*self.pulsevarUpdate(self.acc_vec))
            # --- 
            if self.varmat_change and (self.step%self.varmat_change)==0:
                self.varmat = numpy.array(self.multi['cov'](self.parmat[:,1+self.inferpar]),ndmin=2)
                a = numpy.diag(self.varmat)<EPS_VARMAT_MIN
                self.varmat[a,a] = EPS_VARMAT_MIN
        # --- 
        if self.print_iter and (self.step%self.print_iter)==0:
            print "parMatAcc.chain.%s: %s" %(self.label,join([self.step,self.chain_id,ldet(self.multi['cov'](self.parmat[:,1+self.inferpar])),ldet(self.varmat),numpy.mean(self.acc_vec),self.pulsevar],","))

    def iterateSingle(self):
        self.step += 1
        self.index_acc = (self.index_acc+1)%self.pulse_collect
        # ---
        acc_steps = False
        f0 = self.parmat[self.index,0]
        param0 = self.parmat[self.index,1:].copy()
        for param_id in numpy.arange(len(self.inferpar)):
            param1 = param0.copy()
            param1[self.inferpar[param_id]] = self.newParamSingle(param1[self.inferpar[param_id]],
                                                                  self.varmat[param_id,param_id]*self.pulsevar[param_id])
            f1 = self.fitFun(param1)
            acc = self.checkMove(f0,f1)
            if acc:
                acc_steps = True
                f0 = f1
                param0[self.inferpar[param_id]] = numpy.copy(param1[self.inferpar[param_id]])
            self.acc_vecs[param_id][self.index_acc] = acc
            # ---
        if acc_steps:
            self.index = (self.index+1)%self.varmat_collect
            self.parmat[self.index,0] = f0
            self.parmat[self.index,1:] = param0
            if numpy.isnan(f0) or numpy.isinf(f0):
                Abort("Iterate single failed with %g: %s" %(f0,join(param0,",")))
        # ---
        if self.print_iter and (self.step%self.print_iter)==0:
            print "param.mat.chain.%s: %d,%d,%s" %(self.label,self.step,self.chain_id,join(self.parmat[self.index,:],","))
        # ---
        if self.step>1:
            # --- 
            if self.pulse_change and (self.step%self.pulse_change)==0:
                for param_id in numpy.arange(len(self.inferpar)):
                    self.pulsevar[param_id] = min(EPS_PULSE_MAX,max(EPS_PULSE_MIN,self.pulsevar[param_id]*self.pulsevarUpdate(self.acc_vecs[param_id])))
            # --- 
            if self.varmat_change and (self.step%self.varmat_change)==0:
                for param_id in numpy.arange(len(self.inferpar)):
                    self.varmat[param_id,param_id] = max(EPS_VARMAT_MIN,self.single['cov'](self.parmat[:,1+self.inferpar[param_id]]))
        # --- 
        if self.print_iter and (self.step%self.print_iter)==0:
            print "parMatAcc.chain.%s: %s" %(self.label,join([self.step,self.chain_id,ldet(self.multi['cov'](self.parmat[:,1+self.inferpar])),ldet(self.varmat)],","))
            print "parMatAcc.chain.accs.%s: %d,%d,%s" %(self.label,self.step,self.chain_id,join([numpy.mean(acc_vec) for acc_vec in self.acc_vecs],","))
            print "parMatAcc.chain.pulses.%s: %d,%d,%s" %(self.label,self.step,self.chain_id,join(self.pulsevar,","))
