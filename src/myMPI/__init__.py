"""
NO WARRANTY DISCLAIMER

This software is provided "as is," without any express or implied warranties. 
In no event shall the authors or copyright holders be liable for any claim, 
damages, or other liability, whether in an action of contract, tort, or otherwise, 
arising from, out of, or in connection with the software or the use or other dealings 
in the software.

By using this software, you agree to these terms. If you do not agree to these terms, 
do not use the software.

The authors make no representations about the suitability of this software for any 
purpose. It is provided "as is" without express or implied warranty.

COPYRIGHT NOTICE

Copyright (c) 2024 Kamil Erguler - kerguler@gmail.com
All rights reserved.
"""

import numpy
from mpi4py import MPI
MPI_SIZE = MPI.COMM_WORLD.Get_size()
MPI_RANK = MPI.COMM_WORLD.Get_rank()
MPI_MASTER = 0
MPI_NONE = 0
MPI_AVAIL = 1
MPI_BUSY = 2
MPI_DONE = 3
worker_indices = numpy.delete(numpy.arange(MPI_SIZE),MPI_MASTER)
requests = [MPI_NONE for i in range(MPI_SIZE)]
def MPI_TEST(wrk):
    if requests[wrk] == MPI_NONE:
        return MPI_AVAIL
    if requests[wrk].test()[0]:
        requests[wrk] = MPI_NONE
        return MPI_DONE
    return MPI_BUSY
print("Process %d of %d is running" %(MPI_RANK,MPI_SIZE),flush=True)

class mpi:
    def __init__(self, master, slave, opt={}) -> None:
        #
        # fun_master(mpi, opt)
        #  mpi.exec(tasks, multiple=True, verbose=True)
        #  mpi.clean()
        # fun_slave(mpi, task, opt)
        # mympi.mpi(fun_master, fun_slave)
        #
        self.opt = opt
        self.master = master
        self.slave = slave
        #
        if MPI_RANK != MPI_MASTER:
            while True:
                cmd = MPI.COMM_WORLD.recv(source=MPI_MASTER, tag=1)
                if "break" in cmd:
                    break
                ret = self.slave(self,cmd,opt=self.opt)
                req = MPI.COMM_WORLD.isend(ret, dest=MPI_MASTER, tag=1)
                req.Wait()
        else: # MPI_RANK == MPI_MASTER
            self.master(self,opt=self.opt)
        #
    def clean(self):
        for worker in worker_indices:
            req = MPI.COMM_WORLD.isend({
                "break": True
            }, dest=worker, tag=1)
            req.Wait()
        #
    def exec(self, jobs, verbose=False, multiple=False):
        if MPI_SIZE == 1:
            if verbose:
                rets = []
                while len(jobs):
                    cmd = jobs.pop()
                    print("Remaining: ",len(jobs),flush=True)
                    ret = self.slave(self,cmd,opt=self.opt)
                    rets.append(ret)
            else:
                rets = [self.slave(self,cmd,opt=self.opt) for cmd in jobs]
            return rets
        #
        rets = []
        busy = False
        while len(jobs) > 0 or busy:
            busy = False
            for worker in worker_indices:
                test = MPI_TEST(worker)
                if test == MPI_BUSY:
                    busy = True
                elif test == MPI_DONE:
                    try:
                        ret = MPI.COMM_WORLD.recv(source=worker, tag=1)
                        rets.append(ret)
                    except Exception as error:
                        print("ERROR: Scrambled communication!",error,flush=True)
                        MPI.COMM_WORLD.Abort()
                else: # MPI_AVAIL
                    if len(jobs):
                        cmd = jobs.pop()
                        if verbose:
                            print("Remaining: ",len(jobs),flush=True)
                        requests[worker] = MPI.COMM_WORLD.isend(cmd, dest=worker, tag=1)
                        requests[worker].Wait()
                        busy = True
        #
        if not multiple:
            self.clean()
        return rets
