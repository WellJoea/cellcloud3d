import multiprocessing.pool
from functools import reduce
from multiprocessing import Process, Pool

import multiprocessing
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from joblib import Parallel, delayed
num_cores = multiprocessing.cpu_count()


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def hybpara(alist, funct,
            
            args=(),kwds={},
            backend='loky',
             verbose=10,
            n_jobs=None):
    n_jobs = 1 if n_jobs is None else n_jobs 
    ielm =''
    if backend=='Threading':
        with ThreadPool(processes=n_jobs) as pool:
            result = [ pool.apply_async(funct, args=(*args,), kwds=kwds) for elm in tqdm(alist) ]
            pool.close()
            pool.join()
            result = [ar.get() for ar in result]

    elif backend=='Multiprocessing':
        with Pool(processes=n_jobs) as pool:
            result = [ pool.apply_async(funct, args=(*args,), kwds=kwds) for elm in tqdm(alist) ]
            pool.close()
            pool.join()
        result = [ar.get() for ar in result]

    else:
        result = Parallel(n_jobs= n_jobs, backend=backend, verbose=verbose)\
                     (delayed(funct)(*args, **kwds) for ielm in tqdm(alist))
    return result

def test(self, bamfile):
    pool   = Pool(processes=self.arg.pool)
    allbam = pool.map(self.Bam, bamfile)
    #allbam = [pool.apply_async(self.Bam, (bam, strbed, record_dict, )) for bam in bamfile]
    pool.close()
    pool.join()

def SUB(self, files):
    file = open(self.arg.input, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=self.arg.pool)
    pool.map(self.GATK, file)
    pool.close()
    pool.join()


    pool  = NoDaemonProcessPool(processes=self.arg.pool)
    pool.map(self.GET, files)
    pool.close()
    pool.join()

def test1():
    fov_dir =  f'{rootdir}/processed_data/register_dir'
    keep_fov = fovs
    keep_fov = retrn_fov
    ref_hyb = 5

    import multiprocessing
    from multiprocessing.pool import ThreadPool
    from tqdm import tqdm
    from joblib import Parallel, delayed
    num_cores = multiprocessing.cpu_count()

    # simi = Parallel( n_jobs= -1, backend='threading')\
    #                     (delayed(simitif)(ihyb, ifov, fov_dir, ref_hyb=refhyb) for ihyb in hybs for ifov in keep_fov)

    # def hybparaj(hybs, ifov, fov_dir, ref_hyb):
    #     hsimi = Parallel(n_jobs= 32, backend='threading')\
    #                     (delayed(simitif)(ihyb, ifov, fov_dir, ref_hyb=ref_hyb) for ihyb in hybs)
    #     return list(hsimi)

    def hybpara(hybs, ifov, fov_dir, ref_hyb):
        with ThreadPool(processes=len(hybs)) as pool:
            result = [ pool.apply_async(simitif, args=(ihyb, ifov, fov_dir,), 
                                        kwds={'ref_hyb':ref_hyb}) for ihyb in hybs ]
            pool.close()
            pool.join()
        results = [ar.get() for ar in result]
        return results

    simi = Parallel(n_jobs= 2, backend='loky')\
                        (delayed(hybpara)(hybs, ifov, fov_dir, refhyb) for ifov in tqdm(keep_fov))

def hybpara(hybs, ifov, fov_dir, ref_hyb, backend='threading', njob=None):
    njob = len(hybs) if njob is None else njob 
    if backend=='threading':
        with ThreadPool(processes=njob) as pool:
            result = [ pool.apply_async(simitif, args=(ihyb, ifov, fov_dir,), 
                                        kwds={'ref_hyb':ref_hyb}) for ihyb in hybs ]
            pool.close()
            pool.join()
    elif backend=='multiprocessing':
        with Pool(processes=njob) as pool:
            result = [ pool.apply_async(simitif, args=(ihyb, ifov, fov_dir,), 
                                        kwds={'ref_hyb':ref_hyb}) for ihyb in hybs ]
            pool.close()
            pool.join()
    results = [ar.get() for ar in result]
    return results