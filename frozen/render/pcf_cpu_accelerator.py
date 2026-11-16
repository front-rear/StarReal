"""Optional CPU acceleration, preserving the NumPy reference parameters."""
import ctypes
from pathlib import Path
import numpy as np


class CPUTrace:
    def __init__(self,threads=4):
        if not 1<=threads<=8:
            raise ValueError('bounded CPU thread count required')
        self.threads=threads
        self.lib=ctypes.CDLL(str(Path(__file__).with_name('pcf_trace_cpu.so')))
        arr=np.ctypeslib.ndpointer
        self.lib.trace_pcf.argtypes=[arr(np.float64,flags='C_CONTIGUOUS'),arr(np.float64,flags='C_CONTIGUOUS'),
            ctypes.c_int64,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,
            arr(np.float64,flags='C_CONTIGUOUS'),arr(np.float64,flags='C_CONTIGUOUS'),arr(np.uint8,flags='C_CONTIGUOUS')]
        self.lib.trace_pcf.restype=None

    def __call__(self,origin,directions,spacing=.5,far_cells=64.,steps=4096):
        if spacing<=0:
            raise ValueError('positive consistent relative-world spacing required')
        directions=np.ascontiguousarray(directions,dtype=np.float64)
        phase=np.ascontiguousarray(np.remainder(np.asarray(origin)/spacing+.5,1.))
        depth=np.empty(len(directions),np.float64);terminal=np.empty_like(depth)
        unresolved=np.empty(len(directions),np.uint8)
        self.lib.trace_pcf(phase,directions,len(directions),spacing,far_cells,steps,self.threads,depth,terminal,unresolved)
        return depth,unresolved.astype(bool),phase,terminal


def verify():
    import json,time
    import depth_fusion as ref
    kernel=CPUTrace(4)
    rng=np.random.default_rng(20261112)
    rays=rng.normal(size=(16000,3));rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    cases=[([0,0,0],.5,64.,4096),([1.7,-.25,9.3],.5,64.,4096),
           ([-1.,.1,-2.],.35,20.,4096),([.17,.33,.29],.5,64.,2)]
    for origin,spacing,far,steps in cases:
        expected=ref.trace_lattice(origin,rays,spacing,far,steps)
        actual=kernel(origin,rays,spacing,far,steps)
        for left,right in zip(expected,actual):
            np.testing.assert_array_equal(left,right)
    C=np.eye(4);C[:3,3]=[.17,-.12,.31]
    K=np.array([[80.,0,31.5],[0,80.,23.5],[0,0,1.]])
    boxes=[ref.Box([.2,.1,1.],[.23,.18,.1],np.eye(3),1),ref.Box([-.15,.05,1.3],[.1,.3,.12],np.eye(3),2)]
    reference=ref.render(C,K,boxes,64,48,2)
    old=ref.trace_lattice;ref.trace_lattice=kernel
    try:accelerated=ref.render(C,K,boxes,64,48,2)
    finally:ref.trace_lattice=old
    np.testing.assert_array_equal(reference[0],accelerated[0])
    for name in reference[1]:np.testing.assert_array_equal(reference[1][name],accelerated[1][name])
    start=time.time();kernel([.17,-.12,.31],rays)
    report={'status':'reference_exact_on_analytic_and_random_checks',
            'ray_cases':len(cases),'rays_each':len(rays),'full_RGB_and_all_depth_aux_arrays_exact':True,
            'no_parameter_changes':True,'CPU_threads':4,'benchmark_seconds_16000_rays':time.time()-start}
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':verify()
