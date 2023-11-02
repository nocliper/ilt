def read_file(Path, dt=150, proc = True):
    """Returns data from file

    Parameters:
    --------------
    Path : str
        Path to file
    dt : float
        Step between time points in ms
    proc: boolean
        Process transient if True

    Returns:
    --------------
    time : array
        Time domain points in s
    C : array of [F1(time), F2(time), ...]
        Contains transients experimental data 
        for range of temperatures
    T : array
        Contains experimental temperature points
    """
    import numpy as np

    def process(C, proc):
        """Returns processed transient C_p if proc is True"""

        def get_Baseline(C):
            """Returns baseline of transient C"""
            l = len(C)
            c1, c2, c3 = C[0], C[int(l/2)-1], C[l-1]
            return (c1*c3 - c2**2)/(c1+c3 - 2*c2)

        if proc:
            C_p = C
            for i, _C in enumerate(C):
                F = _C
                F = F/np.average(F[-1])
                if F[0] > F[-1]:
                    F = F - min(F)
                else:
                    F = F - max(F)
                F = np.abs(F)
                F = F + np.average(F)*2
                F = F - get_Baseline(F)*0
                C_p[i] = F
            return np.asarray(C_p)

        else:
            C = C/C[-1]
            return C + np.average(C)*2

    Path = str(Path)

    txt  = np.genfromtxt(Path, delimiter='\t')
    if len(txt.shape) == 2:
        T    = txt[:,0]
        cut  = len(T)

        C    = []
        time = []
        for i in range(0,cut):
            C.append(txt[i][3:-2])

        for i in range(0, len(C[0])):
            time.append(dt/1000*(i+1))
    else:
        T    = txt[0]
        C    = txt[1:]
        time = np.arange(dt, dt*len(C), dt)*1e-3




    C    = np.asarray(C)
    C    = process(C, proc)
    time = np.asarray(time)
    T    = np.asarray(T)
    #print(time)

    return time, C, T
