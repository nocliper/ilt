def read_file(Path, dt=150):
    """Rerurns data from file

    Path – path to file

    returns:
    time – time points (s-domain)
    C – all transients
    T – Tempetarures

    """
    import numpy as np

    Path = str(Path)

    txt  = np.genfromtxt(Path, delimiter='\t')
    T    = txt[:,0]
    cut  = len(T)

    C    = []
    time = []
    for i in range(0,cut):
        C.append(txt[i][3:-2])

    for i in range(0, len(C[0])):
        time.append(dt/1000*(i+2))

    C    = np.asarray(C)
    time = np.asarray(time)
    #print(T)

    return time, C, T
