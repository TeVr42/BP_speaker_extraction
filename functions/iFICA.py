import numpy as np

def iFICA(x, wini, soiinfo = None, nonln = "rati"):
    epsilon = 0.000001
    d, N, K = x.shape
    MaxIt = 100
    realvalued = np.isrealobj(x)

    if soiinfo.shape[0] > 1:
        soiinfo = soiinfo.reshape((1,1,soiinfo.shape[0]))

    x = x.transpose(2,0,1)
    w = wini.transpose(2,0,1)

    x_conj = np.conj(x)
    Cx = np.matmul(x, x_conj.transpose(0,2,1)) / N

    w = w / np.sqrt(np.sum(w * np.conj(w), axis=1)).reshape(K,1,1)
    NumIt = 0
    crit = 0

    if soiinfo is not None:
        normalized_x = x / (np.abs(soiinfo) + 0.001)
        H = np.matmul(normalized_x, np.conj(np.transpose(x, (0, 2, 1)))) / N
    else:
        H = Cx

    while crit < 1-epsilon and NumIt < MaxIt:
        NumIt += 1
        wold = w

        a = np.matmul(Cx, w) 
        sigma2 = np.sum(np.conj(w) * a, axis=1)
        sigma2 = sigma2.reshape(K,1,1)
        # Mixing vector
        a = a / sigma2

        w_conj_transpose = np.conj(w).transpose(0, 2, 1)       
        soi = np.matmul(w_conj_transpose, x)
        sigma = np.sqrt(sigma2)

        # Normalized SOI
        soin = soi / sigma
        if realvalued:
            psi, psihpsi = realnonln(soin, nonln)
        else:
            psi, psihpsi = complexnonln(soin, nonln)

        xpsi = (np.matmul(x, psi.transpose(0, 2, 1)) / sigma) / N
        rho = np.mean(psihpsi, axis=2)
        rho = rho.reshape(K,1,1)

        grad_a = rho * a - xpsi
        w = np.zeros_like(grad_a)

        for k in range(H.shape[0]):
            w[k] = np.linalg.inv(H[k]) @ grad_a[k]

        w = w / np.sqrt(np.sum(w * np.conj(w), axis=1).reshape(K,1,1))
        crit = np.min(np.abs(np.sum(w * np.conj(wold), axis=1)), axis=0)

    a = np.matmul(Cx, w)
    sigma2 = np.sum(np.conj(w) * a, axis=1)
    a = a / sigma2.reshape(K, 1, 1)

    a1_mic = a[:, 0, 0].reshape(K, 1, 1)
    w_conj_transpose = np.conj(np.transpose(w, (0, 2, 1)))
    soi = a1_mic * np.matmul(w_conj_transpose, x)
    w = np.conj(a1_mic)*w
    a = a / a1_mic

    return w, a, soi, NumIt

def realnonln(s, nonln):
    if nonln == "sign":
        if s.shape[0] == 1:
            raise Exception('Nonlinearity "sign" cannot be used for the real-valued ICA/ICE.')
        
        aux = 1 / np.sqrt(np.sum(s**2, axis=2))
        psi = s * aux
        psipsi = aux * (1 - psi**2)
    
    elif nonln == "tanh":
        if s.shape[0] > 1:
            aux = 1 / np.sqrt(np.sum(s**2, axis=0))
            th = np.tanh(s)
            psi = th * aux
            psipsi = aux * (1 - th**2 - psi*aux)
        else:
            psi = np.tanh(s)
            psipsi = 1 - psi**2
    elif nonln == "rati":
        aux = 1 / (1 + np.sum(s**2, axis=0))
        psi = s * aux
        psipsi = aux - 2*psi**2
    return psi, psipsi

def complexnonln(s, nonln):
    if nonln == "sign":
        sp2 = s*np.conj(s)
        aux = 1 / np.sqrt(np.sum(sp2, axis=0))
        psi = np.conj(s)*aux
        psipsi = aux * (1-psi * np.conj(psi)/2)
    elif nonln == "rati":
        sp2 = s*np.conj(s)
        aux = 1 / (1 + np.sum(sp2, axis=0))
        psi = np.conj(s)*aux
        psipsi = aux - psi * np.conj(psi)
    return psi, psipsi
