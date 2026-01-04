import torch as th
import numpy as np
from functools import partial
from torch.nn.functional import conv1d, pad
# from pysptools.abundance_maps.amaps import FCLS
# from pysptools.abundance_maps.amaps import NNLS

_H_cache = None

def cal_conditional_gradient_W(W, Y, bar_alpha, alpha, var, t, mask, type="dps", __cache_H=False):
    W = W[:, 0]
    W = (W + 1)/2
    if mask is None:
        W_masked = W
        Y_masked = Y
    else:
        W_masked = W[:, ~mask]
        Y_masked = Y

    H = solve_H(Y_masked, W_masked, t, __cache_H=__cache_H)
    N, R = H.shape
    if type == "dmps":
        grad = H.T @ th.inverse(var*th.eye(N, device=H.device)+(1-bar_alpha)/bar_alpha * H@H.T)/np.sqrt(bar_alpha) @ (Y - H@W) * (1-alpha)/np.sqrt(alpha)/2
        A = Y_masked - H @ W_masked
        delta = th.tensor(1.75)
    elif type == "diffun":
        HT = H.T
        grad = HT @ (Y_masked - H @ W_masked)
        A = Y_masked - H @ W_masked
        B = H @ grad
        delta = (A.T @ B).trace()/ ((B.T @ B).trace()+1e-5)
        delta = delta* np.sqrt(bar_alpha)/2
    else:
        raise NotImplementedError
    assert not th.isnan(delta)
    grad = grad[:, None] * delta
    H = H[:, None]
    log = {'res': th.norm(A).item(), "H": H.cpu().numpy(), "W": W.cpu().numpy()}
    return grad, H, log


def solve_H(Y, W, t, __cache_H=True):
    global _H_cache
    R = W.shape[0]
    if t % 5 == 0 or _H_cache is None or __cache_H is False:
        H = FCLS(Y.cpu().numpy(), W.cpu().numpy())
        _H_cache = H
    else:
        H = _H_cache
    return th.from_numpy(H).float().to(Y.device)


def denoising_fn(sigma=1):
    blur = partial(
        conv1d,
        bias=None,
        stride=1,
        padding=0,
    )

    k = 7
    gaussian = np.exp(-((np.arange(k)-k//2)**2/(2*sigma**2)))
    def filter(W):
        weight = th.from_numpy(gaussian)
        weight = weight/th.sum(weight)
        weight = weight[None, None].to(W.device).float()
        W = pad(W, (k//2, k//2), mode="reflect")
        return blur(W, weight)

    return filter


# def SAD(s1, s2):
#     return np.arccos(np.sum(s1*s2)/(np.linalg.norm(s1+1e-9)*np.linalg.norm(s2+1e-9)))

# def SAD_matrix(A, B=None):
#     if B is None:
#         B = A
#     N = A.shape[0]
#     M = B.shape[0]
#     dist_matrix = np.zeros([N, M], dtype=np.float32)
#     for i in range(N):
#         for j in range(M):
#             dist_matrix[i, j] = SAD(A[i], B[j])

#     return dist_matrix

def NSAD(A, n, B=None):
    sad_matrix = SAD_matrix(A, B)
    n_sad = np.sort(sad_matrix, axis=1)[:, n]
    return n_sad


def sample_from(A, S, n, thre=0.1):
    R1 = len(S)
    used_idx = []
    idx = np.arange(len(A))
    np.random.shuffle(idx)
    if R1 < 1:
        S = A[[idx[0]]]
        used_idx.append(idx[0])
        R1 = 1
    R2 = n - R1
    for i in idx:
        if len(S) > R2:
            break
        sad = np.arccos(S @ A[i].T / np.linalg.norm(S+1e-9, axis=1)/np.linalg.norm(A[i]+1e-9)).min()
        # sad = np.max(np.abs(S - A[i]), axis=1).min()
        if sad > thre:
            S = np.concatenate([S , A[[i]]], axis=0)
            used_idx.append(i)
    return S, used_idx


class UnmixingUtils:
    def __init__(self, A, S):
        self.A = A
        self.S = S
        pass

    def hyperSAD(self, A_est):
        Rt = self.A.shape[1]
        Re = A_est.shape[1]
        P = np.zeros([Rt, Re])
        for i in range(Rt):
            d = np.arccos(np.clip(A_est.T @ self.A[:, i] / np.linalg.norm(A_est, axis=0)/np.linalg.norm(self.A[:, i]), -1, 1))
            P[i, np.argmin(d)] = 1

        Ap = A_est @ P.T
        dist = np.zeros(Rt)
        for i in range(Rt):
            dist[i] = np.arccos(np.clip(Ap.T[i] @ self.A[:, i] / np.linalg.norm(Ap.T[i], axis=0)/np.linalg.norm(self.A[:, i]), -1, 1))

        mean_dist = np.mean(np.sort(dist)[:A_est.shape[1]])
        return dist, mean_dist, P

    def hyperRMSE(self, S_est, P):
        # print(P)
        N = np.size(self.S, 0)
        Sp = S_est @ P.T
        # Sp = Sp / np.sum(Sp, axis=1, keepdims=True)
        rmse = self.S - Sp
        rmse = rmse * rmse
        rmse = (np.sqrt(np.sum(rmse, 0) / N))
        # print(rmse)
        armse = np.mean(np.sort(rmse)[:Sp.shape[1]])
        return rmse, armse


def analyse(W, dir):
    from guided_diffusion.spectral_datasets import _list_spectral_files, load_spectral_from_txts
    path = np.array(_list_spectral_files(dir))
    A = load_spectral_from_txts(path, False)
    A = A[~np.any(np.isnan(A), axis=1)]
    names = []
    for wi in W:
        sad = np.arccos(np.clip(wi @ A.T / np.linalg.norm(A.T, axis=0)/np.linalg.norm(wi), -1, 1))
        idx = np.argmin(sad)
        name = path[idx].split("/")[-1]
        names.append(name)
    return names

import numpy as np
import scipy.linalg as splin

def estimate_snr(Y, r_m, x):
    [L, N] = Y.shape
    [p, N] = x.shape

    P_y = np.sum(Y ** 2) / float(N)
    P_x = np.sum(x ** 2) / float(N) + np.sum(r_m ** 2)
    snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))
    return snr_est

def vca(Y, R, verbose=True, snr_input=0):
    if len(Y.shape) != 2:
        sys.exit('Input data must be of size L (number of bands) by N (number of pixels)')

    [L, N] = Y.shape
    R = int(R)
    if (R < 0 or R > L):
        sys.exit('ENDMEMBER parameter must be integer between 1 and L')

    if snr_input == 0:
        y_m = np.mean(Y, axis=1, keepdims=True)
        Y_o = Y - y_m
        Ud = np.linalg.svd(Y_o, full_matrices=False)[0][:, :R]
        x_p = np.dot(Ud.T, Y_o)
        SNR = estimate_snr(Y, y_m, x_p)
    else:
        SNR = snr_input

    SNR_th = 15 + 10 * np.log10(R)

    if SNR < SNR_th:
        if verbose:
            pass  # print("... Select proj. to R-1")
        d = R - 1
        if snr_input == 0:
            Ud = Ud[:, :d]
        else:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m
            Ud = np.linalg.svd(Y_o, full_matrices=False)[0][:, :d]
            x_p = np.dot(Ud.T, Y_o)

        Yp = np.dot(Ud, x_p[:d, :]) + y_m
        x = x_p[:d, :]
        c = np.amax(np.sum(x ** 2, axis=0)) ** 0.5
        y = np.vstack((x, c * np.ones((1, N))))
    else:
        d = R
        Ud = np.linalg.svd(Y, full_matrices=False)[0][:, :d]
        x_p = np.dot(Ud.T, Y)
        Yp = np.dot(Ud, x_p[:d, :])
        x = np.dot(Ud.T, Y)
        u = np.mean(x, axis=1, keepdims=True)
        y = x / np.dot(u.T, x)

    indice = np.zeros((R,), dtype=int)
    A = np.zeros((R, R))
    A[-1, 0] = 1

    for i in range(R):
        w = np.random.rand(R, 1)
        f = w - np.dot(A, np.dot(np.linalg.pinv(A), w))
        f = f / np.linalg.norm(f)
        v = np.dot(f.T, y)
        indice[i] = np.argmax(np.abs(v))
        A[:, i] = y[:, indice[i]]

    Ae = Yp[:, indice]
    return Ae, indice, Yp