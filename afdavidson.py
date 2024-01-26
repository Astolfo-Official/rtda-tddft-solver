import numpy as np
from scipy.linalg import eigh
import time
import scipy.linalg
from pyscf.lib import logger
from functools import reduce

def TDA_diag_preconditioner(residual, sub_eigenvalue, hdiag):
    '''
    DX - XΩ = r
    '''
    N_states = np.shape(residual)[0]
    t = 1e-8
    D = np.repeat(hdiag.reshape(-1,1), N_states, axis=1) - sub_eigenvalue
    '''
    force all small values not in [-t,t]
    '''
    D = np.where( abs(D) < t, np.sign(D)*t, D)
    D = D.transpose(1,0)
    new_guess = residual/D

    return new_guess

def Gram_Schmidt_fill_holder(V, count, vecs, double = True):
    '''V is a vectors holder
    count is the amount of vectors that already sit in the holder
    nvec is amount of new vectors intended to fill in the V
    count will be final amount of vectors in V
    '''
    nvec = np.shape(vecs)[0]
    for j in range(nvec):
        vec = vecs[j,:].reshape(-1,1)
        vec = Gram_Schmidt_bvec(V[:count,:], vec)   #single orthonormalize
        if double == True:
            vec = Gram_Schmidt_bvec(V[:count, :], vec)   #double orthonormalize
        norm = np.linalg.norm(vec)
        if  norm > 1e-14:
            vec = vec/norm
            V[count,:] = vec.transpose(1,0)
            count += 1
    new_count = count
    return V, new_count

def Gram_Schmidt_bvec(A, bvec):
    '''orthonormalize vector b against all vectors in A
    b = b - A.T*(A*b)
    suppose A is orthonormalized
    '''
    if A.shape[0] != 0:
        projections_coeff = np.dot(A, bvec)
        bvec -= np.dot(A.T, projections_coeff)
    return bvec

def af_davidson(mf, vind, hdiag, nstate=1, conv_tol=1e-5, max_cycle=100, verbose=10):
    log = logger.new_logger(verbose)
    log.debug('====== Davidson Diagonalization Beginning ======')
    time_start = time.time()

    mo_occ = mf.mo_occ
    occidx = np.where(mo_occ==2)[0]
    viridx = np.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    
    A_size = hdiag.shape[0]
    log.debug('size of A matrix = %4.3g', A_size)
    log.debug('conv_tol = %4.3g', conv_tol)
    size_old = 0
    size_new = min([nstate+8, 2*nstate, A_size])

    max_N_mv = max_cycle*nstate + size_new
    v   = np.zeros((max_N_mv, A_size))
    Av  = np.zeros_like(v)
    vAv = np.zeros((max_N_mv,max_N_mv))

    from pydmet.solvers.rtda import TDA as tda
    init = tda(mf).init_guess(mf=mf, nstates=size_new)
    log.debug('full initial guess size = %4.3g', np.shape(init)[0])
    v[size_old:size_new,:] = init[size_old:size_new,:]

    for cycle in range(max_cycle):
        Av[size_old:size_new,:] = vind(v[size_old:size_new,:])
        vAv = np.dot(v,Av.T)
        sub_vAv = vAv[:size_new,:size_new]
        assert np.shape(sub_vAv) == (size_new,size_new)

        lambda_vAv, alpha_vAv = eigh(sub_vAv)
        subset_lambda = lambda_vAv[:nstate]
        subset_alpha = alpha_vAv[:,:nstate]

        alpha_Av = np.einsum('ji,jk->ik', subset_alpha, Av[:size_new,:])
        alpha_v = np.einsum('ji,jk->ik', subset_alpha, v[:size_new,:])
        error = alpha_Av - np.einsum('i,ik->ik', subset_lambda, alpha_v)
        r_norms = np.linalg.norm(error, axis=1).tolist()
        max_norm = np.max(r_norms)
        log.debug('step = %3d, sub_A.shape = %3d, max|r| = %8.3g, e = %s', cycle+1, sub_vAv.shape[0], max_norm, subset_lambda)
        
        if max_norm < conv_tol or cycle == (max_cycle-1):
            break

        index = [r_norms.index(i) for i in r_norms if i>conv_tol]

        new_guess = TDA_diag_preconditioner(residual = error[index,:],
                                            sub_eigenvalue = subset_lambda[index],
                                            hdiag = hdiag)

        size_old = size_new
        v, size_new = Gram_Schmidt_fill_holder(v, size_old, new_guess, double=True)

    if cycle == max_cycle-1:
        log.debug('=== TDA Failed Due to Iteration Limit ===')
        log.debug('current residual norms %4.3g', r_norms)
        
    time_end = time.time()
    log.debug('Maximum residual norm = %4.3g', max_norm)
    log.debug('Final subspace size = %d', sub_vAv.shape[0])
    log.debug('Cost time = %6.6gs', time_end - time_start)
    log.debug('====== Davidson Diagonalization End ======\n')

    return subset_lambda, alpha_v


def TDDFT_diag_preconditioner(R_x, R_y, omega, hdiag):
    '''
    preconditioners for each corresponding residual (state)
    '''
    N_states = R_x.shape[0]
    t = 1e-14
    d = np.repeat(hdiag.reshape(-1,1), N_states, axis=1)

    D_x = d - omega
    D_x = np.where(abs(D_x) < t, np.sign(D_x)*t, D_x)
    D_x_inv = D_x**-1
    
    D_y = d + omega
    D_y = np.where(abs(D_y) < t, np.sign(D_y)*t, D_y)
    D_y_inv = D_y**-1

    D_x_inv = D_x_inv.transpose(1,0)
    D_y_inv = D_y_inv.transpose(1,0)
    X_new = R_x*D_x_inv
    Y_new = R_y*D_y_inv

    return X_new, Y_new

def VW_Gram_Schmidt_fill_holder(V_holder, W_holder, m, X_new, Y_new, double = False):
    '''
    put X_new into V, and Y_new into W
    m: the amount of vectors that already on V or W
    nvec: amount of new vectors intended to put in the V and W
    '''
    nvec = np.shape(X_new)[0]
    for j in range(0, nvec):
        V = V_holder[:m,:]
        W = W_holder[:m,:]

        x_tmp = X_new[j,:].reshape(-1,1)
        y_tmp = Y_new[j,:].reshape(-1,1)

        x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)
        if double == True:
            x_tmp,y_tmp = VW_Gram_Schmidt(x_tmp, y_tmp, V, W)

        x_tmp,y_tmp = S_symmetry_orthogonal(x_tmp,y_tmp)

        xy_norm = (np.dot(x_tmp.T, x_tmp)+np.dot(y_tmp.T, y_tmp))**0.5

        if  xy_norm > 1e-14:
            x_tmp = x_tmp/xy_norm
            y_tmp = y_tmp/xy_norm

            V_holder[m,:] = x_tmp.transpose(1,0)
            W_holder[m,:] = y_tmp.transpose(1,0)
            m += 1
        else:
            print('vector kicked out during GS orthonormalization')

    new_m = m

    return V_holder, W_holder, new_m

def VW_Gram_Schmidt(x, y, V, W):
    '''orthonormalize vector |x,y> against all vectors in |V,W>'''
    m = np.dot(V,x) + np.dot(W,y)
    n = np.dot(W,x) + np.dot(V,y)
    x = x - np.dot(V.T,m) - np.dot(W.T,n)
    y = y - np.dot(W.T,m) - np.dot(V.T,n)
    return x, y

def S_symmetry_orthogonal(x,y):
    '''symmetrically orthogonalize the vectors |x,y> and |y,x>
    as close to original vectors as possible
    '''
    x_p_y = x + y
    x_p_y_norm = np.linalg.norm(x_p_y)

    x_m_y = x - y
    x_m_y_norm = np.linalg.norm(x_m_y)

    a = x_p_y_norm/x_m_y_norm

    x_p_y /= 2
    x_m_y *= a/2

    new_x = x_p_y + x_m_y
    new_y = x_p_y - x_m_y

    return new_x, new_y

def TDDFT_subspace_eigen_solver(a, b, sigma, pi, k):
    ''' [ a b ] x - [ σ   π] x  Ω = 0 '''
    ''' [ b a ] y   [-π  -σ] y    = 0 '''

    d = abs(np.diag(sigma))
    d_mh = d**(-0.5)

    s_m_p = d_mh.reshape(-1,1) * (sigma - pi) * d_mh.reshape(1,-1)

    '''LU = d^−1/2 (σ − π) d^−1/2'''
    ''' A = PLU '''
    ''' if A is diagonally dominant, P is identity matrix (in fact not always) '''
    P_permutation, L, U = scipy.linalg.lu(s_m_p)

    L = np.dot(P_permutation, L)

    L_inv = np.linalg.inv(L)
    U_inv = np.linalg.inv(U)

    # L_inv = np.linalg.cholesky(np.linalg.inv(s_m_p))
    # U_inv = L_inv.T
    ''' a ̃−b ̃= U^-T d^−1/2 (a−b) d^-1/2 U^-1 = GG^T '''
    dambd =  d_mh.reshape(-1,1)*(a-b)*d_mh.reshape(1,-1)
    GGT = np.linalg.multi_dot([U_inv.T, dambd, U_inv])

    G = scipy.linalg.cholesky(GGT, lower=True)
    G_inv = np.linalg.inv(G)

    ''' M = G^T L^−1 d^−1/2 (a+b) d^−1/2 L^−T G '''
    dapbd = d_mh.reshape(-1,1)*(a+b)*d_mh.reshape(1,-1)
    M = np.linalg.multi_dot([G.T, L_inv, dapbd, L_inv.T, G])

    omega2, Z = np.linalg.eigh(M)
    omega = (omega2**0.5)[:k]
    Z = Z[:,:k]

    ''' It requires Z^T Z = 1/Ω '''
    ''' x+y = d^−1/2 L^−T GZ Ω^-0.5 '''
    ''' x−y = d^−1/2 U^−1 G^−T Z Ω^0.5 '''

    x_p_y = d_mh.reshape(-1,1)\
            *np.linalg.multi_dot([L_inv.T, G, Z])\
            *(np.array(omega)**-0.5).reshape(1,-1)

    x_m_y = d_mh.reshape(-1,1)\
            *np.linalg.multi_dot([U_inv, G_inv.T, Z])\
            *(np.array(omega)**0.5).reshape(1,-1)

    x = (x_p_y + x_m_y)/2
    y = x_p_y - x

    return omega, x, y

def af_davidson_hybird(mf, vind, hdiag, nstate=1, conv_tol=1e-5, max_cycle=114, verbose=None):
    '''
    [ A B ] X - [1   0] Y Ω = 0
    [ B A ] Y   [0  -1] X   = 0

    '''
    log = logger.new_logger(verbose)
    log.debug('======= TDDFT eiegn solver statrs =======')
    time_start = time.time()

    mo_occ = mf.mo_occ
    occidx = np.where(mo_occ==2)[0]
    viridx = np.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    
    A_size = hdiag.shape[0]
    log.debug('size of A matrix = %4.3g', A_size)
    log.debug('conv_tol = %4.3g', conv_tol)
    size_old = 0
    size_new = min([nstate+8, 2*nstate, A_size])

    max_N_mv = max_cycle*nstate + size_new
    X  = np.zeros((max_N_mv, A_size))
    Y  = np.zeros_like(X)
    AXpBY = np.zeros_like(X)
    AYpBX = np.zeros_like(X)
    XAXpXBY = np.zeros((max_N_mv,max_N_mv))
    YAXpYBY = np.zeros_like(XAXpXBY)
    XAYpXBX = np.zeros_like(XAXpXBY)
    YAYpYBX = np.zeros_like(XAXpXBY)
    XX = np.zeros_like(XAXpXBY)
    XY = np.zeros_like(XAXpXBY)
    YY = np.zeros_like(XAXpXBY)
    '''
    set up initial guess X = TDA initila guess, Y=0
    '''
    from rtda import TDA as tda
    init = tda(mf).init_guess(mf=mf, nstates=size_new)
    X[size_old:size_new,:] = init

    for cycle in range(max_cycle):
        AXpBY[size_old:size_new,:], AYpBX[size_old:size_new,:] = vind(X=X[size_old:size_new,:],Y=Y[size_old:size_new,:])
        '''
        [U1] = [A B][V]
        [U2]   [B A][W]

        a = [V.T W.T][A B][V] = [V.T W.T][U1] = VU1 + WU2
                    [B A][W]            [U2]
        '''
        XAXpXBY = np.dot(X,AXpBY.T)
        YAXpYBY = np.dot(Y,AXpBY.T)
        XAYpXBX = np.dot(X,AYpBX.T)
        YAYpYBX = np.dot(Y,AYpBX.T)
        XX = np.dot(X,X.T)
        YY = np.dot(Y,Y.T)
        XY = np.dot(X,Y.T)
        sub_A = XAXpXBY[:size_new, :size_new] + YAYpYBX[:size_new, :size_new]
        sub_B = YAXpYBY[:size_new, :size_new] + XAYpXBX[:size_new, :size_new]
        sigma = XX[:size_new, :size_new] - YY[:size_new, :size_new]
        pi = XY[:size_new, :size_new] - XY[:size_new, :size_new].T
        '''
        solve the eigenvalue omega in the subspace
        '''
        omega, x, y = TDDFT_subspace_eigen_solver(sub_A, sub_B, sigma, pi, nstate)
        '''
        compute the residual
        R_x = U1x + U2y - X_full*omega
        R_y = U2x + U1y + Y_full*omega
        X_full = Vx + Wy
        Y_full = Wx + Vy
        '''
        X_full  = np.einsum('ji,jk->ik', x, X[:size_new,:])
        X_full += np.einsum('ji,jk->ik', y, Y[:size_new,:])
        Y_full  = np.einsum('ji,jk->ik', x, Y[:size_new,:])
        Y_full += np.einsum('ji,jk->ik', y, X[:size_new,:])

        R_x     = np.einsum('ji,jk->ik', x, AXpBY[:size_new,:])
        R_x    += np.einsum('ji,jk->ik', y, AYpBX[:size_new,:])
        R_x    -= np.einsum('ij,i->ij', X_full, omega)
        R_y     = np.einsum('ji,jk->ik', x, AYpBX[:size_new,:])
        R_y    += np.einsum('ji,jk->ik', y, AXpBY[:size_new,:])
        R_y    += np.einsum('ij,i->ij', Y_full, omega)

        residual = np.hstack((R_x, R_y))
        r_norms = np.linalg.norm(residual, axis=1).tolist()
        max_norm = np.max(r_norms)
        log.debug('step = %3d, sub_A.shape = %3d, max|r| = %8.3g, e = %s', cycle+1, sub_A.shape[0], max_norm, omega)
        if max_norm < conv_tol or cycle == (max_cycle -1):
            break

        index = [r_norms.index(i) for i in r_norms if i > conv_tol]
        '''
        preconditioning step
        '''
        X_new, Y_new = TDDFT_diag_preconditioner(R_x = R_x[index,:],
                                                R_y = R_y[index,:],
                                                omega = omega[index],
                                                hdiag = hdiag)
        '''
        GS and symmetric orthonormalizations
        '''
        size_old = size_new
        X, Y, size_new = VW_Gram_Schmidt_fill_holder(V_holder=X,
                                                    W_holder=Y,
                                                    X_new=X_new,
                                                    Y_new=Y_new,
                                                    m=size_old,
                                                    double=False)

        if size_new == size_old:
            print('All new guesses kicked out during GS orthonormalization')
            break

    time_end = time.time()

    if cycle == max_cycle-1:
        log.debug('=== TDDFT Failed Due to Iteration Limit ===')
        log.debug('current residual norms %4.3g', r_norms)
        
    time_end = time.time()
    log.debug('Maximum residual norm = %4.3g', max_norm)
    log.debug('Final subspace size = %d', sub_A.shape[0])
    log.debug('Cost time = %6.6gs', time_end - time_start)
    log.debug('======= TDDFT eigen solver Done =======\n')

    return omega, X_full, Y_full
