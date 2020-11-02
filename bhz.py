###

import numpy as np
import scipy.sparse.linalg as ssl
import scipy.sparse as ss
import matplotlib.pyplot as plt

# pauli matrices
def sig(n):
    # pauli matrices
    # n = 0 is identity, n = 1,2,3 is x,y,z resp.
    if n == 0:
        a = np.identity(2, dtype = complex)
    if n == 1:
        a = np.array([[0 , 1],[1 , 0]], dtype = complex)
    if n == 2:
        a = np.array([[0 , -1j],[1j , 0]], dtype = complex)
    if n == 3:
        a = np.array([[1 , 0],[0 , -1]], dtype = complex)
    return a

# fully FT matrix

def BHZ_FT(p,q,kx,ky,M,A,B,C,D):
    """
    2L x 2L Hamiltonian for the BHZ model
    """
    size = q
    Phi = p/q
    
    # make diagonals
    cos_phis = [2*np.cos(ky - 2*np.pi*(Phi)*m) for m in range(size)]
    sin_phis = [np.sin(ky - 2*np.pi*(Phi)*m) for m in range(size)]
    diags_cos_phis = np.kron(np.diag(cos_phis), B * sig(3) + D * sig(0))
    diags_sin_phis = np.kron(np.diag(sin_phis), A * sig(2))
    mass_const = (M - 4*B) * sig(3) + (C - 4*D) * sig(0)
    diags_const = np.kron(np.identity(n=size),mass_const)
    A_diags = diags_const + diags_sin_phis + diags_cos_phis
    
    # off diagonals m -> m+1 & h.c.
    # note: m -> m+1 is lower diagonal
    hop = B * sig(3) + 1j * A/2 * sig(1) + D * sig(0)
    
    # put into off diagonals
    A_bot_diag = np.kron(np.diag(np.ones(size-1),k=-1),hop)    
    A_bot_diag[0:2,2*(size-1):2*size] = np.exp(1j * kx * q) * hop
    
    A_off_diags = A_bot_diag + A_bot_diag.conj().T
    
    A = A_diags + A_off_diags
    
    return A

# partially FT in y -> ky
# real space model for the BHZ

def BHZ_edge(size, p, q, ky, M=3, A=5, B=1, C=0, D=0):
    """
    Block of X-dependencies corresponding to one Ky to go into full BHZ matrix
    """
    Phi = p/q
    
    # make diagonals
    cos_phis = [2*np.cos(ky - 2*np.pi*(Phi)*x) for x in range(size)]
    sin_phis = [np.sin(ky - 2*np.pi*(Phi)*x) for x in range(size)]
    diags_cos_phis = np.kron(np.diag(cos_phis), B * sig(3) + D * sig(0))
    diags_sin_phis = np.kron(np.diag(sin_phis), A * sig(2))
    mass_const = (M - 4*B) * sig(3) + (C - 4*D) * sig(0)
    diags_const = np.kron(np.identity(n=size),mass_const)
    A_diags = diags_const + diags_sin_phis + diags_cos_phis
    
    # off diagonals m -> m+1 & h.c.
    # note: x -> x+1 is lower diagonal
    hop = B * sig(3) + 1j * A/2 * sig(1) + D * sig(0)
    
    # put into off diagonals
    A_bot_diag = np.kron(np.diag(np.ones(size-1),k=-1),hop)
    
    # impose open boundary condition
    A_bot_diag[0:2,2*(size-1):2*size] = np.zeros((2,2),dtype=complex)
    
    A_off_diags = A_bot_diag + A_bot_diag.conj().T
    
    A_total = A_diags + A_off_diags
    
    return A_total

def BHZ_edge_full(size, p, q, M=3, A=5, B=1, C=0, D=0):
    """
    Full matrix for the PFT BHZ model matrix
    """
    # put blocks into big diag
    
    # make blocks array with dims (size, 2 x size, 2 x size)
    blocks = np.zeros((size,2*size,2*size),dtype=complex) 
    
    # fill up
    kys = np.linspace(-np.pi,np.pi,num=size)
    for i in range(size):
        ky = kys[i]
        blocks[i,:,:] = BHZ_edge(size=size, p=p, q=q, ky=ky, M=M, A=A, B=B, C=C, D=D)

    # put in diagonal
    MAT = ss.block_diag(blocks).toarray()

    return MAT

def BHZ_real_block(size, p, q, x, M=3, A=5, B=1, C=0, D=0):
    """
    Block of Y-dependencies corresponding to one X to go into full BHZ matrix
    Note that we are now making our space X x Y
    """
    Phi = p/q

    # make diagonals
    mass_const = (M - 4*B) * sig(3) + (C - 4*D) * sig(0)
    diags_const = np.kron(np.identity(n=size),mass_const)
    
    # off diagonals m -> m+1 & h.c.
    # note: x -> x+1 is lower diagonal
    hop_y = (B * sig(3) + 1j * A/2 * sig(2) + D * sig(0)) * np.exp(1j * 2*np.pi * Phi * x)
    
    # put into off diagonals
    A_bot_diag = np.kron(np.diag(np.ones(size-1),k=-1),hop_y)
    
    # impose open boundary condition
    # could just leave blank but better to write out
    # for physical reasons
    A_bot_diag[0:2,2*(size-1):2*size] = np.zeros((2,2),dtype=complex)
    
    A_off_diags = A_bot_diag + A_bot_diag.conj().T
    
    A_total = diags_const + A_off_diags
    
    return A_total

def BHZ_real(size, p , q, M=3, A=5, B=1, C=0, D=0):
    """
    Full Hamiltonian matrix for the real-space BHZ model
    Ordered as X x Y --> each block corresponds to a fixed X
    And within these blocks we have Y-dependencies
    """
    # put blocks into big diag
    
    # make blocks array with dims (size, 2 x size, 2 x size)
    blocks = np.zeros((size,2*size,2*size),dtype=complex) 
    
    # fill up
    xs = np.linspace(0,size,num=size)
    for i in range(size):
        x = xs[i]
        blocks[i,:,:] = BHZ_real_block(size=size, p=p, q=q, x=x, M=M, A=A, B=B, C=C, D=D)
        
    # put in diagonal
    M_diags = ss.block_diag(blocks).toarray()
    
    # off diagonals x -> x+1 & h.c.
    hop_x = B * sig(3) + 1j * A/2 * sig(1) + D * sig(0)
    
    # fill up to identity
    hop_x_mat = np.kron(np.identity(n=size), hop_x)
    
    # put these "identity" matrices on the off-diagonals
    ### double check the math for this section please
    M_top_diag = np.kron(np.diag(np.ones(size-1), k=1), hop_x_mat)
    M_bot_diag = M_top_diag.conj().T
    
    M_off_diags = M_top_diag + M_bot_diag
    
    MAT = M_diags + M_off_diags
    
    return MAT

# functions that analyze the models

def spectrum_FT(k_choice, p, q, k_other = 0, k_res = 1000, k_start = -2*np.pi, k_end = 2*np.pi, M=3, A=5, B=1, C=0, D=0):
    """
    Spectrum for BHZ real space model
    k_choice can be either 1 (kx) or 2 (ky)
    """
    # kz
    ks = np.linspace(k_start,k_end,num=k_res,endpoint=False) # to get pretty mid point

    # for each kz, get Es from soti_block
    k_ret = []
    Es = []

    # set the number of eigs returned
    num_eigs = int(2*q) # <- 2 comes from spin dof
    
    # get the energies as a function of k

    # kx
    if k_choice == 1: 
    	for k in ks:
        	H_k = BHZ_FT(p=p,q=q,kx=k,ky=k_other,M=M,A=A,B=B,C=C,D=D)
        	E_k = np.linalg.eigvalsh(H_k)
        	Es.extend(E_k)
        	k_ret.extend([k]*num_eigs)

    # ky
    elif k_choice == 2:
    	for k in ks:
        	H_k = BHZ_FT(p=p,q=q,kx=k_other,ky=k,M=M,A=A,B=B,C=C,D=D)
        	E_k = np.linalg.eigvalsh(H_k)
        	Es.extend(E_k)
        	k_ret.extend([k]*num_eigs)

    return k_ret, Es

def spectrum_edge(size, p, q, k_res = 1000, k_start = -2*np.pi, k_end = 2*np.pi, M=3, A=5, B=1, C=0, D=0):
    """
    Spectrum for BHZ real space model
    """
    # kz
    ks = np.linspace(k_start,k_end,num=k_res,endpoint=False) # to get pretty mid point

    # for each kz, get Es from soti_block
    k_ret = []
    Es = []

    # set the number of eigs returned
    num_eigs = int(2*size) # <- 2 comes from spin dof
    
    # get the energies as a function of k

    # ky
    for k in ks:
        H_k = BHZ_edge(size=size,p=p,q=q,ky=k,M=M,A=A,B=B,C=C,D=D)
        E_k = np.linalg.eigvalsh(H_k)
        Es.extend(E_k)
        k_ret.extend([k]*num_eigs)

    return k_ret, Es

def spectrum_real(size, p, q, M=3, A=5, B=1, C=0, D=0):
    """
    Spectrum for BHZ real space model
    choice can be either 1 (x) or 2 (y)
    """
    # pos
    pos = np.linspace(0, size, num=size, endpoint=False)

    # for each pos, get Es from BHZ_real
    pos_ret = []
    Es = []

    # set the number of eigs returned
    num_eigs = int(2*size) # <- 2 comes from spin dof
    
    # get the energies as a function of x

    # x
    for i in range(size):
        x = pos[i] # symbolic - could be y
        H = BHZ_real_block(size=size,p=p,q=q,x=x,M=M,A=A,B=B,C=C,D=D)
        E, waves = np.linalg.eigh(H)
        Es.extend(E)
        pos_ret.extend([x]*num_eigs)

    return pos_ret, Es

def spectrum_plots_edge(size, ps=[0,1,10], qs=[30,30,30], k_res = 1000, k_start = -2*np.pi, k_end = 2*np.pi, M=3, A=5, B=1, C=0, D=0):
    """
    Plots of Energy as a function of k for the BHZ model for various magnetic flux 
    """
    # initialize
    all_ks=[]
    all_Es=[]

    # fill them up
    for i in range(len(ps)):
        p = ps[i]
        q = qs[i]
        ks, Es = spectrum_edge(size, p, q, k_res=k_res,k_start=k_start,k_end=k_end,M=M,A=A,B=B,C=C,D=D)

        all_ks.append(ks)
        all_Es.append(Es)

    return np.asarray(all_ks), np.asarray(all_Es)

def plot_spectrum(all_ks, all_Es, k_start = -2*np.pi, k_end = 2*np.pi, Phi_labels=["0","1/30","1/10"]):
    """
    Plot the spectrum data
    """
    futura = {'fontname':'Futura'}
    # get length from all_ks
    n = all_ks.shape[0]
    # set up subplots
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 10), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    for i in range(n):
        phi_label = Phi_labels[i]
        # set labels
        if i == 0:
            ax[i].set_ylabel(r"$E$", fontsize=15, **futura)
        ax[i].set_xlabel(r"$k_y$", fontsize= 15, **futura)
    
        # plot
        ax[i].scatter(all_ks[i,:], all_Es[i,:],c='k',s=0.01)
        ax[i].axvline(x=0,ls='--',c='orange')
        ax[i].set_title(r"$\Phi = {}$".format(phi_label), **futura)

def get_phis_eps(qmax=10,ucsize=1,ky=0,M=3,A=5,B=1,C=0,D=0):
    """
    Phis and Energies required to plot Hofstadter's butterfly for 
    a given kz
    """
    # initialize
    phi = []
    eps = []

    # fill up rest
    for q in range(1,qmax):
        size = q * ucsize
        H_dim = 2*size
        for p in range(0,q):
            # add phi
            phi.extend([p/q]*H_dim + [(q-p)/q]*H_dim)
            # get H and eps
            H_pq = BHZ_real(size=size,p=p,q=q,M=M,A=A,B=B,C=C,D=D)
            eigs_pq = np.linalg.eigvalsh(H_pq)
            eps.extend([eigs_pq]*2)
    
    # convert into ndarray
    phi = np.asarray(phi)
    eps = np.concatenate(eps) # necessary evil
    eps = np.asarray(eps)

    return phi, eps

# functions to work with wavefunctions (eigenstates)

def find_plateaus(Es):
    """
    Returns wavefunctions and Energy (singular) of all the plateau
    A point on a plateau is defined as having one neighbour at the same (close) energy
    """
    # round energies to impose tolerance TOL = e-2
    Es_rounded = np.around(Es,decimals=2)
    
    # get unique energies, indices, and counts
    Es_unique, idx_unique, counts_unique = np.unique(Es_rounded, return_index=True, return_counts=True)
    
    # for each unique energy, impose that a plateau must have a count of 50
    idx_plateau = idx_unique[counts_unique > 200]
    counts_plateau = counts_unique[counts_unique > 200]
    Es_plateau = Es_unique[counts_unique > 200]
    
    return Es_plateau, idx_plateau, counts_plateau

def batch_energies(Es,Ws):
    """
    Combine states with the same energy by adding up their complex amplitudes
    """
    # get the indices and counts
    Es_unique, idx_plateau, counts_plateau = find_plateaus(Es)
    Es_plateau = Es_unique
    n = len(idx_plateau)
    
    # array of waves vector_length x number of vectors
    waves_plateau = np.zeros((Ws.shape[0],n), dtype=complex)
    
    # for each plateau, add up the complex amplitudes
    for i in range(n):
        idx = idx_plateau[i]
        counts = counts_plateau[i]
        wave_added = np.sum(Ws[:,idx:idx+counts], axis=1)
        wave_added_reshaped = wave_added.flatten() # because H_real is a matrix so it returns Ws as a matrix -- fixed now
        waves_plateau[:,i] = wave_added_reshaped
            
    return Es_plateau, waves_plateau

def split_real_imaginary(waves):
    """
    Returns real and imaginary part of waves
    """
    # split waves into real and imaginary part
    waves_real = np.real(waves)
    waves_imaginary = np.imag(waves)

    return waves_real, waves_imaginary

def split_spins(waves):
    """
    Separates spin up and spin down part of wave
    Assumes up is (1,0)^T and down is (0,1)^T
    Opposite of batch spins
    """
    # split up and down
    waves_up = waves[0::2,:]
    waves_down = waves[1::2,:]

    return waves_up, waves_down

def split_full(waves):
    """
    Combines spin and real/imaginary split
    """
    # split spins
    waves_up, waves_down = split_spins(waves)

    # split re/im
    waves_up_re, waves_up_im = split_real_imaginary(waves_up)
    waves_down_re, waves_down_im = split_real_imaginary(waves_down)

    return waves_up_re, waves_up_im, waves_down_re, waves_down_im










# main functions for cluster

import sys
ks = np.linspace(-np.pi,np.pi,101,endpoint=True)

def main_butterfly(qmax=10,ucsize=1,M=3,A=5,B=1,C=0,D=0):
    """
    Main function to be run on cluster
    """
    # get kz from argv
    args = sys.argv
    k_idx = int(args[1])
    k = ks[k_idx]
    # run program
    phi, eps = get_phis_eps(qmax=qmax,ucsize=ucsize,ky=k,M=M,A=A,B=B,C=C,D=D)
    # save in file
    with open("bhz_butterfly_data.csv","a") as f:
        np.savetxt(f,(phi,eps),delimiter=',')

def main_energies(size=90,p=1,q=30):
    """
    Main function for ordered energies to be run on cluster
    """
    H_real = BHZ_real(size=size,p=p,q=q)
    Es, Ws = np.linalg.eigh(H_real)
    with open("bhz_energies_data.csv","a") as f:
        np.savetxt(f,(Es,Ws),delimiter=',')

def main_energies_edges(size=90,p=1,q=30):
    """
    Main function for ordered energies to be run on cluster
    """
    H_edge = BHZ_edge_full(size=size,p=p,q=q)
    Es, Ws = np.linalg.eigh(H_edge)
    with open("bhz_energies_edge_data.csv","a") as f:
        np.savetxt(f,(Es,Ws),delimiter=',')

if __name__ == "__main__":
    # run program
    # main_energies(size = 90, p = 1, q = 30)
    #main_butterfly(qmax=50,ucsize=3,M=3,A=5,B=1,C=0,D=0)
    main_energies_edges(size=90,p=1,q=30)

### Leo PBG