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

def BHZ_FT(p,q,kx,ky,M=3,A=5,B=1,C=0,D=0):
    """
    2L x 2L Hamiltonian for the BHZ model
    """
    size = q
    Phi = p/q
    
    # make diagonals
    cos_phis = [2*np.cos(ky - 2*np.pi*(Phi)*m) for m in range(q)]
    sin_phis = [np.sin(ky - 2*np.pi*(Phi)*m) for m in range(q)]
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

def get_phis_eps_FT(qmax=10,kx=0,ky=0,M=3,A=5,B=1,C=0,D=0):
    """
    Phis and Energies required to plot Hofstadter's butterfly for 
    a given kz
    """
    # initialize
    phi = []
    eps = []

    # fill up rest
    for q in range(1,qmax):
        H_dim = 2*q
        for p in range(0,q):
            # get H and eps
            H_pq=BHZ_FT(p=p,q=q,kx=kx,ky=ky,M=M,A=A,B=B,C=C,D=D)
            eigs_pq=ssl.eigsh(H_pq,k=H_dim,sigma=0,return_eigenvectors=False)
            # add phi
            phi.extend([p/q]*H_dim + [(q-p)/q]*H_dim)
            eps.extend([eigs_pq]*2)
    
    # convert into ndarray
    # phi = np.concatenate(phi)
    phi = np.asarray(phi)
    eps = np.concatenate(eps) # necessary evil
    eps = np.asarray(eps)

    return phi, eps

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

def plot_spectrum(all_ks, all_Es, k_start = -2*np.pi, k_end = 2*np.pi, Phis = [0,1/30,1/10], Phi_labels=["0","1/30","1/10"]):
    """
    Plot the spectrum data
    """
    # get length from all_ks
    n = all_ks.shape[0]
    # set up subplots
    fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(20, 10), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    for i in range(n):
        phi_label = Phi_labels[i]
        phi = Phis[i]
        # set labels
        if i == 0:
            ax[i].set_ylabel(r"$E$", fontsize=15)
        ax[i].set_xlabel(r"$k_y$", fontsize= 15)
    
        # plot
        ax[i].scatter(all_ks[i,:], all_Es[i,:],c='k',s=0.01)
        ax[i].axhline(y=-3*2*np.pi*Phis[i]/2)
        #ax[i].axvline(x=-np.pi*phi,ls='--',c='orange')
        ax[i].set_title(r"$\Phi = {}$".format(phi_label))

def get_indices(waves):
    """
    Returns surface and bulk indices for a given set of waves
    """
    prob_ = np.abs(waves)**2
    # batch
    prob = [np.sum(prob_[i:i+4,:], axis=0) for i in range(0, len(waves[:,0]), 4)]
    prob = np.asarray(prob)
    prob_tot = np.sum(prob, axis=0)
    
    # cutoff
    length = np.size(prob[:,0])
    len10 = int(length/10)
    flags = np.zeros((prob.shape[1]), dtype=int)
    # hinges
    # 50% within 10% of corners

    # surface
    # 50% within 10% of surfaces
    # not already labelled hinges
    prob_left = np.sum(prob[0:len10,:], axis=0)
    frac_left = prob_left/prob_tot

    prob_right = np.sum(prob[length-len10:length,:], axis=0)
    frac_right = np.divide(prob_right, prob_tot)

    for i in range(len(flags)):
        if frac_left[i]>0.5 or frac_right[i]>0.5:
            flags[i] = 1
            
    indices = [i for i, x in enumerate(flags) if x == 1]
    indices0 = [i for i, x in enumerate(flags) if x == 0]
    
    return indices, indices0

def get_phis_eps(qmax=10,ucsize=1,ky=0,M=3,A=5,B=1,C=0,D=0):
    """
    Phis and Energies required to plot Hofstadter's butterfly for 
    a given kz
    """
    # initialize
    phi_surf = []
    eps_surf = []
    phi_bulk = []
    eps_bulk = []

    # fill up rest
    for q in range(1,qmax):
        size = q * ucsize
        H_dim = 2*size
        for p in range(0,q):
            # get H and eps
            H_pq=BHZ_edge(size=size,p=p,q=q,ky=ky,M=M,A=A,B=B,C=C,D=D)
            eigs_pq,waves=ssl.eigsh(H_pq,k=H_dim,sigma=0,return_eigenvectors=True)
            idx,idx0=get_indices(waves)
            # add phi
            phi_surf.extend([p/q]*len(idx) + [(q-p)/q]*len(idx))
            eps_surf.extend([eigs_pq[idx]]*2)
            phi_bulk.extend([p/q]*len(idx0) + [(q-p)/q]*len(idx0))
            eps_bulk.extend([eigs_pq[idx0]]*2)
    
    # convert into ndarray
    # phi = np.concatenate(phi)
    phi_surf = np.asarray(phi_surf)
    phi_bulk = np.asarray(phi_bulk)
    eps_surf = np.concatenate(eps_surf) # necessary evil
    eps_surf = np.asarray(eps_surf)
    eps_bulk = np.concatenate(eps_bulk) # necessary evil
    eps_bulk = np.asarray(eps_bulk)

    return phi_surf, eps_surf, phi_bulk, eps_bulk

def variable_M_LL(size,p,q,ky=1.5,M_start=-4,M_end=4,A=5,B=1,C=0,D=0):
    """
    Plots how the Landau levels change with M
    """
    # make array of Ms we'll want to investigate
    num=100
    Ms = np.linspace(M_start,M_end,num=num)

    Masses = []
    Energies0 = []
    Energiesp1 = []
    Energiesm1 = []

    # loop over Ms
    for M in Ms:
        # Hamiltonian(M)
        H_M = BHZ_edge(size=size,p=p,q=q,ky=ky,M=M,A=A,B=B,C=C,D=D)
        Es_M, Ws_M = ssl.eigsh(H_M,k=80,sigma=0,return_eigenvectors=True)
        Es_M_levels, degeneracy = get_levels(Es_M,Ws_M)
        Masses.extend([M]*degeneracy)
        Energies0.extend(Es_M_levels[0:degeneracy])
        Energiesp1.extend(Es_M_levels[degeneracy:2*degeneracy])
        Energiesm1.extend(Es_M_levels[2*degeneracy:3*degeneracy])
        #Energies.extend(Es_M_levels)

    # don't return waves for now
    return Masses, Energies0, Energiesp1, Energiesm1

def get_levels(Es,Ws):
    """
    Returns 0th, +/- 1 level energies (inclusive)
    """
    Es_rounded = np.around(Es,decimals=1)
    # the degeneracy is the count of the 0th LL
    _, count = np.unique(np.abs(Es_rounded),return_counts=True)
    #degeneracy = int(count[0])
    # alternatively, manually impose degeneracy for size/q=90/30=3
    degeneracy = 3

    # initialize Es_levels
    # n = 1 for noe
    n_levels = int(degeneracy*(2*1+1))
    Es_levels = np.zeros((n_levels),dtype=float)
    Ws_levels = np.zeros((Ws.shape[0],n_levels),dtype=complex)

    # 0th level energies
    Es_levels[0:degeneracy] = Es[0:degeneracy]
    Ws_levels[:,0:degeneracy] = Ws[:,0:degeneracy]

    # positive and negative energies
    Es_no0 = Es[degeneracy:]
    Ws_no0 = Ws[:,degeneracy:]
    plus_idx = np.argwhere(Es_no0>0).T[0]
    minus_idx = np.argwhere(Es_no0<0).T[0]
    Es_plus = Es_no0[plus_idx]
    Ws_plus = Ws_no0[:,plus_idx]
    Es_minus = Es_no0[minus_idx]
    Ws_minus = Ws_no0[:,minus_idx]

    # +/- 1 levels
    # for +, we start ahead because of the 0th level
    Es_levels[degeneracy:2*degeneracy] = Es_plus[0:degeneracy]
    Ws_levels[:,degeneracy:2*degeneracy] = Ws_plus[:,0:degeneracy]
    Es_levels[2*degeneracy:3*degeneracy] = Es_minus[0:degeneracy]
    Ws_levels[:,2*degeneracy:3*degeneracy] = Ws_minus[:,0:degeneracy]


    return Es_levels, degeneracy

# functions to work with wavefunctions (eigenstates)

def find_plateaus(Es):
    """
    Returns wavefunctions and Energy (singular) of all the plateau
    A point on a plateau is defined as having one neighbour at the same (close) energy
    Assumes Es is sorted
    """
    # round energies to impose tolerance TOL = e-2
    Es_rounded = np.around(Es,decimals=2)
    
    # get unique energies, indices, and counts
    Es_unique, idx_unique, counts_unique = np.unique(Es_rounded, return_index=True, return_counts=True)
    
    # for each unique energy, impose that a plateau must have a count of 50
    deg = 200
    idx_plateau = idx_unique[counts_unique > deg]
    counts_plateau = counts_unique[counts_unique > deg]
    Es_plateau = Es_unique[counts_unique > deg]
    
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
    waves_up = waves[0::2]
    waves_down = waves[1::2]

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

# input the wavefunction index (corresponding to the energy)
# return the spin up (re and im) and spin down (re and im) parts
def plot_waves(wave):
    """
    wave is a slice of the returned waves array from ssl.eigsh
    """
    spin_up, spin_down = split_spins(wave)
    spin_up_re, spin_up_im = split_real_imaginary(spin_up)
    spin_down_re, spin_down_im = split_real_imaginary(spin_down)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10,5))
    fig.subplots_adjust(wspace=0.1, hspace=0.25)
    
    pos = np.arange(0,len(spin_down_re))
    
    ax[0].set_title(r"spin $\uparrow$ wavefunction for $k_y = 1.5$ and $\Phi = 1/30$")
    ax[0].plot(pos,spin_up_re,label="Re",c='cornflowerblue')
    ax[0].plot(pos,spin_up_im,label="Im",c='orange')
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$\psi$")
    ax[0].legend()
    
    ax[1].set_title(r"spin $\downarrow$ wavefunction for $k_y = 1.5$ and $\Phi = 1/30$")
    ax[1].plot(pos,spin_down_re,label="Re",c='cornflowerblue')
    ax[1].plot(pos,spin_down_im,label="Im",c='orange')
    ax[1].set_xlabel("$x$")
    ax[1].set_ylabel("$\psi$")
    ax[1].legend()
    
    plt.show()

def get_phase(psi):
    """
    Returns phase of wavefunction (array)
    """
    phases = np.angle(psi) # <- down is a choice (Gaussian is cleaner)
    counts = np.unique(np.around(phases,decimals=5),return_counts=True)
    phase = phases[np.argmax(counts)]
    
    return phase

def adjust_for_phase(wave):
    """
    Adjusts the spin-up/down wave to make spin down fully real
    """
    spin_up, spin_down = split_spins(wave)

    phase = get_phase(spin_down)
    factor = np.exp(-1j*phase)

    spin_down *= factor
    spin_up *= factor

    re_up_fixed, im_up_fixed = split_real_imaginary(spin_up)
    re_do_fixed, im_do_fixed = split_real_imaginary(spin_down)

    return re_up_fixed,im_up_fixed,re_do_fixed,im_do_fixed

def get_coefficients(wave):
    """
    Returns u and v for the LL-form eigenstate
    |n^th LL > = (u|up >, v|down >)^T
    """
    # split everything
    spin_up, spin_down = split_spins(wave)
    spin_up_re, spin_up_im = split_real_imaginary(spin_up)
    spin_down_re, spin_down_im = split_real_imaginary(spin_down)
    
    # fix phase
    phase = get_phase(spin_down)
    factor = np.exp(-1j*phase)
    
    # fix waves
    spin_up *= factor
    spin_down *= factor
    
    # get v
    v_s = np.sum(np.abs(spin_down)**2)
    v = np.sqrt(v_s)
    
    # get u
    u_s = np.sum(np.abs(spin_up)**2)
    u = np.sqrt(u_s)
    phase_u = get_phase(spin_up)
    u *= np.exp(1j*phase_u)
    
    return u, v






# main functions for cluster

import sys
ks = np.linspace(-np.pi,np.pi,5,endpoint=True)

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
    with open("bhz_energies_data_edge.csv","a") as f:
        np.savetxt(f,Es,delimiter=',')
    with open("bhz_waves_data_edge.csv","a") as f:
        np.savetxt(f,Ws,delimiter=',')

if __name__ == "__main__":
    # run program
    # main_energies(size = 90, p = 1, q = 30)
    #main_butterfly(qmax=50,ucsize=3,M=3,A=5,B=1,C=0,D=0)
    main_energies_edges(size=90,p=1,q=30)

### Leo PBG