import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time


start = time.time()
omega = 1
omega_l = 2
gamma_h = 0.1
gamma_l = 0.1
dipole_h = 0.2
dipole_l = 0.1
E = 1
a_h= np.zeros((3,3),dtype=np.complex64)
a_h[0,1] = 1
# a_h[1,2] = 1
a_h_d = np.conjugate(a_h).T

a_l= np.zeros((3,3),dtype=np.complex64)
a_l[0,2] = 1
a_l_d = np.conjugate(a_l).T

rho0 =  np.zeros((3,3),dtype=np.complex64)
rho0[0,0]=1
H_s =  omega*(a_h_d@a_h)+ omega_l*(a_l_d@a_l)
mu =  dipole_h *(a_h+a_h_d)+dipole_l*(a_l+a_l_d)
H_d = 0
H = H_s+H_d
# H[0,0] = 0.1
# Only decay
L_ = [np.sqrt(gamma_l)*a_l,np.sqrt(gamma_h)*a_h,np.sqrt(gamma_l*0.1)*a_l_d,np.sqrt(gamma_h*0.1)*a_h_d]
    
rho0 =  np.zeros((3,3),dtype=np.complex64)
rho0[2,2]=1
rho0_vec = rho0.reshape(9)


def steadystate_(H, c_ops=[], sparse=False):
    """
    NumPy equivalent of QuTiP's steadystate() for Lindblad master equations.
    
    Parameters
    ----------
    H : np.ndarray
        Hamiltonian (NxN complex matrix)
    c_ops : list of np.ndarray
        List of collapse operators
    sparse : bool
        Placeholder (unused)
    
    Returns
    -------
    rho_ss : np.ndarray
        Steady-state density matrix (NxN)
    """
    N = H.shape[0]
    I = np.eye(N, dtype=complex)

    # --- Build Liouvillian superoperator L ---
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))

    # Add dissipators
    for L_op in c_ops:
        Ld = L_op.conj().T
        L += np.kron(L_op, L_op.conj()) \
             - 0.5 * np.kron(I, (Ld @ L_op).T) \
             - 0.5 * np.kron((Ld @ L_op), I)

    # --- Solve L vec(rho) = 0 with Tr(rho)=1 ---
    A = np.copy(L)
    b = np.zeros(N*N, dtype=complex)

    # Replace last equation with trace constraint
    trace_constraint = np.zeros(N*N, dtype=complex)
    for i in range(N):
        trace_constraint[i*N + i] = 1.0  # diagonal elements
    A[-1, :] = trace_constraint
    b[-1] = 1.0

    # Solve linear system
    rho_vec, *_ = np.linalg.lstsq(A, b, rcond=None)
    rho_ss = rho_vec.reshape((N, N))

    # Normalize (numerical safety)
    rho_ss /= np.trace(rho_ss)
    return rho_ss


def lindblad_rhs(t, rho_vec):
    rho = rho_vec.reshape(3,3)
    drho = -1j*(H @ rho - rho @ H)  
    for L in L_:
        drho += L @ rho @ L.conj().T - 0.5*(L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return drho.reshape(9)

t_eval = np.linspace(0,50,500)
sol = solve_ivp(lindblad_rhs, (0,50), rho0_vec, t_eval=t_eval)

rho_t = sol.y.T.reshape(-1,3,3)

Ground = [rho[0,0].real for rho in rho_t]
Excited_h = [rho[1,1].real for rho in rho_t]
Excited_l = [rho[2,2].real for rho in rho_t]


plt.plot(t_eval, Excited_h, label='Excited_h', color='r')
plt.plot(t_eval, Excited_l, label='Excited_l', color='k')
plt.plot(t_eval, Ground, label='Ground', color='b')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()




rho0 =  np.zeros((3,3),dtype=np.complex64)
rho0[0,0]=1
rho0 = steadystate_(H, L_)

N = 51
fmin, fmax = -5, 5

# time grid (only positive times)
dt = 10 / (N * ( (fmax - fmin) / N ))  # = 0.1
times = np.arange(N) * dt             # [0, 0.1, ..., 99.9]


# mu = a_h + a_l + a_h_d + a_l_d
t1 = np.arange(N) * dt+dt   
t2 = np.linspace(0.1,5,num=1)
t3 = np.arange(N) * dt+dt

t_p = [0,10,20]
tresh = 1e-4
def a_p(rho,tp,t):
    # print(tp)
    if abs(t-tp[0])<tresh:
        rho = rho@a_h
    if abs(t-tp[1])<tresh:
        rho = a_l_d@rho
    if abs(t-tp[2])<tresh:
        rho = rho@a_h_d
    return rho

dt = 0.1    

freqs = np.fft.fftfreq(100, d=dt)
# dt = freqs[1] - freqs[0]               # Time step
freqs2 = np.fft.fftfreq(100, d=dt)       # Frequency grid
  
# mu = a_h+a_h_d+a_l+a_l_d
def propagate(rho, t):
    """Propagate density matrix for time t using Lindblad equation."""
    rho_vec = rho.reshape(9)
    
    sol = solve_ivp(lindblad_rhs, (0, t), rho_vec, t_eval=[t])
    # print(t)
    return sol.y[:, -1].reshape(3,3)

def sp_2D(t1_vals, t2_vals, t3_vals, rho0):
    signals = np.zeros((len(t1_vals), len(t3_vals)), dtype=complex)

    for i, t1_ in enumerate(t1_vals):
        for j, t3_ in enumerate(t3_vals):
            rho = rho0@mu            # first interaction
            rho = propagate(rho, t1_)     # evolution t1
            rho = mu@rho             # second interaction
            rho = propagate(rho, t2_vals[0])  # evolution t2 (single value here)
            rho = rho@mu            # third interaction
            rho = propagate(rho, t3_)     # evolution t3
            signals[i,j] = np.trace(mu @ rho)   # detection
            
    return -1j*signals
                                
        
s2d  = sp_2D(t1,t2,t3,rho0)

# -----------------------------
# Fourier transform to frequency domain
# -----------------------------
S2 = np.fft.fftshift(np.fft.fft2(s2d))
w1 = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(len(t1), d=t1[1]-t1[0]))
w3 = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(len(t3), d=t3[1]-t3[0]))


# -----------------------------
# Plot 2D spectrum magnitude
# -----------------------------


plt.pcolormesh(w1, w3,  np.real(S2), shading='auto', cmap='seismic')  # 'seismic' good for spectra
plt.xlabel(r'$\omega_1$ (rad)')
plt.ylabel(r'$\omega_3$ (rad)')
plt.title('TLS R1 2D Spectrum (Lindblad)')
plt.colorbar(label='|R1(ω3,ω1)|')
plt.show()

end = time.time() 
print(f"Execution time: {end - start:.6f} seconds")