import jax.numpy as jnp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import diffrax
import time

start = time.time()
omega = 1.5
omega_l = 2
gamma_h = 0.1
gamma_l = 0.05
dipole_h = 0.1
dipole_l = 0.1


a_h= jnp.zeros((3,3),dtype=jnp.complex64)
a_h = a_h.at[0,1].set(1)
a_h_d = jnp.conjugate(a_h).T

a_l= jnp.zeros((3,3),dtype=jnp.complex64)
a_l =  a_l.at[0,2].set(1)
a_l_d = jnp.conjugate(a_l).T


H_s =  omega*(a_h_d@a_h)+ omega_l*(a_l_d@a_l)
H_d =  dipole_h *(a_h+a_h_d)+dipole_l*(a_l+a_l_d)
H_d = 0
H = H_s+H_d
# Only decay
L_ = [jnp.sqrt(gamma_l)*a_l,jnp.sqrt(gamma_h)*a_h]
    
rho0 =  jnp.zeros((3,3),dtype=jnp.complex64)
rho0 = rho0.at[2,2].set(1)
# rho_0 = steadystate_(H, L_)
rho0_vec = rho0.reshape(9)


def steadystate_(H, c_ops=[], sparse=False):

    N = H.shape[0]
    I = jnp.eye(N, dtype=complex)

    # --- Build Liouvillian superoperator L ---
    L = -1j * (jnp.kron(I, H) - jnp.kron(H.T, I))

    # Add dissipators
    for L_op in c_ops:
        Ld = L_op.conj().T
        L += jnp.kron(L_op, L_op.conj()) \
             - 0.5 * jnp.kron(I, (Ld @ L_op).T) \
             - 0.5 * jnp.kron((Ld @ L_op), I)

    # --- Solve L vec(rho) = 0 with Tr(rho)=1 ---
    A = jnp.copy(L)
    b = jnp.zeros(N*N, dtype=complex)

    # Replace last equation with trace constraint
    trace_constraint = jnp.zeros(N*N, dtype=complex)
    for i in range(N):
        trace_constraint =  trace_constraint.at[i*N + i].set(1.0)  # diagonal elements
    A.at[-1, :].set( trace_constraint)
    b.at[-1].set(1)

    # Solve linear system
    rho_vec, *_ = jnp.linalg.lstsq(A, b, rcond=None)
    rho_ss = rho_vec.reshape((N, N))

    # Normalize (numerical safety)
    rho_ss /= jnp.trace(rho_ss)
    return rho_ss

def lindblad_rhs(t, rho_vec,args):

    rho = rho_vec.reshape(3,3)
    drho = -1j*(H @ rho - rho @ H)  
    for L in L_:
        drho += L @ rho @ L.conj().T - 0.5*(L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return drho.reshape(9)

def lindblad_rhs_h(t, rho_vec,args):

    rho = rho_vec.reshape(3,3)
    
    drho = -1j*((H+dipole_h *(a_h+a_h_d)) @ rho - rho @ (H+dipole_h *(a_h+a_h_d)))  
    for L in L_:
        drho += L @ rho @ L.conj().T - 0.5*(L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return drho.reshape(9)

def lindblad_rhs_l(t, rho_vec,args):

    rho = rho_vec.reshape(3,3)
    drho = -1j*((H+dipole_l *(a_l+a_l_d)) @ rho - rho @ (H+dipole_l *(a_l+a_l_d)))  
    for L in L_:
        drho += L @ rho @ L.conj().T - 0.5*(L.conj().T @ L @ rho + rho @ L.conj().T @ L)
    return drho.reshape(9)

rho_0 = steadystate_(H, L_)
rho0_vec = rho0.reshape(9)

t_eval = jnp.linspace(0,50,500)
solver = diffrax.Tsit5()
term = diffrax.ODETerm(lindblad_rhs)
# ts = jnp.linspace(0,t,int(t/0.1))
saveat = diffrax.SaveAt(ts=t_eval)

sol = diffrax.diffeqsolve(term, solver, 0, 50, dt0=50/500, y0=rho0_vec,saveat=saveat)

rho_t = sol.ys.reshape(-1,3,3)

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



rho0 =  jnp.zeros((3,3),dtype=jnp.complex64)
rho0 = rho0.at[0,0].set(1)

rho_0 = steadystate_(H, L_)


N = 51
fmin, fmax = -5, 5


dt = 5 / (N * ( (fmax - fmin) / N ))  
times = jnp.arange(N) * dt           


mu = a_h + a_l + a_h_d + a_l_d
t1 = jnp.arange(N) * dt+dt   
t2 = jnp.linspace(0.1,5,num=1)
t3 = jnp.arange(N) * dt+dt

t_p = [0,10,20]
tresh = 1e-4

dt = 0.1    

freqs = jnp.fft.fftfreq(100, d=dt)
dt = freqs[1] - freqs[0]               # Time step
freqs2 = jnp.fft.fftfreq(100, d=dt)       # Frequency grid
  
def propagate(rho, t):
    """Propagate density matrix for time t using Lindblad equation."""
    rho_vec = rho.reshape(9)
    term = diffrax.ODETerm(lindblad_rhs)
    ts = jnp.linspace(0,t,int(t/0.1))
    saveat = diffrax.SaveAt(ts=ts)
    
    sol = diffrax.diffeqsolve(term, solver, 0, t, dt0=0.1, y0=rho_vec,saveat=saveat)

    return sol.ys[-1].reshape(3,3)

def propagate_h(rho, t):
    """Propagate density matrix for time t using Lindblad equation."""
    rho_vec = rho.reshape(9)
    term = diffrax.ODETerm(lindblad_rhs_h)
    ts = jnp.linspace(0,t,int(t/0.1))
    saveat = diffrax.SaveAt(ts=ts)
    
    sol = diffrax.diffeqsolve(term, solver, 0, t, dt0=0.1, y0=rho_vec,saveat=saveat)

    return sol.ys[-1].reshape(3,3)


def propagate_l(rho, t):
    """Propagate density matrix for time t using Lindblad equation."""
    rho_vec = rho.reshape(9)
    term = diffrax.ODETerm(lindblad_rhs_l)
    ts = jnp.linspace(0,t,int(t/0.1))
    saveat = diffrax.SaveAt(ts=ts)
    
    sol = diffrax.diffeqsolve(term, solver, 0, t, dt0=0.1, y0=rho_vec,saveat=saveat)

    return sol.ys[-1].reshape(3,3)



def sp_2D(t1_vals, t2_vals, t3_vals, rho0):
    signals = jnp.zeros((len(t1_vals), len(t3_vals)), dtype=complex)

    for i, t1_ in enumerate(t1_vals):
        for j, t3_ in enumerate(t3_vals):
            rho = a_h_d@rho0              
            # rho = propagate(rho,dt,H+dipole_h *(a_h+a_h_d))
            rho = propagate_h(rho, dt) 
            rho = propagate(rho, t1_) 
            
            rho = rho@a_h     
            # rho = propagate(rho,dt,H+dipole_h *(a_h+a_h_d))
            rho = propagate_h(rho, dt)  
            rho = propagate(rho, t2_vals[0]) 
            
            rho = a_l_d@rho
            # rho = propagate(rho,dt,H+dipole_l *(a_l+a_l_d))              
            
            rho = propagate_l(rho, dt)     
            rho = propagate(rho, t3_) 
            
            signals = signals.at[i,j].set(jnp.trace(a_l@rho))  

    return 1j*signals
                                
        
s2d  = sp_2D(t1,t2,t3,rho0)

# -----------------------------
# Fourier transform to frequency domain
# -----------------------------
S2 = jnp.fft.fftshift(jnp.fft.fft2(s2d))
w1 = 2*jnp.pi*jnp.fft.fftshift(jnp.fft.fftfreq(len(t1), d=t1[1]-t1[0]))
w3 = 2*jnp.pi*jnp.fft.fftshift(jnp.fft.fftfreq(len(t3), d=t3[1]-t3[0]))


# -----------------------------
# Plot 2D spectrum magnitude
# -----------------------------


plt.pcolormesh(w1, w3,  jnp.imag(S2), shading='auto', cmap='seismic')  # 'seismic' good for spectra
plt.xlabel(r'$\omega_1$ (rad)')
plt.ylabel(r'$\omega_3$ (rad)')
plt.title('TLS R1 2D Spectrum (Lindblad)')
plt.colorbar(label='|R1(ω3,ω1)|')
plt.show()
end = time.time() 

print(f"Execution time: {end - start:.6f} seconds")