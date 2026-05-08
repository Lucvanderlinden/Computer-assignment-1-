# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:15:06 2026

@author: daisy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import CubicSpline

def harmonic_potential(rho):
    return 36*(rho - 1)**2 - 1


def harmonic_force(rho):
    return -72*(rho-1)


def velocity_verlet_integrator(
    initial_position, force_function, dt, max_steps=5_000_000
):
    """
    Integrate one closed orbit using the velocity-Verlet algorithm.

    Starting from (q(0), p(0)) = (q0, 0), the update rule is:

        p_{n+1/2} = p_n       + (dt/2) * F(q_n)
        q_{n+1}   = q_n       + dt     * p_{n+1/2}
        p_{n+1}   = p_{n+1/2} + (dt/2) * F(q_{n+1})

    Integration stops when p changes sign for the second time
    (one full oscillation). Returns None if no orbit is found.
    """

    q, p, t = float(initial_position), 0.0, 0.0
    qs, ps, ts = [q], [p], [t]
    p_prev = 0.0
    zero_crossings = 0

    for _ in range(max_steps):
        # velocity verlet step
        p_half = p + 0.5 * dt * force_function(q)
        q = q + dt * p_half
        p = p_half + 0.5 * dt * force_function(q)
        t += dt

        # accumulate values
        qs.append(q)
        ps.append(p)
        ts.append(t)

        # check momentum sign-change
        if len(qs) > 2 and p_prev * p < 0.0:
            zero_crossings += 1
            # break after complete orbit
            if zero_crossings == 2:
                break
        p_prev = p
    else:
        return None

    return np.array(qs), np.array(ps), np.array(ts)


positions, momenta, time = velocity_verlet_integrator(
    initial_position=1.3,
    force_function=harmonic_force,
    dt=2e-4
)

def total_energy(p, x):
    Kin = 0.5*(p**2)
    Pot = harmonic_potential(x)
    T = np.add(Kin,Pot)
    return T

q0_values=np.linspace(1.05,2,100)
energies=[]
trajectories=[]
T=[]
A=[]

for q0 in q0_values:
    result=velocity_verlet_integrator(q0, harmonic_force, 2e-4)
    if result==None:
        continue
    x,p,t = result
    E=total_energy(p, x)[0]
    energies.append(E)
    trajectories.append((x,p))
    T.append(t[-1])
    
    A.append( 0.5 * np.abs(np.sum(x[:-1] * p[1:] - x[1:] * p[:-1])))
    
energies = np.array(energies)

plt.figure()

plt.plot(energies, A)
plt.title('')
plt.ylabel('Area (A)')
plt.xlabel('Energy')
plt.show()

J=np.array(A)/(2*np.pi)
dJ_dE=np.gradient(J,energies)

#for i in range(len(J)-1):
 #   dE = energies[i+1] - energies[i]
  #  dJ = J[i+1] - J[i]
   # dJdE = dJ/dE
    #dJ_dE.append(dJdE)

    
plt.figure()
plt.plot(energies, dJ_dE)
plt.title('multiplicity')
plt.xlabel('Energy, E')
plt.ylabel(r'multiplicity $\Omega(E)$')
plt.savefig('multiplicity_evenly_spaced_energy.png')
plt.show()


