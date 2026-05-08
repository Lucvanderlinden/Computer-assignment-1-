# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:58:02 2026

@author: daisy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

for q0 in q0_values:
    result=velocity_verlet_integrator(q0, harmonic_force, 2e-4)
    if result==None:
        continue
    x,p,t = result
    E=total_energy(p, x)[0]
    energies.append(E)
    trajectories.append((x,p))
    T.append(t[-1])
    
energies = np.array(energies)
norm = (energies - energies.min()) / (energies.max() - energies.min())

plt.figure()

for i, (q, p) in enumerate(trajectories):
    color = cm.turbo(norm[i])
    plt.plot(q, p, color=color)

plt.title('Phase-space orbits, evenly spaced in position')
plt.xlabel('Position, q(t)')
plt.ylabel('Momentum, p(t)')
plt.show()


plt.figure()
plt.plot(energies, T)
plt.ylabel('Time (t)')
plt.xlabel('Energy (E)')
plt.show()

